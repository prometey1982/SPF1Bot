"""Импорт истории из экспорта Telegram (ТЗ п. 9.8).

Файл экспорта лежит на хосте бота и читается только из `import.allowed_dir`
(проверка по realpath, защита от `..`/симлинков наружу). Строки пишутся в
`user_raw` c `source='export'`, `username=NULL`, `author_name` = отображаемое
имя; дедуп — `UNIQUE(chat_id, message_id)` (INSERT OR IGNORE), повторный импорт
идемпотентен. Одна job импорта одновременно (состояние — в памяти процесса).

После импорта для затронутых пользователей запускается backfill wiki
(bootstrap `from_import` для новых, иначе drain необработанных строк).
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

from . import config
from . import db
from . import manager
from . import redact

logger = logging.getLogger(__name__)

MB = 1024 * 1024


# --- Сериализация/парсинг сообщений экспорта ---

def flatten_export_text(text) -> str | None:
    """`text` — строка или массив сегментов {type,text}; склейка в плоскую строку."""
    if isinstance(text, str):
        return text or None
    if isinstance(text, list):
        parts = []
        for seg in text:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                value = seg.get('text')
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.append(flatten_export_text(value) or '')
        flat = ' '.join(' '.join(parts).split())
        return flat or None
    return None


def extract_export_user_id(value) -> int | None:
    """Числовой Telegram user id из from_id/actor_id/sender_id.

    'user406526542' → 406526542; каналы ('channel…'), анонимы (None) → None.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith('user'):
        rest = text[4:]
        if rest.isdigit():
            return int(rest)
        return None
    if text.lstrip('-').isdigit():
        return int(text)
    return None


def _unixtime_to_sql(ts) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OverflowError):
        return None


def message_content_type(msg: dict) -> str:
    return 'caption' if msg.get('media_type') else 'text'


def build_export_row(import_cfg: dict, msg: dict, *, user_id: int,
                     author_name: str | None, chat_id: int, thread_id) -> dict | None:
    """Сборка строки user_raw для импорта. None — сохранять нечего/запрещено."""
    content = flatten_export_text(msg.get('text'))
    if not content:
        return None
    if import_cfg.get('skip_commands', True) and content.lstrip().startswith('/'):
        return None

    content_type = message_content_type(msg)
    if import_cfg.get('redact', True):
        patterns = import_cfg.get('redact_patterns') or redact.default_patterns(
            {'redact': True})
        content, _ = redact.mask_text(content, patterns)

    max_chars = import_cfg.get('max_content_chars', 4000)
    truncated = 0
    if max_chars and len(content) > max_chars:
        content = content[:max_chars]
        truncated = 1

    message_id = msg.get('id')
    if message_id is None or chat_id is None:
        return None
    return {
        'user_id': user_id,
        'author_name': author_name or None,
        'chat_id': chat_id,
        'thread_id': thread_id,
        'message_id': int(message_id),
        'content': content,
        'content_type': content_type,
        'truncated': truncated,
    }


# --- Маппинг чатов ---

def find_chat_entry(chat_map: list, chat_name: str | None, export_id) -> dict | None:
    """Запись chat_map для экспортного чата (по export_id, затем по name)."""
    if export_id is not None:
        for entry in chat_map:
            if entry.get('export_id') == export_id:
                return entry
    if chat_name:
        for entry in chat_map:
            if entry.get('name') == chat_name:
                return entry
    return None


# --- Безопасность пути (п. 9.8/10.8) ---

def resolve_allowed_path(user_arg: str, allowed_dir: str):
    """Резолвит путь файла строго внутри allowed_dir (realpath).

    Возвращает (абсолютный_путь|None, ошибка|None). Выход через `..`/симлинки
    наружу и абсолютные пути вне allowed_dir отклоняются без чтения.
    """
    if not os.path.isdir(allowed_dir):
        return None, f"Директория allowed_dir не существует: {allowed_dir}"
    allowed_real = os.path.realpath(allowed_dir)

    raw_path = user_arg
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(allowed_dir, raw_path)
    real = os.path.realpath(raw_path)

    def inside(path: str) -> bool:
        try:
            return os.path.commonpath([path, allowed_real]) == allowed_real
        except ValueError:
            return False

    if not inside(real):
        return None, f"Путь вне {allowed_dir}: {user_arg}"
    if not os.path.isfile(real):
        return None, f"Файл не найден: {user_arg}"
    return real, None


# --- Счётчики и импорт ---

class ImportStats:
    def __init__(self):
        self.seen = 0
        self.inserted = 0
        self.duplicates = 0
        self.skipped = 0
        self.skip_reasons: dict[str, int] = {}
        self.users: set[int] = set()
        self.error = None

    def _skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _summary_text(stats: ImportStats) -> str:
    lines = [f"Обработано сообщений: {stats.seen}",
             f"Вставлено: {stats.inserted}", f"Дубли (игнор): {stats.duplicates}",
             f"Пропущено: {stats.skipped}"]
    if stats.skip_reasons:
        reasons = ', '.join(f"{k}={v}" for k, v in stats.skip_reasons.items())
        lines.append(f"Причины пропуска: {reasons}")
    if stats.users:
        lines.append(f"Затронуто пользователей: {len(stats.users)}")
    if stats.error:
        lines.append(f"Ошибка: {stats.error}")
    return '\n'.join(lines)


def perform_import(db_path: str, file_path: str, chat_key: str | int,
                   *, dry_run: bool = False) -> ImportStats:
    """Синхронный импорт одного экспортного файла (вызывается в executor).

    Возвращает ImportStats. При dry_run ничего не пишет.
    """
    settings = config.settings()
    import_cfg = settings.get('import', {})
    stats = ImportStats()

    size_mb = os.path.getsize(file_path) / MB
    max_mb = import_cfg.get('max_file_mb', 200)
    if max_mb and size_mb > max_mb:
        stats.error = f"Файл {size_mb:.1f} МБ больше лимита max_file_mb={max_mb}"
        return stats

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            export = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        stats.error = f"Не удалось прочитать файл: {e}"
        return stats

    chats = (export or {}).get('chats', {}).get('list', [])
    # Выбор экспортного чата по --chat (name или export_id)
    chat_key_int = chat_key if isinstance(chat_key, int) else None
    chat_key_str = None if isinstance(chat_key, int) else str(chat_key)

    export_chat = None
    for c in chats:
        if chat_key_int is not None and c.get('id') == chat_key_int:
            export_chat = c
            break
        if chat_key_str is not None and c.get('name') == chat_key_str:
            export_chat = c
            break
    if export_chat is None:
        names = [f"{c.get('name')!r} (id={c.get('id')})" for c in chats][:20]
        stats.error = (f"Чат {chat_key!r} не найден в экспорте. Доступны: "
                       f"{', '.join(names) or '(нет чатов)'}")
        return stats

    entry = find_chat_entry(import_cfg.get('chat_map', []),
                            export_chat.get('name'), export_chat.get('id'))
    if entry is None:
        stats.error = (f"Чат {export_chat.get('name')!r} (id={export_chat.get('id')}) "
                       "не сопоставлен в import.chat_map. Импорт без маппинга запрещён.")
        return stats
    chat_id = entry.get('chat_id')
    thread_id = entry.get('thread_id')
    if chat_id is None:
        stats.error = f"В chat_map для чата {export_chat.get('name')!r} нет chat_id."
        return stats

    exclude_types = set(import_cfg.get('exclude_types', ['service']))
    rows_to_insert: list = []
    user_batch_rows: list[tuple[dict, str | None]] = []

    for msg in export_chat.get('messages', []):
        stats.seen += 1
        if msg.get('type') in exclude_types:
            stats._skip('type:' + str(msg.get('type')))
            continue
        user_id = (extract_export_user_id(msg.get('from_id'))
                   or extract_export_user_id(msg.get('actor_id'))
                   or extract_export_user_id(msg.get('sender_id')))
        if user_id is None:
            stats._skip('нет user-id автора')
            continue
        author_name = msg.get('from') or msg.get('actor')
        if isinstance(author_name, str):
            author_name = author_name.strip() or None
        else:
            author_name = None
        row = build_export_row(import_cfg, msg, user_id=user_id,
                               author_name=author_name, chat_id=chat_id,
                               thread_id=thread_id)
        if row is None:
            stats._skip('пустой/команда')
            continue
        ts = _unixtime_to_sql(msg.get('date_unixtime'))
        user_batch_rows.append((row, ts))

    if dry_run:
        stats.inserted = len(user_batch_rows)
        stats.users = {r[0]['user_id'] for r in user_batch_rows}
        return stats

    # Вставка батчами, каждая в своей транзакции (on_batch_error)
    batch_size = max(1, int(import_cfg.get('batch_size', 500)))
    on_batch_error = import_cfg.get('on_batch_error', 'stop')
    for start in range(0, len(user_batch_rows), batch_size):
        batch = user_batch_rows[start:start + batch_size]
        try:
            for row, ts in batch:
                inserted = db.insert_export_row(db_path, row, ts)
                if inserted:
                    stats.inserted += 1
                    stats.users.add(row['user_id'])
                else:
                    stats.duplicates += 1
        except Exception as e:
            logger.warning("import: сбой батча (%s): %s", file_path, e)
            stats.error = f"Сбой батча: {e}"
            if on_batch_error == 'stop':
                return stats
            # continue: пропускаем батч (осознанное восстановление)
    return stats


# --- Job-менеджер ---

class ImportJobManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.active = False
        self.started_at: float | None = None
        self.last_text: str | None = None
        self.last_users: list[int] = []
        self.last_ok: bool = False
        self.current_file: str | None = None

    def is_busy(self) -> bool:
        return self.active

    def status_text(self) -> str:
        lines = []
        if self.active:
            lines.append(f"Активная задача: {self.current_file or '?'}")
            elapsed = (time.time() - self.started_at) if self.started_at else 0
            lines.append(f"Выполняется... ({int(elapsed)} c)")
        else:
            lines.append("Активной задачи нет.")
        if self.last_text:
            lines.append("--- Последняя задача ---")
            lines.append(self.last_text)
        return '\n'.join(lines)

    async def run_import(self, file_path: str, chat_key, *, dry_run=False,
                         no_wiki=False) -> str:
        async with self._lock:
            if self.active:
                return 'Импорт уже выполняется. Дождитесь завершения.'
            self.active = True
            self.started_at = time.time()
            self.current_file = os.path.basename(file_path)
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, lambda: perform_import(config.db_path(), file_path,
                                             chat_key, dry_run=dry_run))
            if stats.error and stats.inserted == 0:
                self.last_text = _summary_text(stats)
                self.last_ok = False
                return self.last_text

            if not dry_run and not no_wiki and not stats.error and stats.users:
                users = sorted(stats.users)
                ok_all = True
                notes = []
                for user_id in users:
                    try:
                        ok_all = (await manager.wiki_manager.backfill_user(user_id)) and ok_all
                    except Exception as e:
                        logger.warning("import: backfill user_id=%d ошибка: %s", user_id, e)
                        ok_all = False
                self.last_users = users
                if not ok_all:
                    notes.append('backfill выполнен частично — повторите /reconcile_wiki')

            self.last_text = _summary_text(stats)
            self.last_ok = stats.error is None
            return self.last_text
        finally:
            self.active = False
            self.current_file = None


job_manager = ImportJobManager()
