"""Слой БД для сырья БЗ бота (таблица bot_kb_raw, ТЗ п. 7.1).

Захват — дешёвый синхронный `INSERT OR IGNORE`. Никакого ORM: как и в bot.py,
свежее соединение на вызов. Все функции принимают `db_path` явно
(тест-френдли); bot.py передаёт путь из своего конфига.

Таблица глобальная (единая БЗ на инстанс бота) — в отличие от user_raw здесь
нет разбиения по user_id: watermark один на всю таблицу.
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)

BOT_KB_RAW_DDL = [
    """
    CREATE TABLE IF NOT EXISTS bot_kb_raw (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      speaker TEXT NOT NULL DEFAULT 'human',       -- human | bot
      user_id INTEGER,                              -- автор (для speaker='bot' — id бота)
      username TEXT,                                -- @alias (справочно, только для human)
      author_name TEXT,                             -- отображаемое имя (справочно)
      chat_id INTEGER NOT NULL,
      thread_id INTEGER,
      message_id INTEGER NOT NULL,
      reply_to_message_id INTEGER,                  -- на что реплика/ход
      content TEXT,
      content_type TEXT DEFAULT 'text',             -- text | caption
      truncated INTEGER DEFAULT 0,
      source TEXT NOT NULL DEFAULT 'live',          -- live | export (этап 7)
      inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      ts DATETIME DEFAULT CURRENT_TIMESTAMP         -- время сообщения Telegram (message.date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_botkb_raw_ts ON bot_kb_raw(ts)",
    "CREATE INDEX IF NOT EXISTS idx_botkb_raw_speaker_ts ON bot_kb_raw(speaker, ts)",
    "CREATE INDEX IF NOT EXISTS idx_botkb_raw_reply "
    "ON bot_kb_raw(chat_id, reply_to_message_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_botkb_raw_unique_message "
    "ON bot_kb_raw(chat_id, message_id)",
]

# Колонки, заполняемые при вставке (ts — только если передан явно).
_INSERT_COLUMNS = (
    'speaker', 'user_id', 'username', 'author_name', 'chat_id', 'thread_id',
    'message_id', 'reply_to_message_id', 'content', 'content_type', 'truncated',
    'source',
)


def init_raw_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for statement in BOT_KB_RAW_DDL:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def append_raw(db_path: str, row: dict) -> bool:
    """Вставляет строку bot_kb_raw (INSERT OR IGNORE). Возвращает True, если вставлено.

    `ts` записывается только когда передан не-None (иначе действует DEFAULT
    CURRENT_TIMESTAMP): ретенция/окна считаются по ts, и NULL здесь недопустим.
    """
    values = {key: row.get(key) for key in _INSERT_COLUMNS}
    if values.get('speaker') is None:
        values['speaker'] = 'human'
    if values.get('content_type') is None:
        values['content_type'] = 'text'
    values['truncated'] = int(bool(values.get('truncated', 0)))
    if values.get('source') is None:
        values['source'] = 'live'

    columns = list(_INSERT_COLUMNS)
    ts = row.get('ts')
    if ts is not None:
        columns.append('ts')
        values['ts'] = ts

    conn = connect(db_path)
    try:
        cursor = conn.execute(
            f"""
            INSERT OR IGNORE INTO bot_kb_raw ({', '.join(columns)})
            VALUES ({', '.join(':' + key for key in columns)})
            """,
            values,
        )
        conn.commit()
        inserted = cursor.rowcount > 0
        if not inserted:
            logger.info("bot_kb_raw: строка пропущена (дубль chat_id=%s message_id=%s)",
                        row.get('chat_id'), row.get('message_id'))
        return inserted
    finally:
        conn.close()


def fetch_unprocessed(db_path: str, watermark: int,
                      max_messages: int | None = None,
                      max_chars: int | None = None) -> list[dict]:
    """Снимок необработанных строк (id > watermark) в лимитах снимка (ТЗ 9.3).

    Глобальный (без user_id); берутся первые по id строки, пока не исчерпан
    один из лимитов.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, speaker, user_id, username, author_name, chat_id, thread_id,
                   message_id, reply_to_message_id, content, content_type,
                   truncated, source, inserted_at, ts
            FROM bot_kb_raw
            WHERE id > ?
            ORDER BY id ASC
            """,
            (watermark,),
        ).fetchall()

        result = []
        total_chars = 0
        for row in rows:
            content = row['content'] or ''
            if max_messages is not None and len(result) >= max_messages:
                break
            if max_chars is not None and total_chars + len(content) > max_chars:
                break
            result.append(dict(row))
            total_chars += len(content)
        return result
    finally:
        conn.close()


def watermark(db_path: str) -> int | None:
    """Максимальный id строки bot_kb_raw (для bootstrap/watermark). None, если строк нет."""
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT MAX(id) AS max_id FROM bot_kb_raw").fetchone()
        return row['max_id'] if row is not None else None
    finally:
        conn.close()


def count_rows(db_path: str, speaker: str | None = None, source: str | None = None) -> int:
    conn = connect(db_path)
    try:
        clauses = []
        params: list = []
        if speaker is not None:
            clauses.append("speaker = ?")
            params.append(speaker)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = conn.execute(f"SELECT COUNT(*) AS c FROM bot_kb_raw{where}", params).fetchone()
        return row['c'] if row else 0
    finally:
        conn.close()


def count_unprocessed(db_path: str, watermark: int) -> int:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM bot_kb_raw WHERE id > ?", (watermark,)).fetchone()
        return row['c'] if row else 0
    finally:
        conn.close()


def fetch_window_rows(db_path: str, limit: int) -> list[dict]:
    """Последние `limit` строк в порядке возрастания id (для limited_window/окон)."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, speaker, chat_id, reply_to_message_id, content, ts FROM (
                SELECT id, speaker, chat_id, reply_to_message_id, content, ts
                FROM bot_kb_raw ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_rows_until(db_path: str, upto_id: int, limit: int) -> list[dict]:
    """Последние `limit` строк с id <= upto_id по возрастанию id.

    Используется для построения блоков диалогов self (строки снимка + уже
    обработанный хвост в пределах лимита).
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, speaker, user_id, chat_id, thread_id, message_id,
                   reply_to_message_id, content, content_type, ts
            FROM (
                SELECT id, speaker, user_id, chat_id, thread_id, message_id,
                       reply_to_message_id, content, content_type, ts
                FROM bot_kb_raw WHERE id <= ? ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (upto_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_processed_human_window(db_path: str, watermark: int, limit: int) -> list[dict]:
    """Обработанные (id <= watermark) строки-человек по возрастанию id.

    Окно для создания тематических страниц (п. 9.4): знания извлекаются только
    из сообщений участников, ответы бота в детектор не попадают.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, speaker, chat_id, content, ts FROM (
                SELECT id, speaker, chat_id, content, ts FROM bot_kb_raw
                WHERE speaker = 'human' AND id <= ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (watermark, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_row_by_message(db_path: str, chat_id: int, message_id: int) -> dict | None:
    """Строку по (chat_id, message_id) — любую (human/bot). None — нет."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT id, speaker, chat_id, thread_id, message_id,
                   reply_to_message_id, content, ts
            FROM bot_kb_raw WHERE chat_id = ? AND message_id = ?
            """,
            (chat_id, message_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def fetch_rows_by_messages(db_path: str, chat_id: int,
                           message_ids: list[int]) -> dict[int, dict]:
    """Строки по (chat_id, message_ids) как {message_id: row} (батчево).

    Используется для проверки наличия/подтягивания родительских сообщений
    (знание из ответов бота, K2a) без точечных запросов на каждый ход.
    """
    if not message_ids:
        return {}
    conn = connect(db_path)
    try:
        placeholders = ','.join('?' * len(message_ids))
        rows = conn.execute(
            f"""
            SELECT id, speaker, chat_id, thread_id, message_id,
                   reply_to_message_id, content, content_type, ts
            FROM bot_kb_raw
            WHERE chat_id = ? AND message_id IN ({placeholders})
            """,
            [chat_id] + list(message_ids),
        ).fetchall()
        return {r['message_id']: dict(r) for r in rows}
    finally:
        conn.close()


def fetch_bot_turn(db_path: str, chat_id: int, target_message_id: int) -> list[dict]:
    """Ход бота: строки-бот (куски одного ответа) с общим reply_to_message_id.

    Связка строго в пределах chat_id (п. 7.1/9.3.3): численное совпадение
    reply_to_message_id из другого чата не даёт ложной связки.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, speaker, chat_id, thread_id, message_id,
                   reply_to_message_id, content, ts
            FROM bot_kb_raw
            WHERE chat_id = ? AND speaker = 'bot' AND reply_to_message_id = ?
            ORDER BY id ASC
            """,
            (chat_id, target_message_id),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_replies_to(db_path: str, chat_id: int, message_ids: list[int]) -> list[dict]:
    """Строки-человек, реплики на любые из message_ids (в пределах чата).

    «Обратная связь» (п. 2): реплики людей на сообщения/куски бота.
    """
    if not message_ids:
        return []
    conn = connect(db_path)
    try:
        placeholders = ','.join('?' * len(message_ids))
        rows = conn.execute(
            f"""
            SELECT id, speaker, user_id, chat_id, thread_id, message_id,
                   reply_to_message_id, content, content_type, ts
            FROM bot_kb_raw
            WHERE chat_id = ? AND speaker = 'human'
              AND reply_to_message_id IN ({placeholders})
            ORDER BY id ASC
            """,
            [chat_id] + message_ids,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def tail_watermark(db_path: str, keep_newest: int) -> int:
    """Watermark, оставляющий непокрытыми ровно `keep_newest` самых новых строк.

    Считается по ЧИСЛУ строк, а не арифметикой id (в автоинкременте возможны
    дырки от INSERT OR IGNORE-дублей и будущего импорта, п. 9.7). Если строк
    <= keep_newest — возвращает 0 (непокрыты все); иначе — id строки, стоящей
    на (keep_newest+1)-й позиции от самой новой.
    """
    conn = connect(db_path)
    try:
        if keep_newest <= 0:
            return 0
        row = conn.execute(
            "SELECT id FROM bot_kb_raw ORDER BY id DESC LIMIT 1 OFFSET ?",
            (int(keep_newest),),
        ).fetchone()
        return row['id'] if row is not None else 0
    finally:
        conn.close()


def delete_all_rows(db_path: str) -> int:
    """Удаляет все строки bot_kb_raw (/kb_clear). Возвращает число удалённых."""
    conn = connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM bot_kb_raw")
        conn.commit()
        return cursor.rowcount or 0
    finally:
        conn.close()


