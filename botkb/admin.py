"""Admin-команды БЗ `/kb_*` (ТЗ п. 13).

Функции возвращают готовый текст для отправки админу (приватные чаты; проверку
is_admin делает bot.py). /kb_* не трогают user-wiki/dossier. Тяжёлые операции
(reconcile, merge) делегируются глобальному KBManager.
"""

import os
import logging

from . import config
from . import db
from . import index as index_mod
from . import pages as pageio
from . import inject as inject_mod

logger = logging.getLogger(__name__)


def _manager():
    """Глобальный KBManager пакета (определяется в __init__ после импорта)."""
    from . import kb_manager  # ленивый импорт: пакет уже инициализирован
    return kb_manager

SERVICE_LABELS = {'Home': 'О боте', 'Style': 'Стиль'}


def _db() -> str:
    return config.db_path()


def status_text() -> str:
    """/kb_status: индекс, режим, watermark, страницы, raw, конвейер (п. 13)."""
    root = pageio.kb_root()
    db_path = _db()
    mode = config.mode()
    index_data, status = index_mod.ensure_index(root, db_path)
    if index_data is None:
        return (f"БЗ бота: индекса нет (status={status}). Режим: {mode}. "
                "Bootstrap не выполнен (capture_only или нет сырья).")

    watermark = index_data.get('watermark', 0)
    message_count = index_data.get('message_count', 0)
    lines = [
        f"БЗ бота ({root})",
        f"Индекс: валиден (status={status}), режим: {mode}",
        f"watermark: {watermark}",
        f"message_count (с последнего reconcile): {message_count}",
        f"last_update: {index_data.get('last_update') or '—'}",
        f"last_reconcile: {index_data.get('last_reconcile') or '—'}",
        f"last_error: {index_data.get('last_error') or '—'}",
    ]
    kn = config.settings().get('knowledge', {})
    lines.append(
        f"Обучение из ответов бота: bot_turns={kn.get('bot_turns', False)} "
        f"(мин. длина хода {kn.get('bot_turn_min_chars', 120)}; "
        f"skip-фраз: {len(kn.get('bot_turn_skip_phrases') or [])})")
    lines.append(
        f"Сырьё bot_kb_raw: всего={db.count_rows(db_path)}, "
        f"human={db.count_rows(db_path, speaker='human')}, "
        f"bot={db.count_rows(db_path, speaker='bot')}, "
        f"необработанных={db.count_unprocessed(db_path, watermark)}",
    )
    lines.append("Страницы:")
    if not index_data.get('pages'):
        lines.append("  (нет)")
    for page in sorted(index_data.get('pages', []), key=lambda p: p.get('slug', '')):
        slug = page.get('slug', '')
        content = pageio.read_page(slug, root) or ''
        markers = []
        if page.get('quarantined'):
            markers.append('карантин')
        if page.get('last_error'):
            markers.append(f"ошибка: {str(page.get('last_error'))[:60]}")
        lines.append(
            f"  {slug} [{page.get('kind')}] {page.get('status')} "
            f"({len(content)} симв.){' (' + ', '.join(markers) + ')' if markers else ''}")
    # Состояние конвейера (пауза/backoff/счётчик сбоев — этап 6; в рантайме не ведётся)
    lines.append("Конвейер: пауза — нет; backoff — нет; счётчик сбоев — не ведётся (этап 6)")
    return "\n".join(lines)


def show_text() -> str:
    """/kb_show: активные страницы одним документом."""
    root = pageio.kb_root()
    index_data, _ = index_mod.ensure_index(root, _db())
    if index_data is None:
        return 'БЗ бота ещё не создана (bootstrap не выполнен).'
    parts = []
    for page in sorted(index_data.get('pages', []), key=lambda p: (p.get('status') != 'active', p.get('slug', ''))):
        if page.get('status') != 'active':
            continue
        slug = page.get('slug', '')
        content = pageio.read_page(slug, root)
        if not content:
            continue
        label = SERVICE_LABELS.get(slug) or page.get('title') or slug
        parts.append(f"=== {slug} ({page.get('kind')}) ===\n{content}")
    return "\n\n".join(parts) if parts else 'Активных страниц нет.'


def show_page_text(slug: str) -> str:
    """/kb_show_page <slug>: одна страница."""
    if not pageio.is_safe_slug(slug):
        return 'Небезопасный slug.'
    root = pageio.kb_root()
    content = pageio.read_page(slug, root)
    if content is None:
        return f'Страница «{slug}» не найдена.'
    return f"=== {slug} ===\n{content}"


async def reconcile_text(slug: str | None = None) -> str:
    """/kb_reconcile [<slug>]: ручной reconcile (снимает карантин выбранных)."""
    result = await _manager().reconcile(slug=slug, manual=True, reason='ручной')
    prefix = '✅ ' if result.get('success') else '❌ '
    return prefix + result.get('message', 'Reconcile не выполнен.')


async def merge_text(slug1: str, slug2: str) -> str:
    """/kb_merge <slug1> <slug2>: слияние страниц знаний slug2 → slug1."""
    result = await _manager().merge_pages(slug1, slug2)
    prefix = '✅ ' if result.get('ok') else '❌ '
    return prefix + result.get('message', 'Слияние не выполнено.')


def clear_text(confirm: bool) -> str:
    """/kb_clear: удалить файлы БЗ и строки bot_kb_raw (только с confirm)."""
    if not confirm:
        return ('Удалит все файлы БЗ (wiki/bot_kb) и строки bot_kb_raw. '
                'Подтвердите: /kb_clear confirm')
    root = pageio.kb_root()
    removed_files = 0
    for slug in pageio.list_slugs(root):
        path = pageio.page_path(slug, root)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
                removed_files += 1
            except OSError as e:
                logger.warning("kb_clear: не удалось удалить %s: %s", path, e)
    for name in (index_mod.INDEX_FILE, index_mod.INDEX_BAK):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed_files += 1
            except OSError as e:
                logger.warning("kb_clear: не удалось удалить %s: %s", path, e)
    try:
        removed_rows = db.delete_all_rows(_db())
    except Exception as e:
        removed_rows = 0
        logger.warning("kb_clear: ошибка очистки bot_kb_raw: %s", e)
    inject_mod.invalidate_cache(root)  # кэш горячего пути больше не действителен
    return f'БЗ очищена: удалено файлов={removed_files}, строк={removed_rows}.'
