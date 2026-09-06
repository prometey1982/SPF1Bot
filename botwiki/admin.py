"""Команды admin для wiki (ТЗ п. 13): тексты/действия без Telegram-клея.

bot.py регистрирует CommandHandler'ы и вызывает эти функции, формируя ответ.
Доступ (is_admin + приватный чат) проверяется в bot.py.
"""

import logging
import os
import shutil

from . import config
from . import db
from . import index as index_mod
from . import pages as pageio

logger = logging.getLogger(__name__)

_SERVICE_TITLES = {'Home': 'Сводка', 'Style': 'Стиль'}


def show_wiki(db_path: str, user_id: int) -> str | None:
    """Активные страницы wiki одним markdown-документом. None — wiki нет."""
    user_dir = pageio.user_wiki_dir(user_id)
    index_data, status = index_mod.ensure_index(user_dir, db_path, user_id)
    if index_data is None:
        return None
    sections = []
    actives = [p for p in index_data.get('pages', []) if p.get('status') == 'active']
    actives.sort(key=lambda p: (p['slug'] not in ('Home', 'Style'), p['slug']))
    for page in actives:
        slug = page['slug']
        md = pageio.read_page(user_dir, slug) or ''
        title = page.get('title') or _SERVICE_TITLES.get(slug, slug)
        sections.append(f"## {title} (/{slug})\n\n{md.strip()}\n")
    if not sections:
        return f"Wiki пользователя {user_id} активных страниц не содержит."
    return "\n".join(sections).strip()


def show_wiki_page(db_path: str, user_id: int, slug: str) -> str:
    user_dir = pageio.user_wiki_dir(user_id)
    index_data, status = index_mod.ensure_index(user_dir, db_path, user_id)
    if index_data is None:
        return f"У пользователя {user_id} нет валидной wiki."
    if index_mod.find_page(index_data, slug) is None:
        return f"Страница «{slug}» не найдена."
    md = pageio.read_page(user_dir, slug)
    if md is None:
        return f"Файл страницы «{slug}» отсутствует."
    return md.strip()


def wiki_status(db_path: str, user_id: int) -> str:
    user_dir = pageio.user_wiki_dir(user_id)
    index_data, status = index_mod.ensure_index(user_dir, db_path, user_id)
    if index_data is None:
        return (f"Wiki пользователя {user_id}: отсутствует (статус: {status}). "
                f"Захват продолжается, в ответах — dossier-фолбэк.")

    lines = [
        f"Wiki пользователя {user_id}",
        f"Индекс: валиден (источник: {status})",
        f"watermark: {index_data.get('watermark', 0)}",
        f"message_count: {index_data.get('message_count', 0)}",
        f"last_update: {index_data.get('last_update') or '—'}",
        f"last_reconcile: {index_data.get('last_reconcile') or '—'}",
        f"last_error: {index_data.get('last_error') or '—'}",
    ]
    uncovered = db.count_uncovered_export(db_path, user_id, index_data.get('watermark', 0))
    lines.append(f"Непокрытые export-строки: {uncovered}")

    lines.append("Страницы:")
    for page in index_data.get('pages', []):
        slug = page['slug']
        path = pageio.page_path(user_dir, slug)
        size = os.path.getsize(path) if path and os.path.isfile(path) else 0
        lines.append(
            f"- /{slug} [{page.get('status')}] ({size} б) "
            f"обновлена {page.get('updated') or '—'}"
            f"{', seen ' + page['last_seen'] if page.get('last_seen') else ''}")
    return "\n".join(lines)


def clear_wiki(db_path: str, user_id: int) -> str:
    """Удаляет wiki-файлы + user_raw + связанные mentions (dossier не трогает)."""
    user_dir = pageio.user_wiki_dir(user_id)
    # Алиасы нужны до удаления user_raw (они хранятся в его строках)
    aliases = db.username_aliases(db_path, user_id)
    deleted_raw = db.delete_raw_all(db_path, user_id)
    deleted_mentions = db.delete_mentions_for_aliases(db_path, aliases)

    removed_dir = False
    if os.path.isdir(user_dir):
        try:
            shutil.rmtree(user_dir, ignore_errors=True)
            removed_dir = True
        except OSError as e:
            logger.warning("clear_wiki: не удалось удалить %s: %s", user_dir, e)

    return (f"Wiki пользователя {user_id} очищена: сырьё={deleted_raw} строк, "
            f"упоминания={deleted_mentions}, файлы={'удалены' if removed_dir else '—'}. "
            f"USER_INFO.dossier не тронут.")
