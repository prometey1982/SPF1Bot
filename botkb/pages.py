"""Файловый слой БЗ бота: страницы .md в `bot_kb.dir` (ТЗ п. 7.2).

БЗ — единая глобальная wiki, поэтому директория одна (`wiki/bot_kb/`), а не
по-пользовательская. Вся работа с файлами (атомарная запись, слаги, защита
от path traversal) переиспользуется из `botwiki.pages` без изменений — здесь
только «клей»: корень БЗ, определение `kind` по slug и функции-обёртки,
принимающие корень явно (тест-френдли), по умолчанию — из bot_kb-конфига.
"""

import os

from botwiki import pages as pageio
from botwiki.pages import is_safe_slug, is_thematic_slug, normalize_slug

from . import config

# Служебные (self) страницы — их имена зарезервированы (kind='self').
SERVICE_PAGES = ('Home', 'Style')

PAGE_EXT = '.md'


def kb_root() -> str:
    """Корень БЗ (wiki/bot_kb по умолчанию)."""
    return config.kb_dir()


def kind_for_slug(slug: str) -> str:
    """kind по имени: Home/Style → 'self', всё остальное → 'knowledge'."""
    return 'self' if slug in SERVICE_PAGES else 'knowledge'


def page_path(slug: str, root: str | None = None) -> str | None:
    """Путь к файлу страницы. None — slug небезопасен."""
    return pageio.page_path(root if root is not None else kb_root(), slug)


def page_exists(slug: str, root: str | None = None) -> bool:
    """Существует ли файл страницы."""
    path = page_path(slug, root)
    return bool(path) and os.path.isfile(path)


def read_page(slug: str, root: str | None = None) -> str | None:
    """Читает страницу. None — нет файла или небезопасный slug."""
    return pageio.read_page(root if root is not None else kb_root(), slug)


def write_page(slug: str, content: str, root: str | None = None) -> bool:
    """Атомарная запись страницы (tmp+rename с повторами на блокировки)."""
    return pageio.atomic_write_page(root if root is not None else kb_root(),
                                    slug, content)


def list_slugs(root: str | None = None) -> list[str]:
    """Slug'и существующих .md-страниц (без служебных файлов индекса)."""
    return pageio.list_page_slugs(root if root is not None else kb_root())


def list_pages_with_mtime(root: str | None = None) -> list[tuple[str, float]]:
    """Страницы с mtime их .md-файлов (для кэш-сигнатуры, п. 12.2)."""
    root = root if root is not None else kb_root()
    result = []
    for slug in list_slugs(root):
        path = page_path(slug, root)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        result.append((slug, mtime))
    return result
