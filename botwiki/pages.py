"""Файловый слой wiki: страницы .md и слаги (ТЗ п. 7.2, 8.2).

Структура директории пользователя:
    wiki/<user_id>/                # корень из конфига wiki.dir
      _index.yaml
      _index.yaml.bak
      Home.md
      Style.md
      <slug>.md
      ...

Слаги:
- «IO-безопасный»: ^[A-Za-z0-9_-]{1,64}$ — допускается для имени файла и
  записей индекса (Home/Style пишутся с заглавной буквы).
- «Тематический» (создание по правилам п. 9.4): ^[a-z0-9_-]{1,64}$ + запрет
  служебных имён (Home, Style, _index и т.п.). Проверяется отдельным
  предикатом; валидацию предложений делает менеджер (этап 3+).

Никаких вложенных путей/`..`/спецсимволов: это единственная защита от
path traversal при чтении файлов страниц по slug из админских команд.
"""

import os
import re
import logging
import time

from . import config

logger = logging.getLogger(__name__)

# Сколько раз и с какой задержкой повторять финальный переименовывающий шаг
# записи при транзиентных блокировках файла (Windows: антивирус/индексатор,
# WinError 5 «Отказано в доступе»).
_RENAME_ATTEMPTS = 6
_RENAME_BASE_DELAY = 0.3

# IO-безопасный slug: пригоден как имя файла и как значение в _index.yaml.
SLUG_IO_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
# Тематический slug (lowercase) — только для вновь создаваемых страниц.
SLUG_THEMATIC_RE = re.compile(r'^[a-z0-9_-]{1,64}$')

# Имена, запрещённые для тематических страниц (служебные).
RESERVED_NAMES = {
    'home', 'style', '_index', 'index', '_index.yaml', 'index.yaml',
    '_index.yaml.bak', 'config',
}

PAGE_EXT = '.md'


def is_safe_slug(slug) -> bool:
    """IO-безопасность slug (filename-safe). Принимает и Home/Style."""
    return isinstance(slug, str) and bool(SLUG_IO_RE.match(slug))


def is_thematic_slug(slug) -> bool:
    """Slug пригоден для создания тематической страницы (п. 7.2, 9.4)."""
    return (isinstance(slug, str) and bool(SLUG_THEMATIC_RE.match(slug))
            and slug.lower() not in RESERVED_NAMES)


def normalize_slug(slug) -> str | None:
    """Приводит небезопасный/некорректный slug к нижнему регистру; None — нельзя использовать."""
    if not isinstance(slug, str):
        return None
    candidate = slug.strip().lower()
    if not is_thematic_slug(candidate):
        return None
    return candidate


def user_wiki_dir(user_id: int) -> str:
    """Абсолютный/относительный путь к директории wiki пользователя."""
    return os.path.join(config.wiki_dir(), str(user_id))


def page_path(user_dir: str, slug: str) -> str | None:
    """Путь к файлу страницы. None — slug небезопасен."""
    if not is_safe_slug(slug):
        logger.warning("page: небезопасный slug %r отклонён", slug)
        return None
    return os.path.join(user_dir, slug + PAGE_EXT)


def page_exists(user_dir: str, slug: str) -> bool:
    path = page_path(user_dir, slug)
    return bool(path) and os.path.isfile(path)


def read_page(user_dir: str, slug: str) -> str | None:
    """Читает содержимое страницы. None — нет файла или небезопасный slug."""
    path = page_path(user_dir, slug)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError as e:
        logger.warning("page: не удалось прочитать %s: %s", path, e)
        return None


def _replace_with_retry(tmp_path: str, path: str) -> bool:
    """os.replace(tmp, path) с повторами на транзиентные OSError (блокировки)."""
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            os.replace(tmp_path, path)
            return True
        except OSError as e:
            if attempt < _RENAME_ATTEMPTS - 1:
                delay = _RENAME_BASE_DELAY * (attempt + 1)
                logger.warning("page: повторить замену %s (попытка %d/%d): %s",
                               path, attempt + 1, _RENAME_ATTEMPTS, e)
                time.sleep(delay)
            else:
                logger.warning("page: не удалось заменить %s после %d попыток: %s",
                               path, _RENAME_ATTEMPTS, e)
    return False


def atomic_write_page(user_dir: str, slug: str, content: str) -> bool:
    """Атомарная запись страницы (tmp + rename в той же директории).

    Возвращает True при успехе; при OSError (в т.ч. транзиентные блокировки
    файла на Windows) — повторы; после исчерпания логирует и возвращает False.
    """
    path = page_path(user_dir, slug)
    if not path:
        return False
    tmp_path = os.path.join(user_dir, f'.{slug}{PAGE_EXT}.tmp')
    try:
        os.makedirs(user_dir, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        ok = _replace_with_retry(tmp_path, path)
        if ok:
            return True
    except OSError as e:
        logger.warning("page: не удалось записать %s: %s", path, e)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return False


def list_page_slugs(user_dir: str) -> list[str]:
    """Slug'и существующих страниц (.md) в директории, отсортированные.

    Нечитаемые/небезопасные имена (служебные файлы, мусор) пропускаются с логом.
    """
    if not os.path.isdir(user_dir):
        return []
    slugs = []
    try:
        for name in os.listdir(user_dir):
            if not name.endswith(PAGE_EXT):
                continue
            slug = name[:-len(PAGE_EXT)]
            if is_safe_slug(slug):
                slugs.append(slug)
            else:
                logger.warning("page: пропущен файл с небезопасным именем: %s", name)
    except OSError as e:
        logger.warning("page: не удалось прочитать директорию %s: %s", user_dir, e)
        return []
    return sorted(slugs)
