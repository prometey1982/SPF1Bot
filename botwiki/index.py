"""Индекс wiki `_index.yaml` (схема v3): загрузка, валидация, бэкап, восстановление.

ТЗ п. 7.2 (структура/поля), 8.2 (валидация/.bak/пересбор), 8.4 (атомарная запись).

Правила:
- перед каждой заменой `_index.yaml` предыдущая версия сохраняется в `.bak`
  (atomic: tmp + rename в одной директории);
- при невалидном/отсутствующем индексе: восстановление из `.bak` → пересбор по
  `.md`-файлам (watermark = MAX(id) user_raw, message_count=0, last_error) →
  None (фолбэк на dossier);
- расхождение файлов и индекса (п. 8.2): файл без записи → добавляется;
  запись без файла → status: archived (Home/Style не архивируются автоматически).
"""

import os
import logging
from datetime import datetime, timezone

import yaml

from . import config, db
from . import pages as pageio

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

INDEX_FILE = '_index.yaml'
INDEX_BAK = '_index.yaml.bak'
INDEX_TMP = '_index.yaml.tmp'

PAGE_STATUSES = ('active', 'archived')
SERVICE_PAGES = ('Home', 'Style')
DEFAULT_TITLES = {'Home': 'Сводка', 'Style': 'Стиль'}


def now_iso() -> str:
    """UTC timestamp в формате _index.yaml (naive)."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def index_path(user_dir: str) -> str:
    return os.path.join(user_dir, INDEX_FILE)


def bak_path(user_dir: str) -> str:
    return os.path.join(user_dir, INDEX_BAK)


def new_index() -> dict:
    return {
        'schema_version': SCHEMA_VERSION,
        'watermark': 0,
        'message_count': 0,
        'last_update': None,
        'last_reconcile': None,
        'last_error': None,
        'page_proposal_cooldowns': {},
        'pages': [],
    }


def _minimal_page(slug: str) -> dict:
    """Минимальная запись страницы (используется при пересборе/синхронизации)."""
    return {
        'slug': slug,
        'title': DEFAULT_TITLES.get(slug, slug),
        'status': 'active',
        'keywords': [],
        'aliases': [],
        'created': now_iso(),
        'updated': now_iso(),
        'last_seen': None,
        'hits': 0,
    }


# --- Валидация ---

def validate_index(data) -> list[str]:
    """Возвращает список ошибок валидности индекса (пустой — валиден)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ['_index.yaml: не является словарём']

    if data.get('schema_version') != SCHEMA_VERSION:
        errors.append(f"schema_version: ожидается {SCHEMA_VERSION}, получено {data.get('schema_version')!r}")
    for key in ('watermark', 'message_count'):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{key}: ожидается целое число >= 0, получено {value!r}")

    cooldowns = data.get('page_proposal_cooldowns')
    if cooldowns is not None and not isinstance(cooldowns, dict):
        errors.append("page_proposal_cooldowns: ожидается словарь")

    pages = data.get('pages')
    if not isinstance(pages, list):
        errors.append("pages: ожидается список")
        return errors

    seen: set = set()
    for i, page in enumerate(pages):
        prefix = f"pages[{i}]"
        if not isinstance(page, dict):
            errors.append(f"{prefix}: не является словарём")
            continue
        slug = page.get('slug')
        if not isinstance(slug, str) or not pageio.is_safe_slug(slug):
            errors.append(f"{prefix}: небезопасный slug {slug!r}")
        elif slug in seen:
            errors.append(f"{prefix}: дублирующийся slug {slug!r}")
        seen.add(slug)

        if page.get('status') not in PAGE_STATUSES:
            errors.append(f"{prefix}: status должен быть в {PAGE_STATUSES}, получено {page.get('status')!r}")
        if not isinstance(page.get('title'), str):
            errors.append(f"{prefix}: title обязателен (строка)")
        for ts_key in ('created', 'updated', 'last_seen'):
            value = page.get(ts_key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{prefix}.{ts_key}: ожидается строка или null, получено {value!r}")
        if not isinstance(page.get('hits', 0), int):
            errors.append(f"{prefix}.hits: ожидается целое число")
        for key in ('keywords', 'aliases'):
            value = page.get(key, [])
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                errors.append(f"{prefix}.{key}: ожидается список строк")
    return errors


def is_usable_index(data) -> bool:
    return not validate_index(data)


def service_pages_present(user_dir: str) -> bool:
    """Существуют ли файлы Home.md и Style.md (необходимо для валидности wiki)."""
    return all(pageio.page_exists(user_dir, slug) for slug in SERVICE_PAGES)


def wiki_valid(db_path: str, user_id: int) -> bool:
    """«Валидная wiki» (ТЗ 6.2): валидный/восстановимый индекс + Home.md и Style.md.

    Используется в горячем пути ответа (решение dossier vs wiki), поэтому без
    записи на диск: ensure_index read-only + проверка наличия файлов.
    """
    user_dir = pageio.user_wiki_dir(user_id)
    index_data, status = ensure_index(user_dir, db_path, user_id)
    if index_data is None:
        return False
    return service_pages_present(user_dir)


def find_page(index: dict, slug: str) -> dict | None:
    return next((p for p in index.get('pages', []) if p.get('slug') == slug), None)


# --- Чтение ---

def _read_yaml_file(path: str):
    """Читает YAML-файл. Возвращает (значение|None, причина|None).

    причина 'missing' — файла нет; 'error' — не прочитался/не распарсился.
    """
    if not os.path.isfile(path):
        return None, 'missing'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        value = yaml.safe_load(text)
        return value, None
    except (OSError, yaml.YAMLError) as e:
        logger.warning("index: не удалось прочитать %s: %s", path, e)
        return None, 'error'


def load_index(user_dir: str):
    """Читает и валидирует _index.yaml. (index|None, errors:list[str])."""
    data, reason = _read_yaml_file(index_path(user_dir))
    if reason == 'missing':
        return None, ['файл отсутствует']
    if reason == 'error':
        return None, ['ошибка чтения/парсинга']
    errors = validate_index(data)
    if errors:
        return None, errors
    return data, []


def load_bak(user_dir: str):
    """Читает и валидирует _index.yaml.bak. (index|None, errors)."""
    data, reason = _read_yaml_file(bak_path(user_dir))
    if reason == 'missing':
        return None, ['бэкап отсутствует']
    if reason == 'error':
        return None, ['ошибка чтения/парсинга .bak']
    errors = validate_index(data)
    if errors:
        return None, errors
    return data, []


# --- Синхронизация файлов и индекса (п. 8.2) ---

def sync_index_with_files(user_dir: str, index: dict) -> dict:
    """Приводит индекс в соответствие с .md-файлами (на копии).

    Файл без записи → добавляется; запись (не Home/Style) без файла → archived.
    Home/Style без файла не архивируются — их отсутствие обрабатывает bootstrap.
    """
    result = {
        'pages': list(index.get('pages', [])),
        'page_proposal_cooldowns': dict(index.get('page_proposal_cooldowns', {}) or {}),
    }
    for key, value in index.items():
        if key not in result:
            result[key] = value

    file_slugs = set(pageio.list_page_slugs(user_dir))
    index_slugs = {p.get('slug') for p in result['pages']}

    for page in result['pages']:
        slug = page.get('slug')
        if slug in file_slugs:
            continue
        if slug in SERVICE_PAGES:
            logger.warning("index: отсутствует файл служебной страницы %s.md", slug)
            continue
        if page.get('status') != 'archived':
            logger.info("index: страница %s без файла → archived", slug)
            page['status'] = 'archived'

    for slug in sorted(file_slugs - index_slugs):
        logger.info("index: файл %s.md без записи → добавлен в индекс", slug)
        result['pages'].append(_minimal_page(slug))

    return result


# --- Восстановление / пересбор ---

def rebuild_from_files(user_dir: str, db_path: str, user_id: int) -> dict:
    """Пересбор индекса по .md-файлам (п. 8.2, шаг 2). Ничего не пишет на диск."""
    index = new_index()
    index['watermark'] = db.watermark(db_path, user_id) or 0
    index['message_count'] = 0
    index['last_error'] = 'index rebuilt without backup'
    for slug in pageio.list_page_slugs(user_dir):
        index['pages'].append(_minimal_page(slug))
    return index


def ensure_index(user_dir: str, db_path: str, user_id: int):
    """Гарантирует валидный индекс. Возвращает (index|None, status).

    status: ok | restored_bak | rebuilt | missing (нет ни индекса, ни .bak, ни страниц).
    Не пишет на диск (read-only): сохранение выполняет вызывающий.
    """
    index, errors = load_index(user_dir)
    if index is not None:
        return sync_index_with_files(user_dir, index), 'ok'

    bak, bak_errors = load_bak(user_dir)
    if bak is not None:
        logger.info("index: восстановлен из .bak (user=%s): %s", user_id, errors)
        return sync_index_with_files(user_dir, bak), 'restored_bak'

    if pageio.list_page_slugs(user_dir):
        logger.info("index: пересбор по файлам (user=%s): %s", user_id, errors)
        return rebuild_from_files(user_dir, db_path, user_id), 'rebuilt'

    logger.warning("index: нет ни индекса, ни бэкапа, ни страниц (user=%s): %s",
                   user_id, errors or bak_errors)
    return None, 'missing'


# --- Запись (атомарная, с бэкапом) ---

def save_index(user_dir: str, index: dict) -> bool:
    """Пишет _index.yaml атомарно (tmp+rename); перед заменой — .bak.

    Возвращает True при успехе. Индекс валидируется перед записью; невалидный
    индекс не сохраняется (watermark двигается только после успешного батча).
    """
    errors = validate_index(index)
    if errors:
        logger.error("index: не сохраняю невалидный индекс (%s): %s", user_dir, errors)
        return False

    try:
        os.makedirs(user_dir, exist_ok=True)
        current = index_path(user_dir)
        tmp = os.path.join(user_dir, INDEX_TMP)

        with open(tmp, 'w', encoding='utf-8') as f:
            yaml.safe_dump(index, f, allow_unicode=True, default_flow_style=False,
                           sort_keys=False)
        if os.path.exists(current):
            if not pageio._replace_with_retry(current, bak_path(user_dir)):
                raise OSError(f"не удалось создать бэкап {bak_path(user_dir)}")
        if not pageio._replace_with_retry(tmp, current):
            raise OSError(f"не удалось заменить {current}")
        return True
    except OSError as e:
        logger.error("index: не удалось сохранить %s: %s", index_path(user_dir), e)
        try:
            if os.path.exists(os.path.join(user_dir, INDEX_TMP)):
                os.remove(os.path.join(user_dir, INDEX_TMP))
        except OSError:
            pass
        return False


# --- Обнаружение пользователей с wiki (для ретенции, п. 7.1/2.8) ---

def discover_watermarks(db_path: str, base_dir: str | None = None) -> dict[int, int]:
    """user_id → watermark для пользователей с восстановимым индексом.

    «Восстановимый» = _index.yaml валиден, либо восстановлен из .bak, либо
    пересобран с предсказуемым watermark (статусы ok/restored_bak/rebuilt).
    Пользователи без wiki сюда не попадают → ретенция применяет no-wiki-политику.
    """
    base = base_dir if base_dir is not None else config.wiki_dir()
    result: dict[int, int] = {}
    if not os.path.isdir(base):
        return result
    try:
        entries = sorted(os.listdir(base))
    except OSError as e:
        logger.warning("index: не удалось прочитать %s: %s", base, e)
        return result

    for name in entries:
        if not name.isdigit():
            continue
        user_dir = os.path.join(base, name)
        if not os.path.isdir(user_dir):
            continue
        try:
            user_id = int(name)
        except ValueError:
            continue
        index, status = ensure_index(user_dir, db_path, user_id)
        if index is not None:
            result[user_id] = index.get('watermark', 0)
    return result
