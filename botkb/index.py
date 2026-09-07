"""Индекс БЗ бота `wiki/bot_kb/_index.yaml` (схема v3, kind): загрузка, валидация,
бэкап, восстановление.

ТЗ п. 7.2 (структура/поля/kind), 8.2 (валидация/.bak/пересбор), 8.4 (атомарная
запись). Индекс один на весь инстанс (глобальная БЗ), как и таблица bot_kb_raw,
поэтому все функции работают с одним корнем, а watermark берётся из глобальной
таблицы.

Правила — как в botwiki.index, плюс `kind`:
- записи несут kind: self (только Home/Style) или knowledge (тематические);
- перед заменой `_index.yaml` предыдущая версия уходит в `.bak` (atomic);
- при невалидном/отсутствующем индексе: из `.bak` → пересбор по `.md` (watermark
  = MAX(id) bot_kb_raw, message_count=0) → None;
- расхождение файлов и индекса: файл без записи → добавляется (kind по имени);
  запись без файла → status: archived (Home/Style не архивируются автоматически).
"""

import os
import logging
from datetime import datetime, timezone

import yaml

from botwiki.pages import _replace_with_retry

from . import config, db
from . import pages as pageio

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

INDEX_FILE = '_index.yaml'
INDEX_BAK = '_index.yaml.bak'
INDEX_TMP = '_index.yaml.tmp'

PAGE_STATUSES = ('active', 'archived')
KINDS = ('self', 'knowledge')
DEFAULT_TITLES = {'Home': 'О боте', 'Style': 'Стиль'}


def now_iso() -> str:
    """UTC timestamp в формате _index.yaml (naive)."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def index_path(root: str) -> str:
    return os.path.join(root, INDEX_FILE)


def bak_path(root: str) -> str:
    return os.path.join(root, INDEX_BAK)


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
    """Минимальная запись страницы (пересбор/синхронизация). kind — по имени."""
    return {
        'slug': slug,
        'title': DEFAULT_TITLES.get(slug, slug),
        'kind': pageio.kind_for_slug(slug),
        'status': 'active',
        'keywords': [],
        'aliases': [],
        'created': now_iso(),
        'updated': now_iso(),
        'last_seen': None,
        'hits': 0,
        'quarantined': False,
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
        kind = page.get('kind')
        if kind not in KINDS:
            errors.append(f"{prefix}: kind должен быть в {KINDS}, получено {kind!r}")
        else:
            is_service = slug in pageio.SERVICE_PAGES
            if kind == 'self' and not is_service:
                errors.append(f"{prefix}: kind=self допустим только для "
                              f"{list(pageio.SERVICE_PAGES)}, получен slug {slug!r}")
            if is_service and kind != 'self':
                errors.append(f"{prefix}: служебная страница {slug!r} обязана иметь kind=self")
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
        if page.get('quarantined') is not None and not isinstance(page.get('quarantined'), bool):
            errors.append(f"{prefix}.quarantined: ожидается boolean или null")
    return errors


def is_usable_index(data) -> bool:
    return not validate_index(data)


def service_pages_present(root: str | None = None) -> bool:
    """Существуют ли файлы Home.md и Style.md (необходимо для валидности БЗ)."""
    root = root if root is not None else pageio.kb_root()
    return all(pageio.page_exists(slug, root) for slug in pageio.SERVICE_PAGES)


def kb_valid(db_path: str | None = None, root: str | None = None) -> bool:
    """«Валидная БЗ»: валидный/восстановимый индекс + файлы Home.md и Style.md.

    Используется в горячем пути ответа (решение «инжектить БЗ»), поэтому без
    записи на диск: ensure_index read-only + проверка наличия файлов.
    """
    root = root if root is not None else pageio.kb_root()
    db_path = db_path if db_path is not None else config.db_path()
    index_data, status = ensure_index(root, db_path)
    if index_data is None:
        return False
    return service_pages_present(root)


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
        logger.warning("bot_kb index: не удалось прочитать %s: %s", path, e)
        return None, 'error'


def load_index(root: str):
    """Читает и валидирует _index.yaml. (index|None, errors:list[str])."""
    data, reason = _read_yaml_file(index_path(root))
    if reason == 'missing':
        return None, ['файл отсутствует']
    if reason == 'error':
        return None, ['ошибка чтения/парсинга']
    errors = validate_index(data)
    if errors:
        return None, errors
    return data, []


def load_bak(root: str):
    """Читает и валидирует _index.yaml.bak. (index|None, errors)."""
    data, reason = _read_yaml_file(bak_path(root))
    if reason == 'missing':
        return None, ['бэкап отсутствует']
    if reason == 'error':
        return None, ['ошибка чтения/парсинга .bak']
    errors = validate_index(data)
    if errors:
        return None, errors
    return data, []


# --- Синхронизация файлов и индекса (п. 8.2) ---

def sync_index_with_files(root: str, index: dict) -> dict:
    """Приводит индекс в соответствие с .md-файлами (на копии).

    Файл без записи → добавляется (kind по имени). Запись (не Home/Style) без
    файла → archived (kind сохраняется). Home/Style без файла не архивируются —
    их отсутствие обрабатывает bootstrap.
    """
    result = {
        'pages': list(index.get('pages', [])),
        'page_proposal_cooldowns': dict(index.get('page_proposal_cooldowns', {}) or {}),
    }
    for key, value in index.items():
        if key not in result:
            result[key] = value

    file_slugs = set(pageio.list_slugs(root))
    index_slugs = {p.get('slug') for p in result['pages']}

    for page in result['pages']:
        slug = page.get('slug')
        if slug in file_slugs:
            continue
        if slug in pageio.SERVICE_PAGES:
            logger.warning("bot_kb index: отсутствует файл служебной страницы %s.md", slug)
            continue
        if page.get('status') != 'archived':
            logger.info("bot_kb index: страница %s без файла → archived", slug)
            page['status'] = 'archived'

    for slug in sorted(file_slugs - index_slugs):
        logger.info("bot_kb index: файл %s.md без записи → добавлен в индекс", slug)
        result['pages'].append(_minimal_page(slug))

    return result


# --- Восстановление / пересбор ---

def rebuild_from_files(root: str, db_path: str) -> dict:
    """Пересбор индекса по .md-файлам (п. 8.2). Ничего не пишет на диск."""
    index = new_index()
    index['watermark'] = db.watermark(db_path) or 0
    index['message_count'] = 0
    index['last_error'] = 'index rebuilt without backup'
    for slug in pageio.list_slugs(root):
        index['pages'].append(_minimal_page(slug))
    return index


def ensure_index(root: str | None = None, db_path: str | None = None):
    """Гарантирует валидный индекс. Возвращает (index|None, status).

    status: ok | restored_bak | rebuilt | missing (нет ни индекса, ни .bak,
    ни страниц). Не пишет на диск (read-only): сохранение делает вызывающий.
    """
    root = root if root is not None else pageio.kb_root()
    db_path = db_path if db_path is not None else config.db_path()

    index, errors = load_index(root)
    if index is not None:
        return sync_index_with_files(root, index), 'ok'

    bak, bak_errors = load_bak(root)
    if bak is not None:
        logger.info("bot_kb index: восстановлен из .bak: %s", errors)
        return sync_index_with_files(root, bak), 'restored_bak'

    if pageio.list_slugs(root):
        logger.info("bot_kb index: пересбор по файлам: %s", errors)
        return rebuild_from_files(root, db_path), 'rebuilt'

    logger.warning("bot_kb index: нет ни индекса, ни бэкапа, ни страниц: %s",
                   errors or bak_errors)
    return None, 'missing'


# --- Запись (атомарная, с бэкапом) ---

def save_index(root: str, index: dict) -> bool:
    """Пишет _index.yaml атомарно (tmp+rename); перед заменой — .bak.

    Возвращает True при успехе. Индекс валидируется перед записью; невалидный
    индекс не сохраняется (watermark двигается только после успешного батча).
    """
    errors = validate_index(index)
    if errors:
        logger.error("bot_kb index: не сохраняю невалидный индекс (%s): %s", root, errors)
        return False

    try:
        os.makedirs(root, exist_ok=True)
        current = index_path(root)
        tmp = os.path.join(root, INDEX_TMP)

        with open(tmp, 'w', encoding='utf-8') as f:
            yaml.safe_dump(index, f, allow_unicode=True, default_flow_style=False,
                           sort_keys=False)
        if os.path.exists(current):
            if not _replace_with_retry(current, bak_path(root)):
                raise OSError(f"не удалось создать бэкап {bak_path(root)}")
        if not _replace_with_retry(tmp, current):
            raise OSError(f"не удалось заменить {current}")
        return True
    except OSError as e:
        logger.error("bot_kb index: не удалось сохранить %s: %s", index_path(root), e)
        try:
            if os.path.exists(os.path.join(root, INDEX_TMP)):
                os.remove(os.path.join(root, INDEX_TMP))
        except OSError:
            pass
        return False


# --- Watermark для ретенции (п. 7.3) ---

def watermark_for_retention(db_path: str | None = None,
                            root: str | None = None) -> int | None:
    """Watermark валидного/восстановимого индекса для ретенции bot_kb_raw.

    None — индекса нет (страницы не созданы / capture_only) → ретенция работает
    по no-kb-политике. Для «восстановимого» индекса (из .bak/пересбор) watermark
    детерминирован, поэтому сохранять индекс здесь не нужно.
    """
    index, status = ensure_index(root, db_path)
    if index is None:
        return None
    return index.get('watermark', 0)
