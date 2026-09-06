"""Настройки фичи «wiki пользователя» (ТЗ docs/user_wiki_tz.md).

bot.py держит глобальный `config` из config.yaml. Пакет botwiki НЕ импортирует
bot.py (это позволило бы тестам не триггерить import-time инициализацию БД),
поэтому bot.py передаёт сырой верхнеуровневый конфиг через `configure()`.
`configure()` вызывается при старте и в `/reload_config`.
"""

import copy
import logging

logger = logging.getLogger(__name__)


class WikiConfigError(ValueError):
    """Некорректная секция `wiki:` в конфигурации."""


# Полный блок `wiki:` из ТЗ п. 15. Значения по умолчанию используются всегда,
# когда ключ отсутствует в config.yaml — бот работает и без секции `wiki:`.
WIKI_DEFAULTS = {
    'enabled': True,                 # false ≡ mode: disabled (имеет приоритет)
    'dir': 'wiki',
    'mode': 'primary',               # disabled | capture_only | shadow | primary

    'shadow': {
        'disable_dossier_updates': False,
    },

    'capture': {
        'include_text': True,
        'include_captions': True,
        'include_service_messages': False,
        'include_bot_commands': False,
        'max_content_chars': 4000,
        'redact': True,              # маска phone/email при захвате
        'redact_patterns': ['phone', 'email'],
    },

    'update': {
        'responded_only': True,
        'min_interval_seconds': 60,
        'debounce_seconds': 20,
        'max_pages_per_update': 3,
        'max_raw_messages_per_update': 50,
        'max_raw_chars_per_update': 20000,
        'skip_trivial_messages': True,
        'trivial_min_chars': 3,
        'max_batch_retries': 3,
        'on_persistent_failure': 'pause',   # pause | retry_later | drop_old
    },

    'pages': {
        'start': ['Home', 'Style'],
        'max_page_chars': 2000,
        'home_target_chars': 900,
        'style_target_chars': 600,
        'create_repeats': 3,
        'create_window_messages': 100,
        'create_cooldown_hours': 24,
        'max_cooldown_entries': 100,
        'max_count': 20,
        'slug_pattern': '^[a-z0-9_-]{1,64}$',
        'archive_after_days': 90,
    },

    'bootstrap': {
        'mode': 'from_dossier',      # from_dossier | current_watermark | limited_window | from_import
        'limited_window_messages': 50,
        'max_dossier_chars': 8000,
        'sanitize_dossier': True,
    },

    'reconcile': {
        'every_n_messages': 100,
        'include_mentions': True,
        'stale_fact_days': 90,
        'window_messages': 300,
        'max_raw_chars': 100000,
        'max_mentions': 50,
        'max_pages_per_reconcile': 10,
    },

    'router': {
        'mode': 'keywords',          # keywords | llm
        'normalize': True,
        'lemmatize': False,
        'min_score': 0.35,
        'top_k': 2,
        'llm_close_score_delta': 0.1,
        'llm_timeout_ms': 800,
    },

    'inject': {
        'max_chars': 3000,
        'reserve_home_style_chars': 1500,
        'home_max_chars': 900,
        'style_max_chars': 600,
        'page_max_chars': 1000,
        'include_home': True,
        'include_style': True,
    },

    'raw': {
        'ttl_hours': 168,
        'no_wiki_ttl_hours': 72,
        'max_rows_per_user': 500,
        'delete_only_processed': True,
        'delete_unprocessed_without_wiki': True,
        'import_ttl_hours': 720,
        'import_ttl_basis': 'inserted_at',   # inserted_at | ts
        'import_uncovered_max_days': 30,
    },

    'budgets': {
        'max_updates_per_user_per_day': 20,
        'max_llm_calls_per_user_per_hour': 5,
        'router_llm_calls_per_user_per_hour': 30,
        'reconcile_llm_calls_per_user_per_day': 20,
        'allow_manual_reconcile_over_budget': True,
        'import_updates_per_user_per_day': 100,
        'import_llm_calls_per_user_per_day': 200,
        'import_allow_over_normal_budget': True,
        'max_concurrent_backfills': 3,
        'over_limit_policy': 'delay',        # delay | drop_old
        'max_backlog_messages': 300,
    },

    'import': {                      # импорт экспорта Telegram (этап 8)
        'enabled': True,
        'allowed_dir': 'data/imports',
        'chat_map': [],
        'batch_size': 500,
        'max_file_mb': 200,
        'warn_file_mb': 100,
        'on_batch_error': 'stop',    # stop | continue
        'max_content_chars': 4000,
        'redact': True,
        'exclude_types': ['service'],
        'skip_commands': True,
        'backfill_budget_mode': 'separate',  # separate | normal | bypass
        'backfill_window_messages': 2000,
        'max_backfill_messages_per_user': None,
    },

    'prompts': {
        'update_home_style_prompt': '',
        'update_page_prompt': '',
        'create_page_prompt': '',
        'reconcile_prompt': '',
    },
}


# Секции/поля, которые будут реализованы на более поздних этапах и потому
# валидируются позже (файловая безопасность импорта — этап 8, bootstrap
# limited_window — этап 3 и т.п.). Алгебраические проверки п. 5.2 выполняются
# сразу, т.к. зависят только от значений в конфиге.
_IMPORT_FS_CHECK_ENABLED = False

_mode_values = ('disabled', 'capture_only', 'shadow', 'primary')
_budget_policy_values = ('delay', 'drop_old')
_on_failure_values = ('pause', 'retry_later', 'drop_old')
_on_batch_error_values = ('stop', 'continue')
_backfill_budget_mode_values = ('separate', 'normal', 'bypass')
_ttl_basis_values = ('inserted_at', 'ts')

_positive_int_keys = [
    ('capture', 'max_content_chars'),
    ('raw', 'ttl_hours'),
    ('raw', 'no_wiki_ttl_hours'),
    ('raw', 'max_rows_per_user'),
    ('raw', 'import_ttl_hours'),
    ('update', 'max_raw_messages_per_update'),
    ('update', 'max_raw_chars_per_update'),
    ('pages', 'max_page_chars'),
    ('pages', 'home_target_chars'),
    ('pages', 'style_target_chars'),
    ('pages', 'create_repeats'),
    ('pages', 'create_window_messages'),
    ('pages', 'create_cooldown_hours'),
    ('pages', 'max_cooldown_entries'),
    ('pages', 'max_count'),
    ('pages', 'archive_after_days'),
    ('inject', 'max_chars'),
    ('inject', 'reserve_home_style_chars'),
    ('inject', 'home_max_chars'),
    ('inject', 'style_max_chars'),
    ('inject', 'page_max_chars'),
]

_bool_keys = [
    ('capture', 'include_text'),
    ('capture', 'include_captions'),
    ('capture', 'include_service_messages'),
    ('capture', 'include_bot_commands'),
    ('capture', 'redact'),
    ('raw', 'delete_only_processed'),
    ('raw', 'delete_unprocessed_without_wiki'),
    ('update', 'responded_only'),
]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate(cfg: dict) -> list[str]:
    """Возвращает список ошибок валидации секции wiki (пустой — всё ок)."""
    errors: list[str] = []
    cap = cfg.get('capture', {})
    upd = cfg.get('update', {})
    pages = cfg.get('pages', {})
    boot = cfg.get('bootstrap', {})
    inj = cfg.get('inject', {})
    raw = cfg.get('raw', {})
    budgets = cfg.get('budgets', {})
    imp = cfg.get('import', {})

    def bad(section: str, key: str, why: str):
        errors.append(f"wiki.{section}.{key}: {why}")

    # Типы/диапазоны простых ключей
    for section, key in _positive_int_keys:
        value = cfg.get(section, {}).get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            bad(section, key, f"ожидается целое число > 0, получено {value!r}")

    for section, key in _bool_keys:
        value = cfg.get(section, {}).get(key)
        if not isinstance(value, bool):
            bad(section, key, f"ожидается boolean, получено {value!r}")

    if isinstance(cfg.get('enabled'), bool) is False and cfg.get('enabled') is not None:
        bad('', 'enabled', f"ожидается boolean, получено {cfg.get('enabled')!r}")
    if cfg.get('mode') not in _mode_values:
        bad('', 'mode', f"ожидается одно из {_mode_values}, получено {cfg.get('mode')!r}")
    if isinstance(cfg.get('dir'), str) is False:
        bad('', 'dir', f"ожидается строка, получено {cfg.get('dir')!r}")

    # Алгебраические проверки (ТЗ п. 5.2)
    def n(section: str, key: str) -> int:
        return section.get(key, 0)

    if n(inj, 'home_max_chars') + n(inj, 'style_max_chars') > n(inj, 'reserve_home_style_chars'):
        bad('inject', 'reserve_home_style_chars',
            'home_max_chars + style_max_chars > reserve_home_style_chars')
    if n(inj, 'reserve_home_style_chars') > n(inj, 'max_chars'):
        bad('inject', 'reserve_home_style_chars', 'reserve_home_style_chars > max_chars')
    if n(inj, 'page_max_chars') > n(inj, 'max_chars'):
        bad('inject', 'page_max_chars', 'page_max_chars > max_chars')
    if n(pages, 'max_page_chars') < n(inj, 'page_max_chars'):
        bad('pages', 'max_page_chars', 'pages.max_page_chars < inject.page_max_chars')
    if n(pages, 'home_target_chars') > n(inj, 'home_max_chars'):
        bad('pages', 'home_target_chars', 'pages.home_target_chars > inject.home_max_chars')
    if n(pages, 'style_target_chars') > n(inj, 'style_max_chars'):
        bad('pages', 'style_target_chars', 'pages.style_target_chars > inject.style_max_chars')
    if n(inj, 'home_max_chars') > n(pages, 'max_page_chars'):
        bad('inject', 'home_max_chars', 'inject.home_max_chars > pages.max_page_chars')
    if n(inj, 'style_max_chars') > n(pages, 'max_page_chars'):
        bad('inject', 'style_max_chars', 'inject.style_max_chars > pages.max_page_chars')
    if pages.get('max_count') is not None and isinstance(pages.get('max_count'), int) and pages.get('max_count') < 2:
        bad('pages', 'max_count', 'max_count должен быть >= 2 (Home и Style обязательны)')
    if upd.get('on_persistent_failure') not in _on_failure_values:
        bad('update', 'on_persistent_failure',
            f"ожидается одно из {_on_failure_values}, получено {upd.get('on_persistent_failure')!r}")
    if budgets.get('over_limit_policy') not in _budget_policy_values:
        bad('budgets', 'over_limit_policy',
            f"ожидается одно из {_budget_policy_values}, получено {budgets.get('over_limit_policy')!r}")

    # Bootstrap limited_window (этап 3) — проверка закладывается сразу
    if boot.get('mode') == 'limited_window':
        v = boot.get('limited_window_messages')
        if not isinstance(v, int) or v <= 0:
            bad('bootstrap', 'limited_window_messages',
                'должен быть > 0 при bootstrap.mode=limited_window')

    # Импорт (этап 8) — только чисто конфигурационные проверки, FS — на этапе 8
    if imp.get('enabled'):
        if _IMPORT_FS_CHECK_ENABLED:
            allowed_dir = imp.get('allowed_dir')
            if not allowed_dir:
                bad('import', 'allowed_dir', 'пустой путь')
        if not isinstance(imp.get('max_file_mb'), int) or imp.get('max_file_mb') <= 0:
            bad('import', 'max_file_mb', 'ожидается целое число > 0')
        if imp.get('on_batch_error') not in _on_batch_error_values:
            bad('import', 'on_batch_error',
                f"ожидается одно из {_on_batch_error_values}, получено {imp.get('on_batch_error')!r}")
        if imp.get('backfill_budget_mode') not in _backfill_budget_mode_values:
            bad('import', 'backfill_budget_mode',
                f"ожидается одно из {_backfill_budget_mode_values}, получено {imp.get('backfill_budget_mode')!r}")
        seen: dict = {}
        for entry in imp.get('chat_map', []):
            if not isinstance(entry, dict):
                bad('import', 'chat_map', f"элемент не является словарём: {entry!r}")
                continue
            name = entry.get('name')
            export_id = entry.get('export_id')
            if name is None and export_id is None:
                bad('import', 'chat_map', f"элемент без name и export_id: {entry!r}")
            for key in ('name', 'export_id'):
                val = entry.get(key)
                if val is None:
                    continue
                if val in seen and seen[val] is not key:
                    bad('import', 'chat_map', f"ключ {key}={val!r} повторяется")
                seen[val] = key

    # Ретенция и импорт
    if raw.get('import_ttl_basis') not in _ttl_basis_values:
        bad('raw', 'import_ttl_basis',
            f"ожидается одно из {_ttl_basis_values}, получено {raw.get('import_ttl_basis')!r}")

    return errors


# --- Состояние пакета (устанавливается из bot.py) ---

_top: dict = {}
_merged: dict = copy.deepcopy(WIKI_DEFAULTS)


def configure(top_config: dict | None) -> None:
    """Принимает верхнеуровневый конфиг бота, мёржит секцию wiki поверх дефолтов.

    Бросает WikiConfigError при невалидной секции — валидность проверяется до
    применения новых значений (некорректный /reload_config не ломает рабочую).
    """
    global _top, _merged
    top = top_config if isinstance(top_config, dict) else {}
    merged = _deep_merge(WIKI_DEFAULTS, top.get('wiki') if isinstance(top.get('wiki'), dict) else {})
    errors = _validate(merged)
    if errors:
        raise WikiConfigError("Некорректная конфигурация wiki:\n- " + "\n- ".join(errors))
    _top = copy.deepcopy(top)
    _merged = merged
    logger.info("wiki: конфигурация применена (mode=%s, enabled=%s)", merged.get('mode'), merged.get('enabled'))


def settings() -> dict:
    """Живой merged-конфиг секции wiki. Не мутировать результат."""
    return _merged


def top_config() -> dict:
    """Верхнеуровневый конфиг бота (для доступа к спискам доступа и db)."""
    return _top


def db_path() -> str:
    return _top.get('db', 'bot.db')


def wiki_dir() -> str:
    return _merged.get('dir', 'wiki')


def mode() -> str:
    if not _merged.get('enabled', True):
        return 'disabled'
    return _merged.get('mode', 'primary')


def capture_active() -> bool:
    """Захват сырья включён? (disabled → нет; capture_only/shadow/primary → да)."""
    return mode() != 'disabled'


def allowed_group_thread_ids() -> list:
    return _top.get('allowed_group_chat_ids', [])


def allowed_private_usernames() -> list:
    return _top.get('allowed_private_users', [])


def validate_now() -> list[str]:
    """Прогон валидации по текущему merged-конфигу (для /reload_config)."""
    return _validate(_merged)
