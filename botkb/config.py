"""Настройки фичи «база знаний бота» (ТЗ docs/bot_kb_tz.md).

bot.py держит глобальный `config` из config.yaml. Пакет botkb НЕ импортирует
bot.py (это позволило бы тестам не триггерить import-time инициализацию БД),
поэтому bot.py передаёт сырой верхнеуровневый конфиг через `configure()`.
`configure()` вызывается при старте и в `/reload_config`.
"""

import copy
import logging

logger = logging.getLogger(__name__)


class KBConfigError(ValueError):
    """Некорректная секция `bot_kb:` в конфигурации."""


# Полный блок `bot_kb:` из ТЗ п. 15. Значения по умолчанию используются всегда,
# когда ключ отсутствует в config.yaml — бот работает и без секции `bot_kb:`.
BOT_KB_DEFAULTS = {
    'enabled': True,                 # false ≡ mode: disabled (имеет приоритет)
    'dir': 'wiki/bot_kb',
    'mode': 'primary',               # disabled | capture_only | shadow | primary
    # Модель для генерации страниц БЗ (без reasoning-режима, как wiki.llm_model).
    'llm_model': 'deepseek-chat',

    'capture': {
        'include_text': True,
        'include_captions': True,
        'include_bot_answers': True, # писать ответы бота (speaker='bot'), по куску на сообщение
        'include_service_messages': False,
        'include_bot_commands': False,
        'max_content_chars': 4000,
        'redact': True,
        'redact_patterns': ['phone', 'email'],
    },

    'update': {
        'respond_only': False,        # учиться и без ответа бота (по любому сырью)
        'self_update_on_feedback': True,
        'debounce_seconds': 20,
        'min_interval_seconds': 60,
        'max_self_pages_per_update': 2,
        'max_pages_per_update': 3,
        'self_context_turns': 3,
        'max_raw_messages_per_update': 50,
        'max_raw_chars_per_update': 20000,
        'skip_trivial_messages': True,
        'trivial_min_chars': 3,
        'max_batch_retries': 3,
        'max_persistent_batch_failures': 3,
        'max_backoff_seconds': 600,
        'page_quarantine_failures': 5,
        'pause_cooldown_minutes': 30,
        'on_persistent_failure': 'pause',   # pause | retry_later | drop_old
    },

    'pages': {
        'start': ['Home', 'Style'],
        'max_page_chars': 2400,
        'home_target_chars': 900,
        'style_target_chars': 700,
        'knowledge_target_chars': 1400,
        'create_repeats': 3,
        'create_window_messages': 100,
        'create_cooldown_hours': 24,
        'max_cooldown_entries': 100,
        'max_count': 30,
        'slug_pattern': '^[a-z0-9_-]{1,64}$',
        'archive_after_days': 90,
    },

    'bootstrap': {
        'mode': 'from_system_prompt',   # from_system_prompt | limited_window | empty
        'history': 'discard',           # discard | backlog
        'max_history_messages': 500,
        'limited_window_messages': 80,
        'max_seed_chars': 12000,
        'seed': '',
    },

    'reconcile': {
        'every_n_messages': 150,
        'retry_after_minutes': 60,
        'stale_fact_days': 60,
        'window_messages': 400,
        'max_raw_chars': 120000,
        'page_max_raw_chars': 30000,
        'max_pages_per_reconcile': 10,
    },

    'router': {
        'mode': 'keywords',          # keywords | llm
        'min_score': 0.3,
        'top_k': 3,
        'llm_close_score_delta': 0.1,
        'llm_timeout_ms': 800,
    },

    'inject': {
        'max_chars': 4000,
        'reserve_home_style_chars': 1600,
        'home_max_chars': 900,
        'style_max_chars': 700,
        'page_max_chars': 1500,
        'include_home': True,
        'include_style': True,
        'combined_warn_factor': 0.9,   # 0 = выкл; иначе доля от 1
    },

    'raw': {
        'ttl_hours': 720,              # окно сырья для reconcile (30 дней), по ts
        'no_kb_ttl_hours': 720,        # строки без индекса / в capture_only (по ts)
        'max_rows': 10000,             # глобальный кап строк (обработанных)
        'delete_only_processed': True,
        'delete_unprocessed_without_kb': True,
    },

    'budgets': {
        'max_updates_per_day': 60,
        'max_llm_calls_per_hour': 12,
        'reconcile_llm_calls_per_day': 30,
        'allow_manual_reconcile_over_budget': True,
        'over_limit_policy': 'delay',  # delay | drop_old
        'max_backlog_rows': 5000,
    },

    'prompts': {
        'bootstrap_home_prompt': '',
        'bootstrap_style_prompt': '',
        'update_self_prompt': '',
        'update_knowledge_prompt': '',
        'create_knowledge_prompt': '',
        'reconcile_self_prompt': '',
        'reconcile_knowledge_prompt': '',
        'merge_page_prompt': '',
    },
}


_mode_values = ('disabled', 'capture_only', 'shadow', 'primary')
_bootstrap_mode_values = ('from_system_prompt', 'limited_window', 'empty')
_bootstrap_history_values = ('discard', 'backlog')
_budget_policy_values = ('delay', 'drop_old')
_on_failure_values = ('pause', 'retry_later', 'drop_old')
_router_mode_values = ('keywords', 'llm')

# Ключи, которые обязаны быть целыми > 0 (ТЗ п. 5.3 — явный список).
_positive_int_keys = [
    ('capture', 'max_content_chars'),
    ('update', 'max_raw_messages_per_update'),
    ('update', 'max_raw_chars_per_update'),
    ('update', 'max_self_pages_per_update'),
    ('update', 'max_pages_per_update'),
    ('update', 'self_context_turns'),
    ('update', 'max_batch_retries'),
    ('update', 'max_persistent_batch_failures'),
    ('update', 'max_backoff_seconds'),
    ('update', 'page_quarantine_failures'),
    ('update', 'pause_cooldown_minutes'),
    ('pages', 'max_page_chars'),
    ('pages', 'home_target_chars'),
    ('pages', 'style_target_chars'),
    ('pages', 'knowledge_target_chars'),
    ('pages', 'create_repeats'),
    ('pages', 'create_window_messages'),
    ('pages', 'create_cooldown_hours'),
    ('pages', 'max_cooldown_entries'),
    ('pages', 'max_count'),
    ('pages', 'archive_after_days'),
    ('bootstrap', 'max_history_messages'),
    ('reconcile', 'every_n_messages'),
    ('reconcile', 'retry_after_minutes'),
    ('reconcile', 'stale_fact_days'),
    ('reconcile', 'window_messages'),
    ('reconcile', 'max_raw_chars'),
    ('reconcile', 'page_max_raw_chars'),
    ('reconcile', 'max_pages_per_reconcile'),
    ('inject', 'max_chars'),
    ('inject', 'reserve_home_style_chars'),
    ('inject', 'home_max_chars'),
    ('inject', 'style_max_chars'),
    ('inject', 'page_max_chars'),
    ('raw', 'ttl_hours'),
    ('raw', 'no_kb_ttl_hours'),
    ('raw', 'max_rows'),
    ('budgets', 'max_updates_per_day'),
    ('budgets', 'max_llm_calls_per_hour'),
    ('budgets', 'reconcile_llm_calls_per_day'),
    ('budgets', 'max_backlog_rows'),
]

# Целые ключи, допускающие 0 (тайминги троттлинга).
_nonneg_int_keys = [
    ('update', 'debounce_seconds'),
    ('update', 'min_interval_seconds'),
]

_bool_keys = [
    (None, 'enabled'),
    ('capture', 'include_text'),
    ('capture', 'include_captions'),
    ('capture', 'include_bot_answers'),
    ('capture', 'include_service_messages'),
    ('capture', 'include_bot_commands'),
    ('capture', 'redact'),
    ('update', 'respond_only'),
    ('update', 'self_update_on_feedback'),
    ('update', 'skip_trivial_messages'),
    ('inject', 'include_home'),
    ('inject', 'include_style'),
    ('raw', 'delete_only_processed'),
    ('raw', 'delete_unprocessed_without_kb'),
    ('budgets', 'allow_manual_reconcile_over_budget'),
]

_enum_keys = [
    (None, 'mode', _mode_values),
    (None, 'dir', None),
    ('bootstrap', 'mode', _bootstrap_mode_values),
    ('bootstrap', 'history', _bootstrap_history_values),
    ('router', 'mode', _router_mode_values),
    ('update', 'on_persistent_failure', _on_failure_values),
    ('budgets', 'over_limit_policy', _budget_policy_values),
]

_string_keys = [
    (None, 'dir'),
    ('bootstrap', 'seed'),
    ('prompts', 'bootstrap_home_prompt'),
    ('prompts', 'bootstrap_style_prompt'),
    ('prompts', 'update_self_prompt'),
    ('prompts', 'update_knowledge_prompt'),
    ('prompts', 'create_knowledge_prompt'),
    ('prompts', 'reconcile_self_prompt'),
    ('prompts', 'reconcile_knowledge_prompt'),
    ('prompts', 'merge_page_prompt'),
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
    """Возвращает список ошибок валидации секции bot_kb (пустой — всё ок)."""
    errors: list[str] = []
    upd = cfg.get('update', {})
    pages = cfg.get('pages', {})
    boot = cfg.get('bootstrap', {})
    inj = cfg.get('inject', {})
    recon = cfg.get('reconcile', {})

    def bad(section: str, key: str, why: str):
        errors.append(f"bot_kb.{section}.{key}: {why}" if section else f"bot_kb.{key}: {why}")

    def _get(section, key):
        if section is None:
            return cfg.get(key)
        return cfg.get(section, {}).get(key)

    # Типы/диапазоны простых ключей
    for section, key in _positive_int_keys:
        value = _get(section, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            bad(section, key, f"ожидается целое число > 0, получено {value!r}")

    for section, key in _nonneg_int_keys:
        value = _get(section, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            bad(section, key, f"ожидается целое число >= 0, получено {value!r}")

    for section, key in _bool_keys:
        value = _get(section, key)
        if not isinstance(value, bool):
            bad(section, key, f"ожидается boolean, получено {value!r}")

    for section, key, allowed in _enum_keys:
        if allowed is None:
            continue
        value = _get(section, key)
        if value not in allowed:
            bad(section, key,
                f"ожидается одно из {allowed}, получено {value!r}")

    for section, key in _string_keys:
        value = _get(section, key)
        if not isinstance(value, str):
            bad(section, key, f"ожидается строка, получено {value!r}")

    # combined_warn_factor — число в [0, 1] (0 = выкл)
    warn_factor = inj.get('combined_warn_factor')
    if not isinstance(warn_factor, (int, float)) or isinstance(warn_factor, bool) \
            or not (0 <= warn_factor <= 1):
        bad('inject', 'combined_warn_factor',
            f"ожидается число в [0, 1] (0 = выкл), получено {warn_factor!r}")

    # Алгебраические проверки (ТЗ п. 5.3)
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
    if n(inj, 'home_max_chars') > n(pages, 'max_page_chars'):
        bad('inject', 'home_max_chars', 'inject.home_max_chars > pages.max_page_chars')
    if n(inj, 'style_max_chars') > n(pages, 'max_page_chars'):
        bad('inject', 'style_max_chars', 'inject.style_max_chars > pages.max_page_chars')
    if pages.get('max_count') is not None and isinstance(pages.get('max_count'), int) \
            and pages.get('max_count') < 2:
        bad('pages', 'max_count', 'max_count должен быть >= 2 (Home и Style обязательны)')

    # Bootstrap: ограничения, зависящие от выбранного режима/политики
    if boot.get('mode') == 'limited_window':
        v = boot.get('limited_window_messages')
        if not isinstance(v, int) or v <= 0:
            bad('bootstrap', 'limited_window_messages',
                'должен быть > 0 при bootstrap.mode=limited_window')
    if boot.get('history') == 'backlog':
        v = boot.get('max_history_messages')
        if not isinstance(v, int) or v <= 0:
            bad('bootstrap', 'max_history_messages',
                'должен быть > 0 при bootstrap.history=backlog')
    if boot.get('mode') == 'from_system_prompt':
        v = boot.get('max_seed_chars')
        if not isinstance(v, int) or v <= 0:
            bad('bootstrap', 'max_seed_chars',
                'должен быть > 0 при bootstrap.mode=from_system_prompt')

    # Reconcile: stale-горизонт должен быть меньше архивного (иначе warning в
    # _log_warnings; здесь это не ошибка, но проверяем только типы).
    _ = recon

    return errors


# --- Состояние пакета (устанавливается из bot.py) ---

_top: dict = {}
_merged: dict = copy.deepcopy(BOT_KB_DEFAULTS)


def _log_warnings(cfg: dict):
    """Не-фатальные предупреждения о конфигурации (ТЗ п. 5.3 — warning, не ошибка)."""
    raw = cfg.get('raw', {})
    pages_cfg = cfg.get('pages', {})
    recon = cfg.get('reconcile', {})
    upd = cfg.get('update', {})
    mode = cfg.get('mode')
    enabled = cfg.get('enabled', True)

    if enabled and mode == 'primary' and upd.get('respond_only'):
        logger.warning("bot_kb: update.respond_only=true при mode=primary — БЗ учится только "
                       "по ответам бота (обычно нежелательно)")
    ttl = raw.get('ttl_hours')
    if enabled and mode in ('shadow', 'primary') and isinstance(ttl, int) and ttl < 168:
        logger.warning("bot_kb: raw.ttl_hours=%d < 168 — окно сырья для reconcile меньше недели: "
                       "авто-reconcile может не найти релевантное окно", ttl)
    stale = recon.get('stale_fact_days')
    archive = pages_cfg.get('archive_after_days')
    if isinstance(stale, int) and isinstance(archive, int) and stale >= archive:
        logger.warning("bot_kb: reconcile.stale_fact_days (%d) >= pages.archive_after_days (%d): "
                       "архив спящей темы сработает раньше, чем reconcile пометит факты устаревшими",
                       stale, archive)
    p_max = recon.get('page_max_raw_chars')
    if isinstance(p_max, int) and p_max > 40000:
        logger.warning("bot_kb: reconcile.page_max_raw_chars=%d > 40000 — риск переполнения "
                       "контекста LLM (подокно + страница + промпт должны заведомо влезать)", p_max)


def configure(top_config: dict | None) -> None:
    """Принимает верхнеуровневый конфиг бота, мёржит секцию bot_kb поверх дефолтов.

    Бросает KBConfigError при невалидной секции — валидность проверяется до
    применения новых значений (некорректный /reload_config не ломает рабочую).
    """
    global _top, _merged
    top = top_config if isinstance(top_config, dict) else {}
    merged = _deep_merge(BOT_KB_DEFAULTS, top.get('bot_kb') if isinstance(top.get('bot_kb'), dict) else {})
    errors = _validate(merged)
    if errors:
        raise KBConfigError("Некорректная конфигурация bot_kb:\n- " + "\n- ".join(errors))
    _top = copy.deepcopy(top)
    _merged = merged
    _log_warnings(merged)
    logger.info("bot_kb: конфигурация применена (mode=%s, enabled=%s)",
                merged.get('mode'), merged.get('enabled'))


def settings() -> dict:
    """Живой merged-конфиг секции bot_kb. Не мутировать результат."""
    return _merged


def top_config() -> dict:
    """Верхнеуровневый конфиг бота (для доступа к спискам доступа и db)."""
    return _top


def db_path() -> str:
    return _top.get('db', 'bot.db')


def kb_dir() -> str:
    return _merged.get('dir', 'wiki/bot_kb')


def mode() -> str:
    if not _merged.get('enabled', True):
        return 'disabled'
    return _merged.get('mode', 'primary')


def capture_active() -> bool:
    """Захват сырья включён? (disabled → нет; capture_only/shadow/primary → да)."""
    return mode() != 'disabled'


def validate_now() -> list[str]:
    """Прогон валидации по текущему merged-конфигу (для /reload_config)."""
    return _validate(_merged)
