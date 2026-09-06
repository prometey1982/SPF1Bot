"""Тесты конфигурации и валидации wiki (botwiki/config.py, ТЗ п. 5.2)."""

import copy

import pytest

from botwiki import config as wc
from botwiki.config import WikiConfigError


def _top(override_section=None):
    top = {
        'db': 'bot.db',
        'wiki': copy.deepcopy(wc.WIKI_DEFAULTS),
    }
    if override_section:
        for section, value in override_section.items():
            if isinstance(value, dict) and isinstance(top['wiki'].get(section), dict):
                top['wiki'][section].update(value)
            else:
                top['wiki'][section] = value
    return top


def test_defaults_are_valid():
    assert wc.validate_now() == []
    wc.configure({})
    assert wc.settings()['mode'] == 'primary'
    assert wc.capture_active() is True


def test_mode_disabled_equivalent_when_enabled_false():
    top = _top({'enabled': False, 'mode': 'primary'})
    wc.configure(top)
    assert wc.mode() == 'disabled'
    assert wc.capture_active() is False


def test_invalid_mode_rejected():
    top = _top({'mode': 'nonsense'})
    with pytest.raises(WikiConfigError):
        wc.configure(top)


def test_reserve_invariant():
    top = _top({'inject': {'home_max_chars': 1500, 'style_max_chars': 600,
                           'reserve_home_style_chars': 1000}})
    with pytest.raises(WikiConfigError):
        wc.configure(top)


def test_bad_value_type_rejected():
    top = _top({'raw': {'ttl_hours': -5}})
    with pytest.raises(WikiConfigError):
        wc.configure(top)
    top = _top({'raw': {'ttl_hours': '168'}})
    with pytest.raises(WikiConfigError):
        wc.configure(top)


def test_invalid_config_does_not_replace_current(wiki_config):
    """Провалившаяся configure не трогает рабочее состояние."""
    good = _top()
    wc.configure(good)
    before = wc.settings()['raw']['ttl_hours']
    bad = _top({'raw': {'ttl_hours': -1}})
    with pytest.raises(WikiConfigError):
        wc.configure(bad)
    assert wc.settings()['raw']['ttl_hours'] == before


def test_configure_ignores_missing_wiki_section():
    wc.configure({'db': 'x.db'})
    assert wc.capture_active() is True  # defaults
    assert wc.db_path() == 'x.db'


def test_partial_section_merges_over_defaults():
    top = {'db': 'bot.db', 'wiki': {'capture': {'max_content_chars': 123}}}
    wc.configure(top)
    assert wc.settings()['capture']['max_content_chars'] == 123
    # неупомянутые ключи остаются дефолтными
    assert wc.settings()['capture']['redact'] is True
    assert wc.settings()['mode'] == 'primary'


def test_llm_router_mode_is_warning_not_error():
    """router.mode=llm не блокирует старт (MVP: fallback на keywords + warning)."""
    top = _top({'router': {'mode': 'llm'}})
    wc.configure(top)  # не должно бросить WikiConfigError
    assert wc.settings()['router']['mode'] == 'llm'


def test_invalid_backfill_budget_mode_rejected():
    top = _top({'import': {'backfill_budget_mode': 'always'}})
    with pytest.raises(WikiConfigError):
        wc.configure(top)


def test_duplicate_chat_map_keys_rejected():
    top = _top({'import': {'chat_map': [
        {'name': 'A', 'chat_id': 1},
        {'name': 'A', 'export_id': 2},
    ]}})
    with pytest.raises(WikiConfigError):
        wc.configure(top)


def test_drop_old_without_backlog_warns_not_raises():
    top = _top({'budgets': {'over_limit_policy': 'drop_old', 'max_backlog_messages': 0}})
    wc.configure(top)  # warning, не ошибка
    assert wc.settings()['budgets']['over_limit_policy'] == 'drop_old'
