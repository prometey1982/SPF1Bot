"""Тесты конфигурации и валидации bot_kb (botkb/config.py, ТЗ п. 5.3)."""

import copy

import pytest

from botkb import config as kc
from botkb.config import KBConfigError


def _top(override_section=None):
    top = {
        'db': 'bot.db',
        'bot_kb': copy.deepcopy(kc.BOT_KB_DEFAULTS),
    }
    if override_section:
        for section, value in override_section.items():
            if isinstance(value, dict) and isinstance(top['bot_kb'].get(section), dict):
                top['bot_kb'][section].update(value)
            else:
                top['bot_kb'][section] = value
    return top


def test_defaults_are_valid():
    assert kc.validate_now() == []
    kc.configure({})
    assert kc.settings()['mode'] == 'primary'
    assert kc.capture_active() is True


def test_mode_disabled_equivalent_when_enabled_false():
    top = _top({'enabled': False, 'mode': 'primary'})
    kc.configure(top)
    assert kc.mode() == 'disabled'
    assert kc.capture_active() is False


def test_invalid_mode_rejected():
    top = _top({'mode': 'nonsense'})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_invalid_bootstrap_history_rejected():
    top = _top({'bootstrap': {'history': 'keep_all'}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_reserve_invariant():
    top = _top({'inject': {'home_max_chars': 1500, 'style_max_chars': 700,
                           'reserve_home_style_chars': 1000}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_page_max_chars_not_above_inject():
    top = _top({'inject': {'page_max_chars': 5000, 'max_chars': 4000}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_combined_warn_factor_range():
    for bad in (-0.1, 1.5, '0.9'):
        top = _top({'inject': {'combined_warn_factor': bad}})
        with pytest.raises(KBConfigError):
            kc.configure(top)
    for good in (0, 1, 0.9):
        top = _top({'inject': {'combined_warn_factor': good}})
        kc.configure(top)
        assert kc.settings()['inject']['combined_warn_factor'] == good


def test_backlog_history_requires_positive_messages():
    top = _top({'bootstrap': {'history': 'backlog', 'max_history_messages': 0}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_discard_history_ignores_messages_value():
    top = _top({'bootstrap': {'history': 'discard'}})  # дефолт max_history_messages>0
    kc.configure(top)


def test_limited_window_requires_positive_messages():
    top = _top({'bootstrap': {'mode': 'limited_window', 'limited_window_messages': 0}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_bad_value_type_rejected():
    top = _top({'raw': {'ttl_hours': -5}})
    with pytest.raises(KBConfigError):
        kc.configure(top)
    top = _top({'raw': {'ttl_hours': '720'}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_negative_nonneg_timing_rejected():
    top = _top({'update': {'debounce_seconds': -1}})
    with pytest.raises(KBConfigError):
        kc.configure(top)


def test_invalid_config_does_not_replace_current(kb_config):
    """Провалившаяся configure не трогает рабочее состояние."""
    good = _top()
    kc.configure(good)
    before = kc.settings()['raw']['ttl_hours']
    bad = _top({'raw': {'ttl_hours': -1}})
    with pytest.raises(KBConfigError):
        kc.configure(bad)
    assert kc.settings()['raw']['ttl_hours'] == before


def test_configure_ignores_missing_section():
    kc.configure({'db': 'x.db'})
    assert kc.capture_active() is True  # defaults
    assert kc.db_path() == 'x.db'
    assert kc.kb_dir() == 'wiki/bot_kb'


def test_partial_section_merges_over_defaults():
    top = {'db': 'bot.db', 'bot_kb': {'capture': {'max_content_chars': 123}}}
    kc.configure(top)
    assert kc.settings()['capture']['max_content_chars'] == 123
    # неупомянутые ключи остаются дефолтными
    assert kc.settings()['capture']['redact'] is True
    assert kc.settings()['mode'] == 'primary'


def test_capture_only_not_active_in_disabled():
    top = _top({'mode': 'disabled'})
    kc.configure(top)
    assert kc.capture_active() is False


# --- Секция knowledge (ТЗ bot_kb_knowledge_tz.md, K1) ---

def test_knowledge_defaults_off():
    top = _top()
    kc.configure(top)
    kn = kc.settings()['knowledge']
    assert kn['bot_turns'] is False
    assert kn['bot_turn_min_chars'] == 120
    assert kn['bot_turn_skip_phrases'] == []


def test_knowledge_enabled_valid():
    top = _top({'knowledge': {'bot_turns': True, 'bot_turn_min_chars': 200,
                              'bot_turn_skip_phrases': ['слушай сюда']}})
    kc.configure(top)
    kn = kc.settings()['knowledge']
    assert kn['bot_turns'] is True
    assert kn['bot_turn_min_chars'] == 200
    assert kn['bot_turn_skip_phrases'] == ['слушай сюда']


def test_knowledge_min_chars_validated_always():
    """Валидация bot_turn_min_chars — всегда, независимо от флага."""
    for bad in (0, -1, '120'):
        top = _top({'knowledge': {'bot_turn_min_chars': bad}})
        with pytest.raises(KBConfigError):
            kc.configure(top)


def test_knowledge_skip_phrases_type_checked():
    top = _top({'knowledge': {'bot_turn_skip_phrases': 'спасибо'}})
    with pytest.raises(KBConfigError):
        kc.configure(top)
    top = _top({'knowledge': {'bot_turn_skip_phrases': [1, 2]}})
    with pytest.raises(KBConfigError):
        kc.configure(top)
    top = _top({'knowledge': {'bot_turn_skip_phrases': ['спасибо']}})
    kc.configure(top)  # валидно


def test_knowledge_bot_turns_type_checked():
    top = _top({'knowledge': {'bot_turns': 'yes'}})
    with pytest.raises(KBConfigError):
        kc.configure(top)
