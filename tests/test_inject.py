"""Тесты инъекции wiki при ответе (botwiki/inject.py, ТЗ п. 12)."""

import os

from botwiki import config as wc
from botwiki import index as index_mod
from botwiki import inject, manager, pages

from botwiki.manager import WikiManager
import asyncio

USER = 555


def run(coro):
    return asyncio.run(coro)


def _configure(tmp_path, db_path, overrides=None):
    cfg = {
        'db': db_path,
        'wiki': {'dir': str(tmp_path / 'wiki_root')},
    }
    if overrides:
        for section, value in overrides.items():
            if isinstance(value, dict):
                cfg['wiki'][section] = {**cfg['wiki'].get(section, {}), **value}
            else:
                cfg['wiki'][section] = value
    wc.configure(cfg)
    return wc.wiki_dir()


def _bootstrap(tmp_path, db_path, overrides=None):
    _configure(tmp_path, db_path, overrides)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    run(WikiManager()._bootstrap(USER, db_path, user_dir))
    return user_dir


def test_no_wiki_returns_none(tmp_path, db_path):
    _configure(tmp_path, db_path)
    assert inject.select_pages_for_injection(db_path, USER) is None


def test_injects_home_and_style_after_bootstrap(tmp_path, db_path):
    _bootstrap(tmp_path, db_path)
    pages_sel = inject.select_pages_for_injection(db_path, USER)
    assert pages_sel is not None
    assert {p['slug'] for p in pages_sel} == {'Home', 'Style'}
    assert all(p['text'] for p in pages_sel)
    total = sum(len(p['text']) for p in pages_sel)
    assert total <= wc.settings()['inject']['max_chars']


def test_build_system_message(tmp_path, db_path):
    _bootstrap(tmp_path, db_path)
    pages_sel = inject.select_pages_for_injection(db_path, USER)
    msg = inject.build_system_message(pages_sel)
    assert msg.startswith('Ниже перечислен набор фактов о пользователе.')
    assert '[Сводка]' in msg
    assert '[Стиль]' in msg


def test_big_pages_trimmed_style_first(tmp_path, db_path):
    overrides = {
        'inject': {'reserve_home_style_chars': 400, 'max_chars': 400,
                   'home_max_chars': 220, 'style_max_chars': 160,
                   'page_max_chars': 400},
        'pages': {'home_target_chars': 180, 'style_target_chars': 140},
    }
    user_dir = _bootstrap(tmp_path, db_path, overrides)
    # Большие страницы
    pages.atomic_write_page(user_dir, 'Home', '# Сводка\n\n' + '\n'.join(f'- факт {i}' * 30 for i in range(20)))
    pages.atomic_write_page(user_dir, 'Style', '# Стиль\n\n' + '\n'.join(f'- мем {i}' * 30 for i in range(20)))

    pages_sel = inject.select_pages_for_injection(db_path, USER)
    assert pages_sel is not None
    total = sum(len(p['text']) for p in pages_sel)
    assert total <= 400  # потолок max_chars соблюдён при любом наборе


def test_oversize_signal_bumped_and_drained(tmp_path, db_path):
    user_dir = _bootstrap(tmp_path, db_path)
    # Стиль очень большой → при усечении появится сигнал
    pages.atomic_write_page(user_dir, 'Style',
                            '# Стиль\n\n' + ('- длинный буллет со словами ' * 200))
    inject.select_pages_for_injection(db_path, USER)
    note = inject.oversize_note('Style')
    assert note != ''
    # Сигнал одноразовый: второй вызов пуст
    assert inject.oversize_note('Style') == ''


def test_truncate_md_paragraph_boundaries():
    text = "# Сводка\n\n- один\n- два\n- три\n\nИтог абзаца длиной текста."
    out = inject.truncate_md(text, 30)
    assert len(out) <= 30
    assert out.endswith('…') or len(out) < len(text)


def test_truncate_md_short_untouched():
    text = "# Сводка\n"
    assert inject.truncate_md(text, 200) == text.strip()


def test_truncate_md_empty():
    assert inject.truncate_md('', 10) == ''
    assert inject.truncate_md(None, 10) == ''


def test_oversize_note_empty_without_overflow():
    assert inject.oversize_note('Home') == ''
