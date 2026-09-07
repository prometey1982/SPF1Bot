"""Тесты инъекции БЗ (botkb/inject.py, ТЗ п. 12): выбор, лимиты, кэш по mtime."""

import os

from botkb import config as kc
from botkb import db
from botkb import index as index_mod
from botkb import inject
from botkb import pages as pageio


def _page_entry(slug, title, kind, keywords=()):
    return {
        'slug': slug, 'title': title, 'kind': kind, 'status': 'active',
        'keywords': list(keywords), 'aliases': [],
        'created': '2026-01-01T00:00:00', 'updated': '2026-01-01T00:00:00',
        'last_seen': '2026-01-01T00:00:00', 'hits': 0, 'quarantined': False,
    }


def _configure(tmp_path, db_path, overrides=None):
    bot_kb = {'dir': str(tmp_path / 'kb_root')}
    if overrides:
        for section, value in overrides.items():
            if isinstance(value, dict):
                bot_kb.setdefault(section, {}).update(value)
            else:
                bot_kb[section] = value
    kc.configure({'db': db_path, 'bot_kb': bot_kb})
    return bot_kb['dir']


def _build_kb(tmp_path, db_path, *, home='# О боте\n\n- роль бота',
              style='# Стиль\n\n- кратко', topic='# Выхлоп\n\n- меряй противодавление'):
    """Валидная БЗ: индекс + Home.md/Style.md + knowledge-страница vyhlop."""
    root = _configure(tmp_path, db_path)
    idx = index_mod.new_index()
    idx['pages'] = [
        _page_entry('Home', 'О боте', 'self'),
        _page_entry('Style', 'Стиль', 'self'),
        _page_entry('vyhlop', 'Выхлоп', 'knowledge', keywords=('противодавление',)),
    ]
    assert index_mod.save_index(root, idx) is True
    pageio.write_page('Home', home, root)
    pageio.write_page('Style', style, root)
    pageio.write_page('vyhlop', topic, root)
    inject.invalidate_cache(root)
    return root


def test_select_none_without_valid_kb(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path)  # пустая директория — нет страниц
    assert inject.select_pages_for_injection(db_path=kb_db_path, query='противодавление') is None


def test_select_self_and_topic(tmp_path, kb_db_path):
    root = _build_kb(tmp_path, kb_db_path)
    sel = inject.select_pages_for_injection(db_path=kb_db_path,
                                            query='как мерить противодавление')
    assert sel is not None
    slugs = [s['slug'] for s in sel]
    assert 'Home' in slugs and 'Style' in slugs and 'vyhlop' in slugs
    by_slug = {s['slug']: s for s in sel}
    assert by_slug['Home']['title'] == 'О боте'
    assert by_slug['Style']['title'] == 'Стиль'
    assert 'меряй противодавление' in by_slug['vyhlop']['text']
    # суммарно в пределах max_chars
    total = sum(len(s['text']) for s in sel)
    assert total <= kc.settings()['inject']['max_chars']


def test_include_home_style_off_keeps_topic(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path,
                      {'inject': {'include_home': False, 'include_style': False}})
    # файлы Home/Style обязаны существовать (валидность), но не инжектятся
    _build_kb_files_only(root)
    inject.invalidate_cache(root)
    sel = inject.select_pages_for_injection(db_path=kb_db_path, query='противодавление')
    assert sel is not None
    assert all(s['slug'] != 'Home' and s['slug'] != 'Style' for s in sel)
    assert [s['slug'] for s in sel] == ['vyhlop']


def _build_kb_files_only(root):
    idx = index_mod.new_index()
    idx['pages'] = [
        _page_entry('Home', 'О боте', 'self'),
        _page_entry('Style', 'Стиль', 'self'),
        _page_entry('vyhlop', 'Выхлоп', 'knowledge', keywords=('противодавление',)),
    ]
    assert index_mod.save_index(root, idx) is True
    pageio.write_page('Home', '# О боте\n\n', root)
    pageio.write_page('Style', '# Стиль\n\n', root)
    pageio.write_page('vyhlop', '# Выхлоп\n\n- тезис\n', root)


def test_oversize_topic_dropped(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path)
    big = '# Выхлоп\n\n- ' + ('тезис ' * 600)  # больше inject.page_max_chars
    _build_kb(tmp_path, kb_db_path, topic=big)
    inject.invalidate_cache(root)
    sel = inject.select_pages_for_injection(db_path=kb_db_path, query='противодавление')
    assert sel is not None
    assert 'vyhlop' not in [s['slug'] for s in sel]  # не влезла — сброшена
    assert 'Home' in [s['slug'] for s in sel]


def test_cache_refresh_on_mtime_change(tmp_path, kb_db_path):
    root = _build_kb(tmp_path, kb_db_path,
                     home='# О боте\n\n- старая роль')
    sel = inject.select_pages_for_injection(db_path=kb_db_path, query='')
    assert 'старая роль' in [s['text'] for s in sel if s['slug'] == 'Home'][0]

    # Меняем Home.md и принудительно сдвигаем mtime вперёд (детерминированно)
    path = pageio.page_path('Home', root)
    old_ns = os.stat(path).st_mtime_ns
    pageio.write_page('Home', '# О боте\n\n- новая роль (исправлено)', root)
    os.utime(path, ns=(old_ns + 2_000_000_000, old_ns + 2_000_000_000))

    sel = inject.select_pages_for_injection(db_path=kb_db_path, query='')
    home_text = [s['text'] for s in sel if s['slug'] == 'Home'][0]
    assert 'новая роль' in home_text
    assert 'старая роль' not in home_text


def test_build_system_message_format():
    pages = [
        {'slug': 'Home', 'title': 'О боте', 'text': 'роль'},
        {'slug': 'vyhlop', 'title': 'Выхлоп', 'text': 'тезис'},
    ]
    msg = inject.build_system_message(pages)
    assert 'данные из переписки' in msg or 'данные' in msg
    assert '[О боте]' in msg and '[Выхлоп]' in msg
    assert 'роль' in msg and 'тезис' in msg
