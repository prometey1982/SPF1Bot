"""Тесты admin-команд БЗ /kb_* (botkb/admin.py, ТЗ п. 13)."""

import os

from botkb import admin
from botkb import config as kc
from botkb import db
from botkb import index as index_mod
from botkb import pages as pageio


def _page_entry(slug, title, kind, keywords=()):
    return {
        'slug': slug, 'title': title, 'kind': kind, 'status': 'active',
        'keywords': list(keywords), 'aliases': [],
        'created': '2026-01-01T00:00:00', 'updated': '2026-01-01T00:00:00',
        'last_seen': '2026-01-01T00:00:00', 'hits': 0, 'quarantined': False,
    }


def _configure(tmp_path, db_path):
    root = str(tmp_path / 'kb_root')
    kc.configure({'db': db_path, 'bot_kb': {'dir': root}})
    return root


def _build_kb(tmp_path, db_path):
    """Валидная БЗ: Home/Style + knowledge-страница vyhlop + сырьё raw."""
    root = _configure(tmp_path, db_path)
    idx = index_mod.new_index()
    idx['pages'] = [
        _page_entry('Home', 'О боте', 'self'),
        _page_entry('Style', 'Стиль', 'self'),
        _page_entry('vyhlop', 'Выхлоп', 'knowledge', keywords=('противодавление',)),
    ]
    assert index_mod.save_index(root, idx) is True
    pageio.write_page('Home', '# О боте\n\n- роль\n', root)
    pageio.write_page('Style', '# Стиль\n\n- кратко\n', root)
    pageio.write_page('vyhlop', '# Выхлоп\n\n- меряй на холодной\n', root)
    db.append_raw(db_path, dict(speaker='human', chat_id=-100, message_id=1,
                                content='вопрос'))
    db.append_raw(db_path, dict(speaker='bot', chat_id=-100, message_id=2,
                                reply_to_message_id=1, content='ответ'))
    return root


def test_status_text(tmp_path, kb_db_path):
    root = _build_kb(tmp_path, kb_db_path)
    text = admin.status_text()
    assert 'БЗ бота' in text
    assert 'Home' in text and 'Style' in text and 'vyhlop' in text
    assert 'bot_kb_raw' in text
    assert 'human' in text and 'bot' in text


def test_status_no_index(tmp_path, kb_db_path):
    _configure(tmp_path, kb_db_path)  # пустая директория
    text = admin.status_text()
    assert 'индекса нет' in text or 'Bootstrap' in text


def test_show_text(tmp_path, kb_db_path):
    _build_kb(tmp_path, kb_db_path)
    text = admin.show_text()
    assert 'vyhlop' in text and 'Home' in text
    assert 'меряй на холодной' in text


def test_show_page_text(tmp_path, kb_db_path):
    _build_kb(tmp_path, kb_db_path)
    assert 'меряй на холодной' in admin.show_page_text('vyhlop')
    assert 'не найдена' in admin.show_page_text('nope')
    assert 'Небезопасный slug' in admin.show_page_text('../x')


def test_clear_requires_confirm(tmp_path, kb_db_path):
    root = _build_kb(tmp_path, kb_db_path)
    text = admin.clear_text(False)
    assert 'confirm' in text
    # ничего не удалено
    assert pageio.page_exists('Home', root) is True
    assert db.count_rows(kb_db_path) == 2


def test_clear_confirm_removes_all(tmp_path, kb_db_path):
    root = _build_kb(tmp_path, kb_db_path)
    text = admin.clear_text(True)
    assert 'БЗ очищена' in text
    assert not os.path.exists(index_mod.index_path(root))
    assert pageio.list_slugs(root) == []
    assert db.count_rows(kb_db_path) == 0
    # после очистки статус сообщает об отсутствии индекса
    assert 'индекса нет' in admin.status_text()


def test_status_shows_knowledge_flag(tmp_path, kb_db_path):
    _build_kb(tmp_path, kb_db_path)
    text = admin.status_text()
    assert 'bot_turns=False' in text
    assert 'Обучение из ответов бота' in text

    # включённый флаг + skip-фразы отражаются в статусе
    _configure(tmp_path, kb_db_path)  # сброс root не нужен; показываем флаг отдельно
    root = str(tmp_path / 'kb_root2')
    kc.configure({'db': kb_db_path, 'bot_kb': {
        'dir': root, 'knowledge': {'bot_turns': True,
                                   'bot_turn_skip_phrases': ['спасибо']}}})
    idx = index_mod.new_index()
    idx['pages'] = [
        _page_entry('Home', 'О боте', 'self'),
        _page_entry('Style', 'Стиль', 'self'),
    ]
    assert index_mod.save_index(root, idx) is True
    pageio.write_page('Home', '# О боте\n\n', root)
    pageio.write_page('Style', '# Стиль\n\n', root)
    text = admin.status_text()
    assert 'bot_turns=True' in text
    assert 'skip-фраз: 1' in text
