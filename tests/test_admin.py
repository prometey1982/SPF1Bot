"""Тесты admin-команд wiki (botwiki/admin.py, ТЗ п. 13)."""

import asyncio
import os
import sqlite3

from botwiki import config as wc
from botwiki import admin, db, manager, pages

USER = 4242


def run(coro):
    return asyncio.run(coro)


def _configure(tmp_path, db_path):
    wc.configure({'db': db_path, 'wiki': {'dir': str(tmp_path / 'wiki_root')}})


def _bootstrap(tmp_path, db_path):
    _configure(tmp_path, db_path)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    run(manager.WikiManager()._bootstrap(USER, db_path, user_dir))
    return user_dir


def _ensure_mentions_table(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_mentions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "author_id INTEGER, target_username TEXT, chat_id INTEGER, quote TEXT, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        conn.commit()
    finally:
        conn.close()


def test_show_wiki_returns_active(tmp_path, db_path):
    _bootstrap(tmp_path, db_path)
    text = admin.show_wiki(db_path, USER)
    assert text is not None
    assert 'Сводка' in text


def test_show_wiki_page(tmp_path, db_path):
    user_dir = _bootstrap(tmp_path, db_path)
    pages.atomic_write_page(user_dir, 'cars', '# Машины\n- факт')
    text = admin.show_wiki_page(db_path, USER, 'cars')
    assert '# Машины' in text
    assert 'не найдена' in admin.show_wiki_page(db_path, USER, 'nope')


def test_wiki_status_fields(tmp_path, db_path):
    _bootstrap(tmp_path, db_path)
    text = admin.wiki_status(db_path, USER)
    assert 'watermark:' in text
    assert 'message_count:' in text
    assert 'Страницы:' in text
    assert '/Home' in text


def test_clear_wiki_removes_everything_but_dossier(tmp_path, db_path):
    _ensure_mentions_table(db_path)
    user_dir = _bootstrap(tmp_path, db_path)
    db.append_raw(db_path, dict(user_id=USER, username='vasya', chat_id=-1,
                                message_id=1, content='привет'))
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO user_mentions (author_id, target_username, chat_id, quote) "
                 "VALUES (1, 'vasya', -1, 'цитата')")
    conn.commit()
    conn.close()

    # dossier не трогаем — положим и проверим
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO USER_INFO (id, dossier) VALUES (?, 'досье')", (USER,))
    conn.commit()
    conn.close()

    text = admin.clear_wiki(db_path, USER)
    assert 'упоминания=1' in text
    assert not os.path.isdir(user_dir)
    conn = sqlite3.connect(db_path)
    raw = conn.execute("SELECT COUNT(*) FROM user_raw").fetchone()[0]
    mentions = conn.execute("SELECT COUNT(*) FROM user_mentions").fetchone()[0]
    dossier = conn.execute("SELECT dossier FROM USER_INFO WHERE id=?", (USER,)).fetchone()
    conn.close()
    assert raw == 0
    assert mentions == 0
    assert dossier == ('досье',)


def test_wiki_status_missing_wiki(tmp_path, db_path):
    _configure(tmp_path, db_path)
    text = admin.wiki_status(db_path, 999)
    assert 'отсутствует' in text
