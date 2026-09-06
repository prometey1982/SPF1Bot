"""Тесты схемы user_raw (botwiki/db.py, ТЗ п. 7.1)."""

import sqlite3

import pytest

from botwiki import db

from conftest import table_exists


def test_schema_created(db_path):
    assert table_exists(db_path, 'user_raw')


def test_unique_index_on_chat_message(db_path):
    indexes = sqlite3.connect(db_path).execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='user_raw'"
    ).fetchall()
    uniques = [sql for _, sql in indexes if sql and 'UNIQUE' in sql.upper()]
    assert any('chat_id' in sql and 'message_id' in sql and 'source' not in sql for sql in uniques)


def test_insert_and_read(db_path):
    row = {
        'user_id': 111, 'username': 'vasya', 'author_name': 'Вася П.',
        'chat_id': -100, 'thread_id': 42, 'message_id': 9000000001,
        'content': 'привет', 'content_type': 'text', 'truncated': 0, 'source': 'live',
    }
    assert db.append_raw(db_path, row) is True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    stored = conn.execute("SELECT * FROM user_raw").fetchone()
    conn.close()

    assert stored['user_id'] == 111
    assert stored['username'] == 'vasya'
    assert stored['author_name'] == 'Вася П.'
    assert stored['chat_id'] == -100
    assert stored['thread_id'] == 42
    assert stored['message_id'] == 9000000001
    assert stored['content'] == 'привет'
    assert stored['content_type'] == 'text'
    assert stored['truncated'] == 0
    assert stored['source'] == 'live'


def test_insert_or_ignore_dedup(db_path):
    base = dict(user_id=1, username=None, author_name=None, chat_id=-100,
                thread_id=None, message_id=5, content='первый', source='live')
    assert db.append_raw(db_path, base) is True
    # Тот же (chat_id, message_id) из другого источника — пропускается
    dup = dict(base, content='дубль', source='export')
    assert db.append_raw(db_path, dup) is False

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM user_raw").fetchone()[0]
    content = conn.execute("SELECT content FROM user_raw").fetchone()[0]
    conn.close()
    assert count == 1
    assert content == 'первый'


def test_not_null_constraints(db_path):
    row = dict(user_id=1, chat_id=None, message_id=5, content='x')
    assert db.append_raw(db_path, row) is False

    with pytest.raises(sqlite3.IntegrityError):
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO user_raw (user_id, chat_id, message_id, content) VALUES (1, NULL, 5, 'x')"
            )
            conn.commit()
        finally:
            conn.close()


def test_big_message_id_64bit(db_path):
    big = 9007199254740993  # > 2^53, помещается в 64-битный INTEGER
    row = dict(user_id=1, chat_id=-100, message_id=big, content='x')
    assert db.append_raw(db_path, row) is True
    conn = sqlite3.connect(db_path)
    mid = conn.execute("SELECT message_id FROM user_raw").fetchone()[0]
    conn.close()
    assert mid == big


def test_watermark_empty_and_nonempty(db_path):
    assert db.watermark(db_path, 1) is None
    for mid in (1, 2):
        db.append_raw(db_path, dict(user_id=1, chat_id=-100, message_id=mid, content='x'))
    assert db.watermark(db_path, 1) is not None
    assert db.watermark(db_path, 1) == db.watermark(db_path, 1)  # просто читается
    db.append_raw(db_path, dict(user_id=1, chat_id=-100, message_id=2, content='y'))
    conn = sqlite3.connect(db_path)
    max_id = conn.execute("SELECT MAX(id) FROM user_raw WHERE user_id=1").fetchone()[0]
    conn.close()
    assert db.watermark(db_path, 1) == max_id


def test_helpers_count_and_delete(db_path):
    for i in range(3):
        db.append_raw(db_path, dict(user_id=1, chat_id=-100, message_id=i, content=f'm{i}'))
    assert db.count_rows(db_path, 1) == 3
    assert db.delete_rows(db_path, 1, older_than_hours=0) == 0  # удаление по возрасту 0ч ничего не даёт
