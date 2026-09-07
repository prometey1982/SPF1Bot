"""Тесты схемы bot_kb_raw (botkb/db.py, ТЗ п. 7.1)."""

import sqlite3

import pytest

from botkb import db

from conftest import table_exists


def _row(**overrides):
    base = dict(
        speaker='human', user_id=111, username='vasya', author_name='Вася П.',
        chat_id=-100, thread_id=42, message_id=9000000001,
        reply_to_message_id=None, content='привет', content_type='text',
        truncated=0, source='live', ts='2026-01-01 12:00:00',
    )
    base.update(overrides)
    return base


def test_schema_created(kb_db_path):
    assert table_exists(kb_db_path, 'bot_kb_raw')


def test_indexes(kb_db_path):
    indexes = sqlite3.connect(kb_db_path).execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='bot_kb_raw'"
    ).fetchall()
    names = [name for name, _ in indexes]
    assert 'idx_botkb_raw_ts' in names
    assert 'idx_botkb_raw_speaker_ts' in names
    assert 'idx_botkb_raw_reply' in names
    assert 'idx_botkb_raw_unique_message' in names


def test_unique_index_on_chat_message(kb_db_path):
    indexes = sqlite3.connect(kb_db_path).execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='bot_kb_raw'"
    ).fetchall()
    uniques = [sql for (sql,) in indexes if sql and 'UNIQUE' in sql.upper()]
    assert any('chat_id' in sql and 'message_id' in sql for sql in uniques)


def test_reply_index_is_composite_chat_reply(kb_db_path):
    indexes = sqlite3.connect(kb_db_path).execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='bot_kb_raw'"
    ).fetchall()
    reply = [sql for (sql,) in indexes
             if sql and 'idx_botkb_raw_reply' in sql][0]
    # составной (chat_id, reply_to_message_id) — связка строго в пределах чата
    assert 'chat_id' in reply and 'reply_to_message_id' in reply


def test_insert_and_read(kb_db_path):
    assert db.append_raw(kb_db_path, _row()) is True

    conn = sqlite3.connect(kb_db_path)
    conn.row_factory = sqlite3.Row
    stored = conn.execute("SELECT * FROM bot_kb_raw").fetchone()
    conn.close()

    assert stored['speaker'] == 'human'
    assert stored['user_id'] == 111
    assert stored['username'] == 'vasya'
    assert stored['author_name'] == 'Вася П.'
    assert stored['chat_id'] == -100
    assert stored['thread_id'] == 42
    assert stored['message_id'] == 9000000001
    assert stored['reply_to_message_id'] is None
    assert stored['content'] == 'привет'
    assert stored['content_type'] == 'text'
    assert stored['truncated'] == 0
    assert stored['source'] == 'live'
    assert stored['ts'] == '2026-01-01 12:00:00'


def test_default_speaker_is_human(kb_db_path):
    row = _row()
    del row['speaker']
    assert db.append_raw(kb_db_path, row) is True
    conn = sqlite3.connect(kb_db_path)
    speaker = conn.execute("SELECT speaker FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert speaker == 'human'


def test_ts_defaults_to_now_when_omitted(kb_db_path):
    row = _row()
    del row['ts']
    assert db.append_raw(kb_db_path, row) is True
    conn = sqlite3.connect(kb_db_path)
    ts = conn.execute("SELECT ts FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert ts is not None


def test_insert_or_ignore_dedup(kb_db_path):
    assert db.append_raw(kb_db_path, _row(message_id=5, content='первый')) is True
    # Тот же (chat_id, message_id) из другого источника — пропускается
    assert db.append_raw(kb_db_path, _row(message_id=5, content='дубль',
                                          speaker='bot', source='export')) is False

    conn = sqlite3.connect(kb_db_path)
    count = conn.execute("SELECT COUNT(*) FROM bot_kb_raw").fetchone()[0]
    content = conn.execute("SELECT content FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert count == 1
    assert content == 'первый'


def test_human_and_bot_same_message_are_different_rows(kb_db_path):
    """Уникальность по (chat_id, message_id) — сообщение участника и ответ бота
    имеют разные message_id; один человек не должен быть записан дважды."""
    assert db.append_raw(kb_db_path, _row(message_id=100)) is True
    assert db.append_raw(kb_db_path, _row(message_id=100, content='x2')) is False
    assert db.count_rows(kb_db_path) == 1


def test_bot_row_stores_reply_to(kb_db_path):
    row = _row(speaker='bot', user_id=555, username=None, author_name=None,
               message_id=200, reply_to_message_id=100, content='ответ')
    assert db.append_raw(kb_db_path, row) is True
    conn = sqlite3.connect(kb_db_path)
    conn.row_factory = sqlite3.Row
    stored = conn.execute("SELECT * FROM bot_kb_raw").fetchone()
    conn.close()
    assert stored['speaker'] == 'bot'
    assert stored['reply_to_message_id'] == 100
    assert stored['username'] is None


def test_not_null_constraints(kb_db_path):
    row = _row(chat_id=None)
    assert db.append_raw(kb_db_path, row) is False
    row = _row(message_id=None)
    assert db.append_raw(kb_db_path, row) is False

    with pytest.raises(sqlite3.IntegrityError):
        conn = sqlite3.connect(kb_db_path)
        try:
            conn.execute(
                "INSERT INTO bot_kb_raw (chat_id, message_id, content) "
                "VALUES (NULL, 5, 'x')"
            )
            conn.commit()
        finally:
            conn.close()


def test_big_message_id_64bit(kb_db_path):
    big = 9007199254740993  # > 2^53, помещается в 64-битный INTEGER
    assert db.append_raw(kb_db_path, _row(message_id=big)) is True
    conn = sqlite3.connect(kb_db_path)
    mid = conn.execute("SELECT message_id FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert mid == big


def test_watermark_empty_and_nonempty(kb_db_path):
    assert db.watermark(kb_db_path) is None
    for mid in (1, 2):
        db.append_raw(kb_db_path, _row(message_id=mid))
    conn = sqlite3.connect(kb_db_path)
    max_id = conn.execute("SELECT MAX(id) FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert db.watermark(kb_db_path) == max_id


def test_count_rows_by_speaker(kb_db_path):
    db.append_raw(kb_db_path, _row(message_id=1, speaker='human'))
    db.append_raw(kb_db_path, _row(message_id=2, speaker='human'))
    db.append_raw(kb_db_path, _row(message_id=3, speaker='bot'))
    assert db.count_rows(kb_db_path) == 3
    assert db.count_rows(kb_db_path, speaker='human') == 2
    assert db.count_rows(kb_db_path, speaker='bot') == 1


def test_fetch_unprocessed_snapshot_limits(kb_db_path):
    for mid, text in ((1, 'a'), (2, 'bb'), (3, 'ccc')):
        db.append_raw(kb_db_path, _row(message_id=mid, content=text))
    # watermark по MAX(id) первой строки
    conn = sqlite3.connect(kb_db_path)
    wm = conn.execute("SELECT id FROM bot_kb_raw WHERE message_id=1").fetchone()[0]
    conn.close()

    snap = db.fetch_unprocessed(kb_db_path, wm)
    assert len(snap) == 2
    assert [r['content'] for r in snap] == ['bb', 'ccc']

    snap_limited = db.fetch_unprocessed(kb_db_path, wm, max_messages=1)
    assert [r['content'] for r in snap_limited] == ['bb']

    snap_chars = db.fetch_unprocessed(kb_db_path, wm, max_chars=2)
    assert [r['content'] for r in snap_chars] == ['bb']


def test_fetch_unprocessed_human_and_bot_rows(kb_db_path):
    db.append_raw(kb_db_path, _row(message_id=1, speaker='bot', content='ход'))
    db.append_raw(kb_db_path, _row(message_id=2, speaker='human', content='реплика'))
    conn = sqlite3.connect(kb_db_path)
    wm = conn.execute("SELECT id FROM bot_kb_raw WHERE message_id=1").fetchone()[0]
    conn.close()
    snap = db.fetch_unprocessed(kb_db_path, wm)
    assert snap[0]['speaker'] == 'human'
    assert 'reply_to_message_id' in snap[0]
