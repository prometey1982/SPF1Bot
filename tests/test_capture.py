"""Тесты сборки/захвата raw (botwiki/capture.py, ТЗ п. 9.1)."""

import sqlite3

from botwiki import capture, db

from conftest import configure_db


def _capture_cfg(wiki_config, db_path, **overrides):
    settings = configure_db(wiki_config, db_path)
    cfg = settings['capture']
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _row(user_id=111, username='vasya', first='Вася', last='Пупкин',
         chat_id=-100, thread_id=42, message_id=7, content='текст',
         content_type='text'):
    return dict(user_id=user_id, username=username, first_name=first,
                last_name=last, chat_id=chat_id, thread_id=thread_id,
                message_id=message_id, content=content, content_type=content_type)


def test_build_live_row_full(wiki_config):
    cfg = configure_db(wiki_config, ':memory:')['capture']
    row = capture.build_live_row(cfg, **_row())
    assert row is not None
    assert row['user_id'] == 111
    assert row['username'] == 'vasya'
    assert row['author_name'] == 'Вася Пупкин'
    assert row['chat_id'] == -100
    assert row['thread_id'] == 42
    assert row['message_id'] == 7
    assert row['content'] == 'текст'
    assert row['content_type'] == 'text'
    assert row['truncated'] == 0
    assert row['source'] == 'live'


def test_no_author_name(wiki_config):
    cfg = configure_db(wiki_config, ':memory:')['capture']
    row = capture.build_live_row(cfg, **{**_row(), 'first_name': None, 'last_name': None})
    assert row['author_name'] is None


def test_empty_content_skipped(wiki_config):
    cfg = configure_db(wiki_config, ':memory:')['capture']
    for content in (None, '', '   ', '\n\t'):
        row = capture.build_live_row(cfg, **{**_row(), 'content': content})
        assert row is None


def test_command_skipped_by_default(wiki_config):
    cfg = configure_db(wiki_config, ':memory:')['capture']
    row = capture.build_live_row(cfg, **{**_row(), 'content': '/start привет'})
    assert row is None
    cfg['include_bot_commands'] = True
    row = capture.build_live_row(cfg, **{**_row(), 'content': '/start привет'})
    assert row is not None


def test_missing_key_fields_skipped(wiki_config):
    cfg = configure_db(wiki_config, ':memory:')['capture']
    assert capture.build_live_row(cfg, **{**_row(), 'user_id': None}) is None
    assert capture.build_live_row(cfg, **{**_row(), 'chat_id': None}) is None
    assert capture.build_live_row(cfg, **{**_row(), 'message_id': None}) is None


def test_truncation(wiki_config):
    cfg = _capture_cfg(wiki_config, 'unused.db', max_content_chars=10)
    row = capture.build_live_row(cfg, **{**_row(), 'content': 'a' * 25})
    assert row is not None
    assert row['truncated'] == 1
    assert len(row['content']) == 10


def test_no_truncation_within_limit(wiki_config):
    cfg = _capture_cfg(wiki_config, 'unused.db', max_content_chars=10)
    row = capture.build_live_row(cfg, **{**_row(), 'content': 'короткий'})
    assert row['truncated'] == 0
    assert row['content'] == 'короткий'


def test_redact_before_insert(wiki_config, db_path):
    cfg = _capture_cfg(wiki_config, db_path)
    row = capture.build_live_row(cfg, **{**_row(), 'content': 'пиши на a@b.ru быстро'})
    assert row is not None
    assert 'a@b.ru' not in row['content']
    assert db.append_raw(db_path, row) is True

    conn = sqlite3.connect(db_path)
    stored = conn.execute("SELECT content FROM user_raw").fetchone()[0]
    conn.close()
    assert 'a@b.ru' not in stored


def test_redact_disabled(wiki_config):
    cfg = _capture_cfg(wiki_config, 'unused.db', redact=False)
    row = capture.build_live_row(cfg, **{**_row(), 'content': 'почта a@b.ru'})
    assert 'a@b.ru' in row['content']


def test_caption_content_type(wiki_config):
    cfg = _capture_cfg(wiki_config, 'unused.db')
    row = capture.build_live_row(cfg, **{**_row(), 'content_type': 'caption'})
    assert row['content_type'] == 'caption'


def test_append_raw_store_and_dedupe(wiki_config, db_path):
    cfg = _capture_cfg(wiki_config, db_path)
    row = capture.build_live_row(cfg, **_row())
    assert db.append_raw(db_path, row) is True
    assert db.append_raw(db_path, row) is False
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM user_raw").fetchone()[0]
    conn.close()
    assert count == 1
