"""Тесты сборки строк bot_kb_raw (botkb/capture.py, ТЗ п. 9.1)."""

import sqlite3
from datetime import datetime, timezone, timedelta

from botkb import capture, db

from conftest import kb_configure_db


def _capture_cfg(kb_config, db_path, **overrides):
    settings = kb_configure_db(kb_config, db_path)
    cfg = settings['capture']
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _row(user_id=111, username='vasya', first='Вася', last='Пупкин',
         chat_id=-100, thread_id=42, message_id=7, reply_to_message_id=None,
         content='текст', content_type='text', ts='2026-01-01 12:00:00'):
    return dict(user_id=user_id, username=username, first_name=first,
                last_name=last, chat_id=chat_id, thread_id=thread_id,
                message_id=message_id, reply_to_message_id=reply_to_message_id,
                content=content, content_type=content_type, ts=ts)


def test_build_human_row_full(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    row = capture.build_human_row(cfg, **_row())
    assert row is not None
    assert row['speaker'] == 'human'
    assert row['user_id'] == 111
    assert row['username'] == 'vasya'
    assert row['author_name'] == 'Вася Пупкин'
    assert row['chat_id'] == -100
    assert row['thread_id'] == 42
    assert row['message_id'] == 7
    assert row['reply_to_message_id'] is None
    assert row['content'] == 'текст'
    assert row['content_type'] == 'text'
    assert row['truncated'] == 0
    assert row['source'] == 'live'
    assert row['ts'] == '2026-01-01 12:00:00'


def test_no_author_name(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    row = capture.build_human_row(cfg, **{**_row(), 'first_name': None, 'last_name': None})
    assert row['author_name'] is None
    assert row['username'] is not None


def test_reply_to_message_id_preserved(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    row = capture.build_human_row(cfg, **{**_row(), 'reply_to_message_id': 77})
    assert row['reply_to_message_id'] == 77


def test_empty_content_skipped(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    for content in (None, '', '   ', '\n\t'):
        row = capture.build_human_row(cfg, **{**_row(), 'content': content})
        assert row is None


def test_command_skipped_by_default(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    row = capture.build_human_row(cfg, **{**_row(), 'content': '/start привет'})
    assert row is None
    cfg['include_bot_commands'] = True
    row = capture.build_human_row(cfg, **{**_row(), 'content': '/start привет'})
    assert row is not None


def test_missing_key_fields_skipped(kb_config):
    cfg = kb_configure_db(kb_config, ':memory:')['capture']
    assert capture.build_human_row(cfg, **{**_row(), 'user_id': None}) is None
    assert capture.build_human_row(cfg, **{**_row(), 'chat_id': None}) is None
    assert capture.build_human_row(cfg, **{**_row(), 'message_id': None}) is None


def test_truncation(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db', max_content_chars=10)
    row = capture.build_human_row(cfg, **{**_row(), 'content': 'a' * 25})
    assert row is not None
    assert row['truncated'] == 1
    assert len(row['content']) == 10


def test_no_truncation_within_limit(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db', max_content_chars=10)
    row = capture.build_human_row(cfg, **{**_row(), 'content': 'короткий'})
    assert row['truncated'] == 0
    assert row['content'] == 'короткий'


def test_redact_before_insert(kb_config, kb_db_path):
    cfg = _capture_cfg(kb_config, kb_db_path)
    row = capture.build_human_row(cfg, **{**_row(), 'content': 'пиши на a@b.ru быстро'})
    assert row is not None
    assert 'a@b.ru' not in row['content']
    assert db.append_raw(kb_db_path, row) is True

    conn = sqlite3.connect(kb_db_path)
    stored = conn.execute("SELECT content FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert 'a@b.ru' not in stored


def test_redact_disabled(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db', redact=False)
    row = capture.build_human_row(cfg, **{**_row(), 'content': 'почта a@b.ru'})
    assert 'a@b.ru' in row['content']


def test_caption_content_type(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    row = capture.build_human_row(cfg, **{**_row(), 'content_type': 'caption'})
    assert row['content_type'] == 'caption'


def test_ts_aware_datetime_normalized_to_utc(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    aware = datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone(timedelta(hours=3)))
    row = capture.build_human_row(cfg, **{**_row(), 'ts': aware})
    assert row['ts'] == '2026-01-01 12:00:00'


def test_ts_naive_passthrough(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    naive = datetime(2026, 1, 1, 12, 0, 0)
    row = capture.build_human_row(cfg, **{**_row(), 'ts': naive})
    assert row['ts'] == '2026-01-01 12:00:00'


def test_ts_none_omitted(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    row = capture.build_human_row(cfg, **{**_row(), 'ts': None})
    assert row['ts'] is None


# --- Строки-бот (один кусок ответа) ---

def _bot_row(chat_id=-100, thread_id=42, message_id=300, reply_to_message_id=100,
             content='кусок ответа', ts='2026-01-01 12:00:00'):
    return dict(bot_user_id=555, chat_id=chat_id, thread_id=thread_id,
                message_id=message_id, reply_to_message_id=reply_to_message_id,
                content=content, ts=ts)


def test_build_bot_row_full(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    row = capture.build_bot_row(cfg, **_bot_row())
    assert row is not None
    assert row['speaker'] == 'bot'
    assert row['user_id'] == 555
    assert row['username'] is None
    assert row['author_name'] is None
    assert row['chat_id'] == -100
    assert row['message_id'] == 300
    assert row['reply_to_message_id'] == 100
    assert row['content'] == 'кусок ответа'
    assert row['content_type'] == 'text'
    assert row['source'] == 'live'
    assert row['ts'] == '2026-01-01 12:00:00'


def test_bot_row_empty_content_skipped(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    for content in (None, '', '   '):
        assert capture.build_bot_row(cfg, **{**_bot_row(), 'content': content}) is None


def test_bot_row_leading_slash_not_skipped(kb_config):
    """Ответ бота, начинающийся с '/', — не команда-обращение: не пропускаем."""
    cfg = _capture_cfg(kb_config, 'unused.db')
    row = capture.build_bot_row(cfg, **{**_bot_row(), 'content': '/данные важны'})
    assert row is not None


def test_bot_row_requires_reply_to(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db')
    assert capture.build_bot_row(cfg, **{**_bot_row(), 'reply_to_message_id': None}) is None
    assert capture.build_bot_row(cfg, **{**_bot_row(), 'chat_id': None}) is None
    assert capture.build_bot_row(cfg, **{**_bot_row(), 'message_id': None}) is None


def test_bot_row_truncated_to_max_content_chars(kb_config):
    cfg = _capture_cfg(kb_config, 'unused.db', max_content_chars=8)
    row = capture.build_bot_row(cfg, **{**_bot_row(), 'content': 'a' * 20})
    assert row['truncated'] == 1
    assert len(row['content']) == 8


def test_append_bot_row_store(kb_config, kb_db_path):
    cfg = _capture_cfg(kb_config, kb_db_path)
    row = capture.build_bot_row(cfg, **_bot_row())
    assert db.append_raw(kb_db_path, row) is True
    assert db.count_rows(kb_db_path, speaker='bot') == 1


def test_append_human_row_store_and_dedupe(kb_config, kb_db_path):
    cfg = _capture_cfg(kb_config, kb_db_path)
    row = capture.build_human_row(cfg, **_row())
    assert db.append_raw(kb_db_path, row) is True
    assert db.append_raw(kb_db_path, row) is False
    conn = sqlite3.connect(kb_db_path)
    count = conn.execute("SELECT COUNT(*) FROM bot_kb_raw").fetchone()[0]
    conn.close()
    assert count == 1
