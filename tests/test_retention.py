"""Тесты ретенции user_raw (botwiki/retention.py, ТЗ п. 7.1).

Политики:
- пользователь с watermark (валидная wiki): удаляются только обработанные
  (id <= wm) live по ttl_hours, export по import_ttl_hours от inserted_at;
- без wiki + delete_unprocessed_without_wiki: live по no_wiki_ttl_hours,
  export-строки защищены;
- оба флага false: «обработанные-по-умолчанию» — по ttl_hours/max_rows;
- delete_only_processed без wiki: нечего чистить (ничего не удаляется).
"""

import sqlite3

from botwiki import config as wc
from botwiki import db, retention

from conftest import configure_db

U = 1001


def _insert(db_path, user_id, message_id, *, source='live', ts_delta_hours=0,
            inserted_delta_hours=0, content='x', chat_id=-100):
    db.append_raw(db_path, dict(
        user_id=user_id, chat_id=chat_id, message_id=message_id,
        content=content, source=source,
    ))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE user_raw SET ts=datetime('now', ?), inserted_at=datetime('now', ?) "
            "WHERE chat_id=? AND message_id=?",
            (f'-{int(ts_delta_hours)} hours', f'-{int(inserted_delta_hours)} hours',
             chat_id, message_id),
        )
        conn.commit()
    finally:
        conn.close()


def _counts(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute(
            "SELECT source, COUNT(*) AS c FROM user_raw GROUP BY source"
        ).fetchall())
    finally:
        conn.close()


def _total(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM user_raw").fetchone()[0]
    finally:
        conn.close()


def _max_id(db_path, user_id=U):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT MAX(id) FROM user_raw WHERE user_id=?", (user_id,)).fetchone()[0]
    finally:
        conn.close()


def _configure(wiki_config, db_path, override=None):
    cfg = wiki_config(override)
    cfg['db'] = db_path
    wc.configure(cfg)


# --- Без wiki: delete_unprocessed_without_wiki (по умолчанию true) ---

def test_nowiki_old_live_deleted(wiki_config, db_path):
    _configure(wiki_config, db_path)  # no_wiki_ttl_hours=72 по умолчанию
    _insert(db_path, U, 1, ts_delta_hours=200)   # старое → удаляется
    _insert(db_path, U, 2, ts_delta_hours=10)    # свежее → остаётся
    summary = retention.cleanup_processed_raw(db_path)
    assert summary['deleted'] == 1
    assert summary['per_user'][U] == 1
    assert _total(db_path) == 1


def test_nowiki_export_rows_protected(wiki_config, db_path):
    _configure(wiki_config, db_path)
    _insert(db_path, U, 1, source='live', ts_delta_hours=200)
    _insert(db_path, U, 2, source='export', ts_delta_hours=200, inserted_delta_hours=200)
    retention.cleanup_processed_raw(db_path)
    counts = _counts(db_path)
    assert counts.get('live', 0) == 0
    assert counts.get('export', 0) == 1


def test_nowiki_trim_to_max_rows(wiki_config, db_path):
    _configure(wiki_config, db_path, {'raw': {'max_rows_per_user': 3}})
    for mid in range(1, 8):
        _insert(db_path, U, mid)
    retention.cleanup_processed_raw(db_path)
    assert _total(db_path) == 3


# --- Оба флага false: «обработанные-по-умолчанию» ---

def test_both_flags_false_ttl_default(wiki_config, db_path):
    _configure(wiki_config, db_path, {'raw': {'delete_unprocessed_without_wiki': False,
                                              'delete_only_processed': False}})
    _insert(db_path, U, 1, ts_delta_hours=500)  # старше ttl_hours(168) → удаляется
    _insert(db_path, U, 2, ts_delta_hours=10)
    retention.cleanup_processed_raw(db_path)
    assert _total(db_path) == 1


def test_delete_only_processed_without_wiki_deletes_nothing(wiki_config, db_path):
    _configure(wiki_config, db_path, {'raw': {'delete_unprocessed_without_wiki': False,
                                              'delete_only_processed': True}})
    _insert(db_path, U, 1, ts_delta_hours=500)
    retention.cleanup_processed_raw(db_path)
    assert _total(db_path) == 1  # wiki нет, обработанных нет → чистить нечего


# --- С wiki/watermark ---

def test_wiki_user_processed_only(wiki_config, db_path):
    _configure(wiki_config, db_path)
    _insert(db_path, U, 1, ts_delta_hours=500)   # processed (id<=wm), старая live → удалить
    _insert(db_path, U, 2, ts_delta_hours=10)    # processed, свежая → остаётся
    wm = _max_id(db_path)
    retention.cleanup_processed_raw(db_path, wiki_watermarks={U: wm})
    assert _total(db_path) == 1
    conn = sqlite3.connect(db_path)
    left = conn.execute("SELECT message_id FROM user_raw").fetchone()[0]
    conn.close()
    assert left == 2


def test_wiki_user_unprocessed_survive_ttl(wiki_config, db_path):
    """Строки с id > watermark (необработанные) TTL-политикой не удаляются."""
    _configure(wiki_config, db_path)
    _insert(db_path, U, 1, ts_delta_hours=500)   # id > watermark=0 → необработанная
    retention.cleanup_processed_raw(db_path, wiki_watermarks={U: 0})
    assert _total(db_path) == 1


def test_wiki_user_export_ttl_from_inserted_at(wiki_config, db_path):
    """Export: исторический ts стар, но inserted_at свежий — строка живёт."""
    _configure(wiki_config, db_path)  # import_ttl_hours=720, basis=inserted_at
    _insert(db_path, U, 1, source='export', ts_delta_hours=5000, inserted_delta_hours=1)
    wm = _max_id(db_path)
    retention.cleanup_processed_raw(db_path, wiki_watermarks={U: wm})
    counts = _counts(db_path)
    assert counts.get('export', 0) == 1
