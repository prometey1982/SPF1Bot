"""Тесты ретенции bot_kb_raw (botkb/retention.py, ТЗ п. 7.3).

Политики (единая глобальная таблица, watermark один на инстанс):
- watermark есть (валидный индекс): удаляются только обработанные (id <= wm)
  строки старше raw.ttl_hours; кап raw.max_rows вытесняет старые обработанные;
  необработанные (id > wm) ретенцией не удаляются;
- индекса нет + delete_unprocessed_without_kb (дефолт true): строки удаляются
  по raw.no_kb_ttl_hours и капу max_rows;
- флаги-надстройки (delete_only_processed/delete_unprocessed_without_kb) —
  как в user-wiki.
"""

import sqlite3

from botkb import config as kc
from botkb import db, retention


def _insert(db_path, message_id, *, chat_id=-100, ts_delta_hours=0,
            speaker='human', content='x'):
    db.append_raw(db_path, dict(
        speaker=speaker, chat_id=chat_id, message_id=message_id,
        content=content, source='live',
    ))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE bot_kb_raw SET ts=datetime('now', ?) WHERE chat_id=? AND message_id=?",
            (f'-{int(ts_delta_hours)} hours', chat_id, message_id),
        )
        conn.commit()
    finally:
        conn.close()


def _total(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM bot_kb_raw").fetchone()[0]
    finally:
        conn.close()


def _max_id(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT MAX(id) FROM bot_kb_raw").fetchone()[0]
    finally:
        conn.close()


def _configure(kb_config, db_path, override=None):
    cfg = kb_config(override)
    cfg['db'] = db_path
    kc.configure(cfg)


# --- Без индекса: delete_unprocessed_without_kb (по умолчанию true) ---

def test_nokb_old_rows_deleted(kb_config, kb_db_path):
    _configure(kb_config, kb_db_path)  # no_kb_ttl_hours=720 по умолчанию
    _insert(kb_db_path, 1, ts_delta_hours=2000)   # старое → удаляется
    _insert(kb_db_path, 2, ts_delta_hours=10)     # свежее → остаётся
    summary = retention.cleanup_processed_raw(kb_db_path)
    assert summary['deleted'] == 1
    assert summary['watermark'] is None
    assert _total(kb_db_path) == 1


def test_nokb_trim_to_max_rows(kb_config, kb_db_path):
    _configure(kb_config, kb_db_path, {'raw': {'max_rows': 3}})
    for mid in range(1, 8):
        _insert(kb_db_path, mid)
    retention.cleanup_processed_raw(kb_db_path)
    assert _total(kb_db_path) == 3


# --- Флаги-надстройки без индекса ---

def test_both_flags_false_ttl_default(kb_config, kb_db_path):
    _configure(kb_config, kb_db_path, {'raw': {'delete_unprocessed_without_kb': False,
                                               'delete_only_processed': False}})
    _insert(kb_db_path, 1, ts_delta_hours=5000)  # старше ttl_hours(720) → удаляется
    _insert(kb_db_path, 2, ts_delta_hours=10)
    retention.cleanup_processed_raw(kb_db_path)
    assert _total(kb_db_path) == 1


def test_delete_only_processed_without_kb_deletes_nothing(kb_config, kb_db_path):
    _configure(kb_config, kb_db_path, {'raw': {'delete_unprocessed_without_kb': False,
                                               'delete_only_processed': True}})
    _insert(kb_db_path, 1, ts_delta_hours=5000)
    retention.cleanup_processed_raw(kb_db_path)
    assert _total(kb_db_path) == 1  # индекса нет, обработанных нет → чистить нечего


# --- С watermark (валидный индекс) ---

def test_watermark_processed_old_deleted(kb_config, kb_db_path):
    _configure(kb_config, kb_db_path)
    _insert(kb_db_path, 1, ts_delta_hours=2000)  # processed, старая → удалить
    _insert(kb_db_path, 2, ts_delta_hours=10)    # processed, свежая → остаётся
    wm = _max_id(kb_db_path)
    summary = retention.cleanup_processed_raw(kb_db_path, kb_watermark=wm)
    assert summary['deleted'] == 1
    assert summary['watermark'] == wm
    assert _total(kb_db_path) == 1


def test_watermark_unprocessed_survive_ttl(kb_config, kb_db_path):
    """Строки с id > watermark (необработанные) TTL-политикой не удаляются."""
    _configure(kb_config, kb_db_path)
    _insert(kb_db_path, 1, ts_delta_hours=5000)  # id > watermark=0 → необработанная
    retention.cleanup_processed_raw(kb_db_path, kb_watermark=0)
    assert _total(kb_db_path) == 1


def test_watermark_trim_only_processed(kb_config, kb_db_path):
    """Кап вытесняет старые ОБРАБОТАННЫЕ строки, необработанные не трогает."""
    _configure(kb_config, kb_db_path, {'raw': {'max_rows': 2}})
    for mid in range(1, 6):
        _insert(kb_db_path, mid)
    conn = sqlite3.connect(kb_db_path)
    try:
        wm = conn.execute(
            "SELECT id FROM bot_kb_raw WHERE message_id=4").fetchone()[0]
    finally:
        conn.close()
    retention.cleanup_processed_raw(kb_db_path, kb_watermark=wm)
    conn = sqlite3.connect(kb_db_path)
    try:
        ids = sorted(r[0] for r in conn.execute("SELECT message_id FROM bot_kb_raw"))
    finally:
        conn.close()
    # обработанные (1..4) урезаны до 2 самых новых (3,4); 5 — необработанная, жива
    assert ids == [3, 4, 5]


def test_watermark_deletes_processed_and_keeps_bot_rows_consistent(kb_config, kb_db_path):
    """Ход бота и реплики человека живут в одной таблице и чистятся одинаково."""
    _configure(kb_config, kb_db_path)
    _insert(kb_db_path, 1, speaker='bot', ts_delta_hours=5000)
    _insert(kb_db_path, 2, speaker='human', ts_delta_hours=10)
    wm = _max_id(kb_db_path)
    retention.cleanup_processed_raw(kb_db_path, kb_watermark=wm)
    assert _total(kb_db_path) == 1
