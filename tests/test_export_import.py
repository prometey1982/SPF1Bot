"""Тесты импорта экспорта Telegram (botwiki/export_import.py, ТЗ п. 9.8)."""

import json
import os

import sqlite3

from botwiki import config as wc
from botwiki import db, export_import

USER = 111


def _write_export(tmp_path, allowed_dir, messages=None, chat_id=123,
                  chat_name='Гараж'):
    os.makedirs(allowed_dir, exist_ok=True)
    payload = {
        'chats': {'list': [{
            'name': chat_name, 'type': 'private_supergroup', 'id': chat_id,
            'messages': messages if messages is not None else [],
        }]},
    }
    path = os.path.join(allowed_dir, 'result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def _base_messages():
    return [
        {'id': 1, 'type': 'message', 'date_unixtime': 1600000000,
         'from': 'Вася', 'from_id': 'user111', 'text': 'Люблю котиков'},
        {'id': 2, 'type': 'message', 'date_unixtime': 1600000001,
         'from': 'Петя', 'from_id': 'user222', 'text': [
             {'type': 'text', 'text': 'Смотри '},
             {'type': 'mention', 'text': '@вася'},
             {'type': 'text', 'text': ' про машины'},
         ]},
        {'id': 3, 'type': 'message', 'date_unixtime': 1600000002,
         'from': 'Вася', 'from_id': 'user111',
         'text': 'подпись к фото', 'media_type': 'photo'},
        {'id': 4, 'type': 'service', 'text': 'Вася создал группу'},
        {'id': 5, 'type': 'message', 'from_id': 'user111', 'text': '/start привет'},
        {'id': 6, 'type': 'message', 'from_id': 'channel100', 'text': 'канал'},
        {'id': 7, 'type': 'message', 'date_unixtime': 1600000003,
         'from': 'Вася', 'from_id': 'user111', 'text': ''},
    ]


def _configure(tmp_path, db_path, messages=None, chat_map=None, **import_kw):
    allowed_dir = str(tmp_path / 'imports')
    cfg = {
        'db': db_path,
        'wiki': {'import': {
            'enabled': True,
            'allowed_dir': allowed_dir,
            'batch_size': 500,
            'exclude_types': ['service'],
            'chat_map': chat_map if chat_map is not None else [
                {'name': 'Гараж', 'export_id': 123, 'chat_id': -100, 'thread_id': 7},
            ],
        }},
    }
    for k, v in import_kw.items():
        cfg['wiki']['import'][k] = v
    wc.configure(cfg)
    return allowed_dir


def _rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM user_raw ORDER BY id").fetchall()]
    finally:
        conn.close()


# --- Чистые функции ---

def test_flatten_export_text():
    assert export_import.flatten_export_text('просто текст') == 'просто текст'
    parts = [{'type': 'text', 'text': 'A  '}, {'type': 'custom_emoji', 'text': 'B'},
             {'type': 'link', 'text': 'C'}]
    assert export_import.flatten_export_text(parts) == 'A B C'
    assert export_import.flatten_export_text([]) is None
    assert export_import.flatten_export_text(None) is None


def test_extract_export_user_id():
    assert export_import.extract_export_user_id('user406526542') == 406526542
    assert export_import.extract_export_user_id(42) == 42
    assert export_import.extract_export_user_id('channel100') is None
    assert export_import.extract_export_user_id(None) is None
    assert export_import.extract_export_user_id('user') is None


# --- Путь ---

def test_resolve_path(tmp_path):
    allowed = str(tmp_path / 'imports')
    file_path = _write_export(tmp_path, allowed, [])
    resolved, err = export_import.resolve_allowed_path('result.json', allowed)
    assert err is None
    assert resolved == os.path.realpath(file_path)
    # Абсолютный путь внутри allowed_dir разрешён
    resolved, err = export_import.resolve_allowed_path(file_path, allowed)
    assert err is None
    # Выход через .. отклоняется
    _, err = export_import.resolve_allowed_path('../secret.json', allowed)
    assert err is not None
    # Несуществующий файл
    _, err = export_import.resolve_allowed_path('missing.json', allowed)
    assert err is not None


# --- Импорт ---

def test_import_basic(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path, messages=_base_messages())
    file_path = _write_export(tmp_path, allowed, _base_messages())
    stats = export_import.perform_import(db_path, file_path, 123)
    assert stats.error is None
    assert stats.inserted == 3   # ids 1,2,3
    assert stats.skipped == 4    # service / команда / канал / пустой

    rows = _rows(db_path)
    assert len(rows) == 3
    for row in rows:
        assert row['source'] == 'export'
        assert row['username'] is None
        assert row['chat_id'] == -100
        assert row['thread_id'] == 7

    r1 = next(r for r in rows if r['message_id'] == 1)
    assert r1['author_name'] == 'Вася'
    assert r1['content'] == 'Люблю котиков'
    r2 = next(r for r in rows if r['message_id'] == 2)
    assert 'Смотри' in r2['content'] and 'машины' in r2['content']
    r3 = next(r for r in rows if r['message_id'] == 3)
    assert r3['content_type'] == 'caption'
    assert r1['ts'] is not None and r1['ts'].startswith('2020')


def test_import_idempotent(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path)
    file_path = _write_export(tmp_path, allowed, _base_messages())
    export_import.perform_import(db_path, file_path, 123)
    stats2 = export_import.perform_import(db_path, file_path, 123)
    assert stats2.inserted == 0
    assert stats2.duplicates == 3
    assert len(_rows(db_path)) == 3


def test_import_without_chat_map_rejected(tmp_path, db_path):
    _configure(tmp_path, db_path, chat_map=[])
    file_path = _write_export(tmp_path, os.path.join(str(tmp_path), 'imports'),
                              _base_messages())
    stats = export_import.perform_import(db_path, file_path, 123)
    assert stats.error is not None
    assert 'не сопоставлен' in stats.error
    assert len(_rows(db_path)) == 0


def test_import_unknown_chat(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path)
    file_path = _write_export(tmp_path, allowed, _base_messages())
    stats = export_import.perform_import(db_path, file_path, 999)
    assert stats.error is not None
    assert 'не найден' in stats.error


def test_dry_run_writes_nothing(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path)
    file_path = _write_export(tmp_path, allowed, _base_messages())
    stats = export_import.perform_import(db_path, file_path, 123, dry_run=True)
    assert stats.error is None
    assert stats.inserted == 3
    assert len(_rows(db_path)) == 0


def test_file_over_limit_rejected(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path, max_file_mb=1)
    os.makedirs(allowed, exist_ok=True)
    big = os.path.join(allowed, 'big.json')
    with open(big, 'wb') as f:
        f.write(b'x' * (1024 * 1024 + 1))   # > 1 МБ
    stats = export_import.perform_import(db_path, big, 123)
    assert stats.error is not None
    assert 'больше лимита' in stats.error


def test_redact_and_truncate(tmp_path, db_path):
    allowed = _configure(tmp_path, db_path, max_content_chars=20)
    msg = [{'id': 1, 'type': 'message', 'from': 'Вася', 'from_id': 'user111',
            'text': 'Почта a@b.ru и ' + 'x' * 100}]
    file_path = _write_export(tmp_path, allowed, msg)
    stats = export_import.perform_import(db_path, file_path, 123)
    assert stats.inserted == 1
    row = _rows(db_path)[0]
    assert 'a@b.ru' not in row['content']
    assert row['truncated'] == 1
    assert len(row['content']) <= 20
