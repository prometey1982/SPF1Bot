"""Тесты индекса БЗ (botkb/index.py, ТЗ п. 7.2, 8.2)."""

import os

import pytest

from botkb import index as ix
from botkb import pages as pageio
from botkb import db

from conftest import kb_db_path


def _index_with_pages(pages=None):
    index = ix.new_index()
    index['pages'] = pages if pages is not None else [
        ix._minimal_page('Home'),
        ix._minimal_page('Style'),
    ]
    return index


def _minimal(slug):
    return ix._minimal_page(slug)


def test_new_index_shape():
    index = ix.new_index()
    assert index['schema_version'] == ix.SCHEMA_VERSION
    assert index['watermark'] == 0
    assert index['pages'] == []
    assert ix.is_usable_index(index) is True


def test_minimal_page_kind_and_defaults():
    home = _minimal('Home')
    assert home['kind'] == 'self'
    assert home['title'] == 'О боте'
    assert home['status'] == 'active'
    volvo = _minimal('volvo')
    assert volvo['kind'] == 'knowledge'
    assert volvo['title'] == 'volvo'
    assert volvo['quarantined'] is False


# --- Валидация ---

def test_validate_rejects_bad_scaffolding(kb_db_path):
    assert ix.validate_index({'schema_version': 2})  # непусто
    index = ix.new_index()
    index['watermark'] = -1
    assert ix.validate_index(index)
    index = ix.new_index()
    index['pages'] = 'nope'
    assert ix.validate_index(index)


def test_validate_kind_rules():
    # kind вне допустимого множества
    index = _index_with_pages()
    index['pages'].append({**_minimal('volvo'), 'kind': 'topic'})
    assert ix.validate_index(index)

    # служебная страница обязана иметь kind=self
    index = _index_with_pages()
    index['pages'].append({**_minimal('Home'), 'kind': 'knowledge'})
    assert ix.validate_index(index)

    # kind=self разрешён только Home/Style
    index = _index_with_pages()
    index['pages'].append({**_minimal('volvo'), 'kind': 'self'})
    assert ix.validate_index(index)

    # валидный набор: Home/Style self + тематическая knowledge
    index = _index_with_pages([_minimal('Home'), _minimal('Style'),
                               _minimal('volvo')])
    assert ix.validate_index(index) == []


def test_validate_duplicate_slug_and_bad_page():
    index = _index_with_pages()
    index['pages'].append(_minimal('volvo'))
    index['pages'].append(_minimal('volvo'))
    assert ix.validate_index(index)

    bad = _index_with_pages([_minimal('Home')])
    bad['pages'].append({'slug': 'x', 'kind': 'knowledge'})  # без title
    assert ix.validate_index(bad)


# --- Сохранение/чтение/.bak ---

def test_save_and_load_roundtrip(tmp_path):
    root = str(tmp_path)
    index = _index_with_pages([_minimal('Home'), _minimal('Style'), _minimal('volvo')])
    assert ix.save_index(root, index) is True
    assert os.path.isfile(ix.index_path(root))

    loaded, errors = ix.load_index(root)
    assert errors == []
    assert [p['slug'] for p in loaded['pages']] == ['Home', 'Style', 'volvo']
    assert [p['kind'] for p in loaded['pages']] == ['self', 'self', 'knowledge']


def test_save_creates_bak_then_restores(tmp_path):
    root = str(tmp_path)
    first = _index_with_pages([_minimal('Home'), _minimal('Style')])
    first['watermark'] = 5
    assert ix.save_index(root, first) is True
    second = _index_with_pages([_minimal('Home'), _minimal('Style')])
    second['watermark'] = 9
    assert ix.save_index(root, second) is True
    assert os.path.isfile(ix.bak_path(root))

    # ломаем текущий индекс — восстанавливаемся из .bak
    with open(ix.index_path(root), 'w', encoding='utf-8') as f:
        f.write('schema_version: 2\nnot yaml ]')  # невалидный
    loaded, errors = ix.load_index(root)
    assert loaded is None and errors

    index, status = ix.ensure_index(root, ':memory:')
    assert status == 'restored_bak'
    assert index['watermark'] == 5  # из .bak (первая версия)


def test_ensure_index_saves_invalid_guard(tmp_path):
    root = str(tmp_path)
    index = _index_with_pages([_minimal('Home')])
    index['pages'].append({'slug': 'x', 'kind': 'self'})  # невалидно: self для x
    assert ix.save_index(root, index) is False
    assert not os.path.exists(ix.index_path(root))


# --- Пересбор по файлам ---

def test_rebuild_and_ensure_missing(tmp_path, kb_db_path):
    root = str(tmp_path)
    # пусто: ни индекса, ни бэкапа, ни страниц
    index, status = ix.ensure_index(root, kb_db_path)
    assert index is None
    assert status == 'missing'


def test_rebuild_from_files_sets_kind_and_watermark(tmp_path, kb_db_path):
    root = str(tmp_path)
    for slug in ('Home', 'Style', 'volvo'):
        pageio.write_page(slug, f'#{slug}', root)
    # сырьё в глобальной таблице
    db.append_raw(kb_db_path, dict(speaker='human', user_id=1, chat_id=-100,
                                   message_id=1, content='x'))
    db.append_raw(kb_db_path, dict(speaker='bot', chat_id=-100, message_id=2,
                                   reply_to_message_id=1, content='y'))

    index, status = ix.ensure_index(root, kb_db_path)
    assert status == 'rebuilt'
    assert index['watermark'] == db.watermark(kb_db_path)
    assert index['message_count'] == 0
    kinds = {p['slug']: p['kind'] for p in index['pages']}
    assert kinds == {'Home': 'self', 'Style': 'self', 'volvo': 'knowledge'}


# --- Синхронизация файлов и индекса ---

def test_sync_adds_file_without_record(tmp_path):
    root = str(tmp_path)
    index = _index_with_pages([_minimal('Home'), _minimal('Style')])
    pageio.write_page('turbo', '# turbo', root)  # без записи в индексе
    synced = ix.sync_index_with_files(root, index)
    slugs = [p['slug'] for p in synced['pages']]
    assert 'turbo' in slugs
    rec = ix.find_page(synced, 'turbo')
    assert rec['kind'] == 'knowledge'


def test_sync_archives_record_without_file(tmp_path):
    root = str(tmp_path)
    for slug in ('Home', 'Style'):
        pageio.write_page(slug, f'#{slug}', root)
    index = _index_with_pages([_minimal('Home'), _minimal('Style'),
                               _minimal('volvo')])
    synced = ix.sync_index_with_files(root, index)
    volvo = ix.find_page(synced, 'volvo')
    assert volvo['status'] == 'archived'
    assert volvo['kind'] == 'knowledge'
    home = ix.find_page(synced, 'Home')
    assert home['status'] == 'active'  # служебная не архивируется


def test_kb_valid(tmp_path, kb_db_path):
    root = str(tmp_path)
    assert ix.kb_valid(kb_db_path, root) is False
    index = _index_with_pages([_minimal('Home'), _minimal('Style')])
    assert ix.save_index(root, index) is True
    # Home/Style ещё нет файлами
    assert ix.kb_valid(kb_db_path, root) is False
    for slug in ('Home', 'Style'):
        pageio.write_page(slug, f'#{slug}', root)
    assert ix.kb_valid(kb_db_path, root) is True


# --- Watermark для ретенции ---

def test_watermark_for_retention(tmp_path, kb_db_path):
    root = str(tmp_path)
    assert ix.watermark_for_retention(kb_db_path, root) is None
    db.append_raw(kb_db_path, dict(speaker='human', chat_id=-100, message_id=1, content='x'))
    wm = db.watermark(kb_db_path)
    index = _index_with_pages()
    index['watermark'] = wm
    assert ix.save_index(root, index) is True
    assert ix.watermark_for_retention(kb_db_path, root) == wm
