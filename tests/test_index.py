"""Тесты индекса `_index.yaml` v3 (botwiki/index.py, ТЗ п. 8.2, 7.2)."""

import os

import yaml

from botwiki import config as wc
from botwiki import db, index, pages

USER = 123


def _user_dir(tmp_path):
    return str(tmp_path / f"wiki_{USER}")


def _mk(user_dir):
    os.makedirs(user_dir, exist_ok=True)


def _write_index(user_dir, text):
    _mk(user_dir)
    with open(os.path.join(user_dir, index.INDEX_FILE), 'w', encoding='utf-8') as f:
        f.write(text)


def _seed_raw(db_path, user_id, n=3):
    for mid in range(n):
        db.append_raw(db_path, dict(user_id=user_id, chat_id=-100,
                                    message_id=mid + 1, content='x'))


# --- Валидация ---

def test_new_index_valid():
    assert index.validate_index(index.new_index()) == []


def test_validate_duplicate_slug():
    idx = index.new_index()
    idx['pages'] = [index._minimal_page('Home'), index._minimal_page('Home')]
    errors = index.validate_index(idx)
    assert any('дублирующийся' in e for e in errors)


def test_validate_bad_status():
    idx = index.new_index()
    page = index._minimal_page('cars')
    page['status'] = 'deleted'
    idx['pages'] = [page]
    errors = index.validate_index(idx)
    assert any('status' in e for e in errors)


def test_validate_unsafe_slug():
    idx = index.new_index()
    idx['pages'] = [index._minimal_page('../evil')]
    assert any('небезопасный slug' in e for e in index.validate_index(idx))


def test_validate_wrong_schema_version():
    idx = index.new_index()
    idx['schema_version'] = 2
    assert any('schema_version' in e for e in index.validate_index(idx))


def test_validate_last_seen_null_ok():
    idx = index.new_index()
    page = index._minimal_page('Home')
    page['last_seen'] = None
    idx['pages'] = [page]
    assert index.validate_index(idx) == []


# --- Запись/чтение/бэкап ---

def test_save_load_roundtrip(tmp_path, db_path):
    user_dir = _user_dir(tmp_path)
    idx = index.new_index()
    idx['watermark'] = 42
    idx['pages'] = [index._minimal_page('Home'), index._minimal_page('Style')]
    assert index.save_index(user_dir, idx) is True

    loaded, errors = index.load_index(user_dir)
    assert errors == []
    assert loaded['watermark'] == 42
    assert loaded['schema_version'] == 3
    assert {p['slug'] for p in loaded['pages']} == {'Home', 'Style'}
    # После первой записи .bak ещё нет
    assert os.path.exists(index.bak_path(user_dir)) is False


def test_save_creates_bak_of_previous(tmp_path):
    user_dir = _user_dir(tmp_path)
    idx = index.new_index()
    idx['watermark'] = 5
    index.save_index(user_dir, idx)
    idx2 = index.new_index()
    idx2['watermark'] = 9
    index.save_index(user_dir, idx2)

    bak, errors = index.load_bak(user_dir)
    assert errors == []
    assert bak['watermark'] == 5
    loaded, _ = index.load_index(user_dir)
    assert loaded['watermark'] == 9


def test_invalid_index_not_saved(tmp_path):
    user_dir = _user_dir(tmp_path)
    bad = index.new_index()
    bad['schema_version'] = 1
    assert index.save_index(user_dir, bad) is False
    assert not os.path.exists(index.index_path(user_dir))


# --- Восстановление ---

def test_recover_from_bak(tmp_path, db_path):
    user_dir = _user_dir(tmp_path)
    idx = index.new_index()
    idx['watermark'] = 77
    index.save_index(user_dir, idx)
    # Вторая запись создаёт .bak с предыдущей версией
    idx2 = index.new_index()
    idx2['watermark'] = 78
    index.save_index(user_dir, idx2)
    # Ломаем текущий индекс
    _write_index(user_dir, "не: [валидный: yaml\n")

    loaded, errors = index.load_index(user_dir)
    assert loaded is None and errors

    recovered, status = index.ensure_index(user_dir, db_path, USER)
    assert status == 'restored_bak'
    assert recovered['watermark'] == 77


def test_rebuild_from_files_with_watermark(tmp_path, db_path):
    user_dir = _user_dir(tmp_path)
    for slug in ('Home', 'Style', 'Interests'):
        pages.atomic_write_page(user_dir, slug, f'# {slug}\nконтент')
    _seed_raw(db_path, USER, n=5)
    max_id = db.watermark(db_path, USER)

    rebuilt, status = index.ensure_index(user_dir, db_path, USER)
    assert status == 'rebuilt'
    assert rebuilt['watermark'] == max_id
    assert rebuilt['message_count'] == 0
    assert rebuilt['last_error'] == 'index rebuilt without backup'
    assert {p['slug'] for p in rebuilt['pages']} == {'Home', 'Interests', 'Style'}


def test_rebuild_empty_watermark_zero(tmp_path, db_path):
    """Пересбор при пустом user_raw → watermark = 0 (ТЗ 19.28)."""
    user_dir = _user_dir(tmp_path)
    pages.atomic_write_page(user_dir, 'Home', '# Home')
    rebuilt, status = index.ensure_index(user_dir, db_path, USER)
    assert status == 'rebuilt'
    assert rebuilt['watermark'] == 0


def test_missing_everything(tmp_path, db_path):
    user_dir = _user_dir(tmp_path)
    index_result, status = index.ensure_index(user_dir, db_path, USER)
    assert status == 'missing'
    assert index_result is None


# --- Синхронизация файлов и индекса ---

def test_sync_adds_file_without_record(tmp_path):
    user_dir = _user_dir(tmp_path)
    pages.atomic_write_page(user_dir, 'Home', '# Home')
    pages.atomic_write_page(user_dir, 'cars', '# Машины')
    idx = index.new_index()
    idx['pages'] = [index._minimal_page('Home')]
    synced = index.sync_index_with_files(user_dir, idx)
    slugs = {p['slug'] for p in synced['pages']}
    assert slugs == {'Home', 'cars'}
    car = next(p for p in synced['pages'] if p['slug'] == 'cars')
    assert car['status'] == 'active'


def test_sync_archives_record_without_file(tmp_path):
    user_dir = _user_dir(tmp_path)
    pages.atomic_write_page(user_dir, 'Home', '# Home')
    idx = index.new_index()
    idx['pages'] = [index._minimal_page('Home'), index._minimal_page('old_topic')]
    synced = index.sync_index_with_files(user_dir, idx)
    old = next(p for p in synced['pages'] if p['slug'] == 'old_topic')
    assert old['status'] == 'archived'


def test_sync_keeps_service_pages_active_without_file(tmp_path):
    user_dir = _user_dir(tmp_path)
    idx = index.new_index()
    idx['pages'] = [index._minimal_page('Home'), index._minimal_page('Style')]
    synced = index.sync_index_with_files(user_dir, idx)
    statuses = {p['slug']: p['status'] for p in synced['pages']}
    assert statuses['Home'] == 'active'
    assert statuses['Style'] == 'active'


# --- Обнаружение пользователей с wiki (для ретенции) ---

def _configure_wiki_base(tmp_path):
    cfg = {'db': 'bot.db', 'wiki': {'dir': str(tmp_path / 'wiki_root')}}
    wc.configure(cfg)
    return wc.wiki_dir()


def test_discover_watermarks(tmp_path, db_path):
    base = _configure_wiki_base(tmp_path)
    # Пользователь 1: валидный индекс с watermark
    d1 = os.path.join(base, '1')
    idx = index.new_index()
    idx['watermark'] = 50
    idx['pages'] = [index._minimal_page('Home')]
    index.save_index(d1, idx)
    # Пользователь 2: нет wiki вообще
    # Пользователь 3: пересобирается из файлов (есть .md и raw)
    d3 = os.path.join(base, '3')
    pages.atomic_write_page(d3, 'Home', '# Home')
    _seed_raw(db_path, 3, n=4)
    # Пользователь 4: битый индекс без бэкапа и файлов → не включается
    os.makedirs(os.path.join(base, '4'))
    with open(os.path.join(base, '4', index.INDEX_FILE), 'w', encoding='utf-8') as f:
        f.write("broken: [")

    wm = index.discover_watermarks(db_path)
    assert wm.get(1) == 50
    assert wm.get(3) == db.watermark(db_path, 3)
    assert 2 not in wm
    assert 4 not in wm


def test_discover_ignores_nondigit_dirs(tmp_path, db_path):
    base = _configure_wiki_base(tmp_path)
    os.makedirs(os.path.join(base, 'not_a_user'))
    with open(os.path.join(base, 'not_a_user', index.INDEX_FILE), 'w') as f:
        f.write('{}')
    assert index.discover_watermarks(db_path) == {}


def test_retention_uses_discovered_wiki(tmp_path, db_path):
    """Интеграция: wiki-пользователь из discovery не чистится no-wiki-политикой."""
    base = _configure_wiki_base(tmp_path)
    d1 = os.path.join(base, '1')
    idx = index.new_index()
    idx['watermark'] = 0  # ни одна строка не обработана
    index.save_index(d1, idx)

    # Сырьё пользователя 1 (старое) и пользователя 2 (старое), разные чаты
    cfg = {'db': db_path, 'wiki': {'dir': str(tmp_path / 'wiki_root')}}
    wc.configure(cfg)
    for uid, chat in ((1, -100), (2, -200)):
        for mid in range(2):
            db.append_raw(db_path, dict(user_id=uid, chat_id=chat,
                                        message_id=mid + 1, content='x'))
    conn = __import__('sqlite3').connect(db_path)
    conn.execute("UPDATE user_raw SET ts=datetime('now', '-500 hours')")
    conn.commit()
    conn.close()

    from botwiki import retention
    watermarks = index.discover_watermarks(db_path)
    summary = retention.cleanup_processed_raw(db_path, watermarks)
    # user 2 (без wiki) вычищен; user 1 (watermark=0, всё необработано) — жив
    assert summary['per_user'].get(2, 0) == 2
    assert summary['per_user'].get(1, 0) == 0
    assert watermarks.get(1) == 0
