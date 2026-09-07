"""Общие фикстуры для тестов wiki (этап 1).

Тесты НЕ импортируют bot.py (он при импорте читает config.yaml и создаёт
bot.db). Вместо этого пакет botwiki настраивается явно через configure({...})
с собственным db-путём.
"""

import sqlite3

import pytest

import botwiki
from botwiki import inject as inject_mod
import botkb


@pytest.fixture(autouse=True)
def reset_wiki_config():
    """Сбрасывает конфиг и накопленные сигналы инъекции до каждого теста."""
    botwiki.configure({})
    inject_mod._OVERSIZE.clear()
    yield


@pytest.fixture(autouse=True)
def reset_botkb_config():
    """Сбрасывает конфиг bot_kb до каждого теста."""
    botkb.configure({})
    yield


@pytest.fixture
def db_path(tmp_path):
    """Путь к временной БД со схемой user_raw и легаси-таблицей USER_INFO."""
    path = str(tmp_path / "test.db")
    botwiki.db.init_raw_table(path)
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS USER_INFO (id INTEGER PRIMARY KEY, dossier TEXT)")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def wiki_config():
    """Сырой верхнеуровневый конфиг с заданной секцией wiki + db.

    Позволяет переопределить отдельные wiki-ключи через словарь override.
    """
    def _make(override=None, top_extra=None):
        import copy
        top = {
            'db': 'bot.db',
            'allowed_group_chat_ids': [-100, 42],
            'allowed_private_users': ['admin'],
        }
        if top_extra:
            top.update(top_extra)
        cfg = copy.deepcopy(botwiki.config.WIKI_DEFAULTS)
        if override:
            for section, value in override.items():
                if isinstance(value, dict) and isinstance(cfg.get(section), dict):
                    cfg[section].update(value)
                else:
                    cfg[section] = value
        top['wiki'] = cfg
        return top
    return _make


def configure_db(wiki_config, db_path):
    """configure с привязкой БД к временному файлу."""
    cfg = wiki_config()
    cfg['db'] = db_path
    botwiki.configure(cfg)
    return botwiki.settings()


@pytest.fixture
def kb_db_path(tmp_path):
    """Путь к временной БД со схемой bot_kb_raw."""
    path = str(tmp_path / "test_botkb.db")
    botkb.db.init_raw_table(path)
    return path


@pytest.fixture
def kb_config():
    """Сырой верхнеуровневый конфиг с заданной секцией bot_kb + db.

    Позволяет переопределить отдельные bot_kb-ключи через словарь override.
    """
    def _make(override=None, top_extra=None):
        import copy
        top = {
            'db': 'bot.db',
            'allowed_group_chat_ids': [-100, 42],
            'allowed_private_users': ['admin'],
        }
        if top_extra:
            top.update(top_extra)
        cfg = copy.deepcopy(botkb.config.BOT_KB_DEFAULTS)
        if override:
            for section, value in override.items():
                if isinstance(value, dict) and isinstance(cfg.get(section), dict):
                    cfg[section].update(value)
                else:
                    cfg[section] = value
        top['bot_kb'] = cfg
        return top
    return _make


def kb_configure_db(kb_config, db_path):
    """botkb.configure с привязкой БД к временному файлу."""
    cfg = kb_config()
    cfg['db'] = db_path
    botkb.configure(cfg)
    return botkb.settings()


def table_exists(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()
