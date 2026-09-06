"""Слой БД для сырья wiki (таблица user_raw, ТЗ п. 7.1).

Захват — дешёвый синхронный `INSERT OR IGNORE`. Никакого ORM: как и в bot.py,
свежее соединение на вызов. Все функции принимают `db_path` явно (тест-френдли);
bot.py передаёт путь из своего конфига.
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)

USER_RAW_DDL = [
    """
    CREATE TABLE IF NOT EXISTS user_raw (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      username TEXT,
      author_name TEXT,
      chat_id INTEGER NOT NULL,
      thread_id INTEGER,
      message_id INTEGER NOT NULL,
      content TEXT,
      content_type TEXT DEFAULT 'text',
      truncated INTEGER DEFAULT 0,
      source TEXT NOT NULL DEFAULT 'live',
      inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_raw_user_ts ON user_raw(user_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_raw_ts ON user_raw(ts)",
    "CREATE INDEX IF NOT EXISTS idx_raw_user_id ON user_raw(user_id, id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_unique_message ON user_raw(chat_id, message_id)",
]


def init_raw_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for statement in USER_RAW_DDL:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def append_raw(db_path: str, row: dict) -> bool:
    """Вставляет строку raw (INSERT OR IGNORE). Возвращает True, если вставлено."""
    conn = connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO user_raw
                (user_id, username, author_name, chat_id, thread_id, message_id,
                 content, content_type, truncated, source)
            VALUES
                (:user_id, :username, :author_name, :chat_id, :thread_id, :message_id,
                 :content, :content_type, :truncated, :source)
            """,
            {
                'user_id': row.get('user_id'),
                'username': row.get('username'),
                'author_name': row.get('author_name'),
                'chat_id': row.get('chat_id'),
                'thread_id': row.get('thread_id'),
                'message_id': row.get('message_id'),
                'content': row.get('content'),
                'content_type': row.get('content_type', 'text'),
                'truncated': int(bool(row.get('truncated', 0))),
                'source': row.get('source', 'live'),
            },
        )
        conn.commit()
        inserted = cursor.rowcount > 0
        if not inserted:
            logger.info("raw: строка пропущена (дубль chat_id=%s message_id=%s)",
                        row.get('chat_id'), row.get('message_id'))
        return inserted
    finally:
        conn.close()


def user_ids(db_path: str) -> list[int]:
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT user_id FROM user_raw").fetchall()
        return [r['user_id'] for r in rows]
    finally:
        conn.close()


def get_dossier_text(db_path: str, user_id: int) -> str | None:
    """Содержимое USER_INFO.dossier (легаси-досье) для bootstrap from_dossier."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT dossier FROM USER_INFO WHERE id = ?", (user_id,)).fetchone()
        return row['dossier'] if row and row['dossier'] else None
    finally:
        conn.close()


def fetch_unprocessed(db_path: str, user_id: int, watermark: int,
                     max_messages: int | None = None,
                     max_chars: int | None = None) -> list[dict]:
    """Снимок необработанных строк (id > watermark) в лимитах снимка (ТЗ 8.1).

    Берутся первые по id строки, пока не исчерпан один из лимитов.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, username, author_name, chat_id, thread_id,
                   message_id, content, content_type, truncated, source, ts
            FROM user_raw
            WHERE user_id = ? AND id > ?
            ORDER BY id ASC
            """,
            (user_id, watermark),
        ).fetchall()

        result = []
        total_chars = 0
        for row in rows:
            content = row['content'] or ''
            if max_messages is not None and len(result) >= max_messages:
                break
            if max_chars is not None and total_chars + len(content) > max_chars:
                break
            result.append(dict(row))
            total_chars += len(content)
        return result
    finally:
        conn.close()


def count_unprocessed(db_path: str, user_id: int, watermark: int) -> int:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM user_raw WHERE user_id = ? AND id > ?",
            (user_id, watermark)).fetchone()
        return row['c'] if row else 0
    finally:
        conn.close()


def fetch_window_rows(db_path: str, user_id: int, limit: int) -> list[dict]:
    """Последние `limit` строк пользователя в порядке возрастания id.

    Используется для bootstrap.mode=limited_window (п. 9.7).
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, content, ts FROM (
                SELECT id, content, ts FROM user_raw
                WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_processed_window(db_path: str, user_id: int, watermark: int,
                           limit: int) -> list[dict]:
    """Последние `limit` ОБРАБОТАННЫХ строк (id <= watermark) по возрастанию id.

    Окно создания страниц (п. 9.4): только обработанные сообщения.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, content, ts FROM (
                SELECT id, content, ts FROM user_raw
                WHERE user_id = ? AND id <= ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (user_id, watermark, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def watermark(db_path: str, user_id: int) -> int | None:
    """Максимальный id строки пользователя (для bootstrap/watermark). None, если строк нет."""
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT MAX(id) AS max_id FROM user_raw WHERE user_id = ?",
                           (user_id,)).fetchone()
        return row['max_id'] if row is not None else None
    finally:
        conn.close()


def count_rows(db_path: str, user_id: int, source: str | None = None) -> int:
    conn = connect(db_path)
    try:
        if source is not None:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM user_raw WHERE user_id = ? AND source = ?",
                (user_id, source)).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM user_raw WHERE user_id = ?", (user_id,)).fetchone()
        return row['c'] if row else 0
    finally:
        conn.close()


def delete_rows(db_path: str, user_id: int, *, older_than_hours: int | None = None,
                ts_or_inserted: str = 'ts', source: str | None = 'live') -> int:
    """Удаляет строки пользователя; фильтр по возрасту и/или source. Возвращает число удалённых."""
    conn = connect(db_path)
    try:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if older_than_hours is not None:
            if ts_or_inserted not in ('ts', 'inserted_at'):
                raise ValueError(f"ts_or_inserted должен быть ts|inserted_at, а не {ts_or_inserted!r}")
            clauses.append(f"{ts_or_inserted} < datetime('now', ?)")
            params.append(f"-{int(older_than_hours)} hours")
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        cursor = conn.execute(
            f"DELETE FROM user_raw WHERE {' AND '.join(clauses)}", params)
        conn.commit()
        return cursor.rowcount or 0
    finally:
        conn.close()


def trim_rows(db_path: str, user_id: int, max_rows: int, source: str | None = 'live') -> int:
    """Оставляет у пользователя не более max_rows строк (самые новые по id).

    Удаляет самые старые. Возвращает число удалённых.
    """
    conn = connect(db_path)
    try:
        total = count_rows(db_path, user_id, source=source)
        excess = total - max_rows
        if excess <= 0:
            return 0
        where = "user_id = ?"
        params: list = [user_id]
        if source is not None:
            where += " AND source = ?"
            params.append(source)
        cursor = conn.execute(
            f"""
            DELETE FROM user_raw WHERE id IN (
                SELECT id FROM user_raw
                WHERE {where}
                ORDER BY id ASC
                LIMIT ?
            )
            """,
            params + [excess],
        )
        conn.commit()
        return cursor.rowcount or 0
    finally:
        conn.close()


# --- Вспомогательные запросы reconcile / команд (ТЗ п. 9.6, 13) ---

def last_username(db_path: str, user_id: int) -> str | None:
    """Последний известный @alias пользователя (для доступа к user_mentions)."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT username FROM user_raw WHERE user_id = ? AND username IS NOT NULL "
            "ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
        return row['username'] if row else None
    finally:
        conn.close()


def get_mentions_quotes(db_path: str, username: str, ttl_hours: int = 24,
                        limit: int = 50) -> list[str]:
    """Цитаты об упомянутом пользователе (user_mentions, как в bot.py)."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT quote FROM user_mentions
            WHERE target_username = ?
              AND timestamp > datetime('now', ? || ' hours')
            ORDER BY timestamp DESC LIMIT ?
            """, (username, -ttl_hours, limit)).fetchall()
        return [r['quote'] for r in rows]
    except sqlite3.OperationalError:
        # Таблица user_mentions может отсутствовать в тестовой БД
        return []
    finally:
        conn.close()


def username_aliases(db_path: str, user_id: int) -> list[str]:
    """Все встречавшиеся @alias'ы пользователя (для очистки mentions)."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT username FROM user_raw WHERE user_id = ? AND username IS NOT NULL",
            (user_id,)).fetchall()
        return [r['username'] for r in rows]
    finally:
        conn.close()


def delete_mentions_for_aliases(db_path: str, aliases: list[str]) -> int:
    if not aliases:
        return 0
    conn = connect(db_path)
    try:
        placeholders = ','.join('?' * len(aliases))
        cursor = conn.execute(
            f"DELETE FROM user_mentions WHERE target_username IN ({placeholders})",
            aliases)
        conn.commit()
        return cursor.rowcount or 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def delete_raw_all(db_path: str, user_id: int) -> int:
    conn = connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM user_raw WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount or 0
    finally:
        conn.close()


def count_uncovered_export(db_path: str, user_id: int, watermark: int) -> int:
    """Число импортированных строк вне покрытия watermark (наблюдаемость, п. 14)."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM user_raw "
            "WHERE user_id = ? AND source = 'export' AND id > ?",
            (user_id, watermark)).fetchone()
        return row['c'] if row else 0
    finally:
        conn.close()
