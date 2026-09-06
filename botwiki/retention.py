"""Ретенция user_raw (ТЗ п. 7.1): две политики (+ защита export-строк).

Запускается при старте и периодически. Пользователь «с watermark» = валидная
wiki / восстановимый индекс (этап 2 даст словарь watermark'ов); до этого все
пользователи считаются «без wiki» и чистятся по no_wiki_ttl_hours, чтобы
user_raw не рос бесконечно по участникам тредов.
"""

import logging

from . import config
from .db import connect

logger = logging.getLogger(__name__)


def _delete_where(conn, user_id: int, *, source: str,
                  column: str, hours: int, id_le) -> int:
    """Удаляет строки пользователя: источник, возраст по column, необязательно id<=id_le."""
    clauses = ["user_id = ?", "source = ?"]
    params: list = [user_id, source]
    if id_le is not None:
        clauses.append("id <= ?")
        params.append(id_le)
    clauses.append(f"{column} < datetime('now', ?)")
    params.append(f"-{int(hours)} hours")
    cursor = conn.execute(f"DELETE FROM user_raw WHERE {' AND '.join(clauses)}", params)
    return cursor.rowcount or 0


def _trim_rows(conn, user_id: int, max_rows: int, *, source: str, id_le) -> int:
    """Оставляет не более max_rows самых новых (по id) строк пользователя."""
    where = ["user_id = ?"]
    params: list = [user_id]
    if id_le is not None:
        where.append("id <= ?")
        params.append(id_le)
    if source is not None:
        where.append("source = ?")
        params.append(source)
    where_sql = " AND ".join(where)

    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM user_raw WHERE {where_sql}", params).fetchone()
    excess = (row['c'] if row else 0) - max_rows
    if excess <= 0:
        return 0
    cursor = conn.execute(
        f"""
        DELETE FROM user_raw WHERE id IN (
            SELECT id FROM user_raw WHERE {where_sql}
            ORDER BY id ASC LIMIT ?
        )
        """,
        params + [excess],
    )
    return cursor.rowcount or 0


def cleanup_processed_raw(db_path: str, wiki_watermarks: dict[int, int] | None = None) -> dict:
    """Чистит user_raw по политикам. Возвращает сводку {deleted, per_user}.

    wiki_watermarks: user_id -> watermark пользователей с валидной wiki
    (этап 2). Пусто → все пользователи безвики-политики.
    """
    if wiki_watermarks is None:
        wiki_watermarks = {}
    raw_cfg = config.settings().get('raw', {})
    ttl_hours = int(raw_cfg.get('ttl_hours', 168))
    no_wiki_ttl = int(raw_cfg.get('no_wiki_ttl_hours', 72))
    max_rows = int(raw_cfg.get('max_rows_per_user', 500))
    delete_only_processed = bool(raw_cfg.get('delete_only_processed', True))
    delete_unprocessed_without_wiki = bool(raw_cfg.get('delete_unprocessed_without_wiki', True))
    import_ttl = int(raw_cfg.get('import_ttl_hours', 720))
    import_basis = raw_cfg.get('import_ttl_basis', 'inserted_at')
    if import_basis not in ('ts', 'inserted_at'):
        import_basis = 'inserted_at'

    summary: dict = {'deleted': 0, 'per_user': {}}
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT user_id FROM user_raw").fetchall()
        for row in rows:
            user_id = row['user_id']
            deleted = 0
            wm = wiki_watermarks.get(user_id)

            if wm is not None:
                # Пользователь с валидной wiki / watermark: удаляются только
                # обработанные (id <= wm) строки по TTL; export — по import_ttl
                # от inserted_at (не исторического ts).
                deleted += _delete_where(conn, user_id, source='live', column='ts',
                                         hours=ttl_hours, id_le=wm)
                deleted += _delete_where(conn, user_id, source='export',
                                         column=import_basis, hours=import_ttl, id_le=wm)
                # max_rows вытесняет самые старые обработанные строки
                deleted += _trim_rows(conn, user_id, max_rows, source=None, id_le=wm)
            elif delete_unprocessed_without_wiki:
                # Без wiki/watermark: строки удаляются независимо от watermark,
                # чтобы user_raw не рос по участникам, которым wiki не нужна.
                # Export-строки исключаются (защита истории до backfill).
                deleted += _delete_where(conn, user_id, source='live', column='ts',
                                         hours=no_wiki_ttl, id_le=None)
                deleted += _trim_rows(conn, user_id, max_rows, source='live', id_le=None)
            elif not delete_only_processed:
                # Оба флага false: строки живут как «обработанные-по-умолчанию»
                # до ttl_hours / max_rows_per_user (риск роста — осознанный).
                deleted += _delete_where(conn, user_id, source='live', column='ts',
                                         hours=ttl_hours, id_le=None)
                deleted += _trim_rows(conn, user_id, max_rows, source='live', id_le=None)
            # Иначе (delete_only_processed, но wiki нет): обработанных строк нет —
            # чистить нечего, всё сохраняется до появления wiki/bootstrap.

            if deleted:
                summary['deleted'] += deleted
                summary['per_user'][user_id] = deleted
                logger.info("raw retention: user_id=%d удалено строк=%d", user_id, deleted)
        conn.commit()
    finally:
        conn.close()
    return summary
