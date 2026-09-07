"""Ретенция bot_kb_raw (ТЗ п. 7.3): единая глобальная политика.

Есть валидный/восстановимый индекс (kb_watermark из _index.yaml) — удаляются
обработанные (id <= watermark) строки старше `raw.ttl_hours`; глобальный кап
`raw.max_rows` вытесняет самые старые обработанные. Необработанные строки
ретенцией не удаляются никогда (их рост ограничивают бюджеты, п. 9.2).

Индекса нет (страницы не созданы / capture_only) — строки удаляются по
`raw.no_kb_ttl_hours` и капу `max_rows` (no-kb-политика), чтобы таблица не
росла бесконечно, но накопление имело смысл при последующем включении.

В отличие от user-wiki политика одна на всю таблицу (нет per-user итерации).
"""

import logging

from . import config
from .db import connect

logger = logging.getLogger(__name__)


def _delete_older(conn, *, hours: int, id_le) -> int:
    """Удаляет строки старше `hours` по ts; при id_le — только обработанные (id <= id_le)."""
    clauses = []
    params: list = []
    if id_le is not None:
        clauses.append("id <= ?")
        params.append(id_le)
    clauses.append("ts < datetime('now', ?)")
    params.append(f"-{int(hours)} hours")
    cursor = conn.execute(
        f"DELETE FROM bot_kb_raw WHERE {' AND '.join(clauses)}", params)
    return cursor.rowcount or 0


def _trim_rows(conn, max_rows: int, *, id_le) -> int:
    """Оставляет не более max_rows самых новых (по id) строк."""
    where = []
    params: list = []
    if id_le is not None:
        where.append("id <= ?")
        params.append(id_le)
    where_sql = " AND ".join(where) if where else "1=1"

    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM bot_kb_raw WHERE {where_sql}", params).fetchone()
    excess = (row['c'] if row else 0) - max_rows
    if excess <= 0:
        return 0
    cursor = conn.execute(
        f"""
        DELETE FROM bot_kb_raw WHERE id IN (
            SELECT id FROM bot_kb_raw WHERE {where_sql}
            ORDER BY id ASC LIMIT ?
        )
        """,
        params + [excess],
    )
    return cursor.rowcount or 0


def cleanup_processed_raw(db_path: str, kb_watermark: int | None = None) -> dict:
    """Чистит bot_kb_raw по политикам. Возвращает сводку {'deleted', 'watermark'}.

    kb_watermark — watermark валидного индекса (этап 2 даст его из _index.yaml).
    None → индекса нет, работает no-kb-политика (п. 7.3).
    """
    raw_cfg = config.settings().get('raw', {})
    ttl_hours = int(raw_cfg.get('ttl_hours', 720))
    no_kb_ttl = int(raw_cfg.get('no_kb_ttl_hours', 720))
    max_rows = int(raw_cfg.get('max_rows', 10000))
    delete_only_processed = bool(raw_cfg.get('delete_only_processed', True))
    delete_unprocessed_without_kb = bool(raw_cfg.get('delete_unprocessed_without_kb', True))

    summary: dict = {'deleted': 0, 'watermark': kb_watermark}
    deleted = 0
    conn = connect(db_path)
    try:
        if kb_watermark is not None:
            # Валидный индекс: удаляются только обработанные (id <= wm) по TTL;
            # кап вытесняет самые старые обработанные. Необработанные живут.
            deleted += _delete_older(conn, hours=ttl_hours, id_le=kb_watermark)
            deleted += _trim_rows(conn, max_rows, id_le=kb_watermark)
        elif delete_unprocessed_without_kb:
            # Индекса нет: no-kb-политика — строки живут до no_kb_ttl_hours и
            # капа max_rows (накопление имеет смысл, таблица не растёт вечно).
            deleted += _delete_older(conn, hours=no_kb_ttl, id_le=None)
            deleted += _trim_rows(conn, max_rows, id_le=None)
        elif not delete_only_processed:
            # Флаги-надстройки выключены: строки живут как «обработанные-по-
            # умолчанию» до ttl_hours / max_rows (риск роста — осознанный).
            deleted += _delete_older(conn, hours=ttl_hours, id_le=None)
            deleted += _trim_rows(conn, max_rows, id_le=None)
        # Иначе (delete_only_processed=true, индекса нет): строки сохраняются
        # до появления индекса/bootstrap — чистить нечего.

        conn.commit()
        if deleted:
            summary['deleted'] = deleted
            logger.info("bot_kb retention: удалено строк=%d (watermark=%s)",
                        deleted, kb_watermark)
    finally:
        conn.close()
    return summary
