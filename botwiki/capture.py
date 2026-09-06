"""Захват сырья (ТЗ п. 5.1.1, 9.1).

Решение «захватывать ли» принимает bot.py после проверки доступа (иначе для
групповых тредов и приватных списков использовалась бы та же логика в двух
местах). Здесь — чистые функции: нормализация/обрезка/маскировка и сборка
записи для INSERT. Ничего не пишут на диск сами; INSERT делает db.append_raw.
"""

import logging

from . import redact

logger = logging.getLogger(__name__)


def _clean_author_name(first_name, last_name) -> str | None:
    parts = [p for p in (first_name, last_name) if p]
    name = ' '.join(parts).strip()
    return name or None


def build_raw_row(capture_cfg: dict, *, user_id, username, author_name,
                  chat_id, thread_id, message_id, content, content_type='text',
                  source='live') -> dict | None:
    """Собирает запись user_raw по «сырому» содержимому. None — сохранять нечего.

    Применяются правила ТЗ п. 9.1: пустой текст/команды не сохраняются,
    длина ≤ capture.max_content_chars (truncated=1), маскировка redact_patterns.
    Недостающие обязательные поля (user_id/chat_id/message_id) возвращают None —
    строки без них не вставляются (иначе NULL-строки обходили бы UNIQUE-индекс).
    """
    if user_id is None or chat_id is None or message_id is None:
        return None
    if content is None:
        return None

    text = content
    if not text or not text.strip():
        return None

    # Команды боту не сохраняются (защита от мусора; для caption обычно неактуально)
    if not capture_cfg.get('include_bot_commands', False) and text.lstrip().startswith('/'):
        return None

    # Маскировка ДО обрезки: не режем маску посередине
    if capture_cfg.get('redact', True):
        patterns = redact.default_patterns(capture_cfg)
        text, _ = redact.mask_text(text, patterns)

    max_chars = capture_cfg.get('max_content_chars', 4000)
    truncated = 0
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        truncated = 1

    if not text.strip():
        return None

    return {
        'user_id': user_id,
        'username': username or None,
        'author_name': author_name or None,
        'chat_id': chat_id,
        'thread_id': thread_id,
        'message_id': message_id,
        'content': text,
        'content_type': content_type or 'text',
        'truncated': truncated,
        'source': source,
    }


def build_live_row(capture_cfg: dict, *, user_id, username, first_name=None, last_name=None,
                   chat_id, thread_id, message_id, content, content_type='text') -> dict | None:
    """Сборка записи живого захвата из полей сообщения Telegram."""
    return build_raw_row(
        capture_cfg,
        user_id=user_id,
        username=username,
        author_name=_clean_author_name(first_name, last_name),
        chat_id=chat_id,
        thread_id=thread_id,
        message_id=message_id,
        content=content,
        content_type=content_type,
        source='live',
    )
