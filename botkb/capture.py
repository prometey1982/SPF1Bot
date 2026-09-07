"""Захват сырья БЗ бота (ТЗ п. 5.1.1, 9.1).

Решение «захватывать ли» принимает bot.py после проверки доступа. Здесь —
чистые функции: нормализация/обрезка/маскировка и сборка записи для INSERT.
Ничего не пишут на диск сами; INSERT делает db.append_raw.

Строка-человек: сообщение участника (speaker='human'), reply_to_message_id —
на что он ответил, ts = время сообщения Telegram. Строка-бот: один кусок
ответа бота (speaker='bot'), reply_to_message_id — исходное сообщение, на
которое отвечал бот, ts — время отправки куска.
"""

import logging
from datetime import datetime, timezone

from botwiki import redact

logger = logging.getLogger(__name__)


def _clean_author_name(first_name, last_name) -> str | None:
    parts = [p for p in (first_name, last_name) if p]
    name = ' '.join(parts).strip()
    return name or None


def message_date_to_ts(value) -> str | None:
    """Нормализует время сообщения Telegram в naive-UTC строку 'YYYY-MM-DD HH:MM:SS'.

    `message.date` из python-telegram-bot может быть как naive (UTC), так и
    aware: приводим к UTC и снимаем tzinfo. None → None (тогда БД проставит
    CURRENT_TIMESTAMP). Строки (например, из тестов) пропускаются как есть.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


def _mask_and_trim(capture_cfg: dict, text: str) -> tuple[str, int]:
    """Маскировка (ДО обрезки, чтобы не резать маску посередине) + обрезка по лимиту."""
    if capture_cfg.get('redact', True):
        patterns = redact.default_patterns(capture_cfg)
        text, _ = redact.mask_text(text, patterns)

    max_chars = capture_cfg.get('max_content_chars', 4000)
    truncated = 0
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        truncated = 1
    return text, truncated


def build_human_row(capture_cfg: dict, *, user_id, username, first_name=None, last_name=None,
                    chat_id, thread_id, message_id, reply_to_message_id=None,
                    content, content_type='text', source='live', ts=None) -> dict | None:
    """Сборка строки-человек для bot_kb_raw. None — сохранять нечего.

    Применяются правила ТЗ п. 9.1: пустой текст/команды не сохраняются,
    длина ≤ capture.max_content_chars (truncated=1), маскировка
    redact_patterns. Недостающие обязательные поля (user_id/chat_id/message_id)
    возвращают None.
    """
    if user_id is None or chat_id is None or message_id is None:
        return None
    if content is None:
        return None

    text = content
    if not text or not text.strip():
        return None

    # Команды боту не сохраняются (защита от мусора)
    if not capture_cfg.get('include_bot_commands', False) and text.lstrip().startswith('/'):
        return None

    text, truncated = _mask_and_trim(capture_cfg, text)
    if not text.strip():
        return None

    return {
        'speaker': 'human',
        'user_id': user_id,
        'username': username or None,
        'author_name': _clean_author_name(first_name, last_name),
        'chat_id': chat_id,
        'thread_id': thread_id,
        'message_id': message_id,
        'reply_to_message_id': reply_to_message_id,
        'content': text,
        'content_type': content_type or 'text',
        'truncated': truncated,
        'source': source,
        'ts': message_date_to_ts(ts),
    }


def build_bot_row(capture_cfg: dict, *, bot_user_id=None,
                  chat_id, thread_id, message_id, reply_to_message_id,
                  content, ts=None) -> dict | None:
    """Сборка строки-бот (один кусок ответа) для bot_kb_raw. None — сохранять нечего.

    У строки-бот нет авторства человека (username/author_name = None); текст —
    это отправленный кусок. Командный префикс не является причиной пропуска
    (ответы бота редко начинаются с '/', но триггерный слэш уместен).
    """
    if chat_id is None or message_id is None or reply_to_message_id is None:
        return None
    if content is None:
        return None

    text = content
    if not text or not text.strip():
        return None

    text, truncated = _mask_and_trim(capture_cfg, text)
    if not text.strip():
        return None

    return {
        'speaker': 'bot',
        'user_id': bot_user_id,
        'username': None,
        'author_name': None,
        'chat_id': chat_id,
        'thread_id': thread_id,
        'message_id': message_id,
        'reply_to_message_id': reply_to_message_id,
        'content': text,
        'content_type': 'text',
        'truncated': truncated,
        'source': 'live',
        'ts': message_date_to_ts(ts),
    }
