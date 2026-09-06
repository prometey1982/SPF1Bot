"""Маскировка чувствительных данных при захвате raw (ТЗ п. 10.4).

`content` сырья маскируется до вставки в БД, поэтому замаскированными остаются
и будущие ключевые слова/страницы. Паттерны по умолчанию: phone, email.
Список конфигурируем (`capture.redact_patterns`); незнакомые имена трактуются
как сырые регулярные выражения.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Консервативные паттерны (не маскируем обычные числа в тексте).
_PHONE = (
    r'(?<!\w)(?:\+7|8|7)?[\s\-.]*\(?'
    r'(?:\d{3}|\d{4})\)?[\s\-.]*\d{3}[\s\-.]*\d{2}[\s\-.]*\d{2}(?!\w)'
)
_EMAIL = r'(?<![\w.])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?![\w])'

REDACT_PATTERNS = {
    'phone': _PHONE,
    'email': _EMAIL,
}

# Маркер для «встроенного» имени паттерна (конфигурируемость извне).
MASK_TOKEN = {
    'phone': '[телефон скрыт]',
    'email': '[email скрыт]',
}
_GENERIC_TOKEN = '[скрыто]'


def _build_regexes(patterns: list) -> list:
    compiled = []
    for entry in patterns or []:
        if not entry:
            continue
        if isinstance(entry, str) and entry in REDACT_PATTERNS:
            compiled.append((re.compile(REDACT_PATTERNS[entry]), MASK_TOKEN[entry]))
        else:
            # Сырое регулярное выражение
            try:
                compiled.append((re.compile(entry, re.IGNORECASE), _GENERIC_TOKEN))
            except re.error as e:
                logger.warning("redact: невалидный regex-паттерн %r: %s", entry, e)
    return compiled


def mask_text(content: str, patterns: list | None = None) -> tuple[str, int]:
    """Заменяет совпадения паттернов на маски. Возвращает (текст, число масок)."""
    if not content:
        return content or '', 0
    compiled = _build_regexes(patterns)
    if not compiled:
        return content, 0

    masked = content
    total = 0
    for regex, token in compiled:
        masked, count = regex.subn(token, masked)
        total += count
    return masked, total


def default_patterns(capture_cfg: dict) -> list:
    """Список активных паттернов из конфига захвата."""
    if not capture_cfg.get('redact', True):
        return []
    return capture_cfg.get('redact_patterns', ['phone', 'email'])
