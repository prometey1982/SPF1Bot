"""Детерминированное создание тематических страниц (ТЗ п. 9.4, 9.5).

Состояние между апдейтами не накапливается: кластеризация пересчитывается с
нуля по скользящему окну обработанных строк (id <= watermark), поэтому она
детерминирована и устойчива к рестартам. Униграммы (MVP, п. 9.4/3.5).

Cooldown отклонённых предложений хранится в _index.yaml
(page_proposal_cooldowns): ключ — токен-кандидат; просроченные удаляются,
число записей ограничено.
"""

import logging
from datetime import datetime, timezone

import yaml

from . import pages as pageio
from . import textutil

logger = logging.getLogger(__name__)


def detect_candidates(rows: list[dict], min_repeats: int) -> list[dict]:
    """Токены, встреченные в ≥ min_repeats РАЗНЫХ сообщениях окна (п. 9.4).

    Возвращает [{'token', 'count', 'messages': [row_id, ...]}] по убыванию
    числа сообщений (одно сообщение — один повтор).
    """
    if not rows:
        return []
    counts: dict[str, set] = {}
    for row in rows:
        content = row.get('content') or ''
        tokens = textutil.token_set(content)
        for token in tokens:
            counts.setdefault(token, set()).add(row['id'])
    candidates = [
        {'token': token, 'count': len(ids), 'messages': sorted(ids)}
        for token, ids in counts.items()
        if len(ids) >= min_repeats
    ]
    candidates.sort(key=lambda c: (-c['count'], c['token']))
    return candidates


def _page_word_tokens(page: dict) -> tuple[set, set]:
    """(keywords+aliases+title токены, slug-слова) страницы."""
    meta = ' '.join(page.get('keywords') or [])
    meta += ' ' + ' '.join(page.get('aliases') or [])
    meta += ' ' + (page.get('title') or '')
    tokens = textutil.token_set(meta)
    slug_words = {w for w in re_sub_slug(page.get('slug') or '')}
    return tokens, slug_words


def re_sub_slug(slug: str) -> list[str]:
    return [w for w in slug.lower().replace('-', ' ').split() if w]


def check_page_overlap(token: str, pages: list[dict]):
    """Пересечение кандидата-токена со страницей (активной или архивной).

    Критерий (п. 3.6/9.4): совпадение нормализованного slug/слова slug, либо
    значимый общий токен с keywords/aliases/title. Возвращает запись страницы
    при пересечении, иначе None.
    """
    for page in pages:
        meta_tokens, slug_words = _page_word_tokens(page)
        if token in slug_words or token in meta_tokens:
            return page
    return None


def _utcnow_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def is_cooldown_active(index: dict, token: str, cooldown_hours: int) -> bool:
    if cooldown_hours <= 0:
        return False
    ts = (index.get('page_proposal_cooldowns') or {}).get(token)
    if not ts:
        return False
    try:
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return False
    age_hours = (_utcnow_naive() - parsed).total_seconds() / 3600
    return age_hours < cooldown_hours


def prune_cooldowns(index: dict, cooldown_hours: int, max_entries: int) -> int:
    """Удаляет просроченные записи cooldown и ужимает до max_entries. Число удалённых."""
    cooldowns = index.get('page_proposal_cooldowns') or {}
    if not cooldowns:
        index['page_proposal_cooldowns'] = {}
        return 0
    now = _utcnow_naive()
    removed = 0
    entries = []
    for token, ts in list(cooldowns.items()):
        try:
            parsed = datetime.fromisoformat(ts)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            parsed = now
        if (now - parsed).total_seconds() / 3600 >= cooldown_hours:
            del cooldowns[token]
            removed += 1
        else:
            entries.append((parsed, token))
    if len(cooldowns) > max_entries:
        entries.sort(key=lambda x: x[0])
        excess = len(entries) - max_entries
        for _, token in entries[:excess]:
            cooldowns.pop(token, None)
            removed += 1
    return removed


def set_cooldown(index: dict, token: str, when: str):
    index.setdefault('page_proposal_cooldowns', {})[token] = when


def parse_page_proposal(text: str) -> dict | None:
    """Разбирает ответ LLM вида YAML {slug, title, keywords, aliases, content}.

    Возвращает dict с нормализованным slug и содержимым страницы; None при
    невалидном ответе.
    """
    if not text:
        return None
    try:
        data = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    slug = pageio.normalize_slug(data.get('slug'))
    if slug is None:
        logger.info("topics: невалидный slug в предложении: %r", data.get('slug'))
        return None
    content = data.get('content')
    if not isinstance(content, str) or not content.strip():
        logger.info("topics: пустое содержимое в предложении для %r", slug)
        return None
    return {
        'slug': slug,
        'title': str(data.get('title') or slug),
        'keywords': [str(x) for x in (data.get('keywords') or []) if isinstance(x, str)],
        'aliases': [str(x) for x in (data.get('aliases') or []) if isinstance(x, str)],
        'content': content.strip(),
    }


def parse_bulk_proposal(text: str, max_pages: int) -> dict | None:
    """Разбирает bulk-ответ LLM {home, style, pages:[...]} (офлайн-сборка).

    Возвращает {'home', 'style', 'pages': [...]} с нормализованными/дедуп.
    Страницами, или None при невалидном ответе. Предельный размер содержимого
    и проверка секретов выполняются вызывающим.
    """
    if not text:
        return None
    try:
        data = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    def _clean_optional(value):
        return value.strip() if isinstance(value, str) else None

    home = _clean_optional(data.get('home'))
    style = _clean_optional(data.get('style'))

    pages: list[dict] = []
    seen_slugs: set = set()
    seen_tokens: list[set] = []
    for raw_page in data.get('pages') or []:
        if not isinstance(raw_page, dict) or len(pages) >= max_pages:
            continue
        slug = pageio.normalize_slug(raw_page.get('slug'))
        if slug is None or slug in seen_slugs:
            continue
        content = raw_page.get('content')
        if not isinstance(content, str) or not content.strip():
            continue
        keywords = [str(x) for x in (raw_page.get('keywords') or []) if isinstance(x, str)]
        aliases = [str(x) for x in (raw_page.get('aliases') or []) if isinstance(x, str)]
        meta = set(textutil.token_set(' '.join(keywords) + ' ' + ' '.join(aliases)
                                      + ' ' + str(raw_page.get('title') or '')))
        if any(meta & prev for prev in seen_tokens):
            continue  # пересекающиеся темы в одном ответе — оставляем первую
        seen_slugs.add(slug)
        seen_tokens.append(meta)
        pages.append({
            'slug': slug,
            'title': str(raw_page.get('title') or slug),
            'keywords': keywords,
            'aliases': aliases,
            'content': content.strip(),
        })

    if home is None and not pages:
        return None
    return {'home': home, 'style': style, 'pages': pages}
