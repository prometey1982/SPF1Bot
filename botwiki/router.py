"""Router выбора тематических страниц для инъекции (ТЗ п. 11).

Режим keywords (MVP; llm-режим с LLM-выбором — этап качества). Ранжирование:
score, затем last_seen; hits на ранжирование не влияет (диагностика).
Никаких дисковых записей в горячем пути.
"""

import logging

from . import textutil

logger = logging.getLogger(__name__)

# Веса совпадения (п. 11.2): keyword 1.0, alias/словоформа 0.7, title 0.5.
W_KEYWORD = 1.0
W_ALIAS = 0.7
W_TITLE = 0.5


def page_match_tokens(page: dict) -> tuple[set, set, set]:
    """(keywords, aliases, title) множества значимых токенов страницы."""
    keywords = textutil.token_set(' '.join(page.get('keywords') or []))
    aliases = textutil.token_set(' '.join(page.get('aliases') or []))
    title = textutil.token_set(page.get('title') or '')
    return keywords, aliases, title


def score_query(query_tokens: set[str], page: dict) -> float:
    """Скоринг страницы по токенам запроса (п. 11.2)."""
    if not query_tokens:
        return 0.0
    keywords, aliases, title = page_match_tokens(page)
    score = 0.0
    for token in query_tokens:
        if token in keywords:
            score += W_KEYWORD
        elif token in aliases:
            score += W_ALIAS
        elif token in title:
            score += W_TITLE
    return score


def select_topic_pages(index: dict, query: str | None, min_score: float,
                       top_k: int) -> list[dict]:
    """Активные тематические страницы (не Home/Style) для инъекции.

    Возвращает страницы со score >= min_score, не более top_k, по убыванию
    (score, last_seen). Ничего не пишет.
    """
    if not query:
        return []
    query_tokens = textutil.token_set(query)
    if not query_tokens:
        return []

    candidates = []
    for page in index.get('pages', []):
        slug = page.get('slug', '')
        if slug in ('Home', 'Style'):
            continue
        if page.get('status') != 'active':
            continue
        score = score_query(query_tokens, page)
        if score < min_score:
            continue
        candidates.append({
            'score': score,
            'slug': slug,
            'title': page.get('title', slug),
            'page': page,
        })

    candidates.sort(key=lambda c: (-c['score'], c['page'].get('last_seen') or ''))
    return candidates[:top_k]
