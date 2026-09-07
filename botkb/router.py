"""Router выбора тематических страниц БЗ (ТЗ п. 11).

MVP — keywords-режим. Переиспользует скоринг botwiki.router
(keyword 1.0 / alias 0.7 / title 0.5), но выбирает ТОЛЬКО активные страницы
с kind='knowledge' (self Home/Style исключаются признаком kind, а не slug —
ТЗ п. 11.1). Режим llm — будущее (warning+фолбэк на keywords, как user-wiki).
"""

import logging

from botwiki import router as user_router
from botwiki import textutil

logger = logging.getLogger(__name__)


def select_topic_pages(index: dict, query: str | None, min_score: float,
                       top_k: int) -> list[dict]:
    """Активные knowledge-страницы для инъекции/сужения окна (п. 11).

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
        if page.get('status') != 'active':
            continue
        kind = page.get('kind')
        if kind is not None and kind != 'knowledge':
            continue
        score = user_router.score_query(query_tokens, page)
        if score < min_score:
            continue
        candidates.append({
            'score': score,
            'slug': page.get('slug'),
            'title': page.get('title') or page.get('slug'),
            'page': page,
        })

    candidates.sort(key=lambda c: (-c['score'], c['page'].get('last_seen') or ''))
    return candidates[:top_k]
