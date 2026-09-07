"""Тесты router БЗ (botkb/router.py, ТЗ п. 11)."""

from botkb import router


def _page(slug, title, keywords=(), aliases=(), status='active',
          kind='knowledge', last_seen='2026-01-01'):
    return {'slug': slug, 'title': title, 'keywords': list(keywords),
            'aliases': list(aliases), 'status': status, 'kind': kind,
            'last_seen': last_seen}


def _index():
    pages = [
        _page('Home', 'О боте', kind='self'),
        _page('Style', 'Стиль', kind='self'),
        _page('vyhlop', 'Выхлоп', keywords=('противодавление',)),
        _page('turbo', 'Турбины', keywords=('турбина',)),
        _page('archived_topic', 'Старая тема', keywords=('противодавление',),
              status='archived'),
    ]
    return {'pages': pages}


def test_select_only_active_knowledge():
    index = _index()
    selected = router.select_topic_pages(index, 'противодавление', 0.3, 3)
    slugs = [s['slug'] for s in selected]
    assert 'vyhlop' in slugs
    assert 'Home' not in slugs and 'Style' not in slugs
    assert 'archived_topic' not in slugs  # архивные не выбираются


def test_scoring_and_min_score():
    index = _index()
    # aliases (0.7) дают меньший score, чем keywords (1.0)
    assert router.select_topic_pages(index, 'противодавление', 1.0, 3)
    assert router.select_topic_pages(index, 'противодавление', 1.5, 3) == []


def test_top_k_and_ordering():
    index = _index()
    selected = router.select_topic_pages(index, 'противодавление турбина', 0.3, 1)
    assert len(selected) == 1
    assert selected[0]['score'] == 1.0  # оба со score=1.0; стабильная сортировка


def test_empty_query_or_tokens():
    index = _index()
    assert router.select_topic_pages(index, '', 0.3, 3) == []
    assert router.select_topic_pages(index, 'и в не', 0.3, 3) == []  # стоп-слова
