"""Тесты router (botwiki/router.py, ТЗ п. 11)."""

from botwiki import router


def _page(slug, title='Тема', keywords=(), aliases=(), status='active',
          last_seen=None):
    return {'slug': slug, 'title': title, 'status': status,
            'keywords': list(keywords), 'aliases': list(aliases),
            'last_seen': last_seen}


def test_ignores_home_style_and_archived():
    index = {'pages': [
        _page('Home', 'Сводка', keywords=['сводка']),
        _page('Style', 'Стиль'),
        _page('cars', 'Машины', keywords=['машина']),
        _page('old', 'Дача', keywords=['дача'], status='archived'),
    ]}
    got = router.select_topic_pages(index, 'машина сломалась', 0.35, 2)
    assert [p['slug'] for p in got] == ['cars']


def test_scoring_weights():
    index = {'pages': [
        _page('a', 'Машины', keywords=['машина'], aliases=['авто']),
        _page('b', 'Дача', keywords=[], aliases=['дача']),
        _page('c', 'капуччино'),
    ]}
    q = router.textutil.token_set('машина дача капуччино')
    assert router.score_query(q, index['pages'][0]) == 1.0
    assert router.score_query(q, index['pages'][1]) == 0.7
    assert router.score_query(q, index['pages'][2]) == 0.5


def test_top_k_and_ordering_by_score_then_last_seen():
    index = {'pages': [
        _page('high', 'Гараж', keywords=['гараж'], last_seen='2020-01-01T00:00:00'),
        _page('mid', 'Авто', aliases=['авто']),
    ]}
    got = router.select_topic_pages(index, 'гараж авто машина', 0.35, 1)
    assert [p['slug'] for p in got] == ['high']


def test_no_query_or_no_tokens_empty():
    index = {'pages': [_page('cars', 'Машины', keywords=['машина'])]}
    assert router.select_topic_pages(index, None, 0.35, 2) == []
    assert router.select_topic_pages(index, 'и в не', 0.35, 2) == []


def test_below_min_score_excluded():
    index = {'pages': [_page('c', 'Ксилофон')]}
    # Совпадений с запросом нет — score 0, страница не выбирается
    assert router.select_topic_pages(index, 'совершенно иная тема', 0.35, 2) == []
