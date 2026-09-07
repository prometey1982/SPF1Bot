"""Тесты детерминированного создания тем БЗ (botkb/topics.py, ТЗ п. 9.4)."""

from botkb import topics


def _rows(*messages):
    return [{'id': i, 'content': text} for i, text in enumerate(messages, 1)]


def test_detect_candidates_repeats():
    rows = _rows('турбина дует масло', 'турбина свистит', 'опять турбина дует')
    cands = topics.detect_candidates(rows, min_repeats=2)
    assert cands and cands[0]['token'] == 'турбина'
    assert cands[0]['count'] == 3
    # одно сообщение — один повтор
    one = topics.detect_candidates(_rows('турбина турбина турбина'), min_repeats=2)
    assert one == []


def test_check_page_overlap():
    pages = [
        {'slug': 'vyhlop', 'title': 'Выхлоп', 'keywords': ['противодавление', 'выхлоп'],
         'aliases': [], 'status': 'active'},
    ]
    hit = topics.check_page_overlap('противодавление', pages)
    assert hit is not None and hit['slug'] == 'vyhlop'
    miss = topics.check_page_overlap('турбина', pages)
    assert miss is None


def test_cooldown_active_and_prune():
    import datetime
    index = {'page_proposal_cooldowns': {}}
    now = datetime.datetime.now(datetime.timezone.utc)
    topics.set_cooldown(index, 'турбина', now.isoformat())
    assert topics.is_cooldown_active(index, 'турбина', 24) is True
    old = now - datetime.timedelta(hours=30)
    topics.set_cooldown(index, 'масло', old.isoformat())
    assert topics.is_cooldown_active(index, 'масло', 24) is False
    removed = topics.prune_cooldowns(index, cooldown_hours=24, max_entries=100)
    assert removed == 1
    assert 'масло' not in index['page_proposal_cooldowns']
    assert 'турбина' in index['page_proposal_cooldowns']


def test_parse_page_proposal_valid_and_fenced():
    text = (
        "```yaml\n"
        "slug: Volvo_Repair\n"
        "title: Ремонт Volvo\n"
        "keywords: [volvo, ремонт]\n"
        "aliases: [шведа]\n"
        "content: |\n"
        "  # Ремонт Volvo\n"
        "  - тезис\n"
        "```\n"
    )
    prop = topics.parse_page_proposal(text)
    assert prop is not None
    assert prop['slug'] == 'volvo_repair'
    assert prop['keywords'] == ['volvo', 'ремонт']
    assert 'тезис' in prop['content']


def test_parse_page_proposal_rejects_bad():
    assert topics.parse_page_proposal('') is None
    assert topics.parse_page_proposal('slug: home\ncontent: x') is None  # служебный slug
    assert topics.parse_page_proposal('slug: tema\ncontent: "  "') is None
