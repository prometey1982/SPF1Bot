"""Тесты детектора тем и cooldown (botwiki/topics.py, ТЗ п. 9.4/7.2)."""

from botwiki import pages, topics


def _rows(contents):
    return [{'id': i + 1, 'content': c} for i, c in enumerate(contents)]


def test_detect_candidates_repeats():
    rows = _rows([
        'Пишу про гараж мечты',
        'Купил гараж на окраине',
        'Сам строю гараж зимой',
        'Смотрел кино вчера',
    ])
    cands = topics.detect_candidates(rows, min_repeats=3)
    assert cands and cands[0]['token'] == 'гараж'
    assert cands[0]['count'] == 3
    assert len([c for c in cands if c['count'] >= 3]) == 1


def test_detect_candidates_below_repeat_not_returned():
    rows = _rows(['Люблю котиков', 'Люблю кофе'])
    cands = topics.detect_candidates(rows, min_repeats=3)
    assert cands == []


def test_detect_single_message_counts_once():
    """Одно сообщение — один повтор, даже если токен встречается дважды."""
    rows = _rows(['Гараж и ещё раз гараж', 'Другой гараж тоже', 'Третий гараж'])
    cands = topics.detect_candidates(rows, min_repeats=3)
    assert cands[0]['token'] == 'гараж'
    assert cands[0]['count'] == 3


def test_check_page_overlap_active_and_archived():
    active = {'slug': 'cars', 'title': 'Машины', 'status': 'active',
              'keywords': ['машина', 'авто'], 'aliases': []}
    archived = {'slug': 'dacha', 'title': 'Дача', 'status': 'archived',
                'keywords': [], 'aliases': ['дача']}
    assert topics.check_page_overlap('машина', [active, archived]) is active
    assert topics.check_page_overlap('дача', [active, archived]) is archived
    assert topics.check_page_overlap('котики', [active, archived]) is None
    # slug-слово тоже считается
    assert topics.check_page_overlap('cars', [active, archived]) is active


def test_cooldown_active_and_prune():
    index = {'page_proposal_cooldowns': {'тема': '2020-01-01T00:00:00'}}
    # Просроченная запись (старая) не активна
    assert topics.is_cooldown_active(index, 'тема', cooldown_hours=24) is False
    topics.set_cooldown(index, 'тема', '2099-01-01T00:00:00')
    assert topics.is_cooldown_active(index, 'тема', cooldown_hours=24) is True


def test_prune_cooldowns_removes_old_and_caps():
    index = {'page_proposal_cooldowns': {
        'a': '2020-01-01T00:00:00',
        'b': '2099-01-01T00:00:00',
        'c': '2099-01-02T00:00:00',
    }}
    removed = topics.prune_cooldowns(index, cooldown_hours=24, max_entries=1)
    assert removed == 2  # просроченная 'a' + вытесненная 'b' (лимит 1)
    assert len(index['page_proposal_cooldowns']) == 1


def test_parse_page_proposal():
    text = (
        "slug: garazh\n"
        "title: Гараж\n"
        "keywords: [гараж, гаражи]\n"
        "aliases: [гараж-бокс]\n"
        "content: |\n"
        "  # Гараж\n"
        "  - строит гараж мечты\n"
    )
    proposal = topics.parse_page_proposal(text)
    assert proposal is not None
    assert proposal['slug'] == 'garazh'
    assert 'строит гараж мечты' in proposal['content']


def test_parse_page_proposal_rejects_bad():
    assert topics.parse_page_proposal('slug: ../evil\ncontent: x') is None
    assert topics.parse_page_proposal('slug: Home\ncontent: x') is None
    assert topics.parse_page_proposal('не yaml: [') is None
    assert topics.parse_page_proposal('slug: cars') is None  # нет content
    assert topics.parse_page_proposal('') is None
    assert topics.parse_page_proposal(None) is None
