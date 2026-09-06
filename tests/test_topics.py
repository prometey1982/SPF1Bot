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


_BULK_YAML = """
home: |
  # Сводка
  - любит машины
style: |
  # Стиль
pages:
  - slug: Cars
    title: Машины
    keywords: [машина, авто]
    aliases: []
    content: |
      # Машины
      - владеет жигулями
"""


def test_parse_bulk_proposal():
    p = topics.parse_bulk_proposal(_BULK_YAML, max_pages=10)
    assert p is not None
    assert '# Сводка' in p['home']
    assert len(p['pages']) == 1
    assert p['pages'][0]['slug'] == 'cars'
    assert p['pages'][0]['keywords'] == ['машина', 'авто']


def test_parse_bulk_rejects_duplicate_and_bad():
    yaml_text = (_BULK_YAML +
                 "  - slug: cars\n    title: Дубль\n    content: |\n      # Дубль\n      - x\n"
                 "  - slug: ../evil\n    title: Evil\n    content: x\n")
    p = topics.parse_bulk_proposal(yaml_text, max_pages=10)
    assert len(p['pages']) == 1  # дубль slug и небезопасный slug отброшены


def test_parse_bulk_caps_pages():
    titles = ['Космос', 'Кофе', 'Кино', 'Дача', 'Рыбалка']
    many = "home: |\n  # Сводка\npages:\n"
    for i in range(5):
        many += (f"  - slug: p{i}\n    title: {titles[i]}\n    keywords: [k{i}]\n"
                 "    content: |\n      # Т\n      - факт\n")
    p = topics.parse_bulk_proposal(many, max_pages=3)
    assert p is not None
    assert len(p['pages']) == 3


def test_parse_bulk_invalid():
    assert topics.parse_bulk_proposal('не yaml: [', max_pages=5) is None
    assert topics.parse_bulk_proposal('', max_pages=5) is None
    # нет ни home, ни pages
    assert topics.parse_bulk_proposal('style: |\n  # Стиль', max_pages=5) is None


def test_parse_bulk_proposal_fenced():
    """deepseek оборачивает YAML в ```-фенсы — парсер должен их срезать."""
    fenced = '```yaml\n' + _BULK_YAML + '```\n'
    p = topics.parse_bulk_proposal(fenced, max_pages=10)
    assert p is not None
    assert len(p['pages']) == 1
    assert p['pages'][0]['slug'] == 'cars'
    assert '# Сводка' in p['home']


_PAGE_PROPOSAL = (
    "slug: garazh\n"
    "title: Гараж\n"
    "keywords: [гараж, гаражи]\n"
    "content: |\n"
    "  # Гараж\n"
    "  - строит гараж мечты\n"
)


def test_parse_page_proposal_fenced():
    p = topics.parse_page_proposal('```yaml\n' + _PAGE_PROPOSAL + '```')
    assert p is not None
    assert p['slug'] == 'garazh'
    assert 'строит гараж мечты' in p['content']
