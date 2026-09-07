"""Тесты менеджера БЗ (botkb/manager.py): bootstrap, инкремент, узкий триггер
self, изоляция страниц, создание тем (ТЗ п. 8, 9.3, 9.4, 9.7).

Используется свежий KBManager() на каждый тест; LLM — фейк. bot_kb-конфиг
настраивается явно (корень wiki/bot_kb во временной директории).
"""

import asyncio
import os

from botkb import config as kc
from botkb import db
from botkb import index as index_mod
from botkb import manager
from botkb import pages as pageio

SELF_HOME_FRAME = '# О боте\n\n'


def run(coro):
    return asyncio.run(coro)


def _configure(tmp_path, db_path, overrides=None):
    bot_kb = {'dir': str(tmp_path / 'kb_root')}
    if overrides:
        for section, value in overrides.items():
            if isinstance(value, dict):
                bot_kb.setdefault(section, {})
                bot_kb[section].update(value)
            else:
                bot_kb[section] = value
    kc.configure({'db': db_path, 'bot_kb': bot_kb})
    return bot_kb['dir']


def _row(message_id, *, speaker='human', chat=-100, reply=None, content='текст',
         user_id=None):
    return dict(
        speaker=speaker,
        user_id=user_id if user_id is not None else (111 if speaker == 'human' else 555),
        chat_id=chat,
        message_id=message_id,
        reply_to_message_id=reply,
        content=content,
    )


def _seed(db_path, rows):
    inserted = 0
    for r in rows:
        if db.append_raw(db_path, r):
            inserted += 1
    return inserted


def _max_id(db_path):
    return db.watermark(db_path)


def _index(root, db_path):
    idx, _ = index_mod.ensure_index(root, db_path)
    return idx


class FakeLLM:
    """Возвращает markdown страницы; при yaml=True — YAML-предложение темы."""

    _DEFAULT_PROPOSAL = (
        "slug: turbo\n"
        "title: Турбины\n"
        "keywords: [турбина]\n"
        "aliases: []\n"
        "content: |\n"
        "  # Турбины\n"
        "  - дуется после 0.5 бара\n"
    )

    def __init__(self, response='# Обновление\n\n- новый тезис', yaml=False,
                 proposal=None, fail=False):
        self.response = response
        self.yaml = yaml
        self.proposal = proposal if proposal is not None else self._DEFAULT_PROPOSAL
        self.fail = fail
        self.calls = 0

    async def __call__(self, prompt):
        self.calls += 1
        if self.fail:
            return 'Ошибка: что-то сломалось'
        if self.yaml and 'keywords:' in prompt:
            return self.proposal
        return self.response


def _bootstrap_empty(tmp_path, db_path, **overrides):
    """Пустой bootstrap (фреймы, без LLM) с большим backlog-хвостом: накопленные
    до первого запуска строки остаются непокрытыми и разбираются инкрементами.
    Возвращает (root, mgr)."""
    root = _configure(tmp_path, db_path, {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 10000, **overrides},
    })
    mgr = manager.KBManager()
    return root, mgr


# --- Bootstrap ---

def test_bootstrap_empty_creates_frames_and_discard_watermark(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {'bootstrap': {'mode': 'empty'}})
    for mid in range(1, 4):
        _seed(kb_db_path, [_row(mid, content='факт про кофе')])
    fake = FakeLLM()
    mgr = manager.KBManager()
    mgr.set_llm_caller(fake)

    assert run(mgr._bootstrap()) is True

    assert pageio.read_page('Home', root) == SELF_HOME_FRAME
    assert pageio.read_page('Style', root) == '# Стиль\n\n'
    assert fake.calls == 0
    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)  # discard: сырьё не в апдейтах
    assert {p['slug']: p['kind'] for p in idx['pages']} == \
        {'Home': 'self', 'Style': 'self'}


def test_bootstrap_backlog_leaves_newest_uncovered(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 2},
    })
    for mid in range(1, 6):
        _seed(kb_db_path, [_row(mid, content=f'факт {mid}')])
    mgr = manager.KBManager()
    assert run(mgr._bootstrap()) is True

    idx = _index(root, kb_db_path)
    conn = __import__('sqlite3').connect(kb_db_path)
    ids = sorted(r[0] for r in conn.execute("SELECT id FROM bot_kb_raw"))
    conn.close()
    # watermark = id строки на (keep+1)-й позиции от новой → непокрыты 2 новых
    assert idx['watermark'] == ids[-3]
    uncovered = db.count_unprocessed(kb_db_path, idx['watermark'])
    assert uncovered == 2


# --- Инкремент ---

def test_trivial_rows_advance_without_llm(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    fake = FakeLLM()
    mgr.set_llm_caller(fake)
    _seed(kb_db_path, [_row(mid, content='x') for mid in range(1, 5)])  # тривиально

    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)
    assert idx['message_count'] == 4  # по строкам, включая тривиальные
    assert fake.calls == 0


def _add_knowledge_page(root, db_path, slug='vyhlop', title='Выхлоп',
                        keywords=('противодавление',)):
    idx = _index(root, db_path)
    if index_mod.find_page(idx, slug) is None:
        idx['pages'].append({
            'slug': slug, 'title': title, 'kind': 'knowledge', 'status': 'active',
            'keywords': list(keywords), 'aliases': [],
            'created': 't', 'updated': 't', 'last_seen': 't', 'hits': 0,
            'quarantined': False,
        })
        assert index_mod.save_index(root, idx) is True
    pageio.write_page(slug, f'# {title}\n- тезис\n\n', root)


def test_knowledge_increment_updates_page_and_advances_watermark(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    fake = FakeLLM(response='# Выхлоп\n\n- мерь на холодной, а не на прогретой')
    mgr.set_llm_caller(fake)
    run(mgr.run_updates())  # bootstrap: фреймы Home/Style
    _add_knowledge_page(root, kb_db_path)

    _seed(kb_db_path, [
        _row(1, content='померял противодавление на прогретом — врут цифры'),
    ])
    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)
    assert idx['message_count'] == 1
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['last_error'] is None
    md = pageio.read_page('vyhlop', root)
    assert 'на холодной' in md  # страница обновлена
    assert fake.calls >= 1


def test_watermark_advances_on_page_failure_with_last_error(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    mgr.set_llm_caller(FakeLLM(fail=True))
    run(mgr.run_updates())  # bootstrap: фреймы Home/Style
    _add_knowledge_page(root, kb_db_path)
    _seed(kb_db_path, [_row(1, content='противодавление завышено')])

    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)  # watermark продвинут в любом случае
    page = index_mod.find_page(idx, 'vyhlop')
    assert page is not None and page['last_error']
    # контент не изменился (ошибка страницы, а не батча)
    assert '# Выхлоп' in (pageio.read_page('vyhlop', root) or '')


# --- Узкий триггер self (п. 2.4/9.3.3) ---

def test_self_update_only_on_human_feedback(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    fake = FakeLLM(response='# О боте\n\n- исправлено: меряем на холодной')
    mgr.set_llm_caller(fake)

    # A) только сообщение человека — self не трогаем
    _seed(kb_db_path, [_row(1, content='как правильно мерить противодавление?')])
    run(mgr.run_updates())
    assert pageio.read_page('Home', root) == SELF_HOME_FRAME
    assert fake.calls == 0

    # B) ответ бота (ход из двух кусков) без реплик — сам по себе self НЕ обновляет
    _seed(kb_db_path, [
        _row(2, speaker='bot', reply=1, content='Измеряй противодавление, а не гадай'),
        _row(3, speaker='bot', reply=1, content='Держи шланг ровно'),
    ])
    run(mgr.run_updates())
    assert pageio.read_page('Home', root) == SELF_HOME_FRAME
    assert fake.calls == 0

    # C) реплика-человек на один из кусков хода → обновление self (Home/Style)
    _seed(kb_db_path, [_row(4, reply=2, content='только не на прогретом моторе!')])
    run(mgr.run_updates())
    md = pageio.read_page('Home', root)
    assert md != SELF_HOME_FRAME
    assert 'на холодной' in md
    assert fake.calls >= 1


# --- Создание тематической страницы (п. 9.4) ---

def test_topic_creation_by_repeats(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    mgr.set_llm_caller(FakeLLM(yaml=True))
    _seed(kb_db_path, [
        _row(mid, content=f'турбина дует на {mid} передаче, мерял противодавление')
        for mid in range(1, 7)
    ])
    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)
    assert any(p['slug'] == 'turbo' and p['kind'] == 'knowledge'
               for p in idx['pages'])
    md = pageio.read_page('turbo', root)
    assert 'дуется' in md or '0.5 бара' in md


# --- Многокусочный ход: короткий кусок тривиален, но не роняет батч ---

def test_multichunk_bot_turn_trivial_last_chunk(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    fake = FakeLLM()
    mgr.set_llm_caller(fake)
    _seed(kb_db_path, [
        _row(1, content='вопрос про выхлоп'),
        _row(2, speaker='bot', reply=1, content='Первая часть ответа, длинная и содержательная'),
        _row(3, speaker='bot', reply=1, content='ок'),  # короткий кусок — тривиален
    ])
    run(mgr.run_updates())
    idx = _index(root, kb_db_path)
    assert idx['watermark'] == _max_id(kb_db_path)
    assert idx['message_count'] == 3
    assert fake.calls == 0  # фидбека нет — self не обновляется, LLM не тратится


# --- Reconcile (п. 9.6) ---

def test_reconcile_manual_clears_error_and_updates(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 10000},
        'update': {'max_batch_retries': 1},
    })
    mgr = manager.KBManager()
    mgr.set_llm_caller(FakeLLM(fail=True))
    run(mgr.run_updates())  # bootstrap: фреймы
    _add_knowledge_page(root, kb_db_path)
    _seed(kb_db_path, [_row(1, content='противодавление завышено')])
    run(mgr.run_updates())  # инкремент падает → last_error на странице

    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page is not None and page['last_error']
    assert page['failure_count'] and page['failure_count'] >= 1

    # Ручной reconcile (успешный LLM) снимает ошибку и обновляет страницу
    mgr.set_llm_caller(FakeLLM(response='# Выхлоп\n\n- тезис подтверждён окном'))
    res = run(mgr.reconcile(slug='vyhlop', manual=True))
    assert res['success'] is True and res['updated'] >= 1
    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['last_error'] is None
    assert page['quarantined'] is False
    assert page['failure_count'] == 0
    assert 'подтверждён' in pageio.read_page('vyhlop', root)


def test_quarantine_after_repeated_failures(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 10000},
        'update': {'page_quarantine_failures': 2, 'max_batch_retries': 1},
    })
    mgr = manager.KBManager()
    mgr.set_llm_caller(FakeLLM(fail=True))
    run(mgr.run_updates())  # bootstrap: фреймы
    _add_knowledge_page(root, kb_db_path)
    _seed(kb_db_path, [_row(1, content='противодавление завышено')])
    run(mgr.run_updates())  # инкремент (+авто-reconcile) → 2 неудачи подряд

    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['quarantined'] is True

    # Карантинная страница больше не ретраится автоматически
    before = pageio.read_page('vyhlop', root)
    mgr.set_llm_caller(FakeLLM(response='# Выхлоп\n\n- свежий тезис'))
    _seed(kb_db_path, [_row(2, content='опять противодавление врут')])
    run(mgr.run_updates())
    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['quarantined'] is True
    assert pageio.read_page('vyhlop', root) == before

    # Ручной /kb_reconcile снимает карантин и ретраит
    res = run(mgr.reconcile(slug='vyhlop', manual=True))
    assert res['success'] is True
    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['quarantined'] is False
    assert page['last_error'] is None
    assert 'свежий тезис' in pageio.read_page('vyhlop', root)


# --- Этап 6: негативные тесты (безопасность/достоверность) ---

def test_bootstrap_secret_output_falls_back_to_frames(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {
        'bootstrap': {'mode': 'from_system_prompt', 'seed': 'Описание бота'},
    })
    mgr = manager.KBManager()
    fake = FakeLLM(response='# О боте\n\n- пиши на a@b.ru (секрет)')
    mgr.set_llm_caller(fake)
    assert run(mgr._bootstrap()) is True

    assert fake.calls == 2  # Home и Style оба вернули «секрет» → отброшены
    home = pageio.read_page('Home', root)
    style = pageio.read_page('Style', root)
    assert home == SELF_HOME_FRAME and style == '# Стиль\n\n'
    assert 'a@b.ru' not in home + style


def test_increment_rejects_secret_llm_output(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    mgr.set_llm_caller(FakeLLM(fail=True))
    run(mgr.run_updates())  # bootstrap: фреймы
    _add_knowledge_page(root, kb_db_path)
    mgr.set_llm_caller(FakeLLM(response='# Выхлоп\n\n- password=supersecret на холодной'))
    _seed(kb_db_path, [_row(1, content='противодавление завышено')])
    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    page = index_mod.find_page(idx, 'vyhlop')
    assert page['last_error']  # секретный ответ не сохранён
    md = pageio.read_page('vyhlop', root)
    assert 'supersecret' not in md
    assert '- тезис' in md  # содержимое не изменилось


def test_topic_create_rejects_secret_content(tmp_path, kb_db_path):
    root, mgr = _bootstrap_empty(tmp_path, kb_db_path)
    fake = FakeLLM(yaml=True, proposal=(
        "slug: turbo\n"
        "title: Турбины\n"
        "keywords: [турбина]\n"
        "aliases: []\n"
        "content: |\n"
        "  # Турбины\n"
        "  - token=abcd1234 секрет\n"
    ))
    mgr.set_llm_caller(fake)
    _seed(kb_db_path, [
        _row(mid, content=f'турбина дует на {mid} передаче')
        for mid in range(1, 6)
    ])
    run(mgr.run_updates())

    idx = _index(root, kb_db_path)
    assert not any(p['slug'] == 'turbo' for p in idx['pages'])  # отклонена
    assert not pageio.page_exists('turbo', root)
    # секрет не попал ни в одну страницу
    for slug in pageio.list_slugs(root):
        assert 'abcd1234' not in (pageio.read_page(slug, root) or '')


def test_budget_stops_processing_until_reset(tmp_path, kb_db_path):
    root = _configure(tmp_path, kb_db_path, {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 10000},
        'update': {'max_raw_messages_per_update': 1},
        'budgets': {'max_llm_calls_per_hour': 1},
    })
    mgr = manager.KBManager()
    fake = FakeLLM(response='# Выхлоп\n\n- тезис обновлён')
    mgr.set_llm_caller(fake)
    run(mgr.run_updates())  # bootstrap: фреймы
    _add_knowledge_page(root, kb_db_path)
    _seed(kb_db_path, [
        _row(mid, content='противодавление завышено раз')
        for mid in range(1, 4)
    ])
    run(mgr.run_updates())

    # Обработан один снимок (1 llm-вызов исчерпал часовой лимит), дальше стоп
    assert fake.calls == 1
    idx = _index(root, kb_db_path)
    remaining = db.count_unprocessed(kb_db_path, idx['watermark'])
    assert remaining == 2  # хвост ждёт следующего часа/триггера


# --- K2a: знание из ответов бота (bot_kb_knowledge_tz.md) ---

_BOT_ANSWER = ('противодавление выхлопа на S60R меряй на холодной, '
               'показания будут стабильнее на прогретом моторе... ' * 3)


def _setup_with_knowledge(tmp_path, db_path, *, bot_turns, **extra):
    cfg = {
        'bootstrap': {'mode': 'empty', 'history': 'backlog',
                      'max_history_messages': 10000},
        'knowledge': {'bot_turns': bot_turns, 'bot_turn_min_chars': 120,
                      'bot_turn_skip_phrases': []},
    }
    for section, value in extra.items():
        cfg.setdefault(section, {})
        if isinstance(value, dict):
            cfg[section].update(value)
        else:
            cfg[section] = value
    root = _configure(tmp_path, db_path, cfg)
    return root, manager.KBManager()


def test_bot_turn_updates_knowledge_only_when_flag_on(tmp_path):
    # Вопрос о кофе (не совпадает с темой), ответ бота — про противодавление:
    # при флаге материал включает ответ → страница обновляется.
    for flag in (False, True):
        db_path = str(tmp_path / f'kb_{flag}.db')
        db.init_raw_table(db_path)
        root, mgr = _setup_with_knowledge(tmp_path / f'f{flag}', db_path,
                                          bot_turns=flag)
        fake = FakeLLM(response='# Выхлоп\n\n- тезис из ответа бота')
        mgr.set_llm_caller(fake)
        run(mgr.run_updates())  # bootstrap: фреймы
        _add_knowledge_page(root, db_path)
        _seed(db_path, [
            _row(1, content='люблю кофе по утрам с бутербродом'),
            _row(2, speaker='bot', reply=1, content=_BOT_ANSWER),
        ])
        run(mgr.run_updates())
        md = pageio.read_page('vyhlop', root) or ''
        if flag:
            assert 'из ответа бота' in md
            assert fake.calls >= 1
        else:
            assert 'из ответа бота' not in md
            assert '- тезис' in md
            assert fake.calls == 0


def test_bot_turn_without_parent_excluded(tmp_path, kb_db_path):
    root, mgr = _setup_with_knowledge(tmp_path, kb_db_path, bot_turns=True)
    mgr.set_llm_caller(FakeLLM())
    run(mgr.run_updates())  # bootstrap
    _add_knowledge_page(root, kb_db_path)
    # ход отвечает на message_id, которого нет в БД — не материал
    _seed(kb_db_path, [
        _row(2, speaker='bot', reply=99999, content=_BOT_ANSWER),
    ])
    run(mgr.run_updates())
    md = pageio.read_page('vyhlop', root) or ''
    assert '- тезис' in md  # не обновлена


def test_bot_material_skip_phrases_and_min_chars(tmp_path, kb_db_path):
    root, mgr = _setup_with_knowledge(
        tmp_path, kb_db_path, bot_turns=True,
        update={'max_batch_retries': 1})
    # родитель в БД
    _seed(kb_db_path, [_row(1, content='люблю кофе по утрам с бутербродом')])
    # бот-ход: кусок-шаблон + содержательный кусок
    bot_rows = [
        {'id': 50, 'speaker': 'bot', 'chat_id': -100,
         'reply_to_message_id': 1, 'content': 'Спасибо за вопрос!'},
        {'id': 51, 'speaker': 'bot', 'chat_id': -100,
         'reply_to_message_id': 1, 'content': _BOT_ANSWER},
    ]
    kn = kc.settings()['knowledge']
    kn['bot_turn_skip_phrases'] = ['спасибо за вопрос']
    kept, parents, turns = mgr._bot_material_for_snapshot(bot_rows, 3)
    assert turns == 1
    assert len(kept) == 1            # шаблонный кусок исключён
    assert 'Спасибо за вопрос' not in kept[0]['content']
    assert len(parents) == 1 and parents[0]['message_id'] == 1

    # короткий ход (< bot_turn_min_chars) не подходит
    short = [{'id': 52, 'speaker': 'bot', 'chat_id': -100,
              'reply_to_message_id': 1, 'content': 'да'}]
    kept2, _, turns2 = mgr._bot_material_for_snapshot(short, 3)
    assert kept2 == [] and turns2 == 0


def test_render_knowledge_block_labels():
    material = [
        {'id': 10, 'speaker': 'human', 'content': 'вопрос'},
        {'id': 11, 'speaker': 'bot', 'content': 'ответ длинный'},
    ]
    parents = [{'id': 5, 'speaker': 'human', 'content': 'родительский вопрос'}]
    block = manager.KBManager._render_knowledge_block(material, parents)
    assert '[1][human] родительский вопрос' in block
    assert '[2][human] вопрос' in block
    assert '[3][bot] ответ длинный' in block

    human_only = manager.KBManager._render_knowledge_block(
        [{'id': 1, 'speaker': 'human', 'content': 'x'}], [])
    assert human_only == '[1] x'  # без бота — прежний нейтральный формат


# --- K2b: детектор тем по «сообщениям-материалу» (ход = 1) ---

_TURBO = 'турбина ' * 30  # длинный содержательный ход (>= min_chars)


def test_single_multichunk_turn_does_not_create_topic(tmp_path):
    """Один ответ из 2+ кусков с повтором токена не выполняет порог сам."""
    db_path = str(tmp_path / 'kb.db')
    db.init_raw_table(db_path)
    root, mgr = _setup_with_knowledge(
        tmp_path / 'kb', db_path, bot_turns=True,
        pages={'create_repeats': 2})
    mgr.set_llm_caller(FakeLLM(yaml=True))
    run(mgr.run_updates())  # bootstrap
    _seed(db_path, [
        _row(1, content='люблю кофе по утрам'),
        _row(2, speaker='bot', reply=1, content=_TURBO),
        _row(3, speaker='bot', reply=1, content=_TURBO),  # второй кусок того же хода
    ])
    run(mgr.run_updates())

    idx = _index(root, db_path)
    assert not any(p['slug'] == 'turbo' for p in idx['pages'])
    # единица окна одна: человеческое сообщение + один ход = 2 единицы
    window = mgr._processed_material_window(db_path, idx['watermark'], 100)
    assert len(window) == 2


def test_two_bot_turns_create_topic(tmp_path):
    """Два РАЗНЫХ хода бота с одной темой → тема создаётся."""
    db_path = str(tmp_path / 'kb.db')
    db.init_raw_table(db_path)
    root, mgr = _setup_with_knowledge(
        tmp_path / 'kb', db_path, bot_turns=True,
        pages={'create_repeats': 2})
    mgr.set_llm_caller(FakeLLM(yaml=True))
    run(mgr.run_updates())  # bootstrap
    _seed(db_path, [
        _row(1, content='люблю кофе по утрам'),
        _row(2, speaker='bot', reply=1, content=_TURBO),
        _row(3, content='снова кофе вечером'),
        _row(4, speaker='bot', reply=3, content=_TURBO),
    ])
    run(mgr.run_updates())

    idx = _index(root, db_path)
    assert any(p['slug'] == 'turbo' and p['kind'] == 'knowledge'
               for p in idx['pages'])
    assert pageio.page_exists('turbo', root)


# --- K3: reconcile по объединённому окну; авто-цикл требует человеческие строки ---

def test_reconcile_auto_skipped_without_human_material(tmp_path, kb_db_path):
    db_path = kb_db_path
    root, mgr = _setup_with_knowledge(tmp_path, db_path, bot_turns=True)
    fake = FakeLLM(response='# Выхлоп\n\n- тезис после сверки')
    mgr.set_llm_caller(fake)
    run(mgr.run_updates())  # bootstrap: фреймы
    _add_knowledge_page(root, db_path)
    # вопрос о кофе (не релевантен vyhlop), ответ бота — про противодавление
    _seed(db_path, [
        _row(1, content='люблю кофе по утрам с бутербродом'),
        _row(2, speaker='bot', reply=1, content=_BOT_ANSWER),
    ])

    # auto: в подокне vyhlop нет человеческих строк (вопрос про кофе не совпал)
    res = run(mgr.reconcile(slug='vyhlop', manual=False))
    assert res['success'] is True
    assert fake.calls == 0
    assert 'тезис после сверки' not in (pageio.read_page('vyhlop', root) or '')

    # ручной reconcile выполняется всегда и обновляет страницу
    res = run(mgr.reconcile(slug='vyhlop', manual=True))
    assert res['success'] is True and res['updated'] >= 1
    assert 'тезис после сверки' in pageio.read_page('vyhlop', root)
