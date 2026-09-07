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

    def __init__(self, response='# Обновление\n\n- новый тезис', yaml=False,
                 fail=False):
        self.response = response
        self.yaml = yaml
        self.fail = fail
        self.calls = 0

    async def __call__(self, prompt):
        self.calls += 1
        if self.fail:
            return 'Ошибка: что-то сломалось'
        if self.yaml and 'keywords:' in prompt:
            return (
                "slug: turbo\n"
                "title: Турбины\n"
                "keywords: [турбина]\n"
                "aliases: []\n"
                "content: |\n"
                "  # Турбины\n"
                "  - дует после 0.5 бара\n"
            )
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
