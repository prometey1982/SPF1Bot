"""Тесты менеджера wiki (botwiki/manager.py): bootstrap, инкремент, бюджеты.

Используется свежий WikiManager() на каждый тест (модульный синглтон
botwiki.wiki_manager мог бы держать состояние между тестами). LLM — фейк.
"""

import asyncio
import os
import sqlite3

from botwiki import config as wc
from botwiki import db, index as index_mod, manager, pages

USER = 777


def run(coro):
    return asyncio.run(coro)


def _configure(tmp_path, db_path, overrides=None):
    cfg = {
        'db': db_path,
        'allowed_group_chat_ids': [-100],
        'allowed_private_users': ['admin'],
        'wiki': {'dir': str(tmp_path / 'wiki_root')},
    }
    if overrides:
        for section, value in overrides.items():
            if isinstance(value, dict):
                cfg['wiki'][section] = {**cfg['wiki'].get(section, {}), **value}
            else:
                cfg['wiki'][section] = value
    wc.configure(cfg)
    return wc.wiki_dir()


def _seed_raw(db_path, n, start_mid=1, chat=-100, user_id=USER, content='Люблю пить кофе по утрам'):
    for i in range(n):
        db.append_raw(db_path, dict(user_id=user_id, chat_id=chat,
                                    message_id=start_mid + i, content=content))


_DISTINCT = [
    'Люблю пить кофе по утрам',
    'Обсуждали электричку и расписание',
    'Купил новые зимние колёса',
    'Смотрю матчи по хоккею',
    'Читаю новости про электрокары',
    'Готовил шашлыки на даче',
]


def _seed_distinct(db_path, n, start_mid=1, chat=-100, user_id=USER):
    for i in range(n):
        db.append_raw(db_path, dict(user_id=user_id, chat_id=chat,
                                    message_id=start_mid + i,
                                    content=_DISTINCT[i % len(_DISTINCT)]))


def _max_id(db_path, user_id=USER):
    return db.watermark(db_path, user_id)


def _home(user_dir):
    return pages.read_page(user_dir, 'Home')


def _index_wm(db_path, user_dir, user_id=USER):
    idx, _ = index_mod.ensure_index(user_dir, db_path, user_id)
    return idx


def _ensure_legacy_table(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS USER_INFO (id INTEGER PRIMARY KEY, dossier TEXT)")
        conn.commit()
    finally:
        conn.close()


def _set_dossier(db_path, user_id, dossier):
    _ensure_legacy_table(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO USER_INFO (id, dossier) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET dossier = excluded.dossier",
            (user_id, dossier))
        conn.commit()
    finally:
        conn.close()


class FakeLLM:
    """Возвращает markdown по ключу из промпта; умеет имитировать ошибку/сбой."""

    def __init__(self, fail=False, style_response='# Стиль\n\n- говорит «лол», любит мемы',
                 home_response='# Сводка\n\n- любит котиков'):
        self.calls = 0
        self.fail = fail
        self.style_response = style_response
        self.home_response = home_response

    async def __call__(self, prompt):
        self.calls += 1
        if self.fail:
            return 'Ошибка: что-то сломалось'
        if 'Style' in prompt:
            return self.style_response
        return self.home_response


# --- Bootstrap ---

def test_bootstrap_current_watermark(tmp_path, db_path):
    wiki_root = _configure(tmp_path, db_path)
    _seed_raw(db_path, 3)
    mgr = manager.WikiManager()

    user_dir = os.path.join(wiki_root, str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is True

    assert os.path.exists(os.path.join(user_dir, 'Home.md'))
    assert os.path.exists(os.path.join(user_dir, 'Style.md'))
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] == _max_id(db_path)
    assert idx['message_count'] == 0
    assert {p['slug'] for p in idx['pages']} == {'Home', 'Style'}


def test_bootstrap_from_dossier_uses_llm(tmp_path, db_path):
    _configure(tmp_path, db_path)
    _set_dossier(db_path, USER, 'Любит котиков\nВладеет жигулями')
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)

    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is True
    assert fake.calls == 1
    assert 'котиков' in (_home(user_dir) or '')


def test_bootstrap_from_dossier_empty_no_llm(tmp_path, db_path):
    _configure(tmp_path, db_path)
    mgr = manager.WikiManager()
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is True
    assert mgr._llm is None  # llm не вызывался (caller не установлен и не нужен)
    assert (_home(user_dir) or '').startswith('# Сводка')


def test_bootstrap_from_dossier_llm_failure_no_wiki(tmp_path, db_path):
    _configure(tmp_path, db_path)
    _set_dossier(db_path, USER, 'Факты про пользователя')
    fake = FakeLLM(fail=True)
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)

    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is False
    assert not os.path.exists(os.path.join(user_dir, 'Home.md'))


def test_bootstrap_limited_window(tmp_path, db_path):
    _configure(tmp_path, db_path, {
        'bootstrap': {'mode': 'limited_window', 'limited_window_messages': 2},
    })
    # Старые строки (вне окна) + свежие (в окне) — обрабатывается только окно
    for i in range(4):
        db.append_raw(db_path, dict(user_id=USER, chat_id=-420, message_id=i + 1,
                                    content='Люблю старую тему котиков'))
    for i in range(2):
        db.append_raw(db_path, dict(user_id=USER, chat_id=-421, message_id=i + 1,
                                    content='Увлекаюсь электрокарами в гараже'))

    async def fake(prompt):
        if 'электрокар' in prompt:
            return '# Сводка\n\n- увлекается электрокарами'
        return None

    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is True
    idx = _index_wm(db_path, user_dir)
    # watermark = последний id окна (свежие 2 строки), старые сознательно пропущены
    assert idx['watermark'] == _max_id(db_path)
    assert 'электрокар' in (_home(user_dir) or '')


# --- Инкремент ---

def _bootstrap_then(tmp_path, db_path, mgr, **kw):
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    run(mgr._bootstrap(USER, db_path, user_dir))
    return user_dir


def test_increment_updates_home_and_watermark(tmp_path, db_path):
    _configure(tmp_path, db_path)
    _seed_distinct(db_path, 2)
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    wm_after_boot = _index_wm(db_path, user_dir)['watermark']
    _seed_distinct(db_path, 3, start_mid=100, chat=-200)  # новые строки

    assert run(mgr._process_user(USER)) is True
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] == _max_id(db_path)
    assert idx['watermark'] > wm_after_boot
    assert idx['message_count'] == 3
    assert idx['last_error'] is None
    assert fake.calls == 1  # Home обновлялся один раз (стиль не сигналил)


def test_increment_no_new_rows_no_llm(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    before = _index_wm(db_path, user_dir)['message_count']
    assert run(mgr._process_user(USER)) is True
    assert _index_wm(db_path, user_dir)['message_count'] == before
    assert fake.calls == 0


def test_increment_snapshot_limits_no_loss(tmp_path, db_path):
    _configure(tmp_path, db_path, {
        'update': {'max_raw_messages_per_update': 2},
    })
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_distinct(db_path, 5, start_mid=1, chat=-300)
    assert run(mgr._process_user(USER)) is True
    idx = _index_wm(db_path, user_dir)
    # 5 строк доехали несколькими снимками (2+2+1), ни одна не потеряна
    assert idx['watermark'] == _max_id(db_path)
    assert idx['message_count'] == 5
    assert fake.calls == 3  # по вызову на снимок


def test_increment_trivial_rows_no_llm_but_watermark(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    for mid in range(1, 4):
        db.append_raw(db_path, dict(user_id=USER, chat_id=-500,
                                    message_id=mid, content='а'))
    assert run(mgr._process_user(USER)) is True
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] == _max_id(db_path)
    assert idx['message_count'] == 3
    assert fake.calls == 0


def test_increment_llm_error_does_not_move_watermark(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = FakeLLM(fail=True)
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_raw(db_path, 2, start_mid=50, chat=-600)
    wm_before = _index_wm(db_path, user_dir)['watermark']
    assert run(mgr._process_user(USER)) is False
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] == wm_before
    assert idx['message_count'] == 0
    assert idx['last_error'] is not None


def test_pause_after_persistent_failure(tmp_path, db_path):
    _configure(tmp_path, db_path, {'update': {'max_batch_retries': 2}})
    fake = FakeLLM(fail=True)
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_raw(db_path, 1, start_mid=1, chat=-700)
    run(mgr._process_user(USER))  # сбой 1
    _seed_raw(db_path, 1, start_mid=2, chat=-701)
    run(mgr._process_user(USER))  # сбой 2 → пауза
    assert mgr.is_paused(USER) is True

    mgr.unpause(USER)
    assert mgr.is_paused(USER) is False


def test_style_updated_only_on_signal(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    # Без стилевого сигнала — только Home
    _seed_raw(db_path, 1, start_mid=1, chat=-800)
    run(mgr._process_user(USER))
    assert fake.calls == 1

    # Со стилевым сигналом — Home и Style
    db.append_raw(db_path, dict(user_id=USER, chat_id=-801, message_id=1,
                                content='это просто лол, красава'))
    run(mgr._process_user(USER))
    assert fake.calls == 3  # 1 (Home) + 2 (Home+Style)
    style_md = pages.read_page(user_dir, 'Style') or ''
    assert 'Стиль' in style_md


def test_budget_daily_limit_delay(tmp_path, db_path):
    _configure(tmp_path, db_path, {
        'budgets': {'max_updates_per_user_per_day': 1},
    })
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_raw(db_path, 2, start_mid=1, chat=-900)
    assert run(mgr._process_user(USER)) is True
    wm1 = _index_wm(db_path, user_dir)['watermark']

    _seed_raw(db_path, 2, start_mid=3, chat=-901)
    # Дневной лимит исчерпан → delay: watermark не двигается
    assert run(mgr._process_user(USER)) is False
    assert _index_wm(db_path, user_dir)['watermark'] == wm1
    assert _index_wm(db_path, user_dir)['message_count'] == 2


def test_drop_old_over_limit(tmp_path, db_path):
    _configure(tmp_path, db_path, {
        'budgets': {'max_updates_per_user_per_day': 0,
                    'over_limit_policy': 'drop_old',
                    'max_backlog_messages': 2},
    })
    fake = FakeLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_raw(db_path, 5, start_mid=1, chat=-910)
    wm_before = _index_wm(db_path, user_dir)['watermark']
    assert run(mgr._process_user(USER)) is True
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] > wm_before
    assert idx['message_count'] == 0  # сброшенные строки счётчик не увеличивают


def test_is_trivial():
    assert manager.is_trivial('') is True
    assert manager.is_trivial('ок') is True
    assert manager.is_trivial('ок ок ок ок') is True
    assert manager.is_trivial('👍👍👍') is True
    assert manager.is_trivial('Люблю котиков') is False
    assert manager.is_trivial('Да') is True   # 'да' — стоп-слово, значимых токенов нет
    assert manager.is_trivial('и и и') is True


def test_wiki_valid_and_find_page(tmp_path, db_path):
    _configure(tmp_path, db_path)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    mgr = manager.WikiManager()
    run(mgr._bootstrap(USER, db_path, user_dir))
    assert index_mod.wiki_valid(db_path, USER) is True
    idx = _index_wm(db_path, user_dir)
    assert index_mod.find_page(idx, 'Home') is not None
    assert index_mod.find_page(idx, 'Nope') is None


class TopicLLM:
    """Fake: маршрутизирует create/update по содержимому промпта."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, prompt):
        self.calls += 1
        if 'Предложи страницу' in prompt:
            return (
                'slug: garazh\n'
                'title: Гараж\n'
                'keywords: [гараж, гаражи]\n'
                'aliases: [гараж-бокс]\n'
                'content: |\n'
                '  # Гараж\n'
                '  - строит гараж мечты\n'
            )
        return '# Сводка\n\n- любит гаражи'


def _seed_repeating(db_path, n, token='гараж', chat=-50):
    phrases = [
        f'Пишу про {token} мечты',
        f'Купил {token} на окраине',
        f'Сам строю {token} зимой',
        f'Сдаю {token} в аренду',
    ]
    for i in range(n):
        db.append_raw(db_path, dict(user_id=USER, chat_id=chat,
                                    message_id=i + 1,
                                    content=phrases[i % len(phrases)]))


def test_create_topic_page_and_no_duplicate(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = TopicLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    # 3 повторения темы в разных сообщениях → создаётся страница
    _seed_repeating(db_path, 3)
    run(mgr._process_user(USER))
    assert pages.page_exists(user_dir, 'garazh') is True
    idx = _index_wm(db_path, user_dir)
    page = index_mod.find_page(idx, 'garazh')
    assert page is not None
    assert page['status'] == 'active'
    assert page['keywords'] == ['гараж', 'гаражи']

    # Повторная тема уже покрыта активной страницей — дубль не создаётся,
    # но снимок обновляет и Home, и саму тематическую страницу (п. 9.3).
    calls_after_create = fake.calls
    _seed_repeating(db_path, 3, chat=-51)
    run(mgr._process_user(USER))
    idx = _index_wm(db_path, user_dir)
    assert len([p for p in idx['pages'] if p['slug'] == 'garazh']) == 1
    assert fake.calls == calls_after_create + 2  # Home + тематическая garazh


def test_create_topic_skipped_on_invalid_proposal(tmp_path, db_path):
    _configure(tmp_path, db_path)

    async def fake(prompt):
        return '# Сводка\n\n- просто текст без YAML'  # невалидное предложение

    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_repeating(db_path, 3)
    run(mgr._process_user(USER))
    assert pages.page_exists(user_dir, 'garazh') is False

    # cooldown активен → повторная попытка по тому же кандидату не выполняется
    idx = _index_wm(db_path, user_dir)
    cooldowns = idx.get('page_proposal_cooldowns') or {}
    assert 'гараж' in cooldowns


class ReconcileLLM:
    """Fake: Home/Style/темы на reconcile, отдельные маркеры для merge."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, prompt):
        self.calls += 1
        if 'Слей две страницы' in prompt:
            return '# Объединено\n- общий факт'
        if 'Style' in prompt or 'Стиль' in prompt:
            return '# Стиль\n\n- стиль обновлён'
        return '# Сводка\n\n- факты сверены'


def _add_raw_page(user_dir, db_path, slug, title, keywords, md, last_seen=None):
    idx, _ = index_mod.ensure_index(user_dir, db_path, USER)
    pages.atomic_write_page(user_dir, slug, md)
    idx['pages'].append({
        'slug': slug, 'title': title, 'status': 'active',
        'keywords': list(keywords), 'aliases': [], 'created': '2020-01-01T00:00:00',
        'updated': last_seen or '2020-01-01T00:00:00',
        'last_seen': last_seen or '2020-01-01T00:00:00', 'hits': 0,
    })
    index_mod.save_index(user_dir, idx)


def test_reconcile_archives_stale_topic(tmp_path, db_path):
    _configure(tmp_path, db_path, {'pages': {'archive_after_days': 1}})
    fake = ReconcileLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)
    # Темы давно не видели (last_seen 2020) и в окне нет подтверждения
    _add_raw_page(user_dir, db_path, 'garazh', 'Гараж', ['гараж'], '# Гараж\n- старый')

    result = run(mgr.reconcile(USER))
    assert result['success'] is True
    idx = _index_wm(db_path, user_dir)
    assert index_mod.find_page(idx, 'garazh')['status'] == 'archived'
    assert idx['message_count'] == 0
    assert idx['last_reconcile'] is not None
    # Файл архивной страницы сохраняется (п. 9.5)
    assert pages.page_exists(user_dir, 'garazh') is True


def test_reconcile_keeps_fresh_topic(tmp_path, db_path):
    _configure(tmp_path, db_path, {'pages': {'archive_after_days': 90}})
    fake = ReconcileLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)
    # Свежая тема с подтверждением в окне
    _add_raw_page(user_dir, db_path, 'garazh', 'Гараж', ['гараж'], '# Гараж',
                  last_seen=index_mod.now_iso())
    db.append_raw(db_path, dict(user_id=USER, chat_id=-333, message_id=1,
                                content='Пишу про гараж опять'))
    run(mgr.reconcile(USER))
    idx = _index_wm(db_path, user_dir)
    assert index_mod.find_page(idx, 'garazh')['status'] == 'active'


def test_merge_pages(tmp_path, db_path):
    _configure(tmp_path, db_path)
    fake = ReconcileLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)
    _add_raw_page(user_dir, db_path, 'garazh', 'Гараж', ['гараж'], '# Гараж\n- факт1')
    _add_raw_page(user_dir, db_path, 'dacha', 'Дача', ['дача'], '# Дача\n- факт2')

    result = run(mgr.merge_pages(USER, 'garazh', 'dacha'))
    assert result['ok'] is True
    idx = _index_wm(db_path, user_dir)
    assert index_mod.find_page(idx, 'garazh')['status'] == 'active'
    assert index_mod.find_page(idx, 'dacha')['status'] == 'archived'
    # Файл slug2 сохраняется как архивная копия
    assert pages.page_exists(user_dir, 'dacha') is True
    # keywords объединены
    assert 'гараж' in index_mod.find_page(idx, 'garazh')['keywords']
    assert 'дача' in index_mod.find_page(idx, 'garazh')['keywords']


def test_merge_pages_errors(tmp_path, db_path):
    _configure(tmp_path, db_path)
    mgr = manager.WikiManager()
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)
    assert run(mgr.merge_pages(USER, 'nope', 'dacha'))['ok'] is False
    assert run(mgr.merge_pages(USER, 'Home', 'dacha'))['ok'] is False
    assert run(mgr.merge_pages(USER, 'dacha', 'dacha'))['ok'] is False


def test_auto_reconcile_after_messages(tmp_path, db_path):
    _configure(tmp_path, db_path, {'reconcile': {'every_n_messages': 2}})
    fake = ReconcileLLM()
    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_distinct(db_path, 3, start_mid=1, chat=-340)
    run(mgr._process_user(USER))
    idx = _index_wm(db_path, user_dir)
    # message_count достиг порога → авто-reconcile сбросил счётчик и обновил страницы
    assert idx['last_reconcile'] is not None
    assert idx['message_count'] == 0


# --- Этап 7: качество и безопасность ---

def test_secret_in_llm_output_rejected(tmp_path, db_path):
    _configure(tmp_path, db_path)

    async def fake(prompt):
        return '# Сводка\n\n- тел +7 912 000-00-00 для связи'  # секрет → отклонено

    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    user_dir = _bootstrap_then(tmp_path, db_path, mgr)

    _seed_distinct(db_path, 2, start_mid=1, chat=-600)
    wm_before = _index_wm(db_path, user_dir)['watermark']
    assert run(mgr._process_user(USER)) is False
    idx = _index_wm(db_path, user_dir)
    assert idx['watermark'] == wm_before      # батч не прошёл
    assert idx['last_error'] is not None      # ошибка зафиксирована
    home = pages.read_page(user_dir, 'Home') or ''
    assert '912' not in home                  # прежняя версия сохранена


def test_bootstrap_truncates_dossier_to_max(tmp_path, db_path):
    _configure(tmp_path, db_path, {'bootstrap': {'max_dossier_chars': 40}})
    seen = {}

    async def fake(prompt):
        seen['prompt'] = prompt
        return '# Сводка\n\n- сжатые факты'

    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    _set_dossier(db_path, USER, 'A' * 200)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is True
    prompt = seen['prompt']
    assert 'A' * 40 in prompt        # затравка усечена до max_dossier_chars
    assert 'A' * 41 not in prompt


def test_bootstrap_limited_window_failure_no_wiki(tmp_path, db_path):
    _configure(tmp_path, db_path, {
        'bootstrap': {'mode': 'limited_window', 'limited_window_messages': 5},
    })

    async def fake(prompt):
        return None  # генерация окна не удалась

    mgr = manager.WikiManager()
    mgr.set_llm_caller(fake)
    _seed_distinct(db_path, 3, start_mid=1, chat=-610)
    user_dir = os.path.join(wc.wiki_dir(), str(USER))
    assert run(mgr._bootstrap(USER, db_path, user_dir)) is False
    # wiki не создаётся — пользователь остаётся в фолбэке на dossier (п. 9.7)
    assert not os.path.exists(os.path.join(user_dir, 'Home.md'))
    assert not os.path.exists(os.path.join(user_dir, index_mod.INDEX_FILE))


def test_wiki_valid_and_find_page_empty(tmp_path, db_path):
    """Валидность требует Home.md и Style.md."""
    _configure(tmp_path, db_path)
    assert index_mod.wiki_valid(db_path, USER) is False

