"""Менеджер wiki: bootstrap, инкремент со снимком, троттлинг, бюджеты (ТЗ п. 8, 9).

Этап 3 покрывает: bootstrap (Home+Style+_index), инкрементальные обновления
Home/Style по снимку `user_raw` с лимитами, атомарную запись (порядок 8.4),
монотонный watermark (8.1), троттлинг/дебаунс, бюджеты с `over_limit_policy`
(9.2), защиту от повторяющихся сбоев (9.3, MVP — пауза в памяти).

Создание тематических страниц, router и reconcile — этапы 5/6. Импорт — этап 8.

LLM-вызовы не выполняются здесь напрямую: bot.py внедряет `set_llm_caller()`
(callable, оборачивающий существующий провайдер). В тестах — фейковый caller.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

from . import config
from . import db
from . import index as index_mod
from . import pages as pageio
from . import prompts
from . import inject as inject_mod

logger = logging.getLogger(__name__)

STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
    'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если',
    'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять',
    'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они',
    'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была',
    'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
    'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
    'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'кажется',
    'сейчас', 'были', 'куда', 'зачем', 'сказать', 'всех', 'никогда', 'конечно',
    'всю', 'нету', 'при', 'об', 'хоть', 'после', 'над', 'тот', 'через', 'эти',
    # междометия/сленг-реакции: «ок», «ага», «лол» — тривиальные (п. 8.1)
    'ок', 'окей', 'ага', 'ахах', 'угу', 'мда', 'хм', 'ого', 'лол', 'кек', 'лан',
    'неа', 'ауч', 'аа', 'оо', 'мм', 'эх', 'оу', 'вау',
}

TOKEN_RE = re.compile(r'[a-zа-яё]{2,}', re.IGNORECASE)

DEFAULT_HOME_FRAME = "# Сводка\n\n"
DEFAULT_STYLE_FRAME = "# Стиль\n\n"

_STYLE_TOKENS = {
    'лол', 'кек', 'ахах', 'рофл', 'хах', 'хз', 'ага', 'ок', 'оу', 'вау', 'йоу',
    'забей', 'жесть', 'красава', 'норм', 'кринж', 'имба', 'топ',
}


def is_trivial(content: str | None, min_chars: int = 3) -> bool:
    """Тривиальное сообщение (п. 8.1): не попадает в LLM, но считается обработанным.

    Короче `min_chars` ИЛИ после нормализации/удаления стоп-слов не содержит
    значимых токенов (только эмодзи/«ок», «ага», «лол»).
    """
    if not content:
        return True
    stripped = content.strip()
    if len(stripped) < min_chars:
        return True
    tokens = [t.lower() for t in TOKEN_RE.findall(stripped)]
    significant = [t for t in tokens if t not in STOPWORDS]
    return not significant


def has_style_signal(content: str | None) -> bool:
    if not content:
        return False
    lower = content.lower()
    tokens = {t.lower() for t in TOKEN_RE.findall(lower)}
    if tokens & _STYLE_TOKENS:
        return True
    if re.search(r'[!?]{2,}|🙂|😄|😁|🤣|😂|😏|😎|👍|🤡|😭', content):
        return True
    return False


def _build_raw_block(rows: list[dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        content = (row.get('content') or '').strip()
        if not content:
            continue
        content_type = row.get('content_type') or 'text'
        label = f"[{i}][{content_type}]" if content_type != 'text' else f"[{i}]"
        lines.append(f"{label} {content}")
    return "\n".join(lines)


def _validate_page_md(md: str | None, max_chars: int) -> str | None:
    if not md:
        return None
    md = md.strip()
    if not md:
        return None
    if len(md) > max_chars:
        logger.warning("wiki: ответ страницы слишком большой (%d > %d)", len(md), max_chars)
        return None
    return md


def _page_meta_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


class BudgetTracker:
    """Счётчики бюджетов в памяти процесса (ТЗ п. 9.2, допущение 17).

    Перезапуск процесса сбрасывает лимиты — осознанно для MVP.
    """

    def __init__(self):
        self._day: dict[int, tuple[str, int]] = {}    # user -> (yyyymmdd, count)
        self._hour: dict[int, tuple[int, int]] = {}   # user -> (epoch_hour, count)

    @staticmethod
    def _day_key() -> str:
        return datetime.now(timezone.utc).strftime('%Y%m%d')

    @staticmethod
    def _hour_key() -> int:
        return int(time.time() // 3600)

    def update_budget(self, user_id: int, day_limit: int) -> bool:
        """Достигнут ли дневной лимит обновлений (True — можно)."""
        if day_limit <= 0:
            return False
        bucket = self._day_key()
        value = self._day.get(user_id)
        return value is None or value[0] != bucket or value[1] < day_limit

    def consume_update(self, user_id: int):
        bucket = self._day_key()
        value = self._day.get(user_id)
        if value is None or value[0] != bucket:
            self._day[user_id] = (bucket, 1)
        else:
            self._day[user_id] = (bucket, value[1] + 1)

    def llm_budget(self, user_id: int, hour_limit: int) -> bool:
        if hour_limit <= 0:
            return False
        bucket = self._hour_key()
        value = self._hour.get(user_id)
        return value is None or value[0] != bucket or value[1] < hour_limit

    def consume_llm(self, user_id: int, count: int = 1):
        bucket = self._hour_key()
        value = self._hour.get(user_id)
        if value is None or value[0] != bucket:
            self._hour[user_id] = (bucket, count)
        else:
            self._hour[user_id] = (bucket, value[1] + count)


class WikiManager:
    """Очередь-триггер + один активный апдейт на пользователя (ТЗ 8.3)."""

    def __init__(self):
        self._llm = None
        self._running: set[int] = set()
        self._pending: set[int] = set()
        self._paused: set[int] = set()
        self._last_run_mono: dict[int, float] = {}
        self._consecutive_failures: dict[int, int] = {}
        self.budget = BudgetTracker()

    # --- внедрение зависимостей ---

    def set_llm_caller(self, fn):
        """fn: async (prompt: str) -> str | None. Ошибка — строка 'Ошибка…' или None."""
        self._llm = fn

    # --- управление паузой (восстановление — этап 6, команды) ---

    def pause(self, user_id: int):
        self._paused.add(user_id)

    def unpause(self, user_id: int):
        self._paused.discard(user_id)
        self._consecutive_failures.pop(user_id, None)

    def is_paused(self, user_id: int) -> bool:
        return user_id in self._paused

    # --- очередь и воркер ---

    def enqueue(self, user_id: int):
        if user_id in self._paused:
            return
        self._pending.add(user_id)
        if user_id in self._running:
            return
        asyncio.get_event_loop().create_task(self._worker(user_id))

    async def _worker(self, user_id: int):
        if user_id in self._running:
            return
        self._running.add(user_id)
        try:
            while user_id in self._pending:
                self._pending.discard(user_id)
                if not await self._run_once(user_id):
                    break
        finally:
            self._running.discard(user_id)
            if user_id in self._pending:
                asyncio.get_event_loop().create_task(self._worker(user_id))

    # --- один проход обработки пользователя ---

    async def _run_once(self, user_id: int) -> bool:
        if user_id in self._paused:
            return False
        update_cfg = config.settings().get('update', {})
        min_interval = update_cfg.get('min_interval_seconds', 0)
        last = self._last_run_mono.get(user_id)
        now = time.monotonic()
        if last is not None and now - last < min_interval:
            await asyncio.sleep(min_interval - (now - last))
            return True
        self._last_run_mono[user_id] = time.monotonic()

        return await self._process_user(user_id)

    async def _process_user(self, user_id: int) -> bool:
        """Обрабатывает пользователя, пока есть что обрабатывать (лимиты снимка)."""
        db_path = config.db_path()
        user_dir = pageio.user_wiki_dir(user_id)
        settings = config.settings()
        update_cfg = settings.get('update', {})
        pages_cfg = settings.get('pages', {})
        max_messages = update_cfg.get('max_raw_messages_per_update', 50)
        max_chars = update_cfg.get('max_raw_chars_per_update', 20000)
        budgets_cfg = settings.get('budgets', {})

        while True:
            index_data, status = index_mod.ensure_index(user_dir, db_path, user_id)
            if index_data is None:
                # wiki нет → bootstrap (первое создание)
                ok = await self._bootstrap(user_id, db_path, user_dir)
                return ok

            if not index_mod.service_pages_present(user_dir):
                self._ensure_service_frames(user_dir, index_data)

            watermark = index_data.get('watermark', 0)
            rows = db.fetch_unprocessed(db_path, user_id, watermark,
                                        max_messages=max_messages,
                                        max_chars=max_chars)
            if not rows:
                return True  # больше обрабатывать нечего

            budget_cfg_day = budgets_cfg.get('max_updates_per_user_per_day', 20)
            budget_llm_hour = budgets_cfg.get('max_llm_calls_per_user_per_hour', 5)
            if not self.budget.update_budget(user_id, budget_cfg_day):
                return self._over_limit(user_id, 'update')
            if not self.budget.llm_budget(user_id, budget_llm_hour):
                return self._over_limit(user_id, 'llm')

            processed_ok, more = await self._process_batch(
                user_id, user_dir, index_data, rows, update_cfg, pages_cfg)
            if not processed_ok:
                return False  # сбой: watermark не двигался; ждём следующий триггер
            if not more:
                return True

    # --- обработка одного снимка ---

    async def _process_batch(self, user_id, user_dir, index_data, rows,
                             update_cfg, pages_cfg) -> tuple[bool, bool]:
        max_id = rows[-1]['id']
        page_max = pages_cfg.get('max_page_chars', 2000)

        nontrivial = [r for r in rows if not is_trivial(
            r.get('content'), update_cfg.get('trivial_min_chars', 3))]
        to_update = [(slug, current) for slug, current in (('Home', pageio.read_page(user_dir, 'Home') or ''),
                                                           ('Style', pageio.read_page(user_dir, 'Style') or ''))
                     if _slug_wants_update(slug, nontrivial)]

        new_pages: dict[str, str] = {}
        if to_update:
            raw_block = _build_raw_block(nontrivial)
            for slug, current_md in to_update:
                page = index_mod.find_page(index_data, slug)
                title = (page or {}).get('title', slug)
                target = (pages_cfg.get('home_target_chars', 900)
                          if slug == 'Home' else pages_cfg.get('style_target_chars', 600))
                prompt_text = prompts.build_update_page_prompt(
                    config.settings().get('prompts', {}).get('update_home_style_prompt'),
                    slug=slug, title=title, current_md=current_md,
                    target_chars=target, max_chars=page_max, raw_block=raw_block)
                note = inject_mod.oversize_note(slug)
                if note:
                    prompt_text = f"{prompt_text}\n\n{note}"
                self.budget.consume_llm(user_id)
                md = await self._llm_call(prompt_text)
                md = _validate_page_md(md, page_max)
                if md is None:
                    self._record_failure(user_id, user_dir, index_data,
                                         f"невалидный ответ LLM для {slug}")
                    return False, False
                new_pages[slug] = md

        # Порядок записи 8.4: страницы → индекс (с watermark) только при полном успехе.
        for slug, md in new_pages.items():
            if not pageio.atomic_write_page(user_dir, slug, md):
                self._record_failure(user_id, user_dir, index_data,
                                     f"ошибка записи {slug}.md")
                return False, False

        now = _page_meta_now()
        for slug in new_pages:
            page = index_mod.find_page(index_data, slug)
            if page:
                page['updated'] = now
                page['last_seen'] = now

        index_data['watermark'] = max_id
        index_data['message_count'] = index_data.get('message_count', 0) + len(rows)
        index_data['last_update'] = now
        index_data['last_error'] = None
        self._consecutive_failures.pop(user_id, None)
        self.budget.consume_update(user_id)
        if not index_mod.save_index(user_dir, index_data):
            self._record_failure(user_id, user_dir, index_data, "ошибка записи индекса")
            return False, False

        logger.info("wiki update: user_id=%d строк=%d (nontrivial=%d) watermark→%d",
                    user_id, len(rows), len(nontrivial), max_id)
        has_more = db.count_unprocessed(config.db_path(), user_id, max_id) > 0
        return True, has_more

    # --- bootstrap (п. 9.7) ---

    async def _bootstrap(self, user_id: int, db_path: str, user_dir: str) -> bool:
        settings = config.settings()
        boot = settings.get('bootstrap', {})
        pages_cfg = settings.get('pages', {})
        mode = boot.get('mode', 'from_dossier')
        page_max = pages_cfg.get('max_page_chars', 2000)

        if mode == 'limited_window':
            window = db.fetch_window_rows(
                db_path, user_id, boot.get('limited_window_messages', 50))
            if not window:
                home_md = DEFAULT_HOME_FRAME
            else:
                window_content = [r for r in window if not is_trivial(
                    r.get('content'), settings.get('update', {}).get('trivial_min_chars', 3))]
                raw_block = _build_raw_block(window_content)
                home_md = await self._gen_home_from_window(raw_block, page_max)
                if home_md is None:
                    logger.warning("bootstrap limited_window: не удалось сгенерировать Home (user=%d)",
                                   user_id)
                    return False
            watermark = window[-1]['id'] if window else 0
        elif mode in ('from_dossier', 'current_watermark'):
            watermark = db.watermark(db_path, user_id) or 0
            home_md = DEFAULT_HOME_FRAME
            if mode == 'from_dossier':
                dossier = db.get_dossier_text(db_path, user_id)
                if dossier:
                    max_dossier = boot.get('max_dossier_chars', 8000)
                    if max_dossier:
                        dossier = dossier[:max_dossier]
                    generated = await self._gen_home_from_dossier(dossier, page_max)
                    if generated is None:
                        logger.warning("bootstrap from_dossier: генерация Home не удалась (user=%d)",
                                       user_id)
                        return False
                    home_md = generated
        else:
            logger.error("bootstrap: неизвестный режим %r (user=%d)", mode, user_id)
            return False

        style_md = DEFAULT_STYLE_FRAME
        pageio.atomic_write_page(user_dir, 'Home', home_md)
        pageio.atomic_write_page(user_dir, 'Style', style_md)

        index_data = index_mod.new_index()
        index_data['watermark'] = watermark
        index_data['message_count'] = 0
        index_data['last_update'] = _page_meta_now()
        for slug in ('Home', 'Style'):
            page = index_mod._minimal_page(slug)
            page['last_seen'] = index_data['last_update']
            index_data['pages'].append(page)
        if not index_mod.save_index(user_dir, index_data):
            logger.error("bootstrap: не удалось сохранить индекс (user=%d)", user_id)
            return False

        logger.info("wiki bootstrap: user_id=%d mode=%s watermark=%d",
                    user_id, mode, watermark)
        return True

    def _ensure_service_frames(self, user_dir: str, index_data: dict):
        """Достраивает недостающие файлы Home.md/Style.md (битая валидность)."""
        if not pageio.page_exists(user_dir, 'Home'):
            pageio.atomic_write_page(user_dir, 'Home', DEFAULT_HOME_FRAME)
        if not pageio.page_exists(user_dir, 'Style'):
            pageio.atomic_write_page(user_dir, 'Style', DEFAULT_STYLE_FRAME)
        now = _page_meta_now()
        for slug in ('Home', 'Style'):
            page = index_mod.find_page(index_data, slug)
            if page is None:
                index_data['pages'].append(index_mod._minimal_page(slug))
            elif page.get('status') != 'active':
                page['status'] = 'active'
        index_data['last_error'] = 'missing service pages restored'
        index_mod.save_index(user_dir, index_data)

    # --- LLM ---

    async def _llm_call(self, prompt_text: str) -> str | None:
        if not self._llm:
            logger.warning("wiki: LLM-caller не установлен")
            return None
        try:
            result = await self._llm(prompt_text)
        except Exception as e:
            logger.warning("wiki: исключение LLM: %s", e)
            return None
        if not result or result.startswith('Ошибка'):
            return None
        return result

    async def _gen_home_from_dossier(self, dossier: str, page_max: int) -> str | None:
        prompt_text = prompts.build_bootstrap_home_prompt(
            config.settings().get('prompts', {}).get('update_home_style_prompt'),
            target_chars=config.settings().get('pages', {}).get('home_target_chars', 900),
            max_chars=page_max, dossier_seed=dossier, window_block=None)
        if not prompt_text:
            return None
        md = await self._llm_call(prompt_text)
        return _validate_page_md(md, page_max)

    async def _gen_home_from_window(self, raw_block: str, page_max: int) -> str | None:
        prompt_text = prompts.build_bootstrap_home_prompt(
            None, target_chars=config.settings().get('pages', {}).get('home_target_chars', 900),
            max_chars=page_max, dossier_seed=None, window_block=raw_block)
        if not prompt_text:
            return DEFAULT_HOME_FRAME
        md = await self._llm_call(prompt_text)
        return _validate_page_md(md, page_max)

    # --- ошибки / политики исчерпания ---

    def _over_limit(self, user_id: int, kind: str) -> bool:
        settings = config.settings()
        policy = settings.get('budgets', {}).get('over_limit_policy', 'delay')
        if policy == 'drop_old':
            return self._drop_old(user_id, kind)
        logger.info("wiki budget: user_id=%d лимит %s → delay (watermark не двигается)", user_id, kind)
        return False

    def _drop_old(self, user_id: int, kind: str) -> bool:
        """drop_old: watermark двигается мимо старых строк, остаются новейшие backlog."""
        settings = config.settings()
        max_backlog = settings.get('budgets', {}).get('max_backlog_messages', 300)
        user_dir = pageio.user_wiki_dir(user_id)
        index_data, _ = index_mod.ensure_index(user_dir, config.db_path(), user_id)
        if index_data is None:
            return False
        watermark = index_data.get('watermark', 0)
        total = db.count_unprocessed(config.db_path(), user_id, watermark)
        if total <= max_backlog:
            return False
        # Продвигаем watermark до id строки #(total - max_backlog)
        conn = db.connect(config.db_path())
        try:
            row = conn.execute(
                """
                SELECT id FROM user_raw WHERE user_id = ? AND id > ?
                ORDER BY id ASC LIMIT 1 OFFSET ?
                """, (user_id, watermark, total - max_backlog - 1)).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        drop_to = row['id']
        now = _page_meta_now()
        index_data['watermark'] = max(index_data.get('watermark', 0), drop_to)
        index_data['last_update'] = now
        index_data['last_error'] = f"drop_old: {kind}-лимит, сброшено {total - max_backlog} строк"
        if index_mod.save_index(user_dir, index_data):
            logger.warning("wiki drop_old: user_id=%d watermark→%d (%s)", user_id, drop_to, kind)
            return True
        return False

    def _record_failure(self, user_id: int, user_dir: str, index_data: dict, message: str):
        settings = config.settings()
        update_cfg = settings.get('update', {})
        index_data['last_error'] = message
        index_mod.save_index(user_dir, index_data)
        fails = self._consecutive_failures.get(user_id, 0) + 1
        self._consecutive_failures[user_id] = fails
        max_retries = update_cfg.get('max_batch_retries', 3)
        if fails >= max_retries:
            self._paused.add(user_id)
            logger.error("wiki: user_id=%d устойчивая ошибка — автоапдейты приостановлены: %s",
                         user_id, message)
        else:
            logger.warning("wiki: user_id=%d сбой батча (%d/%d): %s",
                           user_id, fails, max_retries, message)


def _slug_wants_update(slug: str, nontrivial: list[dict]) -> bool:
    if not nontrivial:
        return False
    if slug == 'Home':
        return True
    if slug == 'Style':
        return any(has_style_signal(r.get('content')) for r in nontrivial)
    return False


wiki_manager = WikiManager()
