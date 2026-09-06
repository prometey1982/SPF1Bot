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
from . import textutil
from . import topics
from . import router
from . import redact

logger = logging.getLogger(__name__)

DEFAULT_HOME_FRAME = "# Сводка\n\n"
DEFAULT_STYLE_FRAME = "# Стиль\n\n"

_STYLE_TOKENS = {
    'лол', 'кек', 'ахах', 'рофл', 'хах', 'хз', 'ага', 'ок', 'оу', 'вау', 'йоу',
    'забей', 'жесть', 'красава', 'норм', 'кринж', 'имба', 'топ',
}


def is_trivial(content: str | None, min_chars: int = 3) -> bool:
    """Тривиальное сообщение (п. 8.1): не попадает в LLM, но считается обработанным.

    Короче `min_chars` ИЛИ не содержит значимых токенов (стоп-слова/эмодзи).
    """
    if not content:
        return True
    if len(content.strip()) < min_chars:
        return True
    return not textutil.tokenize(content)


def has_style_signal(content: str | None) -> bool:
    if not content:
        return False
    lower = content.lower()
    tokens = set(textutil.tokenize(lower))
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


def _build_reconcile_window(rows: list[dict], max_chars: int) -> str:
    """Блок окна raw для reconcile, ограниченный max_raw_chars."""
    lines = []
    total = 0
    for i, row in enumerate(reversed(rows), 1):  # последние сверху
        content = (row.get('content') or '').strip()
        if not content:
            continue
        if total + len(content) > max_chars:
            break
        lines.append(f"[{i}] {content}")
        total += len(content)
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
    # Анти-injection/приватность (п. 10.2, 10.3): секреты не сохраняются.
    if redact.has_sensitive(md):
        logger.warning("wiki: ответ страницы содержит запрещённые данные (секреты) — отклонён")
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
        self._reconcile_day: dict[int, tuple[str, int]] = {}  # reconcile/день

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

    def reconcile_budget(self, user_id: int, day_limit: int) -> bool:
        """Достигнут ли дневной лимит reconcile (True — можно)."""
        if day_limit <= 0:
            return False
        bucket = self._day_key()
        value = self._reconcile_day.get(user_id)
        return value is None or value[0] != bucket or value[1] < day_limit

    def consume_reconcile(self, user_id: int):
        bucket = self._day_key()
        value = self._reconcile_day.get(user_id)
        if value is None or value[0] != bucket:
            self._reconcile_day[user_id] = (bucket, 1)
        else:
            self._reconcile_day[user_id] = (bucket, value[1] + 1)


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
                # После успешной обработки снимка — детерминированное создание
                # тематических страниц по окну обработанных строк (п. 9.4).
                try:
                    await self._maybe_create_topic_page(user_id, user_dir, index_data, pages_cfg)
                except Exception as e:
                    logger.warning("wiki: ошибка создания страниц user_id=%d: %s", user_id, e)
                # Авто-reconcile по message_count (п. 9.6)
                try:
                    await self.maybe_auto_reconcile(user_id, index_data)
                except Exception as e:
                    logger.warning("wiki: ошибка авто-reconcile user_id=%d: %s", user_id, e)
                return True

    # --- обработка одного снимка ---

    def _select_updates(self, user_dir, index_data, nontrivial,
                        update_cfg, pages_cfg) -> list[str]:
        """Страницы, затронутые снимком (п. 9.3): Home/Style + тематические.

        Home обновляется при наличии нетривиального содержимого, Style — при
        стилевом сигнале; тематические — по router-аффинности снимка, не более
        `max_pages_per_update`.
        """
        updates: list[str] = []
        if nontrivial:
            updates.append('Home')
            if any(has_style_signal(r.get('content')) for r in nontrivial):
                updates.append('Style')

            snapshot_tokens: set[str] = set()
            for row in nontrivial:
                snapshot_tokens |= textutil.token_set(row.get('content'))
            scored = []
            for page in index_data.get('pages', []):
                slug = page.get('slug', '')
                if slug in ('Home', 'Style') or page.get('status') != 'active':
                    continue
                score = router.score_query(snapshot_tokens, page)
                if score > 0:
                    scored.append((score, slug))
            scored.sort(key=lambda x: (-x[0], x[1]))
            limit = update_cfg.get('max_pages_per_update', 3)
            updates.extend(slug for _, slug in scored[:limit])
        return updates

    async def _process_batch(self, user_id, user_dir, index_data, rows,
                             update_cfg, pages_cfg) -> tuple[bool, bool]:
        max_id = rows[-1]['id']
        page_max = pages_cfg.get('max_page_chars', 2000)

        nontrivial = [r for r in rows if not is_trivial(
            r.get('content'), update_cfg.get('trivial_min_chars', 3))]

        updates = self._select_updates(user_dir, index_data, nontrivial,
                                       update_cfg, pages_cfg)
        new_pages: dict[str, str] = {}
        if updates:
            raw_block = _build_raw_block(nontrivial)
            for slug in updates:
                current_md = pageio.read_page(user_dir, slug) or ''
                page = index_mod.find_page(index_data, slug)
                title = (page or {}).get('title', slug)
                target = self._target_chars(slug, pages_cfg)
                template_key = ('update_home_style_prompt'
                                if slug in ('Home', 'Style') else 'update_page_prompt')
                prompt_text = prompts.build_update_page_prompt(
                    config.settings().get('prompts', {}).get(template_key),
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

    @staticmethod
    def _target_chars(slug: str, pages_cfg: dict) -> int:
        if slug == 'Home':
            return pages_cfg.get('home_target_chars', 900)
        if slug == 'Style':
            return pages_cfg.get('style_target_chars', 600)
        return pages_cfg.get('style_target_chars', 600)  # тематическая — универсальный target

    # --- создание тематических страниц (п. 9.4, 9.5) ---

    async def _maybe_create_topic_page(self, user_id, user_dir, index_data,
                                       pages_cfg) -> bool:
        """Создаёт (или реактивирует) ОДНУ тематическую страницу по окну.

        Возвращает True, если что-то создано/реактивировано. Ошибки LLM не
        роняют инкремент — пишется cooldown, попытка повторится позже.
        """
        settings = config.settings()
        budgets_cfg = settings.get('budgets', {})
        page_max = pages_cfg.get('max_page_chars', 2000)
        max_count = pages_cfg.get('max_count', 20)
        cooldown_hours = pages_cfg.get('create_cooldown_hours', 24)
        max_cooldown = pages_cfg.get('max_cooldown_entries', 100)

        active_count = sum(1 for p in index_data.get('pages', [])
                           if p.get('status') == 'active')
        if active_count >= max_count:
            return False
        if not self.budget.llm_budget(user_id, budgets_cfg.get('max_llm_calls_per_user_per_hour', 5)):
            return False

        watermark = index_data.get('watermark', 0)
        window = db.fetch_processed_window(
            config.db_path(), user_id, watermark,
            pages_cfg.get('create_window_messages', 100))
        if not window:
            return False
        candidates = topics.detect_candidates(window, pages_cfg.get('create_repeats', 3))
        if not candidates:
            return False

        topics.prune_cooldowns(index_data, cooldown_hours, max_cooldown)

        for cand in candidates:
            token = cand['token']
            if topics.is_cooldown_active(index_data, token, cooldown_hours):
                continue
            pages = index_data.get('pages', [])
            overlap = topics.check_page_overlap(token, pages)

            if overlap is not None and overlap.get('status') == 'active':
                # Тема уже покрыта активной страницей — новая не нужна.
                continue

            examples = self._candidate_examples(window, cand, token)

            if overlap is not None and overlap.get('status') == 'archived':
                # Возврат темы: реактивируем архивную, дубль не создаём (п. 9.5).
                ok = await self._reactivate_page(user_id, user_dir, index_data,
                                                 overlap, examples, page_max)
            else:
                ok = await self._create_new_page(user_id, user_dir, index_data,
                                                 token, cand, examples, page_max)
            if ok:
                return True
            # Отклонение → cooldown (сохраняем индекс: иначе повторим попытку
            # с тем же кандидатом на каждом проходе)
            topics.set_cooldown(index_data, token, _page_meta_now())
            topics.prune_cooldowns(index_data, cooldown_hours, max_cooldown)
            if not index_mod.save_index(user_dir, index_data):
                logger.warning("wiki: не удалось сохранить cooldown (user=%d)", user_id)
        return False

    @staticmethod
    def _candidate_examples(window: list[dict], cand: dict, token: str) -> str:
        """Строки окна с кандидатом-токеном (для LLM-предложения)."""
        lines = []
        for row in window:
            if row['id'] not in cand['messages']:
                continue
            content = (row.get('content') or '').strip()
            if token.lower() not in textutil.token_set(content):
                continue
            lines.append(content[:300])
            if len(lines) >= 6:
                break
        return "\n".join(lines)

    async def _create_new_page(self, user_id, user_dir, index_data, token, cand,
                               examples, page_max) -> bool:
        settings = config.settings()
        prompts_cfg = settings.get('prompts', {})
        prompt_text = prompts.build_create_page_prompt(
            prompts_cfg.get('create_page_prompt'),
            candidate=token, examples_block=examples, max_chars=page_max)
        self.budget.consume_llm(user_id)
        raw = await self._llm_call(prompt_text)
        proposal = topics.parse_page_proposal(raw) if raw else None
        if proposal is None:
            logger.info("wiki: предложение страницы отклонено (token=%s): пустой/невалидный ответ", token)
            return False
        proposal['content'] = _validate_page_md(proposal['content'], page_max)
        if proposal['content'] is None:
            return False

        # Повторная проверка пересечений уже с предложенными keywords/aliases
        if topics.check_page_overlap(proposal['slug'], index_data.get('pages', [])) is not None:
            logger.info("wiki: предложение пересекается с существующей страницей (%s)", proposal['slug'])
            return False
        if index_mod.find_page(index_data, proposal['slug']) is not None:
            return False
        if not pageio.atomic_write_page(user_dir, proposal['slug'], proposal['content']):
            return False

        now = _page_meta_now()
        index_data['pages'].append({
            'slug': proposal['slug'],
            'title': proposal['title'],
            'status': 'active',
            'keywords': proposal['keywords'],
            'aliases': proposal['aliases'],
            'created': now,
            'updated': now,
            'last_seen': now,
            'hits': 0,
        })
        if index_mod.save_index(user_dir, index_data):
            logger.info("wiki: создана страница %s (token=%s, user=%d)",
                        proposal['slug'], token, user_id)
            return True
        return False

    async def _reactivate_page(self, user_id, user_dir, index_data, page,
                               examples, page_max) -> bool:
        settings = config.settings()
        pages_cfg = settings.get('pages', {})
        slug = page['slug']
        current_md = pageio.read_page(user_dir, slug) or ''
        prompt_text = prompts.build_reactivate_prompt(
            settings.get('prompts', {}).get('update_page_prompt'),
            slug=slug, title=page.get('title', slug), current_md=current_md,
            examples_block=examples,
            target_chars=pages_cfg.get('style_target_chars', 600), max_chars=page_max)
        self.budget.consume_llm(user_id)
        raw = await self._llm_call(prompt_text)
        md = _validate_page_md(raw, page_max)
        if md is None:
            logger.info("wiki: реактивация %s отклонена (невалидный ответ)", slug)
            return False
        if not pageio.atomic_write_page(user_dir, slug, md):
            return False
        now = _page_meta_now()
        page['status'] = 'active'
        page['updated'] = now
        page['last_seen'] = now
        if index_mod.save_index(user_dir, index_data):
            logger.info("wiki: реактивирована архивная страница %s (user=%d)", slug, user_id)
            return True
        return False

    # --- reconcile (п. 9.6) и ручное слияние (п. 13) ---

    async def maybe_auto_reconcile(self, user_id: int, index_data: dict) -> bool:
        """Авто-reconcile по message_count (вызывается после успешного инкремента)."""
        settings = config.settings()
        reconcile_cfg = settings.get('reconcile', {})
        budgets_cfg = settings.get('budgets', {})
        if index_data.get('message_count', 0) < reconcile_cfg.get('every_n_messages', 100):
            return False
        if not self.budget.reconcile_budget(
                user_id, budgets_cfg.get('reconcile_llm_calls_per_user_per_day', 20)):
            return False
        if not self.budget.llm_budget(
                user_id, budgets_cfg.get('max_llm_calls_per_user_per_hour', 5)):
            return False
        result = await self._reconcile(user_id)
        return bool(result.get('success'))

    async def reconcile(self, user_id: int) -> dict:
        """Ручной /reconcile_wiki (вне обычного бюджета при allow_manual_*)."""
        return await self._reconcile(user_id, manual=True)

    async def _reconcile(self, user_id: int, manual: bool = False) -> dict:
        settings = config.settings()
        reconcile_cfg = settings.get('reconcile', {})
        pages_cfg = settings.get('pages', {})
        budgets_cfg = settings.get('budgets', {})
        prompts_cfg = settings.get('prompts', {})
        page_max = pages_cfg.get('max_page_chars', 2000)
        db_path = config.db_path()
        user_dir = pageio.user_wiki_dir(user_id)

        index_data, _ = index_mod.ensure_index(user_dir, db_path, user_id)
        if index_data is None:
            return {'success': False, 'partial': False,
                    'message': 'У пользователя нет валидной wiki.'}

        rec_day_limit = budgets_cfg.get('reconcile_llm_calls_per_user_per_day', 20)
        if not self.budget.reconcile_budget(user_id, rec_day_limit):
            return {'success': False, 'partial': False,
                    'message': 'Дневной лимит reconcile исчерпан.'}
        if not manual and not self.budget.llm_budget(
                user_id, budgets_cfg.get('max_llm_calls_per_user_per_hour', 5)):
            return {'success': False, 'partial': False, 'message': 'Бюджет LLM исчерпан.'}

        # Окно raw (последние строки по id) + упоминания
        rows = db.fetch_window_rows(db_path, user_id, reconcile_cfg.get('window_messages', 300))
        window_block = _build_reconcile_window(rows, reconcile_cfg.get('max_raw_chars', 100000))
        mentions: list[str] = []
        if reconcile_cfg.get('include_mentions', True):
            username = db.last_username(db_path, user_id)
            if username:
                mentions = db.get_mentions_quotes(
                    db_path, username,
                    ttl_hours=settings.get('mentions', {}).get('ttl_hours', 24),
                    limit=reconcile_cfg.get('max_mentions', 50))
        mentions_block = "\n".join(f"Упоминание: {q[:500]}" for q in mentions)

        now = _page_meta_now()
        now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        stale_days = pages_cfg.get('archive_after_days', 90)
        window_tokens = textutil.token_set(window_block)

        active_themes = []
        for page in index_data.get('pages', []):
            if page.get('status') != 'active':
                continue
            slug = page.get('slug', '')
            if slug in ('Home', 'Style'):
                continue
            active_themes.append(page)

        # Устаревание страниц-тем по last_seen (не Home/Style); null не архивирует
        stale_slugs = set()
        for page in active_themes:
            ts = page.get('last_seen') or page.get('updated')
            if ts is None:
                continue
            try:
                parsed = datetime.fromisoformat(ts)
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            except ValueError:
                continue
            age_days = (now_dt - parsed).total_seconds() / 86400
            confirmed = bool(window_tokens and router.score_query(window_tokens, page) > 0)
            if not confirmed and age_days >= stale_days:
                stale_slugs.add(page.get('slug'))

        # Кандидаты на LLM-обновление: Home, Style, подтверждённые темы
        candidates = []
        seen = set()
        for slug in ('Home', 'Style'):
            page = index_mod.find_page(index_data, slug)
            if page is not None and page.get('status') == 'active':
                candidates.append(page)
                seen.add(slug)
        for page in active_themes:
            slug = page['slug']
            if slug in stale_slugs or slug in seen:
                continue
            if window_tokens and router.score_query(window_tokens, page) > 0:
                candidates.append(page)

        cap = reconcile_cfg.get('max_pages_per_reconcile', 10)
        partial = len(candidates) > cap
        chosen = candidates[:cap]

        # Генерируем новые версии (все валидные — иначе не успех, п. 3.4 ревью)
        new_pages: dict[str, str] = {}
        for page in chosen:
            slug = page['slug']
            current_md = pageio.read_page(user_dir, slug) or ''
            target = self._target_chars(slug, pages_cfg)
            prompt_text = prompts.build_reconcile_prompt(
                prompts_cfg.get('reconcile_prompt'),
                slug=slug, title=page.get('title', slug), current_md=current_md,
                target_chars=target, max_chars=page_max,
                window_block=window_block, mentions_block=mentions_block)
            self.budget.consume_llm(user_id)
            md = await self._llm_call(prompt_text)
            md = _validate_page_md(md, page_max)
            if md is None:
                return {'success': False, 'partial': True,
                        'message': f'Reconcile не завершён: невалидный ответ LLM для {slug}. '
                                   'Повторите позже.'}
            new_pages[slug] = md

        for slug, md in new_pages.items():
            if not pageio.atomic_write_page(user_dir, slug, md):
                return {'success': False, 'partial': True,
                        'message': f'Ошибка записи {slug}.md при reconcile.'}

        # Применяем изменения индекса
        for page in index_data.get('pages', []):
            slug = page.get('slug', '')
            if slug in stale_slugs:
                page['status'] = 'archived'
                logger.info("wiki reconcile: страница %s → archived (user=%d)", slug, user_id)
        for slug in new_pages:
            page = index_mod.find_page(index_data, slug)
            if page:
                page['updated'] = now
                page['last_seen'] = now
        # Подтверждённые, но не выбранные из-за cap, темы: last_seen освежается (п. 7.2)
        for page in candidates[len(chosen):]:
            page['last_seen'] = now

        index_data['last_error'] = None
        if not partial:
            index_data['message_count'] = 0
            index_data['last_reconcile'] = now
            self.budget.consume_reconcile(user_id)
            if manual:
                self.unpause(user_id)
        index_data['last_update'] = now

        if not index_mod.save_index(user_dir, index_data):
            return {'success': False, 'partial': True,
                    'message': 'Ошибка записи индекса при reconcile.'}

        logger.info("wiki reconcile: user_id=%d страниц=%d partial=%s archived=%d",
                    user_id, len(new_pages), partial, len(stale_slugs))
        if partial:
            return {'success': False, 'partial': True,
                    'message': f'Reconcile обработал {len(new_pages)} из {len(candidates)} '
                               'страниц; message_count не сброшен — повторите позже.'}
        return {'success': True, 'partial': False,
                'message': f'Reconcile выполнен: обновлено страниц={len(new_pages)}, '
                           f'архивировано={len(stale_slugs)}.'}

    async def merge_pages(self, user_id: int, slug1: str, slug2: str) -> dict:
        """Слияние страниц slug2 → slug1 (п. 13). Возвращает (ok, message)."""
        settings = config.settings()
        pages_cfg = settings.get('pages', {})
        prompts_cfg = settings.get('prompts', {})
        page_max = pages_cfg.get('max_page_chars', 2000)
        user_dir = pageio.user_wiki_dir(user_id)
        index_data, _ = index_mod.ensure_index(user_dir, config.db_path(), user_id)
        if index_data is None:
            return {'ok': False, 'message': 'У пользователя нет валидной wiki.'}
        if slug1 == slug2:
            return {'ok': False, 'message': 'Страницы должны различаться.'}
        if slug1 in ('Home', 'Style'):
            return {'ok': False, 'message': 'slug1 не может быть служебной страницей.'}
        if not pageio.is_safe_slug(slug1) or not pageio.is_safe_slug(slug2):
            return {'ok': False, 'message': 'Небезопасный slug.'}

        page1 = index_mod.find_page(index_data, slug1)
        page2 = index_mod.find_page(index_data, slug2)
        if page1 is None or page2 is None:
            return {'ok': False, 'message': 'Одна из страниц не найдена.'}
        md1 = pageio.read_page(user_dir, slug1) or ''
        md2 = pageio.read_page(user_dir, slug2) or ''

        prompt_text = prompts.build_merge_prompt(
            prompts_cfg.get('merge_page_prompt'),
            target_slug=slug1, target_title=page1.get('title', slug1), target_md=md1,
            source_slug=slug2, source_title=page2.get('title', slug2), source_md=md2,
            max_chars=page_max)
        merged = await self._llm_call(prompt_text)
        merged = _validate_page_md(merged, page_max)
        if merged is None:
            return {'ok': False, 'message': 'Невалидный ответ LLM при слиянии.'}

        now = _page_meta_now()
        page1['status'] = 'active'
        page1['updated'] = now
        page1['last_seen'] = now
        page1['title'] = page1.get('title') or slug1
        # Объединение keywords/aliases без дублей
        for key in ('keywords', 'aliases'):
            merged_list = []
            seen = set()
            for item in list(page1.get(key, [])) + list(page2.get(key, [])):
                if item and item not in seen:
                    seen.add(item)
                    merged_list.append(item)
            page1[key] = merged_list
        page2['status'] = 'archived'

        if not (pageio.atomic_write_page(user_dir, slug1, merged)
                and pageio.atomic_write_page(user_dir, slug2, pageio.read_page(user_dir, slug2) or '')
                and index_mod.save_index(user_dir, index_data)):
            return {'ok': False, 'message': 'Ошибка записи при слиянии (изменения не применены).'}
        logger.info("wiki merge: user_id=%d %s <- %s", user_id, slug1, slug2)
        return {'ok': True,
                'message': f'Страница {slug2} влита в {slug1}; {slug2} архивирована.'}

    # --- backfill после импорта (п. 9.8) ---

    async def backfill_user(self, user_id: int) -> bool:
        """Обрабатывает импортированные строки пользователя (bootstrap/drain).

        Для новой wiki создаёт каркас с watermark=0 (mode from_import) и затем
        докатывает ВСЕ необработанные строки снимками (без обычных бюджетов —
        импортный backfill использует отдельные лимиты, п. 9.8). Возвращает True
        при полном успехе.
        """
        settings = config.settings()
        update_cfg = settings.get('update', {})
        pages_cfg = settings.get('pages', {})
        max_messages = update_cfg.get('max_raw_messages_per_update', 50)
        max_chars = update_cfg.get('max_raw_chars_per_update', 20000)
        db_path = config.db_path()
        user_dir = pageio.user_wiki_dir(user_id)

        index_data, _ = index_mod.ensure_index(user_dir, db_path, user_id)
        if index_data is None:
            if not await self._bootstrap(user_id, db_path, user_dir,
                                         mode_override='from_import'):
                logger.warning("backfill: bootstrap не создан (user_id=%d)", user_id)
                return False

        # Докатываем необработанные строки снимками (watermark двигается только
        # по фактически обработанным). Бюджеты обычных автоапдейтов не действуют.
        while True:
            index_data, _ = index_mod.ensure_index(user_dir, db_path, user_id)
            if index_data is None:
                return False
            if not index_mod.service_pages_present(user_dir):
                self._ensure_service_frames(user_dir, index_data)
            rows = db.fetch_unprocessed(db_path, user_id,
                                        index_data.get('watermark', 0),
                                        max_messages=max_messages, max_chars=max_chars)
            if not rows:
                return True
            ok, more = await self._process_batch(user_id, user_dir, index_data,
                                                 rows, update_cfg, pages_cfg)
            if not ok:
                logger.warning("backfill: сбой батча (user_id=%d)", user_id)
                return False
            if not more:
                return True

    # --- bootstrap (п. 9.7) ---

    async def _bootstrap(self, user_id: int, db_path: str, user_dir: str,
                         mode_override: str | None = None) -> bool:
        settings = config.settings()
        boot = settings.get('bootstrap', {})
        pages_cfg = settings.get('pages', {})
        mode = mode_override or boot.get('mode', 'from_dossier')
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
        elif mode == 'from_import':
            # Импорт экспорта: wiki стартует с watermark=0, чтобы необработанные
            # (импортированные) строки обработались последующим drain'ом (п. 9.7).
            watermark = 0
            home_md = DEFAULT_HOME_FRAME
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


wiki_manager = WikiManager()
