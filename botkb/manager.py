"""Менеджер БЗ бота: bootstrap, инкремент, темы, бюджеты (ТЗ п. 8, 9.3, 9.4, 9.7).

Одна глобальная очередь + один воркер (сериализация апдейтов БЗ, п. 8).
Батч = независимые задачи по страницам (self + knowledge); сбой одной страницы
не останавливает батч: watermark всё равно продвигается до MAX(id) снимка,
страница помечается `last_error` и добирается позже reconcile (п. 8, 9.3).

Этап 3 покрывает: ленивый bootstrap (modes from_system_prompt/limited_window/
empty; history discard/backlog), инкремент со снимком, узкий триггер self
(только реплики людей на ходы бота, п. 2.4/9.3.3), блок диалогов с лимитом
`self_context_turns`, раздельные квоты self/knowledge, пер-страничную изоляцию,
детерминированное создание тематических страниц по повторяемости (п. 9.4),
бюджеты и политику переполнения (delay/drop_old). Механика глобального pause/
backoff/карантина и reconcile — этапы 5/6.

LLM-вызовы не выполняются здесь напрямую: bot.py внедряет `set_llm_caller()`
(callable, оборачивающий провайдер). В тестах — фейковый caller.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from botwiki import router, redact, textutil
from botwiki import inject as inject_mod

from . import config, db
from . import index as index_mod
from . import pages as pageio
from . import prompts
from . import topics

logger = logging.getLogger(__name__)

HOME_FRAME = "# О боте\n\n"
STYLE_FRAME = "# Стиль\n\n"

SERVICE_PAGES = ('Home', 'Style')


def is_trivial(content: str | None, min_chars: int = 3) -> bool:
    """Тривиальное сообщение (п. 8): не идёт в LLM, но считается обработанным."""
    if not content:
        return True
    if len(content.strip()) < min_chars:
        return True
    return not textutil.tokenize(content)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def _process_page_md(raw, page_max: int):
    """Валидирует ответ LLM для страницы → (markdown|None, причина|None).

    Пустой/ошибка провайдера/секреты — отклонение. Превышение размера не роняет
    страницу: обрезается по границам строк до page_max (п. 8).
    """
    text = raw if isinstance(raw, str) else ''
    text = text.strip()
    if not text:
        return None, 'пустой ответ LLM'
    if text.startswith('Ошибка'):
        return None, f'ошибка провайдера: {text[:120]}'
    if redact.has_sensitive(text):
        return None, 'ответ содержит запрещённые данные (секреты)'
    if len(text) > page_max:
        trimmed = inject_mod.truncate_md(text, page_max).strip()
        if not trimmed:
            return None, f'ответ обрезан до пустоты (лимит {page_max})'
        return trimmed, None
    return text, None


class _Budget:
    """Счётчики бюджетов в памяти процесса (ТЗ п. 9.2): день и час, глобально."""

    def __init__(self):
        self._day: tuple[str, int] | None = None
        self._hour: tuple[int, int] | None = None

    @staticmethod
    def _day_key() -> str:
        return datetime.now(timezone.utc).strftime('%Y%m%d')

    @staticmethod
    def _hour_key() -> int:
        return int(time.time() // 3600)

    def can_update(self, day_limit: int) -> bool:
        if day_limit <= 0:
            return False
        bucket = self._day_key()
        return self._day is None or self._day[0] != bucket or self._day[1] < day_limit

    def consume_update(self):
        bucket = self._day_key()
        if self._day is None or self._day[0] != bucket:
            self._day = (bucket, 1)
        else:
            self._day = (bucket, self._day[1] + 1)

    def can_llm(self, hour_limit: int) -> bool:
        if hour_limit <= 0:
            return False
        bucket = self._hour_key()
        return self._hour is None or self._hour[0] != bucket or self._hour[1] < hour_limit

    def consume_llm(self, count: int = 1):
        bucket = self._hour_key()
        if self._hour is None or self._hour[0] != bucket:
            self._hour = (bucket, count)
        else:
            self._hour = (bucket, self._hour[1] + count)


class KBManager:
    """Один воркер на всю БЗ (глобальная сериализация апдейтов)."""

    def __init__(self):
        self._llm = None
        self._running = False
        self._pending = False
        self.budget = _Budget()

    # --- внедрение зависимостей ---

    def set_llm_caller(self, fn):
        """fn: async (prompt: str) -> str | None. Ошибка — строка 'Ошибка…' или None."""
        self._llm = fn

    async def _llm_call(self, prompt: str) -> str | None:
        if self._llm is None:
            logger.warning("bot_kb: LLM-caller не внедрён")
            return None
        try:
            return await self._llm(prompt)
        except Exception as e:
            logger.warning("bot_kb: ошибка LLM-вызова: %s", e)
            return f"Ошибка при вызове LLM: {e}"

    # --- режим ---

    @staticmethod
    def _enabled_for_updates() -> bool:
        """Апдейты/bootstrap возможны только в shadow/primary (ТЗ 6.1, 9.7)."""
        return config.mode() in ('shadow', 'primary')

    # --- очередь и воркер ---

    def enqueue(self):
        if not self._enabled_for_updates():
            return
        self._pending = True
        if self._running:
            return
        asyncio.get_event_loop().create_task(self._worker())

    async def _worker(self):
        if self._running:
            return
        self._running = True
        try:
            while self._pending and self._enabled_for_updates():
                self._pending = False
                await self._drain()
        finally:
            self._running = False
            if self._pending and self._enabled_for_updates():
                asyncio.get_event_loop().create_task(self._worker())

    async def run_updates(self):
        """Прямой проход (для тестов/отладки): обрабатывает до исчерпания."""
        if not self._enabled_for_updates():
            return
        await self._drain()

    # --- главный цикл ---

    async def _drain(self):
        settings = config.settings()
        update_cfg = settings.get('update', {})
        pages_cfg = settings.get('pages', {})
        budgets_cfg = settings.get('budgets', {})
        max_messages = update_cfg.get('max_raw_messages_per_update', 50)
        max_chars = update_cfg.get('max_raw_chars_per_update', 20000)
        root = pageio.kb_root()
        db_path = config.db_path()

        while True:
            index_data, status = index_mod.ensure_index(root, db_path)
            if index_data is None:
                if not await self._bootstrap():
                    return
                continue

            if not index_mod.service_pages_present(root):
                if not self._ensure_service_frames(index_data):
                    return

            watermark = index_data.get('watermark', 0)

            # Политика переполнения непокрытого хвоста (п. 9.2): drop_old двигает
            # watermark поверх избыточного диапазона; delay — no-op.
            backlog_policy = budgets_cfg.get('over_limit_policy', 'delay')
            if backlog_policy == 'drop_old':
                if not self._maybe_drop_backlog(index_data):
                    return

            rows = db.fetch_unprocessed(db_path, watermark,
                                        max_messages=max_messages,
                                        max_chars=max_chars)
            if not rows:
                # Снимок пуст → окно разобрано: детерминированное создание темы
                # (по обработанным строкам) и выход.
                try:
                    await self._maybe_create_topic(index_data, pages_cfg)
                except Exception as e:
                    logger.warning("bot_kb: ошибка создания темы: %s", e)
                return

            if not self.budget.can_update(budgets_cfg.get('max_updates_per_day', 60)) \
                    or not self.budget.can_llm(budgets_cfg.get('max_llm_calls_per_hour', 12)):
                logger.info("bot_kb: бюджет исчерпан, пауза до следующего триггера")
                return

            ok = await self._process_batch(index_data, rows, update_cfg,
                                           pages_cfg, budgets_cfg)
            if not ok:
                return

    def _maybe_drop_backlog(self, index_data: dict) -> bool:
        """drop_old: если непокрытых строк больше max_backlog_rows — сбросить старый
        избыточный диапазон (watermark двигается; строки живут до ttl_hours)."""
        settings = config.settings()
        budgets_cfg = settings.get('budgets', {})
        max_backlog = int(budgets_cfg.get('max_backlog_rows', 5000))
        if max_backlog <= 0:
            return False
        db_path = config.db_path()
        unprocessed = db.count_unprocessed(db_path, index_data.get('watermark', 0))
        if unprocessed <= max_backlog:
            return True
        new_wm = db.tail_watermark(db_path, max_backlog)
        if new_wm <= index_data.get('watermark', 0):
            return True
        logger.info("bot_kb: drop_old: непокрытых=%d > max_backlog=%d, "
                    "watermark %d → %d", unprocessed, max_backlog,
                    index_data.get('watermark', 0), new_wm)
        index_data['watermark'] = new_wm
        return index_mod.save_index(pageio.kb_root(), index_data)

    # --- bootstrap (п. 9.7) ---

    async def _bootstrap(self) -> bool:
        settings = config.settings()
        boot_cfg = settings.get('bootstrap', {})
        pages_cfg = settings.get('pages', {})
        root = pageio.kb_root()
        db_path = config.db_path()

        history = boot_cfg.get('history', 'discard')
        if history == 'backlog':
            keep = int(boot_cfg.get('max_history_messages', 500))
            watermark = db.tail_watermark(db_path, keep)
        else:
            watermark = db.watermark(db_path) or 0

        mode = boot_cfg.get('mode', 'from_system_prompt')
        seed_text = ''
        window_block = ''
        if mode in ('from_system_prompt', 'limited_window'):
            seed_text = self._bootstrap_seed(boot_cfg)
            if mode == 'limited_window':
                window = db.fetch_window_rows(
                    db_path, boot_cfg.get('limited_window_messages', 80))
                window_block = _render_block([{'id': r['id'], 'content': r.get('content'),
                                               'content_type': 'text'} for r in window])
            if not (seed_text and seed_text.strip()) and not window_block:
                mode = 'empty'  # источников нет → каркас без LLM (как user-wiki)

        index = index_mod.new_index()
        index['watermark'] = watermark

        home_md, style_md = HOME_FRAME, STYLE_FRAME
        if mode != 'empty':
            page_max = pages_cfg.get('max_page_chars', 2400)
            prompts_cfg = settings.get('prompts', {})
            target_home = pages_cfg.get('home_target_chars', 900)
            target_style = pages_cfg.get('style_target_chars', 700)
            prompt_home = prompts.build_bootstrap_home_prompt(
                prompts_cfg.get('bootstrap_home_prompt'), target_chars=target_home,
                max_chars=page_max, seed_text=seed_text, window_block=window_block)
            prompt_style = prompts.build_bootstrap_style_prompt(
                prompts_cfg.get('bootstrap_style_prompt'), target_chars=target_style,
                max_chars=page_max, seed_text=seed_text, window_block=window_block)
            if prompt_home:
                home_md, _ = _process_page_md(await self._llm_call(prompt_home), page_max)
                home_md = home_md or HOME_FRAME
            if prompt_style:
                style_md, _ = _process_page_md(await self._llm_call(prompt_style), page_max)
                style_md = style_md or STYLE_FRAME

        if not (pageio.write_page('Home', home_md, root)
                and pageio.write_page('Style', style_md, root)):
            logger.error("bot_kb: bootstrap: не удалось записать Home/Style")
            return False

        now = _now_iso()
        for slug, content in (('Home', home_md), ('Style', style_md)):
            index['pages'].append({
                'slug': slug,
                'title': 'О боте' if slug == 'Home' else 'Стиль',
                'kind': 'self',
                'status': 'active',
                'keywords': [],
                'aliases': [],
                'created': now,
                'updated': now,
                'last_seen': now,
                'hits': 0,
                'quarantined': False,
            })
        index['message_count'] = 0
        if not index_mod.save_index(root, index):
            logger.error("bot_kb: bootstrap: не удалось сохранить индекс")
            return False
        logger.info("bot_kb: bootstrap выполнен (mode=%s, history=%s, watermark=%d)",
                    mode, history, watermark)
        return True

    @staticmethod
    def _bootstrap_seed(boot_cfg: dict) -> str:
        """Затравка для bootstrap: operator seed или ai.system_prompt (усечённая)."""
        seed = boot_cfg.get('seed') or ''
        if not seed:
            top = config.top_config()
            seed = (top.get('ai') or {}).get('system_prompt') or ''
        if not isinstance(seed, str):
            seed = ''
        max_seed = int(boot_cfg.get('max_seed_chars', 12000))
        if max_seed and len(seed) > max_seed:
            seed = seed[:max_seed]
        return seed

    def _ensure_service_frames(self, index_data: dict) -> bool:
        """Восстановление файлов Home/Style (фреймы), если их нет (п. 7.2/8.2)."""
        root = pageio.kb_root()
        for slug, frame in (('Home', HOME_FRAME), ('Style', STYLE_FRAME)):
            if not pageio.page_exists(slug, root):
                pageio.write_page(slug, frame, root)
        if not index_mod.service_pages_present(root):
            return False
        return index_mod.save_index(root, index_data)

    # --- обработка одного снимка ---

    async def _process_batch(self, index_data: dict, rows: list[dict],
                             update_cfg: dict, pages_cfg: dict,
                             budgets_cfg: dict) -> bool:
        max_id = rows[-1]['id']
        page_max = pages_cfg.get('max_page_chars', 2400)
        hour_limit = budgets_cfg.get('max_llm_calls_per_hour', 12)

        trivial_min = update_cfg.get('trivial_min_chars', 3)
        nontrivial = [r for r in rows if not is_trivial(r.get('content'), trivial_min)]
        human_rows = [r for r in nontrivial if r.get('speaker') == 'human']

        # --- выбор страниц под обновление (раздельные квоты, п. 9.3) ---
        updates: dict[str, str] = {}   # slug -> kind
        dialog_block = ''
        if update_cfg.get('self_update_on_feedback', True):
            dialog_block, feedback_present = self._build_feedback_block(rows, update_cfg)
            if feedback_present:
                for slug in SERVICE_PAGES:
                    if len(updates) >= len(SERVICE_PAGES):
                        break
                    page = index_mod.find_page(index_data, slug)
                    if page is not None and page.get('status') == 'active':
                        updates[slug] = 'self'
                    if len(updates) >= update_cfg.get('max_self_pages_per_update', 2):
                        break

        if human_rows:
            tokens: set[str] = set()
            for r in human_rows:
                tokens |= textutil.token_set(r.get('content'))
            scored = []
            for page in index_data.get('pages', []):
                if page.get('status') != 'active' or page.get('kind') == 'self':
                    continue
                score = router.score_query(tokens, page)
                if score > 0:
                    scored.append((score, page.get('slug')))
            scored.sort(key=lambda x: (-x[0], x[1]))
            limit = update_cfg.get('max_pages_per_update', 3)
            for _, slug in scored[:limit]:
                updates.setdefault(slug, 'knowledge')

        # --- LLM-обновление страниц (независимо, с ретраями) ---
        applied: list[str] = []
        knowledge_block = _render_block(human_rows)
        ordered = ([s for s in SERVICE_PAGES if s in updates]
                   + [s for s in updates if s not in SERVICE_PAGES])
        for slug in ordered:
            page = index_mod.find_page(index_data, slug)
            kind = 'self' if slug in SERVICE_PAGES else 'knowledge'
            block = dialog_block if kind == 'self' else knowledge_block
            if not (block or '').strip():
                continue
            current_md = pageio.read_page(slug, root=pageio.kb_root()) or ''
            title = (page or {}).get('title', slug)
            target = self._target_chars(slug, kind, pages_cfg)
            prompts_cfg = config.settings().get('prompts', {})
            if kind == 'self':
                prompt_text = prompts.build_update_self_prompt(
                    prompts_cfg.get('update_self_prompt'), slug=slug, title=title,
                    current_md=current_md, target_chars=target, max_chars=page_max,
                    dialog_block=block)
            else:
                prompt_text = prompts.build_update_knowledge_prompt(
                    prompts_cfg.get('update_knowledge_prompt'), slug=slug, title=title,
                    current_md=current_md, target_chars=target, max_chars=page_max,
                    raw_block=block)

            md, reason = await self._update_page_with_retries(
                prompt_text, page_max, hour_limit)
            if md is None:
                if reason and page is not None:
                    page['last_error'] = reason
                    logger.warning("bot_kb: страница %s: %s", slug, reason)
                continue
            if not pageio.write_page(slug, md, pageio.kb_root()):
                if page is not None:
                    page['last_error'] = 'ошибка записи файла страницы'
                logger.warning("bot_kb: не удалось записать страницу %s", slug)
                continue
            applied.append(slug)
            if page is not None:
                page['updated'] = _now_iso()
                page['last_seen'] = page['updated']
                page['last_error'] = None

        # Порядок записи п. 8: страницы → индекс; watermark = MAX(id) снимка
        # продвигается в любом случае (успех/сбой отдельных страниц).
        now = _now_iso()
        index_data['watermark'] = max_id
        index_data['message_count'] = index_data.get('message_count', 0) + len(rows)
        index_data['last_update'] = now
        index_data['last_error'] = None
        self.budget.consume_update()
        if not index_mod.save_index(pageio.kb_root(), index_data):
            logger.error("bot_kb: не удалось сохранить индекс после батча")
            return False

        logger.info("bot_kb update: строк=%d (nontrivial=%d) страниц=%s watermark→%d",
                    len(rows), len(nontrivial), applied or '—', max_id)
        return True

    async def _update_page_with_retries(self, prompt_text: str, page_max: int,
                                        hour_limit: int):
        """Ретраи одной страницы. Возвращает (markdown|None, причина|None)."""
        settings = config.settings()
        max_retries = int(settings.get('update', {}).get('max_batch_retries', 3))
        last_reason = None
        attempts = 0
        for _ in range(max_retries + 1):
            if not self.budget.can_llm(hour_limit):
                logger.info("bot_kb: часовой LLM-бюджет исчерпан — страницы дальше не трогаем")
                return None, None
            self.budget.consume_llm()
            attempts += 1
            md, reason = _process_page_md(await self._llm_call(prompt_text), page_max)
            if md is not None:
                return md, None
            last_reason = reason
        return None, last_reason or 'не удалось получить валидный ответ'

    @staticmethod
    def _target_chars(slug: str, kind: str, pages_cfg: dict) -> int:
        if slug == 'Home':
            return pages_cfg.get('home_target_chars', 900)
        if slug == 'Style':
            return pages_cfg.get('style_target_chars', 700)
        return pages_cfg.get('knowledge_target_chars', 1400)

    # --- обратная связь и блок диалогов (п. 9.3.3) ---

    def _build_feedback_block(self, rows: list[dict], update_cfg: dict):
        """Возвращает (dialog_block|'', feedback_present: bool).

        Узкий триггер self: строки-бот сами по себе self не обновляют — нужна
        реплика-человек на ход бота (реплика может быть на ход из снимка или из
        уже обработанных строк). Блок строится по последним `self_context_turns`
        таким ходам в пределах `max_raw_chars_per_update`.
        """
        db_path = config.db_path()
        max_turns = int(update_cfg.get('self_context_turns', 3))
        max_chars = int(update_cfg.get('max_raw_chars_per_update', 20000))
        trivial_min = update_cfg.get('trivial_min_chars', 3)

        targets: list[tuple[int, dict]] = []  # (order_row_id, turn info)
        seen_turns: set = set()
        for row in rows:
            if row.get('speaker') != 'human':
                continue
            if is_trivial(row.get('content'), trivial_min):
                continue
            target_id = row.get('reply_to_message_id')
            if not target_id:
                continue
            chunk = db.fetch_row_by_message(db_path, row['chat_id'], target_id)
            if chunk is None or chunk.get('speaker') != 'bot':
                continue  # реплика не на сообщение бота — не обратная связь
            turn_key = (row['chat_id'], chunk.get('reply_to_message_id'))
            if turn_key in seen_turns:
                continue
            seen_turns.add(turn_key)
            targets.append((row['id'], {'chat': row['chat_id'],
                                        'target': chunk.get('reply_to_message_id')}))

        if not targets:
            return '', False

        # Последние (по id реплики) ходы, не более self_context_turns
        targets.sort(key=lambda x: -x[0])
        turns = []
        for _, info in targets[:max_turns]:
            turn_rows = db.fetch_bot_turn(db_path, info['chat'], info['target'])
            if not turn_rows:
                continue
            chunk_ids = [t['message_id'] for t in turn_rows]
            replies = db.fetch_replies_to(db_path, info['chat'], chunk_ids)
            origin = db.fetch_row_by_message(db_path, info['chat'], info['target'])
            merged: dict[int, dict] = {}
            for r in turn_rows:
                merged[r['id']] = r
            for r in replies:
                merged[r['id']] = r
            if origin is not None:
                merged[origin['id']] = origin
            turns.append(sorted(merged.values(), key=lambda r: r['id']))

        block = ''
        used = 0
        # Более свежие ходы сверху; старые отбрасываем, если не влезают
        for turn in reversed(turns):
            rendered = _render_turn(turn)
            if not rendered:
                continue
            if used and used + len(rendered) > max_chars:
                continue
            if not used and len(rendered) > max_chars:
                continue
            block = rendered + ('\n' + block if block else '')
            used += len(rendered)
        return block, True

    # --- создание тематических страниц (п. 9.4) ---

    async def _maybe_create_topic(self, index_data: dict, pages_cfg: dict) -> bool:
        """Создаёт ОДНУ тематическую страницу по повторяемой теме окна.

        Возвращает True, если создано. Ошибки LLM/невалидные предложения не
        роняют инкремент — пишется cooldown, попытка повторится позже.
        """
        settings = config.settings()
        budgets_cfg = settings.get('budgets', {})
        hour_limit = budgets_cfg.get('max_llm_calls_per_hour', 12)
        page_max = pages_cfg.get('max_page_chars', 2400)
        max_count = pages_cfg.get('max_count', 30)
        cooldown_hours = pages_cfg.get('create_cooldown_hours', 24)
        max_cooldown = pages_cfg.get('max_cooldown_entries', 100)

        active_count = sum(1 for p in index_data.get('pages', [])
                           if p.get('status') == 'active')
        if active_count >= max_count:
            return False
        if not self.budget.can_llm(hour_limit):
            return False

        root = pageio.kb_root()
        db_path = config.db_path()
        watermark = index_data.get('watermark', 0)
        window = db.fetch_processed_human_window(
            db_path, watermark, pages_cfg.get('create_window_messages', 100))
        if not window:
            return False
        candidates = topics.detect_candidates(
            window, pages_cfg.get('create_repeats', 3))
        if not candidates:
            return False

        dirty = False
        removed = topics.prune_cooldowns(index_data, cooldown_hours, max_cooldown)
        dirty = dirty or removed > 0

        prompts_cfg = settings.get('prompts', {})
        pages = index_data.get('pages', [])
        for cand in candidates:
            token = cand['token']
            if topics.is_cooldown_active(index_data, token, cooldown_hours):
                continue
            if len(window) >= 20 and cand['count'] / len(window) > 0.6:
                continue
            overlap = topics.check_page_overlap(token, pages)
            if overlap is not None and overlap.get('status') == 'active':
                continue
            if overlap is not None:
                # Архивная тема возвращается — реактивация (этап 5) отложена.
                continue

            examples = self._candidate_examples(window, cand, token)
            prompt_text = prompts.build_create_knowledge_prompt(
                prompts_cfg.get('create_knowledge_prompt'), candidate=token,
                examples_block=examples, max_chars=page_max)
            if not self.budget.can_llm(hour_limit):
                break
            self.budget.consume_llm()
            raw = await self._llm_call(prompt_text)
            proposal = topics.parse_page_proposal(raw) if raw else None
            if proposal is None:
                topics.set_cooldown(index_data, token, _now_iso())
                dirty = True
                continue
            content, reason = _process_page_md(proposal['content'], page_max)
            if content is None:
                topics.set_cooldown(index_data, token, _now_iso())
                dirty = True
                continue
            proposal['content'] = content
            if topics.check_page_overlap(proposal['slug'], pages) is not None:
                continue
            if index_mod.find_page(index_data, proposal['slug']) is not None:
                continue
            if not pageio.write_page(proposal['slug'], content, root):
                logger.warning("bot_kb: не удалось записать новую страницу %s",
                               proposal['slug'])
                return False
            now = _now_iso()
            index_data['pages'].append({
                'slug': proposal['slug'],
                'title': proposal['title'],
                'kind': 'knowledge',
                'status': 'active',
                'keywords': proposal['keywords'],
                'aliases': proposal['aliases'],
                'created': now,
                'updated': now,
                'last_seen': now,
                'hits': 0,
                'quarantined': False,
            })
            dirty = True
            if index_mod.save_index(root, index_data):
                logger.info("bot_kb: создана страница %s (token=%s)", proposal['slug'], token)
                return True
            return False

        if dirty:
            index_mod.save_index(root, index_data)
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


# --- helpers рендера блоков ---

def _render_block(rows: list[dict]) -> str:
    """Строки с нейтральными метками [N] (без метаданных авторов, п. 10)."""
    lines = []
    for i, row in enumerate(rows, 1):
        content = (row.get('content') or '').strip()
        if not content:
            continue
        content_type = row.get('content_type') or 'text'
        label = f"[{i}][{content_type}]" if content_type != 'text' else f"[{i}]"
        lines.append(f"{label} {content}")
    return "\n".join(lines)


def _render_turn(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        content = (row.get('content') or '').strip()
        if not content:
            continue
        speaker = row.get('speaker') or 'human'
        lines.append(f"[{row['id']}][{speaker}] {content}")
    return "\n".join(lines)
