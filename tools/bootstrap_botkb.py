"""Офлайн-сборка/обновление БЗ бота (bot_kb) из bot_kb_raw (этап 1–6, ТЗ п. 9.7).

Запускает ленивый bootstrap (Home.md/Style.md/_index.yaml в wiki/bot_kb/), затем
инкременты по ВСЕМ необработанным строкам bot_kb_raw (снимками) и, опционально,
reconcile. Режим/history/bootstrap.mode берутся из config.yaml (bot_kb:), но
могут быть переопределены флагами. Бюджеты в процессе инструмента поднимаются,
чтобы разобрать весь хвост за один прогон (в рантайм-конфиг не пишется).

Требуется настроенный ai в config.yaml (deepseek-ключ и т.п.).

Примеры:
  python tools/bootstrap_botkb.py --dry-run          # оценка без LLM
  python tools/bootstrap_botkb.py --yes              # bootstrap + полный drain
  python tools/bootstrap_botkb.py --yes --mode limited_window --history backlog --backlog 2000
  python tools/bootstrap_botkb.py --yes --reconcile  # + сверка страниц окном raw
"""

import argparse
import asyncio
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HTTP-обёртки bot.py (requests + прокси) — импортируем ДО переопределения
# конфига инструментом: импорт bot.py сам вызывает configure(config.yaml),
# а main() ниже вызовет botkb.configure(work) с рабочими настройками.
from bot import call_llm_raw, make_async_request  # noqa: E402

import yaml

import botkb
from botkb import config as kbconfig
from botkb import db
from botkb import index as index_mod
from botkb import manager

BIG = 10 ** 9          # бюджет инструмента (не лимит рантайма)
TOPIC_ATTEMPTS = 12    # сколько раз дать менеджеру создать тему подряд


def load_top_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    if os.path.isfile('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def check_ai_configured(top: dict) -> str | None:
    ai = top.get('ai', {})
    provider = ai.get('provider', 'deepseek')
    if provider == 'deepseek' and not ai.get('deepseek_api_key'):
        return 'в config.yaml не задан ai.deepseek_api_key'
    if provider == 'gigachat' and not ai.get('gigachat_api_key'):
        return 'в config.yaml не задан ai.gigachat_api_key'
    if provider == 'yandexgpt' and not ai.get('api_key'):
        return 'в config.yaml не задан ai.api_key'
    if provider not in ('deepseek', 'gigachat', 'yandexgpt') and not ai.get('llama_api_base'):
        return f"для провайдера {provider} задайте ai.llama_api_base (Ollama)"
    return None


def setup_llm(top: dict):
    """Внедряет LLM-caller (как bot.py._setup_botkb_manager, без импорта bot.py)."""
    ai_config = top.get('ai', {})
    provider = ai_config.get('provider', 'deepseek')
    temperature = (top.get('dossier') or {}).get('temperature', 0.1)

    async def _call(prompt: str):
        if provider == 'deepseek':
            # для deepseek — НЕ reasoning-модель (bot_kb.llm_model)
            model = kbconfig.settings().get('llm_model') or 'deepseek-chat'
            return await _call_deepseek_text(ai_config.get('deepseek_api_key'),
                                             prompt, temperature, model=model)
        return await _call_openai_compat(ai_config, prompt, provider, temperature)
    return _call


async def _call_deepseek_text(api_key, prompt, temperature, model: str) -> str:
    if not api_key:
        return 'Ошибка: API ключ для deepseek не настроен'
    url = 'https://api.deepseek.com/chat/completions'
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': 8000,
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    try:
        resp = await make_async_request(url, headers, data)
        if resp.status_code == 200:
            return (resp.json()['choices'][0]['message'].get('content') or '').strip()
        return f'Ошибка DeepSeek API: {resp.status_code} - {resp.text}'
    except Exception as e:
        return f'Ошибка при запросе к DeepSeek: {e}'


async def _call_openai_compat(ai_config, prompt, provider, temperature):
    return await call_llm_raw(ai_config, [{'role': 'user', 'content': prompt}],
                              provider, temperature=temperature)


def raw_stats(db_path: str) -> dict:
    return {
        'total': db.count_rows(db_path),
        'human': db.count_rows(db_path, speaker='human'),
        'bot': db.count_rows(db_path, speaker='bot'),
    }


def index_summary(root: str, db_path: str) -> tuple[str | None, dict]:
    index_data, status = index_mod.ensure_index(root, db_path)
    if index_data is None:
        return None, {'status': status}
    pages = index_data.get('pages', [])
    return index_data, {
        'status': status,
        'watermark': index_data.get('watermark', 0),
        'message_count': index_data.get('message_count', 0),
        'pages_active': sum(1 for p in pages if p.get('status') == 'active'),
        'pages_knowledge': sum(1 for p in pages
                               if p.get('status') == 'active' and p.get('kind') == 'knowledge'),
        'unprocessed': db.count_unprocessed(db_path, index_data.get('watermark', 0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='путь к config.yaml')
    ap.add_argument('--mode', default=None,
                    choices=['shadow', 'primary'],
                    help='переопределить bot_kb.mode на время прогона')
    ap.add_argument('--bootstrap-mode', default=None,
                    choices=['from_system_prompt', 'limited_window', 'empty'])
    ap.add_argument('--history', default=None, choices=['discard', 'backlog'])
    ap.add_argument('--backlog', type=int, default=None,
                    help='max_history_messages при --history backlog')
    ap.add_argument('--max-messages', type=int, default=None,
                    help='max_raw_messages_per_update (снимок; крупнее — меньше LLM-вызовов)')
    ap.add_argument('--max-chars', type=int, default=None,
                    help='max_raw_chars_per_update (снимок)')
    ap.add_argument('--reconcile', action='store_true',
                    help='после drain выполнить reconcile (сверка страниц с окном)')
    ap.add_argument('--dry-run', action='store_true', help='оценка без LLM')
    ap.add_argument('--yes', action='store_true', help='без подтверждения')
    args = ap.parse_args()

    top = load_top_config(args.config)
    if not top.get('bot_kb', {}).get('enabled', True):
        ap.error('bot_kb.enabled=false в конфиге — включите или удалите флаг.')

    # Рабочая копия с инструментальными переопределениями (в файл не пишется)
    work = copy.deepcopy(top)
    bk = work.setdefault('bot_kb', {})
    bk.setdefault('budgets', {}).update(max_updates_per_day=BIG,
                                        max_llm_calls_per_hour=BIG,
                                        reconcile_llm_calls_per_day=BIG)
    if args.mode:
        bk['mode'] = args.mode
    if args.bootstrap_mode:
        bk.setdefault('bootstrap', {})['mode'] = args.bootstrap_mode
    if args.history:
        bk.setdefault('bootstrap', {})['history'] = args.history
    if args.backlog is not None:
        bk.setdefault('bootstrap', {})['max_history_messages'] = args.backlog
    if args.max_messages:
        bk.setdefault('update', {})['max_raw_messages_per_update'] = args.max_messages
    if args.max_chars:
        bk.setdefault('update', {})['max_raw_chars_per_update'] = args.max_chars

    try:
        botkb.configure(work)
    except Exception as e:
        ap.error(f'Некорректная конфигурация bot_kb: {e}')

    if kbconfig.mode() not in ('shadow', 'primary'):
        ap.error('bot_kb.mode должен быть shadow/primary для обновления страниц '
                 '(передайте --mode).')

    db_path = kbconfig.db_path()
    db.init_raw_table(db_path)
    root = kbconfig.kb_dir()
    stats = raw_stats(db_path)
    if stats['total'] == 0:
        ap.error('bot_kb_raw пуст — сначала накопите сырьё (capture_only) или импортируйте.')

    err = check_ai_configured(top)
    if err and not args.dry_run:
        ap.error(err)

    print(f'bot_kb: корень={root}')
    print(f'raw: всего={stats["total"]} (human={stats["human"]}, bot={stats["bot"]})')
    print(f'оценка токенов: ~{stats["total"] * 40:,} симв. (до ~{stats["total"] * 40 // 3:,} ток.)')

    if args.dry_run:
        return

    if not args.yes:
        answer = input('Выполнить сборку? [y/N] ').strip().lower()
        if answer != 'y':
            print('Отменено.')
            return

    mgr = manager.KBManager()
    mgr.set_llm_caller(setup_llm(work))

    async def _run():
        prev_topics = -1
        for attempt in range(TOPIC_ATTEMPTS):
            await mgr.run_updates()
            index_data = index_mod.ensure_index(root, db_path)[0]
            if index_data is None:
                break
            left = db.count_unprocessed(db_path, index_data.get('watermark', 0))
            if left > 0:
                continue  # бюджет инструмента поднят; здесь быть не должно
            topics_now = sum(1 for p in index_data.get('pages', [])
                             if p.get('status') == 'active' and p.get('kind') == 'knowledge')
            if topics_now == prev_topics:
                break
            prev_topics = topics_now
        if args.reconcile:
            idx2 = index_mod.ensure_index(root, db_path)[0]
            if idx2 is not None:
                res = await mgr.reconcile(manual=True, reason='offline')
                print('reconcile:', res.get('message'))

    asyncio.run(_run())

    index_data, summary = index_summary(root, db_path)
    if index_data is None:
        print('Индекс не создан (см. логи выше).')
        return
    print(f'Готово: status={summary["status"]} watermark={summary["watermark"]} '
          f'active={summary["pages_active"]} (knowledge={summary["pages_knowledge"]}) '
          f'необработанных осталось={summary["unprocessed"]}')
    print('Проверка: /kb_status, /kb_show (или wiki/bot_kb/ на диске).')


if __name__ == '__main__':
    main()
