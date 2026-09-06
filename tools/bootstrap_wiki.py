"""Локальный построитель wiki из существующей БД (без запуска Telegram-бота).

Для каждого пользователя из user_raw запускает backfill: если wiki ещё нет —
bootstrap (Home.md/Style.md/_index.yaml, watermark=0), затем докатывает ВСЕ
необработанные строки снимками через LLM (тот же путь, что и в боте). Прогон
идемпотентен и возобновляем: обработанные строки «закрываются» watermark'ом,
повторный запуск продолжит с того же места.

Требуется config.yaml (config.py copy sample_config.yaml) с настроенным ai
(провайдер + API-ключ). Расход токенов зависит от объёма user_raw.

Примеры:
  python tools/bootstrap_wiki.py --dry-run            # план без LLM
  python tools/bootstrap_wiki.py --top 1 --yes        # самый «объёмный» пользователь
  python tools/bootstrap_wiki.py --user 406526542 --yes
  python tools/bootstrap_wiki.py --yes                # все пользователи (дорого)
"""

import argparse
import asyncio
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

import botwiki
from botwiki import db, index as index_mod, manager


def load_top_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    if os.path.isfile('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def per_user_stats(db_path: str) -> list[tuple[int, int]]:
    """(user_id, суммарные символы содержимого) по убыванию."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT user_id, COUNT(*) AS n, COALESCE(SUM(LENGTH(content)),0) AS ch "
            "FROM user_raw GROUP BY user_id ORDER BY ch DESC").fetchall()
        return [(r[0], r[2]) for r in rows]
    finally:
        conn.close()


def check_ai_configured(top: dict) -> str | None:
    ai = top.get('ai', {})
    provider = ai.get('provider', 'deepseek')
    if provider == 'deepseek' and not ai.get('deepseek_api_key'):
        return 'В config.yaml не задан ai.deepseek_api_key'
    if provider == 'gigachat' and not ai.get('gigachat_api_key'):
        return 'В config.yaml не задан ai.gigachat_api_key'
    if provider == 'yandexgpt' and not ai.get('api_key'):
        return 'В config.yaml не задан ai.api_key'
    if provider not in ('deepseek', 'gigachat', 'yandexgpt') and not ai.get('llama_api_base'):
        return f'Провайдер {provider} требует ai.llama_api_base (Ollama)'
    return None


def setup_llm(top: dict):
    """Внедряет LLM-caller в менеджер (как bot._setup_wiki_manager).

    Для deepseek использует НЕ reasoning-модель (wiki.llm_model), иначе
    deepseek-reasoner тратит max_tokens на reasoning и возвращает пустой content.
    """
    # call_llm_raw / _call_openai_text живут в bot.py; импорт не запускает main().
    from bot import call_llm_raw, _call_openai_text

    ai_config = top.get('ai', {})
    provider = ai_config.get('provider', 'deepseek')
    dossier_cfg = top.get('dossier', {})
    temperature = dossier_cfg.get('temperature', 0.1)

    async def _call(prompt: str):
        if provider == 'deepseek':
            model = botwiki.settings().get('llm_model') or 'deepseek-chat'
            return await _call_openai_text(ai_config.get('deepseek_api_key'),
                                           prompt, temperature, model=model)
        return await call_llm_raw(ai_config, [{"role": "user", "content": prompt}],
                                  provider, temperature=temperature)
    return _call


async def build_for_user(wiki_manager: manager.WikiManager, user_id: int,
                         db_path: str) -> str:
    user_dir = os.path.join(botwiki.config.wiki_dir(), str(user_id))
    idx, _ = index_mod.ensure_index(user_dir, db_path, user_id)
    watermark = idx.get('watermark', 0) if idx else 0
    if idx is not None and db.count_unprocessed(db_path, user_id, watermark) == 0:
        return f"user {user_id}: wiki уже готова, необработанных строк нет — пропуск"

    total = db.count_rows(db_path, user_id)
    ok = await wiki_manager.backfill_user(user_id)
    idx2, _ = index_mod.ensure_index(user_dir, db_path, user_id)
    left = (db.count_unprocessed(db_path, user_id, idx2.get('watermark', 0))
            if idx2 else total)
    state = 'OK' if ok else 'СБОЙ (см. last_error в _index.yaml)'
    return f"user {user_id}: {state} | обработано_в_этом_запуске={total} остаток={left}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='путь к config.yaml')
    ap.add_argument('--user', type=int, action='append', default=[],
                    help='собирать только этих пользователей (можно несколько)')
    ap.add_argument('--top', type=int, default=None,
                    help='только N пользователей с наибольшим объёмом raw')
    ap.add_argument('--min-chars', type=int, default=0,
                    help='пропускать пользователей с объёмом меньше N символов')
    ap.add_argument('--dry-run', action='store_true', help='показать план и выйти')
    ap.add_argument('--yes', action='store_true', help='не спрашивать подтверждение')
    args = ap.parse_args()

    top = load_top_config(args.config)
    botwiki.configure(top)
    db_path = botwiki.config.db_path()
    botwiki.db.init_raw_table(db_path)

    err = check_ai_configured(top)
    if err and not args.dry_run:
        ap.error(err)

    stats = per_user_stats(db_path)
    if not stats:
        ap.error('В user_raw нет строк — импортируйте сначала (tools/import_raw.py).')

    wanted = []
    if args.user:
        ids = set(args.user)
        wanted = [(u, ch) for u, ch in stats if u in ids]
    else:
        wanted = list(stats)
    if args.min_chars:
        wanted = [(u, ch) for u, ch in wanted if ch >= args.min_chars]
    if args.top is not None and not args.user:
        wanted = wanted[:args.top]

    total_chars = sum(ch for _, ch in wanted)
    print(f"Пользователей в плане: {len(wanted)}; суммарно символов raw: {total_chars:,}")
    print(f"Ориентировочно входных токенов: ~{total_chars // 3:,} (при ~3 симв/токен)")
    for u, ch in wanted:
        print(f"  user {u}: {ch:,} симв.")

    if args.dry_run:
        return

    if not args.yes:
        answer = input("Запустить сборку? [y/N] ").strip().lower()
        if answer != 'y':
            print('Отменено.')
            return

    mgr = manager.WikiManager()
    mgr.set_llm_caller(setup_llm(top))

    async def _run():
        for user_id, _ch in wanted:
            try:
                msg = await build_for_user(mgr, user_id, db_path)
                print(msg)
            except Exception as e:
                print(f"user {user_id}: ИСКЛЮЧЕНИЕ {e}")
    asyncio.run(_run())
    print('Готово. Смотрите wiki/<user_id>/ и /wiki_status после запуска бота.')


if __name__ == '__main__':
    main()
