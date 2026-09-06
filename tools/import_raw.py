"""Офлайн-импорт экспорта Telegram в user_raw (без запуска бота).

Заполняет ТОЛЬКО таблицу user_raw (source='export'); wiki не строит
(backfill выполняется ботом или /reconcile_wiki). Работает с официальным
JSON-экспортом: chats.list[] с from_id='user<id>'.

Примеры:
  python tools/import_raw.py telegram_data_to_process.json --list
  python tools/import_raw.py telegram_data_to_process.json --chat 1832622632 --map 1832622632=-100123456789 --dry-run
  python tools/import_raw.py telegram_data_to_process.json --chat 1832622632 --map 1832622632=-100123456789:42 --max-file-mb 4000

--chat принимает имя чата или export_id (id из экспорта). Без маппинга в
config.yaml (wiki.import.chat_map) или --map импорт запрещён.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

import botwiki
from botwiki import export_import


def load_top_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    if os.path.isfile('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_map(spec: str):
    """--map 'export_id|name=chat_id[:thread_id]' → dict(export_id|name, chat_id, thread_id)."""
    key, _, value = spec.partition('=')
    if not key or not value:
        raise argparse.ArgumentTypeError(f"--map должен быть вида key=chat_id[:thread_id]: {spec!r}")
    chat_part, _, thread_part = value.partition(':')
    entry = {'chat_id': int(chat_part)}
    if thread_part:
        entry['thread_id'] = int(thread_part)
    if key.lstrip('-').isdigit():
        entry['export_id'] = int(key)
    else:
        entry['name'] = key
    return entry


def merge_chat_map(top: dict, extra: list):
    wiki = top.setdefault('wiki', {})
    imp = wiki.setdefault('import', {})
    chat_map = imp.setdefault('chat_map', [])
    existing_ids = {e.get('export_id') for e in chat_map if e.get('export_id') is not None}
    existing_names = {e.get('name') for e in chat_map if e.get('name')}
    for entry in extra:
        if entry.get('export_id') in existing_ids or entry.get('name') in existing_names:
            continue
        chat_map.append(entry)
        if entry.get('export_id') is not None:
            existing_ids.add(entry['export_id'])
        if entry.get('name') is not None:
            existing_names.add(entry['name'])
    imp['chat_map'] = chat_map


def list_chats(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        export = json.load(f)
    chats = (export or {}).get('chats', {}).get('list', [])
    if not chats:
        print('В экспорте нет чатов.')
        return
    print('Доступные чаты (name | export_id | messages):')
    for c in chats:
        count = len(c.get('messages', []))
        print(f"  {c.get('name')!r} | {c.get('id')} | {count} сообщений")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', help='путь к JSON-экспорту (относительно allowed-dir или абсолютный)')
    ap.add_argument('--chat', help='имя чата или export_id из экспорта')
    ap.add_argument('--list', action='store_true', help='показать чаты в файле и выйти')
    ap.add_argument('--map', action='append', type=parse_map, default=[],
                    help='маппинг key=chat_id[:thread_id] (key: export_id или имя)')
    ap.add_argument('--allowed-dir', default=None, help='override import.allowed_dir')
    ap.add_argument('--max-file-mb', type=int, default=None, help='override import.max_file_mb')
    ap.add_argument('--db', default=None, help='override пути к SQLite (default из config.yaml)')
    ap.add_argument('--config', default=None, help='путь к config.yaml')
    ap.add_argument('--dry-run', action='store_true', help='парсинг и статистика без записи')
    args = ap.parse_args()

    top = load_top_config(args.config)
    if args.db:
        top['db'] = args.db
    merge_chat_map(top, args.map)
    wiki = top.setdefault('wiki', {})
    imp = wiki.setdefault('import', {})
    if args.allowed_dir is not None:
        imp['allowed_dir'] = args.allowed_dir
    if args.max_file_mb is not None:
        imp['max_file_mb'] = args.max_file_mb

    botwiki.configure(top)
    allowed_dir = botwiki.config.settings()['import']['allowed_dir']
    db_path = botwiki.config.db_path()
    botwiki.db.init_raw_table(db_path)  # гарантируем схему user_raw (без запуска бота)

    if args.list:
        real, err = export_import.resolve_allowed_path(args.file, allowed_dir)
        if err:
            print(f'Путь отклонён: {err}')
            sys.exit(1)
        list_chats(real)
        return

    if not args.chat:
        ap.error('Укажите --chat (или --list для просмотра чатов)')
    chat_key = args.chat if not args.chat.lstrip('-').isdigit() else int(args.chat)

    real, err = export_import.resolve_allowed_path(args.file, allowed_dir)
    if err:
        print(f'Путь отклонён: {err}')
        sys.exit(1)

    stats = export_import.perform_import(db_path, real, chat_key, dry_run=args.dry_run)
    print(export_import._summary_text(stats))
    if not args.dry_run and not stats.error and stats.users:
        print('Строки записаны. Для построения wiki запустите бота (backfill) '
              'или /reconcile_wiki для затронутых пользователей.')
    if stats.error:
        sys.exit(2)


if __name__ == '__main__':
    main()
