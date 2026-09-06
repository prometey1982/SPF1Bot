# AGENTS.md

## Project

Single-file Telegram bot (`bot.py`, ~850 lines) that proxies chat messages to LLM providers (DeepSeek, GigaChat, YandexGPT, local Llama/Ollama) with per-chat context, per-user "dossiers" (facts tracked by the LLM), and @mention-quote collection. UI and comments are in Russian.

## Setup

1. Copy `sample_config.yaml` to `config.yaml` and fill in secrets (`bot_token`, API keys, `allowed_group_chat_ids`, `allowed_private_users`).
2. Install dependencies: `pip install pyyaml requests python-telegram-bot[socks]` (the `[socks]` extra is needed for SOCKS5 proxies).

## Run

```
python bot.py
```

No tests, linter, or CI exist. The only verification is running the bot and sending Telegram messages.

## Architecture (all in bot.py)

- **Config** — global `config` dict loaded from `config.yaml` at import time by `load_config()` (bot.py:182). Reloaded at runtime via `/reload_config` without restart. Single source of truth: `sample_config.yaml`.
- **Context** — global `chat_context` (`ChatContext`, bot.py:126), in-memory per-`chat_id` rolling window (default 15 messages, 24h TTL, lost on restart). `/clear_context` and `/show_context` (private only) manage it.
- **DB** — SQLite (`db: bot.db`, created by `init_db()` at import). Two tables: `USER_INFO(id, dossier)` for dossiers and `user_mentions(author_id, target_username, chat_id, quote, timestamp)` for mentions. Raw SQL via a fresh connection per call; no ORM.
- **Dossier manager** — global `dossier_manager` (`DossierManager`, bot.py:344). Fire-and-forget async queue, one worker per user; batches queued messages, appends mentions about that user, asks the LLM (low temperature, own prompt/retry config) to rewrite the dossier, then saves it.
- **Mentions** — `@username` in messages ≥ `min_length` chars are extracted (excluding the bot) and stored as quotes (trimmed to 200 chars) with a per-target cap (`max_quotes`) and TTL (`ttl_hours`). `cleanup_mentions()` runs at startup.
- **LLM dispatch** — `PROVIDER_CONFIGS` (bot.py:562) defines OpenAI-compatible providers (deepseek = default, `deepseek-reasoner` model; gigachat; yandexgpt, which needs `folder_id` and a custom `build_data`). Any other `provider` value falls back to local Llama via Ollama's `/api/chat`. All HTTP is synchronous `requests.post` wrapped with `run_in_executor` in `make_async_request` (bot.py:448); errors are returned as strings starting with `"Ошибка"`.
- **Message handling** — a single `handle_message` (bot.py:626) handles both groups and private chats (no separate group handlers anymore). In groups it triggers only when the bot is mentioned, the message replies to the bot, or the author is in `always_respond_to_users`. Replies to quoted messages are enriched with the quote content before hitting the LLM. Long replies are split by `send_long_message` into ≤4096-char chunks (Markdown; retries without `parse_mode` if formatting fails).

## Commands

`/clear_context`, `/show_context`, `/clear_dossier <user_id>`, `/clear_dossiers` (private chats) and `/reload_config`. Admin gate: `is_admin` = username in `allowed_private_users` (bot.py:443).

## Key facts & gotchas

- `config.yaml` is gitignored; never commit secrets. `bot.db` (runtime user data) and `telegram_data.json` are **not** gitignored — do not commit them.
- Global mutable state (`config`, `chat_context`, `dossier_manager`) is initialized at module import; DB init runs at import too.
- In group chats access is checked against `message_thread_id`, so `allowed_group_chat_ids` must list forum-topic thread IDs, not plain chat IDs.
- `pyproject.toml` is vestigial: named `protobuf_experiments`, no `src/` layout — it does not match the repo (single-file bot at the root). Used only as a dependency/manifest reference.
- `docs/` holds design notes for features (user dossier, mentions, proxy) and `refactoring_plan.md`, a roadmap for further cleanup — code has drifted past parts of it.
