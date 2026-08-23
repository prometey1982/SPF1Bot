import yaml
import random
import re
import requests
import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from telegram.request import BaseRequest, HTTPXRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProxyRotatingRequest(BaseRequest):
    """BaseRequest с автоматическим переключением прокси при ошибке соединения."""

    def __init__(self, proxy_list: list[str], **kwargs):
        self._proxies = proxy_list
        self._current_index = 0
        self._kwargs = kwargs
        self._requests = [HTTPXRequest(proxy=p, **kwargs) for p in proxy_list]

    @property
    def proxy(self) -> str:
        return self._proxies[self._current_index]

    async def do_request(self, url, method, request_data=None, **kwargs):
        last_error = None
        for _ in range(len(self._proxies)):
            req = self._requests[self._current_index]
            try:
                return await req.do_request(url, method, request_data, **kwargs)
            except Exception as e:
                logger.warning("Прокси %s недоступен: %s", self._proxies[self._current_index], e)
                self._current_index = (self._current_index + 1) % len(self._proxies)
        raise last_error

    async def initialize(self):
        for req in self._requests:
            await req.initialize()

    async def shutdown(self):
        for req in self._requests:
            await req.shutdown()

    @property
    def read_timeout(self):
        return self._requests[self._current_index].read_timeout

    @property
    def write_timeout(self):
        return self._requests[self._current_index].write_timeout

    @property
    def connect_timeout(self):
        return self._requests[self._current_index].connect_timeout

    @property
    def pool_timeout(self):
        return self._requests[self._current_index].pool_timeout

    @property
    def http_version(self):
        return self._requests[self._current_index].http_version


def split_text(text: str, max_length: int = 4096) -> list[str]:
    """
    Разбивает текст на части длиной не более max_length,
    стараясь не разрывать слова.
    """
    if not text:
        return []

    parts = []
    while len(text) > max_length:
        # Ищем место для разреза: сначала по переносу строки, потом по пробелу
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, max_length)
        if split_pos == -1:
            # Если ни переноса, ни пробела нет — приходится резать по лимиту
            split_pos = max_length

        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip()  # убираем начальные пробелы/переносы

    parts.append(text)
    return parts

async def send_long_message(update, message_text, parse_mode='Markdown'):
    """
    Асинхронная отправка длинного сообщения с учетом ограничений:
    - длина одного сообщения не может превышать 4 кб
    - между сообщениями должно пройти не меньше 1 секунды
    """
    # Максимальная длина сообщения в байтах
    MAX_MESSAGE_LENGTH = 4096

    parts = split_text(message_text, MAX_MESSAGE_LENGTH)
    
    # Отправляем все части с задержкой
    for i, part in enumerate(parts):
        # Проверяем, есть ли message_thread_id (для супергрупп и тем обсуждений)
        message_thread_id = getattr(update.message, 'message_thread_id', None)
        try:
            if message_thread_id:
                await update.message.reply_text(part, parse_mode=parse_mode, message_thread_id=message_thread_id)
            else:
                await update.message.reply_text(part, parse_mode=parse_mode)
        except Exception:
            if message_thread_id:
                await update.message.reply_text(part, message_thread_id=message_thread_id)
            else:
                await update.message.reply_text(part)

        # Не делаем задержку после последнего сообщения
        if i < len(parts) - 1:
            await asyncio.sleep(0.05)


# Хранилище контекста (в памяти)
class ChatContext:
    def __init__(self, max_context_length=10, ttl_hours=24):
        self.contexts = {}  # {chat_id: [{"role": str, "content": str, "timestamp": datetime}]}
        self.max_context_length = max_context_length
        self.ttl = timedelta(hours=ttl_hours)

    def add_message(self, chat_id, role, content):
        if chat_id not in self.contexts:
            self.contexts[chat_id] = []

        # Очищаем старые сообщения
        self._clean_old_messages(chat_id)

        # Добавляем новое сообщение
        self.contexts[chat_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # Ограничиваем длину контекста
        if len(self.contexts[chat_id]) > self.max_context_length:
            self.contexts[chat_id] = self.contexts[chat_id][-self.max_context_length:]

    def get_context(self, chat_id, include_system=True):
        """Возвращает историю сообщений для чата"""
        if chat_id not in self.contexts:
            return []

        self._clean_old_messages(chat_id)

        context = self.contexts[chat_id].copy()

        # Фильтруем системные сообщения если нужно
        if not include_system:
            context = [msg for msg in context if msg["role"] != "system"]

        return context

    def clear_context(self, chat_id):
        """Очищает контекст для чата"""
        if chat_id in self.contexts:
            del self.contexts[chat_id]

    def _clean_old_messages(self, chat_id):
        """Удаляет сообщения старше TTL"""
        if chat_id not in self.contexts:
            return

        now = datetime.now()
        self.contexts[chat_id] = [
            msg for msg in self.contexts[chat_id]
            if now - msg["timestamp"] <= self.ttl
        ]


def load_config():
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Файл config.yaml не найден!")
        return {}


# Глобальный объект контекста
chat_context = ChatContext(max_context_length=15, ttl_hours=24)

config = load_config()


# --- Database ---

def init_db():
    db_path = config.get('db', 'bot.db')
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS USER_INFO (
            id INTEGER PRIMARY KEY,
            dossier TEXT
        )
    ''')
    conn.commit()
    conn.close()


def _get_conn():
    return sqlite3.connect(config.get('db', 'bot.db'))


def get_dossier(user_id: int) -> str | None:
    conn = _get_conn()
    try:
        row = conn.execute('SELECT dossier FROM USER_INFO WHERE id = ?', (user_id,)).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def save_dossier(user_id: int, dossier: str):
    conn = _get_conn()
    try:
        conn.execute(
            'INSERT INTO USER_INFO (id, dossier) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET dossier = excluded.dossier',
            (user_id, dossier)
        )
        conn.commit()
    finally:
        conn.close()


def clear_dossier(user_id: int):
    conn = _get_conn()
    try:
        conn.execute('DELETE FROM USER_INFO WHERE id = ?', (user_id,))
        conn.commit()
    finally:
        conn.close()


def clear_all_dossiers():
    conn = _get_conn()
    try:
        conn.execute('DELETE FROM USER_INFO')
        conn.commit()
    finally:
        conn.close()


# --- Dossier Manager ---

class DossierManager:
    def __init__(self):
        self._queues: dict[int, asyncio.Queue] = {}
        self._processing: set[int] = set()

    def enqueue(self, user_id: int, message: str):
        logger.info("Запуск обновления досье для user_id=%d", user_id)
        if user_id not in self._queues:
            self._queues[user_id] = asyncio.Queue()
        self._queues[user_id].put_nowait(message)
        if user_id not in self._processing:
            asyncio.create_task(self._process_queue(user_id))

    async def _process_queue(self, user_id: int):
        self._processing.add(user_id)
        try:
            while not self._queues[user_id].empty():
                messages = []
                while not self._queues[user_id].empty():
                    messages.append(self._queues[user_id].get_nowait())

                current_dossier = get_dossier(user_id) or ""
                dossier_config = config.get('dossier', {})
                prompt_template = dossier_config.get('update_prompt', '')
                combined_message = "\n".join(messages)
                prompt = prompt_template.replace('{dossier}', current_dossier).replace('{message}', combined_message)

                new_dossier = await self._call_with_retries(prompt, dossier_config)
                if new_dossier:
                    save_dossier(user_id, new_dossier)
                    logger.info("Досье обновлено для user_id=%d", user_id)
        except Exception as e:
            logger.warning("Ошибка обновления досье для user_id=%d: %s", user_id, e)
        finally:
            self._processing.discard(user_id)

    async def _call_with_retries(self, prompt: str, dossier_config: dict) -> str | None:
        retry_count = dossier_config.get('update_retry_count', 3)
        backoff = dossier_config.get('backoff', 5)
        max_backoff = dossier_config.get('max_backoff', 60)
        temperature = dossier_config.get('temperature', 0.1)

        ai_config = config.get('ai', {})
        provider = ai_config.get('provider', 'deepseek')

        for attempt in range(retry_count):
            try:
                messages = [{"role": "user", "content": prompt}]
                result = await call_llm_raw(ai_config, messages, provider, temperature=temperature)
                if result and not result.startswith("Ошибка"):
                    return result
                logger.warning("Попытка %d/%d обновления досье: LLM вернул ошибку: %s", attempt + 1, retry_count, result)
            except Exception as e:
                logger.warning("Попытка %d/%d обновления досье не удалась: %s", attempt + 1, retry_count, e)

            if attempt < retry_count - 1:
                delay = min(backoff * (2 ** attempt), max_backoff)
                await asyncio.sleep(delay)

        logger.warning("Не удалось обновить досье после %d попыток", retry_count)
        return None


dossier_manager = DossierManager()

init_db()


def is_bot_mentioned(text, bot_username):
    """Проверяет, упомянут ли бот в тексте"""
    if not text:
        return False

    # Регулярное выражение для поиска упоминаний
    # Ищет @username или просто username как отдельное слово
    pattern = r'(?:^|\s)(@?' + re.escape(bot_username) + r')(?:\s|$|[,!?.])'
    return bool(re.search(pattern, text, re.IGNORECASE))


def is_admin(user) -> bool:
    """Проверяет, есть ли у пользователя доступ к административным командам"""
    return user is not None and user.username in config.get('allowed_private_users', [])


async def make_async_request(url, headers, data):
    """Асинхронно выполняет HTTP запрос"""
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=data, timeout=120)
        )
        return response
    except Exception as e:
        raise e


async def call_llm_raw(ai_config: dict, messages: list, provider: str, temperature: float = None) -> str:
    """Вызывает LLM напрямую с переданными сообщениями."""
    if provider in PROVIDER_CONFIGS:
        return await get_openai_compatible_response(ai_config, messages, provider, temperature=temperature)
    else:
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        return await get_llama_response(ai_config, user_msg, temperature=temperature)


async def get_ai_response_with_context(message_text, bot_username, chat_id, user_name="", user_id=None):
    """Получает ответ от нейросети с учетом контекста"""
    ai_config = config.get('ai', {})
    provider = ai_config.get('provider', 'deepseek')

    # Очищаем сообщение от упоминания бота
    if bot_username:
        message_text = re.sub(f'@?{re.escape(bot_username)}', '', message_text, flags=re.IGNORECASE)
    message_text = message_text.strip()

    # Добавляем текущее сообщение пользователя в контекст
    user_message = f"{user_name}: {message_text}" if user_name else message_text
    chat_context.add_message(chat_id, "user", user_message)

    # Получаем историю диалога
    context_messages = chat_context.get_context(chat_id)

    # Инжектим досье в контекст, если есть
    if user_id and config.get('use_ai', False):
        dossier = get_dossier(user_id)
        if dossier:
            dossier_prefix = "Ниже перечислен набор фактов о пользователе:"
            dossier_msg = {"role": "system", "content": f"{dossier_prefix}\n{dossier}"}
            context_messages = [dossier_msg] + context_messages

    # Запускаем обновление досье (fire-and-forget)
    if user_id and config.get('use_ai', False):
        dossier_manager.enqueue(user_id, message_text)

    # Формируем messages для LLM
    system_prompt = ai_config.get('system_prompt', 'Ты полезный ассистент. Отвечай на русском.')
    messages = [{"role": "system", "content": system_prompt}]

    if provider in PROVIDER_CONFIGS:
        # Современные API — передаём историю сообщений
        for msg in context_messages[-15:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return await call_llm_raw(ai_config, messages, provider)
    else:
        # Legacy API (Llama) — собираем контекст в один текст
        context_text = ""
        for msg in context_messages[-5:]:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            context_text += f"{role}: {msg['content']}\n"
        full_prompt = f"Контекст диалога:\n{context_text}\nТекущее сообщение: {message_text}\nОтвет:"
        return await call_llm_raw(ai_config, [{"role": "user", "content": full_prompt}], provider)


async def get_llama_response(ai_config, prompt, temperature=None):
    """Llama API с поддержкой локальных моделей"""
    try:
        api_base = ai_config.get('llama_api_base', 'http://localhost:11434')
        model = ai_config.get('llama_model', 'llama2')
        url = f"{api_base}/api/chat" if api_base.endswith('/api/chat') else f"{api_base}/api/chat"

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else ai_config.get('temperature', config.get('temperature', 0.7)),
                "num_predict": config.get('max_tokens', 1000)
            }
        }

        response = await make_async_request(url, {"Content-Type": "application/json"}, data)

        if response.status_code == 200:
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
            elif 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            elif 'response' in result:
                return result['response']
            else:
                return "Llama API вернул неожиданный формат ответа"
        else:
            return f"Ошибка Llama API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка при запросе к Llama: {str(e)}"


PROVIDER_CONFIGS = {
    'deepseek': {
        'url': 'https://api.deepseek.com/chat/completions',
        'headers': lambda c: {"Authorization": f"Bearer {c.get('deepseek_api_key')}", "Content-Type": "application/json"},
        'model': 'deepseek-reasoner',
        'api_key': lambda c: c.get('deepseek_api_key'),
        'default_temp': 1.3,
        'error_name': 'DeepSeek',
    },
    'gigachat': {
        'url': 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
        'headers': lambda c: {"Authorization": f"Bearer {c.get('gigachat_api_key')}", "Content-Type": "application/json", "Accept": "application/json"},
        'model': 'GigaChat',
        'api_key': lambda c: c.get('gigachat_api_key'),
        'default_temp': 0.7,
        'error_name': 'GigaChat',
    },
    'yandexgpt': {
        'url': 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
        'headers': lambda c: {"Authorization": f"Api-Key {c.get('api_key')}", "Content-Type": "application/json"},
        'model': None,
        'api_key': lambda c: c.get('api_key'),
        'default_temp': 0.6,
        'error_name': 'Yandex GPT',
        'build_data': lambda c, msgs, temp: {
            "modelUri": f"gpt://{c.get('folder_id')}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": temp, "maxTokens": 1000},
            "messages": msgs,
        },
        'parse_response': lambda r: r['result']['alternatives'][0]['message']['text'],
    },
}


async def get_openai_compatible_response(ai_config, messages, provider_name, temperature=None):
    """Универсальный вызов OpenAI-совместимых API (DeepSeek, GigaChat, YandexGPT)."""
    cfg = PROVIDER_CONFIGS[provider_name]
    try:
        api_key = cfg['api_key'](ai_config)
        if not api_key:
            return f"API ключ для {cfg['error_name']} не настроен"

        temp = temperature if temperature is not None else ai_config.get('temperature', cfg['default_temp'])
        data = cfg.get('build_data')(ai_config, messages, temp) if 'build_data' in cfg else {
            "model": cfg['model'],
            "messages": messages,
            "temperature": temp,
            "max_tokens": 2000,
            "stream": False,
        }

        response = await make_async_request(cfg['url'], cfg['headers'](ai_config), data)

        if response.status_code == 200:
            result = response.json()
            if 'parse_response' in cfg:
                return cfg['parse_response'](result)
            return result['choices'][0]['message']['content']
        else:
            return f"Ошибка {cfg['error_name']} API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка при запросе к {cfg['error_name']}: {str(e)}"


async def handle_message(update: Update, context):
    """Единый обработчик сообщений для групп и личных чатов"""
    if update.message is None:
        return
    user = update.message.from_user
    if user is None:
        return

    bot_username = context.bot.username
    chat_id = update.message.chat_id
    is_group = update.message.chat.type in ('group', 'supergroup')

    # Проверка доступа
    if is_group:
        message_thread_id = update.message.message_thread_id
        allowed_chat_ids = config.get('allowed_group_chat_ids', [])
        if message_thread_id not in allowed_chat_ids:
            return

        always_respond = config.get('always_respond_to_users', [])
        mentioned = is_bot_mentioned(update.message.text, bot_username)
        replied_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user.id == context.bot.id
        )
        if not (mentioned or replied_to_bot or user.username in always_respond):
            return
    else:
        if user.username not in config.get('allowed_private_users', []):
            return

    # Формирование сообщения
    message_text = update.message.text
    if is_group:
        quoted_info = await analyze_quoted_message(update.message.reply_to_message)
        message_text = await enhance_message_with_quote(message_text, quoted_info, user.first_name)

    # Вызов AI или случайный ответ
    use_ai = config.get('use_ai', False)
    if use_ai:
        ai_response = await get_ai_response_with_context(
            message_text, bot_username, chat_id,
            user_name=user.first_name, user_id=user.id
        )
        chat_context.add_message(chat_id, "assistant", ai_response)
        await send_long_message(update, ai_response, parse_mode='Markdown')
    else:
        responses = config.get('responses', [])
        if responses:
            await send_long_message(update, random.choice(responses), parse_mode='Markdown')


async def clear_context_command(update: Update, context):
    """Команда для очистки контекста"""
    chat_id = update.message.chat_id
    chat_context.clear_context(chat_id)
    await send_long_message(update, "Контекст диалога очищен!", parse_mode='Markdown')


async def show_context_command(update: Update, context):
    """Команда для показа текущего контекста (для отладки)"""
    chat_id = update.message.chat_id
    context_messages = chat_context.get_context(chat_id)

    if not context_messages:
        await send_long_message(update, "Контекст пуст", parse_mode='Markdown')
        return

    context_text = "Текущий контекст:\n\n"
    for i, msg in enumerate(context_messages[-5:], 1):  # Показываем последние 5 сообщений
        role = "👤" if msg["role"] == "user" else "🤖"
        context_text += f"{role} {msg['content'][:100]}...\n"

    await send_long_message(update, context_text, parse_mode='Markdown')


async def reload_config_command(update: Update, context):
    """Команда для перезагрузки конфигурации"""
    try:
        user = update.message.from_user
        if not is_admin(user):
            return

        global config
        config = load_config()
        await send_long_message(update, "✅ Конфигурация перезагружена!", parse_mode='Markdown')
    except Exception as e:
        await send_long_message(update, f"❌ Ошибка: {str(e)}", parse_mode='Markdown')


async def clear_dossier_command(update: Update, context):
    """Команда для очистки досье конкретного пользователя"""
    user = update.message.from_user
    if not is_admin(user):
        return

    args = context.args if context.args else []
    if not args or not args[0].isdigit():
        await send_long_message(update, "Использование: /clear_dossier <user_id>", parse_mode='Markdown')
        return

    target_user_id = int(args[0])
    clear_dossier(target_user_id)
    await send_long_message(update, f"✅ Досье пользователя {target_user_id} очищено!", parse_mode='Markdown')


async def clear_dossiers_command(update: Update, context):
    """Команда для очистки досье всех пользователей"""
    user = update.message.from_user
    if not is_admin(user):
        return

    clear_all_dossiers()
    await send_long_message(update, "✅ Досье всех пользователей очищены!", parse_mode='Markdown')


async def analyze_quoted_message(quoted_message):
    """Анализирует цитируемое сообщение и возвращает информацию о нем"""
    if not quoted_message:
        return None

    info = {
        'exists': True,
        'user_id': quoted_message.from_user.id,
        'user_name': quoted_message.from_user.first_name,
        'message_id': quoted_message.message_id,
        'date': quoted_message.date,
        'content_type': 'text',
        'content': None
    }

    # Определяем тип контента
    if quoted_message.text:
        info['content'] = quoted_message.text
        info['content_type'] = 'text'
    elif quoted_message.caption:
        info['content'] = quoted_message.caption
        info['content_type'] = 'media_with_caption'
    elif quoted_message.photo:
        info['content_type'] = 'photo'
        info['content'] = "[Изображение]"
    elif quoted_message.video:
        info['content_type'] = 'video'
        info['content'] = "[Видео]"
    elif quoted_message.document:
        info['content_type'] = 'document'
        info['content'] = f"[Документ: {quoted_message.document.file_name}]"
    elif quoted_message.sticker:
        info['content_type'] = 'sticker'
        info['content'] = f"[Стикер: {quoted_message.sticker.emoji}]"
    else:
        info['content_type'] = 'unknown'
        info['content'] = "[Медиа-сообщение]"

    return info


async def enhance_message_with_quote(current_message, quoted_info, user_name):
    """Улучшает сообщение, добавляя информацию о цитате"""
    if not quoted_info:
        return current_message

    quote_text = quoted_info['content'] or "[сообщение без текста]"

    # Формируем контекст в зависимости от типа цитаты
    if quoted_info['content_type'] == 'text':
        enhanced = f"Пользователь {user_name} отвечает на сообщение '{quote_text}': {current_message}"
    else:
        enhanced = f"Пользователь {user_name} отвечает на {quoted_info['content_type']} '{quote_text}': {current_message}"

    return enhanced


def main():
    token = config.get('bot_token', "")
    proxy_list = config.get('telegram_proxy', [])

    if proxy_list:
        custom_request = ProxyRotatingRequest(proxy_list)
        application = (
            Application.builder()
            .token(token)
            .request(custom_request)
            .get_updates_request(custom_request)
            .build()
        )
        logger.info("Бот запущен с прокси: %s", proxy_list)
    else:
        application = Application.builder().token(token).build()

    # Обработчики сообщений
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    # Команды
    application.add_handler(CommandHandler("clear_context", clear_context_command, filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("show_context", show_context_command, filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("reload_config", reload_config_command))
    application.add_handler(CommandHandler("clear_dossier", clear_dossier_command, filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("clear_dossiers", clear_dossiers_command, filters.ChatType.PRIVATE))

    print("Бот запущен с поддержкой контекста!")
    application.run_polling()


if __name__ == "__main__":
    main()
