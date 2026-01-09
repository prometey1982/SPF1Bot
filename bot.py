import yaml
import random
import re
import requests
import asyncio
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters


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
        if message_thread_id:
            await update.message.reply_text(part, parse_mode=parse_mode, message_thread_id=message_thread_id)
        else:
            await update.message.reply_text(part, parse_mode=parse_mode)
        
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


def is_bot_mentioned(text, bot_username):
    """Проверяет, упомянут ли бот в тексте"""
    if not text:
        return False

    # Регулярное выражение для поиска упоминаний
    # Ищет @username или просто username как отдельное слово
    pattern = r'(?:^|\s)(@?' + re.escape(bot_username) + r')(?:\s|$|[,!?.])'
    return bool(re.search(pattern, text, re.IGNORECASE))


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


async def get_ai_response_with_context(message_text, bot_username, chat_id, user_name=""):
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

    # Формируем промпт с контекстом
    if provider in ['deepseek', 'yandexgpt', 'gigachat']:
        return await get_modern_ai_response(ai_config, context_messages, provider)
    else:
        return await get_legacy_ai_response(ai_config, context_messages, message_text, provider)


async def get_modern_ai_response(ai_config, context_messages, provider):
    """Для современных API, поддерживающих историю сообщений"""
    try:
        system_prompt = ai_config.get('system_prompt', 'Ты полезный ассистент. Отвечай на русском.')

        # Формируем messages для API
        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю диалога
        for msg in context_messages[-15:]:  # Берем последние 15 сообщений
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        if provider == 'deepseek':
            return await get_deepseek_response(ai_config, messages)
        elif provider == 'yandexgpt':
            return await get_yandexgpt_response(ai_config, messages)
        elif provider == 'gigachat':
            return await get_gigachat_response(ai_config, messages)

    except Exception as e:
        return f"Ошибка при обработке контекста: {str(e)}"


async def get_legacy_ai_response(ai_config, context_messages, message_text, provider):
    """Для API, которые не поддерживают историю сообщений"""
    # Собираем контекст в один текст
    context_text = ""
    for msg in context_messages[-5:]:  # Берем последние 5 сообщений
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        context_text += f"{role}: {msg['content']}\n"

    full_prompt = f"Контекст диалога:\n{context_text}\nТекущее сообщение: {message_text}\nОтвет:"

    if provider == 'llama':
        return await get_llama_response(ai_config, full_prompt)
    else:
        return await get_deepseek_response(ai_config, [{"role": "user", "content": full_prompt}])


async def get_llama_response(ai_config, prompt):
    """Llama API с поддержкой локальных моделей"""
    try:
        # Получаем конфигурацию для Llama
        api_base = ai_config.get('llama_api_base', 'http://localhost:11434')
        model = ai_config.get('llama_model', 'llama2')

        # Формируем URL для API
        url = f"{api_base}/api/chat" if api_base.endswith('/api/chat') else f"{api_base}/api/chat"

        headers = {
            "Content-Type": "application/json"
        }

        # Формируем данные для запроса в формате Ollama
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": config.get('temperature', 0.7),
                "num_predict": config.get('max_tokens', 1000)
            }
        }

        response = await make_async_request(url, headers, data)

        if response.status_code == 200:
            result = response.json()

            # Обрабатываем разные форматы ответов от разных Llama API
            if 'message' in result and 'content' in result['message']:
                # Формат Ollama
                response_text = result['message']['content']
            elif 'choices' in result and len(result['choices']) > 0:
                # Формат OpenAI-compatible
                response_text = result['choices'][0]['message']['content']
            elif 'response' in result:
                # Прямой ответ
                response_text = result['response']
            else:
                return "Llama API вернул неожиданный формат ответа"

            return response_text
        else:
            return f"Ошибка Llama API: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Ошибка при запросе к Llama: {str(e)}"


async def get_deepseek_response(ai_config, messages):
    """DeepSeek API с поддержкой контекста"""
    try:
        api_key = ai_config.get('deepseek_api_key')
        if not api_key:
            return "API ключ для DeepSeek не настроен"

        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 1.3,
            "max_tokens": 2000,
            "stream": False
        }

        response = await make_async_request(url, headers, data)

        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']

            # Добавляем ответ ассистента в контекст
            # (это делается в основной функции после возврата)
            return response_text
        else:
            return f"Ошибка DeepSeek API: {response.status_code}"

    except Exception as e:
        return f"Ошибка при запросе к DeepSeek: {str(e)}"


async def get_yandexgpt_response(ai_config, messages):
    """Yandex GPT API"""
    try:
        api_key = ai_config.get('api_key')
        folder_id = ai_config.get('folder_id')

        if not api_key or not folder_id:
            return "Не настроен API ключ или folder_id для Yandex GPT"

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": 1000
            },
            "messages": messages
        }

        response = await make_async_request(url, headers, data)

        if response.status_code == 200:
            result = response.json()
            return result['result']['alternatives'][0]['message']['text']
        else:
            return f"Ошибка Yandex GPT API: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Ошибка при запросе к Yandex GPT: {str(e)}"


async def get_gigachat_response(ai_config, messages):
    """GigaChat API с поддержкой контекста"""
    try:
        # Получаем конфигурацию для GigaChat
        api_key = ai_config.get('gigachat_api_key')
        if not api_key:
            return "API ключ для GigaChat не настроен"

        # URL для GigaChat API
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Формируем данные для запроса
        data = {
            "model": "GigaChat",  # или другая модель GigaChat
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }

        response = await make_async_request(url, headers, data)

        if response.status_code == 200:
            result = response.json()
            # GigaChat возвращает ответ в формате choices[0].message.content
            if 'choices' in result and len(result['choices']) > 0:
                response_text = result['choices'][0]['message']['content']
                return response_text
            else:
                return "GigaChat API вернул неожиданный формат ответа"
        else:
            return f"Ошибка GigaChat API: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Ошибка при запросе к GigaChat: {str(e)}"


async def handle_group_message(update: Update, context):
    """Обрабатывает сообщения в группах с учетом контекста"""
    if update.message is None:
        return
    user = update.message.from_user
    bot_username = context.bot.username
    chat_id = update.message.chat_id

    mentioned = is_bot_mentioned(update.message.text, bot_username)
    replied_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user.id == context.bot.id
    )

    print(f"Группа: {update.message.chat.title}")
    print(f"Чат: {chat_id}")
    print(f"От: {user.first_name} (ID: {user.id})")
    print(f"Упоминание: {mentioned}, Ответ боту: {replied_to_bot}")

    always_respond_to_users = config.get('always_respond_to_users')

    if (mentioned or replied_to_bot) or user in always_respond_to_users:
        use_ai = config.get('use_ai', False)

        if use_ai:
            # Получаем ответ с учетом контекста
            ai_response = await get_ai_response_with_context(
                update.message.text,
                bot_username,
                chat_id,
                user_name=user.first_name
            )

            # Добавляем ответ бота в контекст
            chat_context.add_message(chat_id, "assistant", ai_response)

            if chat_id in config.allowed_group_chat_ids:
                await send_long_message(update, ai_response, parse_mode='Markdown')
                print(f"AI ответ: {ai_response}")
        else:
            responses = config.get('responses', [])
            if responses and chat_id in config.allowed_group_chat_ids:
                response = random.choice(responses)
                await send_long_message(update, response, parse_mode='Markdown')
    print("---")


async def handle_private_message(update: Update, context):
    """Обрабатывает личные сообщения с учетом контекста"""
    user = update.message.from_user
    if user is None:
        return
    bot_username = context.bot.username
    chat_id = update.message.chat_id

    allowed_private_users = config.get('allowed_private_users')

    if user.username in allowed_private_users:
        use_ai = config.get('use_ai', False)

        if use_ai:
            ai_response = await get_ai_response_with_context(
                update.message.text,
                bot_username,
                chat_id,
                user_name=user.first_name
            )

            # Добавляем ответ бота в контекст
            chat_context.add_message(chat_id, "assistant", ai_response)

            await send_long_message(update, ai_response, parse_mode='Markdown')
        else:
            responses = config.get('responses', [])
            if responses:
                response = random.choice(responses)
                await send_long_message(update, response, parse_mode='Markdown')


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
    """Команда для показа текущего контекста (для отладки)"""
    try:
        user = update.message.from_user
        if user is None:
            return

        global config

        if user.username not in config.get('allowed_private_users'):
            return

        config = load_config()  # Перезагружаем конфиг
        await send_long_message(update, "✅ Конфигурация перезагружена!", parse_mode='Markdown')
    except Exception as e:
        await send_long_message(update, f"❌ Ошибка: {str(e)}", parse_mode='Markdown')


async def handle_group_message_advanced(update: Update, context):
    """Расширенная обработка с детальным анализом цитируемых сообщений"""
    if update.message is None:
        return

    user = update.message.from_user

    if user is None:
        return

    bot_username = context.bot.username
    chat_id = update.message.chat_id
    message_thread_id = update.message.message_thread_id

    if chat_id is None:
        return

    # Анализ цитируемого сообщения
    quoted_info = await analyze_quoted_message(update.message.reply_to_message)

    mentioned = is_bot_mentioned(update.message.text, bot_username)
    replied_to_bot = (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user.id == context.bot.id
    )

    print(f"Группа: {update.message.chat.title}")
    print(f"Чат: {chat_id}")
    print(f"Message thread id: {message_thread_id}")
    print(f"От: {user.first_name} (ID: {user.id})")
    print(f"Цитирование: {quoted_info}")

    always_respond_to_users = config.get('always_respond_to_users')

    if (mentioned or replied_to_bot or user.username in always_respond_to_users) and message_thread_id in config.get(
            'allowed_group_chat_ids', []):
        use_ai = config.get('use_ai', False)

        if use_ai:
            # Формируем расширенный контекст с цитатой
            enhanced_message = await enhance_message_with_quote(
                update.message.text,
                quoted_info,
                user.first_name
            )

            ai_response = await get_ai_response_with_context(
                enhanced_message,
                bot_username,
                chat_id,
                user_name=user.first_name
            )

            chat_context.add_message(chat_id, "assistant", ai_response)
            await send_long_message(update, ai_response, parse_mode='Markdown')

        else:
            responses = config.get('responses', [])
            if responses:
                response = random.choice(responses)
                await send_long_message(update, response, parse_mode='Markdown')


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
    application = Application.builder().token(config.get('bot_token', "")).build()

    # Обработчики сообщений
    application.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.GROUPS,
        handle_group_message_advanced
    ))
    # application.add_handler(MessageHandler(
    #     filters.TEXT & filters.ChatType.GROUPS,
    #     handle_group_message
    # ))

    application.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
        handle_private_message
    ))

    # Команды для управления контекстом
    application.add_handler(MessageHandler(
        filters.Regex(r'^/clear_context$') & filters.ChatType.PRIVATE,
        clear_context_command
    ))

    application.add_handler(MessageHandler(
        filters.Regex(r'^/show_context$') & filters.ChatType.PRIVATE,
        show_context_command
    ))

    application.add_handler(CommandHandler("reload_config", reload_config_command))

    print("Бот запущен с поддержкой контекста!")
    application.run_polling()


if __name__ == "__main__":
    main()
