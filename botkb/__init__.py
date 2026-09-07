"""Пакет botkb: долговременная база знаний бота (ТЗ docs/bot_kb_tz.md).

Не импортирует bot.py (чтобы тесты не триггерили import-time инициализацию
БД). bot.py вызывает `botkb.configure(config)` при старте и в
/reload_config.

Этап 1: захват сырья в bot_kb_raw + конфиг/валидация + ретенция.
"""

from . import config as config
from . import db as db
from . import capture as capture
from . import retention as retention
from . import pages as pages
from . import index as index
from . import prompts as prompts
from . import topics as topics
from . import router as router
from . import inject as inject
from . import manager as manager
from . import admin as admin

configure = config.configure
settings = config.settings
kb_manager = manager.KBManager()
