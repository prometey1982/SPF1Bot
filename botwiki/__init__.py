"""Пакет botwiki: персональная wiki пользователя (ТЗ docs/user_wiki_tz.md).

Не импортирует bot.py (чтобы тесты не триггерили import-time инициализацию
БД). bot.py вызывает `botwiki.configure(config)` при старте и в
/reload_config.
"""

from . import config as config
from . import db as db
from . import capture as capture
from . import redact as redact
from . import retention as retention
from . import pages as pages
from . import index as index
from . import prompts as prompts
from . import manager as manager

configure = config.configure
settings = config.settings
wiki_manager = manager.wiki_manager
