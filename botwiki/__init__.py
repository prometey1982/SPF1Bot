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

configure = config.configure
settings = config.settings
