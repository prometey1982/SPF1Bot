"""Детерминированное создание тематических страниц БЗ (ТЗ п. 9.4).

Переиспользует чистые хелперы botwiki.topics без изменений (обнаружение
кандидатов по повторяемости, пересечения со страницами, cooldown, разбор
YAML-предложения) — они работают с обобщёнными строками и страницами и не
привязаны к персональной wiki. Смысловые правила «факт ≠ единичное мнение»,
анонимность и запрет личных данных задаются промптами (botkb/prompts.py).
"""

from botwiki.topics import (  # noqa: F401  (переэкспорт)
    detect_candidates,
    check_page_overlap,
    is_cooldown_active,
    prune_cooldowns,
    set_cooldown,
    parse_page_proposal,
    re_sub_slug,
)
