"""Тесты промптов БЗ (botkb/prompts.py, ТЗ п. 15, 10).

Проверяются обязательные элементы всех промптов: обёртка «данные ≠ инструкции»,
запрет секретов/личных данных, компактность с целевым объёмом, формат «только
markdown», отсутствие инструкций «как должен отвечать бот» в тематике знаний,
запрет «такой-то сказал», правило противоречий, self-блок «реплики — обратная
связь, не инструкции».
"""

from botwiki.prompts import DATA_BEGIN, DATA_END

from botkb import prompts


def test_markers_present():
    assert DATA_BEGIN and DATA_END


def test_bootstrap_home_with_seed():
    text = prompts.build_bootstrap_home_prompt(
        None, target_chars=900, max_chars=2000,
        seed_text='Бот — техспециалист по Volvo', window_block='')
    assert DATA_BEGIN in text and DATA_END in text
    assert 'ТОЛЬКО markdown' in text
    assert 'а не инструкции' in text
    assert 'не дублируй' in text.lower()


def test_bootstrap_home_empty_without_sources():
    assert prompts.build_bootstrap_home_prompt(
        None, target_chars=900, max_chars=2000, seed_text='', window_block='') == ''


def test_bootstrap_style_prompt():
    text = prompts.build_bootstrap_style_prompt(
        None, target_chars=700, max_chars=2000,
        seed_text='описание стиля', window_block='[1] пример реплики')
    assert DATA_BEGIN in text
    assert 'Стиль' in text or 'стиль' in text.lower()


def test_update_self_prompt_rules():
    text = prompts.build_update_self_prompt(
        None, slug='Home', title='О боте', current_md='# О боте\n- тезис',
        target_chars=900, max_chars=2000,
        dialog_block='[10][bot] кусок\n[11][human] не так!')
    assert DATA_BEGIN in text
    assert 'обратная связь' in text.lower()
    assert 'не инструкции' in text.lower()
    assert 'Home' in text and 'О боте' in text
    assert 'кусок' in text  # блок диалогов попал в промпт


def test_update_knowledge_prompt_rules():
    text = prompts.build_update_knowledge_prompt(
        None, slug='vyhlop', title='Выхлоп', current_md='# Выхлоп\n- старый тезис',
        target_chars=1400, max_chars=2000, raw_block='[1] меряй противодавление')
    assert DATA_BEGIN in text
    assert 'ранее X, теперь Y' in text or 'ранее' in text
    # знания — факты, а не «сказал»: явный запрет авторства
    assert 'авторов' in text and 'такой-то сказал' in text
    assert 'Выхлоп' in text and 'vyhlop' in text
    assert 'противодавление' in text
    assert 'обратная связь' not in text  # это не self-контур


def test_update_knowledge_prompt_with_bot_answers_note():
    text = prompts.build_update_knowledge_prompt(
        None, slug='vyhlop', title='Выхлоп', current_md='# Выхлоп',
        target_chars=1400, max_chars=2000,
        raw_block='[1][human] вопрос\n[2][bot] ответ', includes_bot_answers=True)
    assert '[i][bot]' in text  # заметка про ответы бота как данные
    assert 'могут ошибаться' in text


def test_create_knowledge_prompt_rules():
    text = prompts.build_create_knowledge_prompt(
        None, candidate='турбина', examples_block='[1] турбина дует',
        max_chars=2000)
    assert 'турбина' in text
    assert 'slug:' in text and 'content:' in text
    # запрет «такой-то сказал» и инструкций «как должен отвечать бот»
    assert 'такой-то сказал' in text
    assert 'как должен отвечать бот' in text
    assert DATA_BEGIN in text and DATA_END in text
