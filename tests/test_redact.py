"""Тесты маскировки (botwiki/redact.py, ТЗ п. 10.4)."""

from botwiki import redact


def test_masks_phone():
    text = "Позвони мне +7 912 345-67-89 завтра"
    masked, count = redact.mask_text(text, ['phone'])
    assert count == 1
    assert "912" not in masked
    assert "345" not in masked
    assert "[телефон скрыт]" in masked


def test_masks_email():
    text = "мой ящик: vasya.pupkin@mail.ru, не звони"
    masked, count = redact.mask_text(text, ['email'])
    assert count == 1
    assert "vasya.pupkin@mail.ru" not in masked
    assert "[email скрыт]" in masked


def test_masks_both_by_default():
    text = "т: +7 (912) 345-67-89 и почта a@b.ru"
    masked, count = redact.mask_text(text, ['phone', 'email'])
    assert count == 2


def test_no_mask_without_patterns():
    text = "обычное число 123456789 и слово"
    masked, count = redact.mask_text(text, [])
    assert masked == text
    assert count == 0


def test_plain_digits_not_masked():
    text = "у меня 1000 сил и 5000 оборотов"
    masked, count = redact.mask_text(text, ['phone'])
    assert masked == text
    assert count == 0


def test_empty_content():
    masked, count = redact.mask_text("", ['phone'])
    assert masked == ""
    assert count == 0
    masked, count = redact.mask_text(None, ['phone'])
    assert masked == ""
    assert count == 0


def test_default_patterns_from_capture_config():
    cfg = {'redact': True, 'redact_patterns': ['phone']}
    assert redact.default_patterns(cfg) == ['phone']
    cfg2 = {'redact': False, 'redact_patterns': ['phone']}
    assert redact.default_patterns(cfg2) == []


def test_bad_regex_ignored():
    masked, count = redact.mask_text("привет", ['['])
    assert masked == "привет"
    assert count == 0
