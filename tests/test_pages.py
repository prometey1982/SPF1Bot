"""Тесты файлового слоя страниц (botwiki/pages.py, ТЗ п. 7.2)."""

import os

from botwiki import pages


def test_safe_slug_cases():
    assert pages.is_safe_slug('Home') is True
    assert pages.is_safe_slug('style') is True
    assert pages.is_safe_slug('Interests_2024') is True
    assert pages.is_safe_slug('cars-2') is True
    assert pages.is_safe_slug('') is False
    assert pages.is_safe_slug('home.md') is False
    assert pages.is_safe_slug('a/b') is False
    assert pages.is_safe_slug('..') is False
    assert pages.is_safe_slug('a' * 65) is False
    assert pages.is_safe_slug('x y') is False
    assert pages.is_safe_slug(None) is False


def test_thematic_slug_cases():
    assert pages.is_thematic_slug('cars') is True
    assert pages.is_thematic_slug('Home') is False
    assert pages.is_thematic_slug('style') is False
    assert pages.is_thematic_slug('_index') is False
    assert pages.is_thematic_slug('_index.yaml') is False
    assert pages.is_thematic_slug('cars.md') is False
    assert pages.is_thematic_slug('Авто') is False  # slug_pattern ^[a-z0-9_-]$
    assert pages.is_thematic_slug('') is False


def test_normalize_slug():
    assert pages.normalize_slug('Cars') == 'cars'
    assert pages.normalize_slug(' Cars ') == 'cars'
    assert pages.normalize_slug('Home') is None
    assert pages.normalize_slug('Авто') is None
    assert pages.normalize_slug('a/b') is None


def test_write_read_roundtrip(tmp_path):
    user_dir = str(tmp_path)
    assert pages.atomic_write_page(user_dir, 'Home', '# Сводка\n- факт') is True
    assert pages.read_page(user_dir, 'Home') == '# Сводка\n- факт'
    assert pages.page_exists(user_dir, 'Home') is True
    assert pages.read_page(user_dir, 'Nope') is None


def test_atomic_overwrite_no_tmp_left(tmp_path):
    user_dir = str(tmp_path)
    pages.atomic_write_page(user_dir, 'Home', 'v1')
    pages.atomic_write_page(user_dir, 'Home', 'v2')
    assert pages.read_page(user_dir, 'Home') == 'v2'
    leftovers = [n for n in os.listdir(user_dir) if n.endswith('.tmp')]
    assert leftovers == []


def test_unsafe_slug_rejected(tmp_path):
    user_dir = str(tmp_path)
    assert pages.atomic_write_page(user_dir, 'a/b', 'x') is False
    assert pages.atomic_write_page(user_dir, '..', 'x') is False
    assert pages.atomic_write_page(user_dir, '', 'x') is False
    assert os.listdir(user_dir) == []


def test_list_page_slugs_filters_junk(tmp_path):
    user_dir = str(tmp_path)
    for name in ('Home.md', 'cars.md', '_index.yaml', 'x y.md', '.DS_Store',
                 'a.md.tmp', 'Style.md'):
        with open(os.path.join(user_dir, name), 'w', encoding='utf-8') as f:
            f.write('x')
    assert pages.list_page_slugs(user_dir) == ['Home', 'Style', 'cars']


def test_list_page_slugs_missing_dir(tmp_path):
    assert pages.list_page_slugs(str(tmp_path / 'nope')) == []
