"""Тесты файлового слоя БЗ (botkb/pages.py, ТЗ п. 7.2)."""

import os

from botkb import pages


def test_kind_for_slug():
    assert pages.kind_for_slug('Home') == 'self'
    assert pages.kind_for_slug('Style') == 'self'
    assert pages.kind_for_slug('volvo') == 'knowledge'
    assert pages.kind_for_slug('VYHLOP') == 'knowledge'


def test_safe_and_thematic_slugs():
    assert pages.is_safe_slug('Home') is True
    assert pages.is_safe_slug('volvo_repair-2') is True
    assert pages.is_thematic_slug('volvo_repair-2') is True
    # служебные имена запрещены для тематических
    for bad in ('home', 'style', '_index', 'index', 'config'):
        assert pages.is_thematic_slug(bad) is False
    # IO-безопасность: никаких путей/спецсимволов
    for bad in ('../x', 'a/b', 'a.b', 'a b', '', 'x' * 65):
        assert pages.is_safe_slug(bad) is False
        assert pages.is_thematic_slug(bad) is False


def test_normalize_slug():
    assert pages.normalize_slug('Volvo_Repair') == 'volvo_repair'
    assert pages.normalize_slug('  Turbo-2  ') == 'turbo-2'
    assert pages.normalize_slug('Home') is None  # служебное
    assert pages.normalize_slug('с пробелом') is None
    assert pages.normalize_slug('../../etc') is None


def test_write_read_roundtrip(tmp_path):
    root = str(tmp_path)
    assert pages.write_page('volvo', '# Volvo\n\nФакты', root) is True
    assert pages.read_page('volvo', root) == '# Volvo\n\nФакты'
    assert pages.page_exists('volvo', root) is True


def test_unsafe_slug_rejected(tmp_path):
    root = str(tmp_path)
    assert pages.page_path('../escape', root) is None
    assert pages.write_page('../escape', 'x', root) is False
    assert pages.read_page('../escape', root) is None
    assert pages.page_exists('../escape', root) is False
    # за пределами корня файлов не появляется
    assert not os.path.exists(str(tmp_path.parent / 'escape.md'))


def test_overwrite_no_tmp_left(tmp_path):
    root = str(tmp_path)
    assert pages.write_page('volvo', 'первая', root) is True
    assert pages.write_page('volvo', 'вторая', root) is True
    assert pages.read_page('volvo', root) == 'вторая'
    leftovers = [n for n in os.listdir(root) if n.endswith('.tmp')]
    assert leftovers == []


def test_list_slugs_filters_junk(tmp_path):
    root = str(tmp_path)
    for slug in ('Home', 'Style', 'volvo'):
        pages.write_page(slug, f'#{slug}', root)
    # мусор/служебные файлы и небезопасные имена не дают слагов
    with open(os.path.join(root, 'некст.md'), 'w', encoding='utf-8') as f:
        f.write('x')           # кириллица — небезопасный slug
    with open(os.path.join(root, 'page.tmp.md'), 'w', encoding='utf-8') as f:
        f.write('x')           # точка в имени — небезопасный slug
    with open(os.path.join(root, '_index.yaml'), 'w', encoding='utf-8') as f:
        f.write('{}')
    assert pages.list_slugs(root) == ['Home', 'Style', 'volvo']


def test_list_pages_with_mtime(tmp_path):
    root = str(tmp_path)
    pages.write_page('volvo', '# v', root)
    pairs = pages.list_pages_with_mtime(root)
    assert [slug for slug, _ in pairs] == ['volvo']
    assert isinstance(pairs[0][1], float)
