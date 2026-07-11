# tests/test_icons.py
def test_unicode_icons_by_default():
    from birdcode.ui.icons import select_icons

    ic = select_icons(unicode_supported=True)
    assert ic.tool == "●" and ic.result == "⎿" and ic.success == "✓"


def test_ascii_fallback():
    from birdcode.ui.icons import select_icons

    ic = select_icons(unicode_supported=False)
    assert all(ord(c) < 128 for c in (ic.tool, ic.result, ic.success, ic.fail, ic.queued))
    assert len(ic.spinner) >= 2 and all(ord(c) < 128 for c in ic.spinner)
