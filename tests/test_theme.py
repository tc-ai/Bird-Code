# tests/test_theme.py


def test_supports_unicode_with_utf8(monkeypatch):
    from birdcode.ui import theme

    class FakeStream:
        encoding = "utf-8"

    monkeypatch.setattr(theme.sys, "stdout", FakeStream())
    assert theme.supports_unicode() is True


def test_get_theme_returns_dark_or_light():
    from birdcode.ui.theme import get_theme

    assert get_theme("dark").name == "dark"
    assert get_theme("light").name == "light"
    assert get_theme("unknown").name == "dark"  # fallback


def test_theme_has_textual_design():
    from birdcode.ui.theme import get_theme

    dark = get_theme("dark")
    light = get_theme("light")
    assert "background" in dark.design and "primary" in dark.design
    assert dark.design != light.design  # 两套配色不同
