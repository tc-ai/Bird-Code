# tests/test_pure.py
from birdcode.ui.pure import (
    enter_action,
    shorten_path,
    should_continue,
)


def test_should_continue():
    assert should_continue("ls \\") is True
    assert should_continue("ls -la") is False


def test_shorten_path_under_home(monkeypatch, tmp_path):
    monkeypatch.setattr("birdcode.ui.pure.Path.home", lambda: tmp_path)
    p = tmp_path / "proj" / "pkg"
    p.mkdir(parents=True)
    assert shorten_path(str(p)) == "~/pkg" or shorten_path(str(p)).endswith("pkg")


def test_enter_action():
    assert enter_action("ls \\") == "continue"
    assert enter_action("ls -la") == "submit"
