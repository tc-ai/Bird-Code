"""记忆目录解析(纯函数)测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.memory.paths import (
    project_memory_dir,
    type_to_dir,
    user_memory_dir,
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


def test_user_memory_dir_under_birdcode(home):
    assert user_memory_dir() == home / ".birdcode" / "memory"


def test_project_memory_dir_under_project_root(tmp_path):
    assert project_memory_dir(tmp_path) == tmp_path / ".birdcode" / "memory"


def test_type_to_dir_user_types_go_user_level(home, tmp_path):
    assert type_to_dir("user", tmp_path) == home / ".birdcode" / "memory"
    assert type_to_dir("feedback", tmp_path) == home / ".birdcode" / "memory"


def test_type_to_dir_project_types_go_project_level(tmp_path):
    assert type_to_dir("project", tmp_path) == tmp_path / ".birdcode" / "memory"
    assert type_to_dir("reference", tmp_path) == tmp_path / ".birdcode" / "memory"


def test_type_to_dir_rejects_unknown_type(tmp_path):
    with pytest.raises(ValueError):
        type_to_dir("secret", tmp_path)
