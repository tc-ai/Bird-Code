import sys
from pathlib import Path

import pytest

from birdcode.permission.sandbox import check_path, resolve_roots


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    return tmp_path


def test_inside_allowed(project: Path):
    assert check_path(str(project / "src" / "x.py"), resolve_roots([project])) is None


def test_absolute_escape(project: Path):
    if sys.platform.startswith("win"):
        target = r"C:\Windows\System32\drivers\etc\hosts"
    else:
        target = "/etc/passwd"
    assert check_path(target, resolve_roots([project])) is not None


def test_dotdot_escape(project: Path):
    assert check_path(str(project / ".." / "evil.py"), resolve_roots([project])) is not None


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Windows 创建符号链接需开发者模式/管理员,CI 不保证",
)
def test_symlink_escape(project: Path):
    (project / "link").symlink_to("/etc")
    assert check_path(str(project / "link" / "hosts"), resolve_roots([project])) is not None


def test_extra_root_allowed(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    extra = tmp_path / "outside"
    extra.mkdir()
    assert check_path(str(extra / "f"), resolve_roots([proj, extra])) is None
