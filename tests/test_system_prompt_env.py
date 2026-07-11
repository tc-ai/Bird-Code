import asyncio
from types import SimpleNamespace  # noqa: F401  (保留与既有风格一致)

import pytest

from birdcode.agent.system_prompt import build_system_reminder
from birdcode.agent.system_prompt.env import (
    _stable_env_cached,
    gather_dynamic_env,
    gather_stable_env,
)
from birdcode.tools.base import Tool
from birdcode.tools.registry import ToolRegistry


def test_stable_env_contains_platform_cwd_home():
    _stable_env_cached.cache_clear()
    s = gather_stable_env()
    assert "平台:" in s
    assert "操作系统:" in s
    assert "主目录:" in s
    assert "工作目录:" in s


def test_stable_env_is_cached_and_stable():
    _stable_env_cached.cache_clear()
    assert gather_stable_env() == gather_stable_env()


@pytest.mark.asyncio
async def test_dynamic_env_has_date_and_git_in_non_git_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 空 tmp 目录 → 非 git 仓库
    s = await gather_dynamic_env()
    assert "日期:" in s
    assert "Git:" in s
    assert "非 git 仓库" in s  # 降级路径


class _FakeProc:
    """伪造 asyncio 子进程:可控 stdout/returncode,记录是否被 kill。"""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self._stdout = stdout_bytes
        self.returncode = returncode
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


@pytest.mark.asyncio
async def test_git_branch_dirty_parses_no_commits_yet(monkeypatch, tmp_path):
    """空仓库(git init 无提交)→ '## No commits yet on main' → branch='main',不是 'No'。"""
    from birdcode.agent.system_prompt import env

    monkeypatch.chdir(tmp_path)

    async def _fake_exec(*args, **kwargs):
        return _FakeProc(b"## No commits yet on main\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    assert await env._git_branch_dirty() == "main(clean)"


@pytest.mark.asyncio
async def test_git_branch_dirty_parses_normal_branch_and_dirty(monkeypatch, tmp_path):
    async def _fake_exec(*args, **kwargs):
        return _FakeProc(b"## main...origin/main\n M src/foo.py\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.chdir(tmp_path)
    from birdcode.agent.system_prompt import env

    assert await env._git_branch_dirty() == "main(有未提交改动)"


@pytest.mark.asyncio
async def test_git_branch_dirty_timeout_degrades_and_kills(monkeypatch, tmp_path):
    """communicate() 超时 → 降级 '非 git 仓库' 且 proc 被 kill。"""
    from birdcode.agent.system_prompt import env

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "_GIT_TIMEOUT", 0.01)  # 加速测试

    class _HungProc(_FakeProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(30)
            return b"", b""

    holder: dict = {}

    async def _fake_exec(*args, **kwargs):
        holder["p"] = _HungProc(b"## main\n")
        return holder["p"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    assert await env._git_branch_dirty() == "非 git 仓库"
    assert holder["p"].killed is True


class _Deferred(Tool):
    name = "mcp__g__q"
    description = "查询"

    def should_defer(self) -> bool:
        return True

    async def execute(self, **args):  # type: ignore[override]
        return ""


@pytest.mark.asyncio
async def test_reminder_includes_deferred_tools():
    reg = ToolRegistry()
    reg.register(_Deferred())
    reminder = await build_system_reminder(registry=reg)
    assert "mcp__g__q" in reminder


@pytest.mark.asyncio
async def test_reminder_no_registry_is_backward_compatible():
    reminder = await build_system_reminder()
    assert isinstance(reminder, str)
