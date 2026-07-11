import pytest

from birdcode.agent.context import CompactionResult
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.ui.commands.builtins import build_builtin_registry
from birdcode.ui.commands.command import CommandType


class FakeCtx:
    """最小 CommandContext 实现,记录调用。"""

    def __init__(
        self,
        *,
        registry=None,
        mcp_manager=None,
        store=None,
        cfg=None,
        controller=None,
        model="mock",
    ):
        self._registry = registry
        self._mcp = mcp_manager
        self._store = store
        self._cfg = cfg
        self._controller = controller
        self._model = model
        self.messages = []
        self.sent = []
        self.exited = False

    async def show_message(self, text, *, kind="info"):
        self.messages.append((kind, text))

    async def send_user_message(self, text):
        self.sent.append(text)

    def get_token_usage(self):
        return None

    def refresh_status(self):
        pass

    def exit_app(self):
        self.exited = True

    async def clear_conversation(self):
        pass

    async def compact_now(self):
        return None, "empty"

    async def switch_profile(self, name):
        return False

    @property
    def store(self):
        return self._store

    @property
    def tools(self):
        return self._registry

    @property
    def mcp_manager(self):
        return self._mcp

    @property
    def controller(self):
        return self._controller

    @property
    def cfg(self):
        return self._cfg

    @property
    def model(self):
        return self._model


@pytest.mark.asyncio
async def test_help_lists_non_hidden_commands():
    reg = build_builtin_registry()
    help_cmd = reg.resolve("/help")
    ctx = FakeCtx(registry=reg)
    await help_cmd.handler(ctx, "")
    assert ctx.messages
    text = ctx.messages[0][1]
    assert "/help" in text and "/review" in text
    assert "/clear" in text
    lines = text.splitlines()
    assert all(not ln.strip().startswith("/?") for ln in lines)


@pytest.mark.asyncio
async def test_exit_calls_exit_app():
    reg = build_builtin_registry()
    ctx = FakeCtx()
    await reg.resolve("/exit").handler(ctx, "")
    assert ctx.exited is True


def test_registry_has_11_commands_and_aliases():
    reg = build_builtin_registry()
    names = reg.names(include_hidden=True)
    assert set(names) == {
        "/help",
        "/server",
        "/tool",
        "/sessions",
        "/compact",
        "/context",
        "/clear",
        "/exit",
        "/profile",
        "/review",
        "/agents",
    }
    assert reg.resolve("/?").name == "/help"
    assert reg.resolve("/quit").name == "/exit"
    assert reg.resolve("/session").name == "/sessions"


def test_command_types():
    reg = build_builtin_registry()
    assert reg.resolve("/help").type is CommandType.LOCAL
    assert reg.resolve("/clear").type is CommandType.STATE
    assert reg.resolve("/compact").type is CommandType.STATE
    assert reg.resolve("/profile").type is CommandType.STATE
    assert reg.resolve("/review").type is CommandType.PROMPT


@pytest.mark.asyncio
async def test_compact_busy_warns():
    reg = build_builtin_registry()
    ctx = FakeCtx()

    async def _busy():
        return None, "busy"

    ctx.compact_now = _busy  # type: ignore[assignment]
    await reg.resolve("/compact").handler(ctx, "")
    assert ctx.messages[0][0] == "warn"
    assert "正在运行" in ctx.messages[0][1]


@pytest.mark.asyncio
async def test_compact_ok_formats_tokens():
    reg = build_builtin_registry()
    ctx = FakeCtx()
    res = CompactionResult(
        trigger="manual",
        pre_tokens=1000,
        post_tokens=200,
        boundary_uuid="b",
        summary_uuid="s",
        fell_back=False,
    )

    async def _ok():
        return res, "ok"

    ctx.compact_now = _ok  # type: ignore[assignment]
    await reg.resolve("/compact").handler(ctx, "")
    assert "1000→200" in ctx.messages[0][1]


@pytest.mark.asyncio
async def test_clear_reports_new_session():
    reg = build_builtin_registry()
    ctx = FakeCtx()

    async def _clear():
        return "abc12345-aaaa-bbbb-cccc-dddddddddddd"

    ctx.clear_conversation = _clear  # type: ignore[assignment]
    await reg.resolve("/clear").handler(ctx, "")
    assert "abc12345" in ctx.messages[0][1]  # 短 id 前 8


@pytest.mark.asyncio
async def test_profile_unknown_lists_available():
    reg = build_builtin_registry()
    cfg = AppConfig(
        providers={"a": ProviderProfile(protocol="anthropic", model="m", base_url="u", api_key="k")}
    )
    ctx = FakeCtx(cfg=cfg)
    await reg.resolve("/profile").handler(ctx, "nope")
    assert ctx.messages[0][0] == "warn"
    assert "a" in ctx.messages[0][1]
    assert not ctx.messages[0][1].startswith("⚠")  # 不自带 ⚠(show_message 统一加,防双图标)


@pytest.mark.asyncio
async def test_profile_no_arg_warns_usage():
    reg = build_builtin_registry()
    ctx = FakeCtx()
    await reg.resolve("/profile").handler(ctx, "   ")
    assert "用法" in ctx.messages[0][1]


@pytest.mark.asyncio
async def test_review_empty_diff_warns(monkeypatch):
    reg = build_builtin_registry()

    async def _empty(_cwd):
        return ""

    monkeypatch.setattr("birdcode.ui.commands.builtins.run_git_diff_head", _empty)
    ctx = FakeCtx()
    await reg.resolve("/review").handler(ctx, "")
    assert ctx.messages[0][0] == "warn"
    assert "无未提交" in ctx.messages[0][1]
    assert ctx.sent == []  # 不送 AI


@pytest.mark.asyncio
async def test_review_sends_prompt_with_diff(monkeypatch):
    reg = build_builtin_registry()

    async def _diff(_cwd):
        return "diff --git a/x b/x\n+new"

    monkeypatch.setattr("birdcode.ui.commands.builtins.run_git_diff_head", _diff)
    ctx = FakeCtx()
    await reg.resolve("/review").handler(ctx, "")
    assert len(ctx.sent) == 1
    assert "diff --git a/x b/x" in ctx.sent[0]
    assert "审查" in ctx.sent[0]


@pytest.mark.asyncio
async def test_review_diff_failure_errors(monkeypatch):
    reg = build_builtin_registry()

    async def _boom(_cwd):
        raise RuntimeError("git broken")

    monkeypatch.setattr("birdcode.ui.commands.builtins.run_git_diff_head", _boom)
    ctx = FakeCtx()
    await reg.resolve("/review").handler(ctx, "")
    assert ctx.messages[0][0] == "error"
    assert ctx.sent == []


@pytest.mark.asyncio
async def test_review_truncates_huge_diff(monkeypatch):
    reg = build_builtin_registry()

    async def _huge(_cwd):
        return "X" * 150_000

    monkeypatch.setattr("birdcode.ui.commands.builtins.run_git_diff_head", _huge)
    ctx = FakeCtx()
    await reg.resolve("/review").handler(ctx, "")
    assert len(ctx.sent[0]) < 150_000
    assert "截断" in ctx.sent[0]


@pytest.mark.asyncio
async def test_profile_takes_first_token_only():
    """/profile gpt-4 extra → 只取首 token "gpt-4"(恢复旧行为),extra 不影响切换。"""
    reg = build_builtin_registry()
    cfg = AppConfig(
        providers={
            "gpt-4": ProviderProfile(protocol="anthropic", model="m", base_url="u", api_key="k")
        }
    )
    ctx = FakeCtx(cfg=cfg)
    captured: dict = {}

    async def _sw(name):
        captured["name"] = name
        return name == "gpt-4"

    ctx.switch_profile = _sw  # type: ignore[assignment]
    await reg.resolve("/profile").handler(ctx, "gpt-4 extra")
    assert captured.get("name") == "gpt-4"


@pytest.mark.asyncio
async def test_review_git_diff_timeout_kills_proc(monkeypatch):
    """run_git_diff_head 超时 → kill 子进程 + 抛 RuntimeError(/review 捕获提示,不误报"无改动")。"""
    import asyncio as _asyncio
    from pathlib import Path as _Path

    from birdcode.ui.commands import builtins

    monkeypatch.setattr(builtins, "_GIT_DIFF_TIMEOUT", 0.01)

    class _HungProc:
        def __init__(self) -> None:
            self.killed = False
            self.returncode = 0

        async def communicate(self):
            await _asyncio.sleep(30)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return 0

    holder: dict = {}

    async def _fake_exec(*args, **kwargs):
        holder["p"] = _HungProc()
        return holder["p"]

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
    with pytest.raises(RuntimeError, match="超时"):
        await builtins.run_git_diff_head(_Path("."))
    assert holder["p"].killed is True
