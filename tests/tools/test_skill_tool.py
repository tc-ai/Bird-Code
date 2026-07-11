# tests/tools/test_skill_tool.py
"""_InlineSkillTool / _ForkSkillTool / register_skill_tools 单测(Task 4)。"""

import types

import pytest

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.registry import CapabilityConflictError
from birdcode.tools.agent_tools import (
    ForkSkillInput,
    InlineSkillInput,
    _ForkSkillTool,
    _InlineSkillTool,
    register_skill_tools,
)
from birdcode.tools.base import Tool
from birdcode.tools.registry import ToolRegistry


def test_inline_skill_attrs():
    defn = AgentDefinition(
        name="commit",
        description="d",
        system_prompt="",
        mode="inline",
        is_transparent=True,
        body="do $ARGUMENTS now",
    )
    tool = _InlineSkillTool(defn=defn)
    assert tool.is_agent_tool is False
    assert tool.kind == "read" and tool.parallel_safe is True
    assert tool.name == "commit"


@pytest.mark.asyncio
async def test_inline_skill_execute():
    defn = AgentDefinition(
        name="commit",
        description="d",
        system_prompt="",
        mode="inline",
        is_transparent=True,
        body="do $ARGUMENTS now",
    )
    tool = _InlineSkillTool(defn=defn)
    assert await tool.execute(args="X") == "do X now"


@pytest.mark.asyncio
async def test_fork_skill_renders_body_as_prompt(monkeypatch):
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kw: object) -> None:
            captured.update(kw)
            # _AgentTool.execute 在 run() 后读 agent_id/sidechain_path(真实
            # SubagentRunner 契约),此处照 _CapturingRunner 模式补齐。
            self.agent_id = "sub-fixed"
            self.sidechain_path = "/tmp/side-sub-fixed.jsonl"
            self.isolation = kw.get("isolation")  # 忠 SubagentRunner 接口

        async def run(self):
            from birdcode.agents.runner import SubagentReport

            return SubagentReport(
                is_completed=True,
                text="done",
                status="completed",
                duration_ms=0,
                tokens=0,
                tool_use_count=0,
            )

    monkeypatch.setattr("birdcode.tools.agent_tools.SubagentRunner", FakeRunner)
    defn = AgentDefinition(
        name="big",
        description="d",
        system_prompt="",
        mode="fork",
        is_transparent=True,
        body="refactor $ARGUMENTS",
    )
    tool = _ForkSkillTool(
        defn=defn,
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        progress_cb=None,
        subagent_mgr=None,
    )
    assert tool.is_agent_tool is True
    await tool.execute(args="module X")
    assert captured["prompt"] == "refactor module X"


def test_register_skill_tools_returns_map_and_registers():
    reg = ToolRegistry()
    inline = AgentDefinition(
        name="ci", description="d", system_prompt="", mode="inline", is_transparent=True, body="b"
    )
    fork = AgentDefinition(
        name="big", description="d", system_prompt="", mode="fork", is_transparent=True, body="b"
    )
    from birdcode.agents.registry import AgentRegistry

    caps = AgentRegistry()
    caps.register(inline)
    caps.register(fork)
    skill_tools = register_skill_tools(
        reg,
        caps,
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        progress_cb=None,
        subagent_mgr=None,
    )
    assert set(skill_tools.keys()) == {"ci", "big"}
    assert reg.get("ci") is skill_tools["ci"]
    assert reg.get("big") is skill_tools["big"]


def test_register_skill_tools_skips_non_transparent():
    """非透明 defn(agent)不在 caps→skill_tool 映射内(它们走 register_agent_tools)。"""
    from birdcode.agents.registry import AgentRegistry

    reg = ToolRegistry()
    caps = AgentRegistry()
    caps.register(
        AgentDefinition(
            name="explore",
            description="d",
            system_prompt="",
            mode="inline",
            is_transparent=False,
            body="b",
        )
    )
    skill_tools = register_skill_tools(
        reg,
        caps,
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        progress_cb=None,
        subagent_mgr=None,
    )
    assert skill_tools == {}
    assert reg.get("explore") is None


def test_skill_input_models_field_default():
    assert InlineSkillInput().args == ""
    assert ForkSkillInput().args == ""


# ---------------------------------------------------------------------------
# Fix I1 — skill 不得静默覆盖核心工具
# ---------------------------------------------------------------------------


class _StubCoreTool(Tool):
    """最小核心工具占位:is_agent_tool 未设(走 getattr 默认 False),有 name 即可。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "core"

    async def execute(self, **kwargs: object) -> str:  # pragma: no cover - 不会被调
        return "core"


def _deps_kwargs():
    return dict(
        cfg=None,
        app=None,
        ctx=None,
        project_root=None,
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        progress_cb=None,
        subagent_mgr=None,
    )


def test_register_skill_tools_core_tool_collision_raises():
    """skill 与同名核心工具 → CapabilityConflictError(I1 fail-fast)。"""
    from birdcode.agents.registry import AgentRegistry

    reg = ToolRegistry()
    reg.register(_StubCoreTool("read_file"))  # 核心 tool,is_agent_tool 缺省 False
    caps = AgentRegistry()
    caps.register(
        AgentDefinition(
            name="read_file",
            description="d",
            system_prompt="",
            mode="inline",
            is_transparent=True,
            body="b",
        )
    )
    with pytest.raises(CapabilityConflictError):
        register_skill_tools(reg, caps, **_deps_kwargs())


def test_register_skill_tools_ask_user_collision_raises():
    """#3:核心工具(含 ask_user/tool_search)先注册后,同名 skill 被 I1 守卫拦下。

    钉住 on_mount 重排的意图——ask_user/ToolSearchTool 现先于 capability 注册,故名为
    ask_user 的 skill 不会再绕过守卫、随后被 AskUserTool 静默覆盖(否则斜杠命令指向 skill、
    模型工具却是真 AskUserTool)。AskUserTool.is_agent_tool 未设 → 视作核心工具 → raise。
    """
    from birdcode.agents.registry import AgentRegistry
    from birdcode.tools.ask_user import AskUserTool

    reg = ToolRegistry()
    reg.register(AskUserTool(app=None))  # 核心 tool 先注册(模拟 on_mount 重排后顺序)
    caps = AgentRegistry()
    caps.register(
        AgentDefinition(
            name="ask_user",
            description="d",
            system_prompt="",
            mode="inline",
            is_transparent=True,
            body="b",
        )
    )
    with pytest.raises(CapabilityConflictError):
        register_skill_tools(reg, caps, **_deps_kwargs())


def test_register_skill_tools_no_collision_registers():
    """正例:无核心工具同名 → 正常注册(inline + fork 都 OK)。"""
    from birdcode.agents.registry import AgentRegistry

    reg = ToolRegistry()
    caps = AgentRegistry()
    caps.register(
        AgentDefinition(
            name="ci",
            description="d",
            system_prompt="",
            mode="inline",
            is_transparent=True,
            body="b",
        )
    )
    caps.register(
        AgentDefinition(
            name="big",
            description="d",
            system_prompt="",
            mode="fork",
            is_transparent=True,
            body="b",
        )
    )
    skill_tools = register_skill_tools(reg, caps, **_deps_kwargs())
    assert set(skill_tools.keys()) == {"ci", "big"}
    assert reg.get("ci") is skill_tools["ci"]
    assert reg.get("big") is skill_tools["big"]


def test_register_skill_tools_capability_overlap_allowed():
    """能力工具(agent tool / fork-skill / inline-skill)间允许层覆盖,不触发 I1。
    预置一个 inline-skill(是 _InlineSkillTool)再注册同名 inline → 不 raise。"""
    from birdcode.agents.registry import AgentRegistry

    reg = ToolRegistry()
    reg.register(
        _InlineSkillTool(
            defn=AgentDefinition(
                name="ci",
                description="d",
                system_prompt="",
                mode="inline",
                is_transparent=True,
                body="old",
            )
        )
    )
    caps = AgentRegistry()
    caps.register(
        AgentDefinition(
            name="ci",
            description="d",
            system_prompt="",
            mode="inline",
            is_transparent=True,
            body="new",
        )
    )
    skill_tools = register_skill_tools(reg, caps, **_deps_kwargs())
    assert "ci" in skill_tools


# ---------------------------------------------------------------------------
# Fix I2 — _ForkSkillTool.invoke_async 直接单测(slash→fork 启动路径)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_skill_invoke_async_launches_and_returns_ack(monkeypatch):
    """slash 路径:invoke_async 渲染 body 作 prompt,恒异步启动,回 ack_text。
    断言传入 launch_async 的 runner 字段(prompt/is_async/tool_use_id)正确。

    用 monkeypatch 换掉 SubagentRunner(真 __init__ 会访问 ctx.session_id 建 jsonl,
    与本测试无关),FakeRunner 仅存字段——launch_async 拿到的就是它,无需运行。
    """
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kw: object) -> None:
            captured.update(kw)
            # 仿真实 SubagentRunner:把构造参数也存为实例属性,
            # 这样 launch_async 收到的 runner 可直接读 .prompt/.is_async/.tool_use_id。
            for k, v in kw.items():
                setattr(self, k, v)

    class _StubMgr:
        async def launch_async(self, runner):
            captured["runner"] = runner
            return types.SimpleNamespace(ack_text="launched-ok")

    monkeypatch.setattr("birdcode.tools.agent_tools.SubagentRunner", FakeRunner)
    defn = AgentDefinition(
        name="big",
        description="d",
        system_prompt="",
        mode="fork",
        is_transparent=True,
        body="refactor $ARGUMENTS",
    )
    tool = _ForkSkillTool(defn=defn, **{**_deps_kwargs(), "subagent_mgr": _StubMgr()})

    ack = await tool.invoke_async("module X")
    assert ack == "launched-ok"
    runner = captured["runner"]
    assert captured["prompt"] == "refactor module X"  # body 渲染了 $ARGUMENTS
    assert runner.prompt == "refactor module X"
    assert runner.is_async is True
    assert runner.tool_use_id == "(skill-cmd)"


@pytest.mark.asyncio
async def test_fork_skill_invoke_async_no_manager_returns_error_string():
    """无 SubagentManager(Phase2 未启用)→ 返回错误串,不抛(防崩溃)。"""
    defn = AgentDefinition(
        name="big",
        description="d",
        system_prompt="",
        mode="fork",
        is_transparent=True,
        body="refactor $ARGUMENTS",
    )
    tool = _ForkSkillTool(defn=defn, **_deps_kwargs())  # subagent_mgr=None

    result = await tool.invoke_async("module X")
    assert isinstance(result, str)
    assert "未启用" in result
