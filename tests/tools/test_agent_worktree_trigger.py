# tests/tools/test_agent_worktree_trigger.py
"""Task 6:AgentToolInput 加 isolation/run_in_background + execute force-async 触发点单测。

验证 _AgentTool.execute 的路由逻辑:
- isolation='worktree' → 强制异步分支(即使 defn.run_in_background=False),并透传 isolation。
- run_in_background=True(call 参数)→ 覆盖 defn.run_in_background=False,走异步,isolation=None。
- AgentToolInput 默认 isolation=None / run_in_background=None。

用 FakeRunner(monkeypatch SubagentRunner)隔离,只测 execute 的路由/透传,不测 SubagentRunner 内部。
"""
from birdcode.tools.agent_tools import AgentToolInput


def _sync_defn():
    """构造一个同步 defn(run_in_background=False),字段取自 AgentDefinition 真实 ctor。"""
    from birdcode.agents.definition import AgentDefinition

    return AgentDefinition(
        name="t", description="t", system_prompt="", body="",
        mode="fork", model="", disallowed_tools=(), kind="write",
        parallel_safe=False, is_transparent=False, run_in_background=False,
    )


class _FakeRunner:
    """记录构造参数(kwargs),暴露 async 分支读取的 agent_id / sidechain_path。"""

    def __init__(self, **kw):
        self.kwargs = kw

    @property
    def agent_id(self):
        return "sub-x"

    @property
    def sidechain_path(self):
        return "/tmp/x.jsonl"

    async def run(self):  # sync 分支不会走到(两用例均强制异步)
        ...


class _CapturingMgr:
    """记录 launch_async 收到的 runner,供断言其构造 kwargs。"""

    def __init__(self):
        self.launched_runner = None

    async def launch_async(self, runner):
        self.launched_runner = runner
        return type("A", (), {"ack_text": "ack", "agent_id": "sub-x"})()


def _tool(mgr):
    from birdcode.tools.agent_tools import _AgentTool

    return _AgentTool(
        defn=_sync_defn(), cfg=None, app=None, ctx=None, project_root=None,
        parent_provider=None, parent_registry=None, parent_gate=None,
        spawn_depth=0, subagent_mgr=mgr,
    )


def test_agent_input_has_isolation_and_run_in_background():
    """AgentToolInput 新增 isolation / run_in_background 字段(默认 None)。"""
    inp = AgentToolInput(prompt="do x")
    assert inp.isolation is None
    assert inp.run_in_background is None


def test_agent_input_accepts_worktree_isolation():
    inp = AgentToolInput(prompt="do x", isolation="worktree", run_in_background=True)
    assert inp.isolation == "worktree"
    assert inp.run_in_background is True


async def test_execute_isolation_worktree_forces_async(monkeypatch):
    """isolation='worktree' → 走异步分支(即使 defn.run_in_background=False),透传 isolation。"""
    monkeypatch.setattr("birdcode.tools.agent_tools.SubagentRunner", _FakeRunner)
    mgr = _CapturingMgr()
    tool = _tool(mgr)
    await tool.execute(prompt="x", isolation="worktree")  # 强制异步
    runner = mgr.launched_runner
    assert runner is not None
    assert runner.kwargs["is_async"] is True
    assert runner.kwargs["isolation"] == "worktree"


async def test_execute_run_in_background_override(monkeypatch):
    """run_in_background=True(call 参数)→ 异步,覆盖 defn.run_in_background=False;isolation=None。"""
    monkeypatch.setattr("birdcode.tools.agent_tools.SubagentRunner", _FakeRunner)
    mgr = _CapturingMgr()
    tool = _tool(mgr)
    await tool.execute(prompt="x", run_in_background=True)  # call 参数覆盖 defn=False
    runner = mgr.launched_runner
    assert runner is not None
    assert runner.kwargs["is_async"] is True
    assert runner.kwargs["isolation"] is None
