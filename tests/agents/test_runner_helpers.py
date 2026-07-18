# tests/agents/test_runner_helpers.py
import pytest

from birdcode.agents.runner import MAX_SPAWN_DEPTH, build_child_registry, resolve_profile
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.tools.echo import EchoTool
from birdcode.tools.registry import ToolRegistry


def _parent_registry_with_agent():
    r = ToolRegistry()
    r.register(EchoTool())
    # 模拟 agent tool:is_agent_tool=True 标记(build_child_registry 据此排除,非按名)。
    # 名字任意——证明排除靠标记不靠名字(自定义 agent 名也排除)。
    class _AgentShim(EchoTool):
        name = "general-purpose"
        is_agent_tool = True
    r.register(_AgentShim())
    return r


def test_child_registry_excludes_agent_and_disallowed():
    parent = _parent_registry_with_agent()
    child = build_child_registry(parent, disallowed=("echo",))
    names = set(child.names())
    assert "general-purpose" not in names  # I3:agent tool(is_agent_tool 标记)排除
    assert "echo" not in names  # 黑名单


def test_child_registry_keeps_others():
    parent = _parent_registry_with_agent()
    child = build_child_registry(parent, disallowed=())
    # 父里只有 echo + agent tool shim;排除 agent tool 后剩 echo
    assert "echo" in child.names()
    assert "general-purpose" not in child.names()


def test_child_registry_excludes_session_scoped_read_history():
    """read_history(session_scoped,持主线 jsonl)不进子工具表 —— 否则子 agent 读主 agent 历史。"""
    from birdcode.tools.executor import default_registry

    parent = default_registry()  # 含 read_history
    child = build_child_registry(parent, disallowed=())
    names = set(child.names())
    assert "read_history" not in names  # 排除:不共享主线会话状态
    assert "read_file" in names  # 普通 read 保留


def test_child_registry_excludes_session_scoped_tool_search():
    """tool_search(session_scoped)不进子表 —— 否则子 agent mark_discovered 改父 registry。"""
    from birdcode.mcp.tool_search import ToolSearchTool

    parent = ToolRegistry()
    parent.register(EchoTool())
    parent.register(ToolSearchTool(registry=parent))
    child = build_child_registry(parent, disallowed=())
    names = set(child.names())
    assert "echo" in names
    assert "tool_search" not in names


def test_child_registry_excludes_session_scoped_mcp_adapter():
    """McpToolAdapter(should_defer,仅经 tool_search 加载)不进子表。

    子 agent 无 tool_search;遗留它会让 deferred_reminder 宣称「可用 tool_search 拉 MCP 工具」,
    而子 agent 既无 tool_search、deferred MCP 工具也永不进其工具列表 → 空耗一轮。
    """
    from birdcode.mcp.adapter import McpToolAdapter

    class _FakeMgr:
        pass

    parent = ToolRegistry()
    parent.register(EchoTool())
    parent.register(McpToolAdapter("srv", "tool", "desc", None, _FakeMgr()))  # type: ignore[arg-type]
    child = build_child_registry(parent, disallowed=())
    names = set(child.names())
    assert "echo" in names
    assert "mcp__srv__tool" not in names  # session_scoped → 排除


def test_child_registry_forks_bash_isolating_cwd():
    """bash 经 fork_for_child 派生全新实例:快照父当前 cwd(继承主 agent 工作目录)、隔离后续改动。"""
    from birdcode.tools.bash_tool import BashTool
    from birdcode.tools.executor import default_registry

    parent = default_registry()
    parent_bash = parent.get("bash")
    assert isinstance(parent_bash, BashTool)
    # 模拟主 agent `cd` 切换工作目录(进程 cwd 不变,BashTool._cwd 变)
    parent_bash._cwd = "/fake/parent/cwd"  # noqa: SLF001
    child = build_child_registry(parent, disallowed=())
    child_bash = child.get("bash")
    assert child_bash is not None
    assert child_bash is not parent_bash  # 全新实例(非共享)
    # 继承父当前 cwd(非进程 os.getcwd())
    assert child_bash._cwd == "/fake/parent/cwd"  # noqa: SLF001
    # 隔离:改子实例 cwd,父实例不受影响
    child_bash._cwd = "/changed/by/child"  # noqa: SLF001
    assert parent_bash._cwd == "/fake/parent/cwd"  # noqa: SLF001


def test_resolve_profile_override_wins():
    cfg = AppConfig(
        providers={
            "default": ProviderProfile(name="default", protocol="anthropic", model="big",
                                       base_url="http://x", api_key="k"),
            "haiku": ProviderProfile(name="haiku", protocol="anthropic", model="small",
                                     base_url="http://x", api_key="k"),
        },
        default="default",
    )

    class _FakeParent:
        profile = cfg.providers["default"]

    assert resolve_profile("", "haiku", cfg, _FakeParent()).model == "small"


def test_resolve_profile_inherit_when_empty():
    cfg = AppConfig(
        providers={"default": ProviderProfile(name="default", protocol="anthropic", model="big",
                                              base_url="http://x", api_key="k")},
        default="default",
    )

    class _FakeParent:
        profile = cfg.providers["default"]

    assert resolve_profile("", "", cfg, _FakeParent()).model == "big"


def test_resolve_profile_unknown_raises():
    cfg = AppConfig(
        providers={"default": ProviderProfile(name="default", protocol="anthropic", model="big",
                                              base_url="http://x", api_key="k")},
        default="default",
    )

    class _FakeParent:
        profile = cfg.providers["default"]

    with pytest.raises(ValueError):
        resolve_profile("", "nope", cfg, _FakeParent())


def test_max_spawn_depth_is_one():
    assert MAX_SPAWN_DEPTH == 1


def test_update_meta_freezes_lost_status(tmp_path):
    """_update_meta 守卫:status==lost 时不再被生命周期写覆盖(防丢弃的 agent 复活)。

    防 review Finding 7:用户 2/No 已落 lost,但 async 子 agent 的 cancel 落盘晚于此时,
    runner except CancelledError 的 _update_meta(status="cancelled") 若无守卫会覆盖 lost →
    cancelled ∈ RESUMABLE_STATUSES → 跨会话再弹,违背"永久丢弃"。
    """
    from birdcode.agents.runner import _update_meta
    from birdcode.session.models import SubagentMeta
    from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta

    p = tmp_path / "agent-sub.meta.json"
    write_subagent_meta(
        p,
        SubagentMeta(
            agent_id="sub",
            agent_type="general-purpose",
            tool_use_id="tu",
            status="lost",
        ),
    )
    # 模拟 runner 的生命周期写(cancel/complete)落到已 lost 的 meta 上
    _update_meta(p, status="cancelled")
    back = read_subagent_meta(p)
    assert back is not None
    assert back.status == "lost"  # 守卫:未被 cancelled 覆盖


def test_child_registry_async_replaces_ask_user_with_stub():
    """is_async=True:ask_user(is_ask_user)→AskUserAsyncStub(不 fork 原 tool);同步保留原 tool。"""
    from birdcode.tools.ask_user import AskUserAsyncStub, AskUserTool

    class _App:
        pass

    parent = ToolRegistry()
    parent.register(EchoTool())
    parent.register(AskUserTool(app=_App()))
    # 同步:保留原 AskUserTool(可交互)
    sync_child = build_child_registry(parent, disallowed=())
    sync_tool = sync_child.get("ask_user")
    assert sync_tool is not None and not isinstance(sync_tool, AskUserAsyncStub)
    # 异步:换成 stub(execute 返回拒)
    async_child = build_child_registry(parent, disallowed=(), is_async=True)
    assert isinstance(async_child.get("ask_user"), AskUserAsyncStub)
    # 其他工具不受影响
    assert "echo" in async_child.names()
