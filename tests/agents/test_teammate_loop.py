# tests/agents/test_teammate_loop.py
"""agent teams:长驻 teammate 循环(history 跨轮累积 + mailbox park/wake)。

teammate 跑第 1 轮→park→mailbox 唤醒跑第 2 轮;第 2 轮 history 含第 1 轮。
预载 mailbox=[第2轮 prompt, SHUTDOWN] → run() 确定性跑完(无需后台 task/时序)。
"""
import asyncio
import json
import shutil
import subprocess

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage, ToolCallStart
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.mailbox import MailboxMessage
from birdcode.agents.runner import _TEAMMATE_SHUTDOWN, SubagentRunner
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.session.models import SessionContext
from birdcode.session.paths import subagent_meta_path
from birdcode.session.subagent_meta import read_subagent_meta

_HAS_GIT = shutil.which("git") is not None


def _git_init(repo):  # type: ignore[no-untyped-def]
    """init git 仓库 + 初始 commit src/foo.py(给 worktree 基线 + src/ 目录)。

    teammate 强制 worktree 隔离 → 需 git 仓库;镜像 test_agent_worktree_e2e._init_repo。
    """
    for c in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=repo, check=True, capture_output=True)
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "foo.py").write_text("# original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)


class _HistoryRecordingProvider:
    """每次 stream 快照记录收到的 history(list[Turn] 只读先验上下文),yield 文本+Done。

    快照(list(...))而非存引用:run_agent_loop 传同一 history list 对象,调用方在轮间
    append → 存引用会让所有记录指向最终态。快照锁定每轮调用时刻的 history。
    """

    def __init__(self, profile):
        self.profile = profile
        self.histories: list[list] = []

    def stream(self, messages, *, history):  # noqa: ARG002
        self.histories.append(list(history))

        async def _gen():
            yield TextDelta(text="ok")
            yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn")

        return _gen()


def _cfg():
    return AppConfig(
        providers={"p": ProviderProfile(name="p", protocol="anthropic", model="m",
                                        base_url="http://x", api_key="k")},
        default="p",
    )


def _ctx():
    return SessionContext(session_id="s1", cwd=".", version="", git_branch=None)


def _defn():
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


async def test_teammate_loop_history_continuity(tmp_path, monkeypatch):
    """轮1(history=[])→park→取"第二轮 prompt"→轮2(history=[轮1])→park→取 SHUTDOWN→终止。"""
    from birdcode.agents import runner

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(MailboxMessage(sender="test", to="bob", content="第二轮 prompt"))
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)

    r = SubagentRunner(
        defn=_defn(), prompt="第一轮 prompt", description="d", tool_use_id="call_t1",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    report = await r.run()

    provider = captured["provider"]
    assert len(provider.histories) == 2  # 两轮
    assert provider.histories[0] == []  # 轮 1:history 空
    turn1_history = provider.histories[1]
    assert len(turn1_history) == 1  # 轮 2:history 含轮 1
    first_turn = turn1_history[0]
    user_texts = [
        getattr(b, "text", "")
        for m in first_turn.messages if m.role == "user"
        for b in m.content
    ]
    assert any("第一轮" in t for t in user_texts)
    assert report.status == "completed"  # SHUTDOWN → 干净终止


async def test_teammate_loop_transcript_accumulates_across_turns(tmp_path, monkeypatch):
    """transcript jsonl 跨轮累积——两轮 user 种子 + assistant 进同一文件。

    store 在 try 内建、finally 只在循环结束后关一次(非每轮关)→ 轮间保持打开;_on_message
    每条 assistant 落盘 + 循环里每轮 store.append(种子) → 两轮消息进同一 sidechain。读
    post-run 文件断言两轮 prompt 都在 + 行数 ≥ 4(2 user 种子 + 2 assistant)。
    注:teammate 专属路径 <sid>/teammates/{name}/ 随命名/TeamManager 后续引入;目前
    复用 subagent sidechain path(r.sidechain_path)验证累积行为本身。
    """
    from birdcode.agents import runner

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(MailboxMessage(sender="test", to="bob", content="第二轮 prompt"))
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)

    r = SubagentRunner(
        defn=_defn(), prompt="第一轮 prompt", description="d", tool_use_id="call_t2",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    await r.run()

    sidechain = r.sidechain_path
    assert sidechain.exists()
    text = sidechain.read_text(encoding="utf-8")
    assert "第一轮 prompt" in text  # 轮 1 user 种子
    assert "第二轮 prompt" in text  # 轮 2 user 种子(证明跨轮累积、同一文件)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 4  # 2 user 种子 + 2 assistant


async def test_teammate_loop_meta_parks_to_idle(tmp_path, monkeypatch):
    """teammate park→meta=idle(需 SubagentMeta.status Literal 含 idle)。

    此前 idle 不在 Literal → _update_meta 校验失败被静默吞 → meta 卡 running、永不 idle。
    本测试后台起 run()(mailbox 空 → park 在 await get 阻塞),轮询 meta 直到 idle,证明
    Literal 修复后 idle 真持久化。finally 投 SHUTDOWN 让 task 收尾,免悬挂。
    """
    from birdcode.agents import runner

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()  # 空:park 在 await get() 阻塞
    r = SubagentRunner(
        defn=_defn(), prompt="第一轮", description="d", tool_use_id="call_t3",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id)

    task = asyncio.create_task(r.run())
    try:
        observed = False
        for _ in range(100):  # 最多 ~1s
            m = read_subagent_meta(meta_path)
            if m is not None and m.status == "idle":
                observed = True
                break
            await asyncio.sleep(0.01)
        assert observed, "teammate 未在 1s 内 park 到 idle(Literal idle 修复可能未生效)"
    finally:
        mailbox.put_nowait(_TEAMMATE_SHUTDOWN)  # 让 task 终止,避免悬挂
        await task


async def _wait_status(meta_path, status, rounds=100):
    """轮询 SubagentMeta 直到 status(默认最多 ~1s)。观测 teammate 瞬态(idle/running)。"""
    for _ in range(rounds):
        m = read_subagent_meta(meta_path)
        if m is not None and m.status == status:
            return True
        await asyncio.sleep(0.01)
    return False


async def test_teammate_spawn_demo_cancel_clean(tmp_path, monkeypatch):
    """端到端:spawn 'bob'(只读)→park→poke→续跑→cancel→干净停。

    内部 spawn 入口(spawn_teammate)+ handle(send/cancel)。cancel 走 runner 的 CancelledError
    分支:_update_meta(cancelled)+store.close()+raise。断言 cancel 后 meta=cancelled、
    transcript 落盘。spawn_teammate 来自 birdcode.agents.teammate(最小入口;后续加 TeamManager)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import spawn_teammate

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    r = SubagentRunner(
        defn=_defn(), prompt="第一轮", description="d", tool_use_id="call_t4",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id)

    handle = await spawn_teammate(name="bob", runner=r)
    assert handle.name == "bob"
    try:
        assert await _wait_status(meta_path, "idle")  # 轮1 跑完→park
        await handle.send("第二轮")  # poke:唤醒续跑
        for _ in range(100):  # 等轮2 跑完(provider 记 2 次 history)
            if len(captured["provider"].histories) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(captured["provider"].histories) >= 2  # 轮2 真跑过(续跑成功)
        handle.cancel()  # Esc 风格取消
        with pytest.raises(asyncio.CancelledError):
            await handle.join()
    finally:
        if not handle.done():  # 早失败时收尾,免悬挂
            handle.cancel()
            try:
                await handle.join()
            except asyncio.CancelledError:
                pass
    m = read_subagent_meta(meta_path)
    assert m is not None and m.status == "cancelled"
    assert r.sidechain_path.exists()


async def test_teammate_loop_message_shows_sender(tmp_path, monkeypatch):
    """MailboxMessage 投递 → recipient 第 2 轮 Turn 含'[来自 sender]:'前缀 + content。"""
    from birdcode.agents import runner

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(MailboxMessage(sender="alice", to="bob", content="hi 从 alice"))
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)

    r = SubagentRunner(
        defn=_defn(), prompt="第一轮", description="d", tool_use_id="call_p1t1",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    await r.run()

    text = r.sidechain_path.read_text(encoding="utf-8")
    assert "[来自 alice]" in text  # sender 前缀
    assert "hi 从 alice" in text  # content


async def test_team_manager_registry_and_lifecycle(tmp_path, monkeypatch):
    """TeamManager register/resolve/unregister + spawn 自动注册 + task 结束自动注销。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    captured: dict = {}

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        captured["provider"] = _HistoryRecordingProvider(profile=profile)
        return captured["provider"]

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    # 纯 registry CRUD
    team.register("lead", lambda _m: None)
    assert team.resolve("lead") is not None
    assert "lead" in team.names()
    team.unregister("lead")
    assert team.resolve("lead") is None

    # spawn 注册 + task 结束自动注销
    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)  # bob 跑完轮1→park→SHUTDOWN→终止
    r = SubagentRunner(
        defn=_defn(), prompt="第一轮", description="d", tool_use_id="call_p1t2",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    handle = await team.spawn("bob", r)
    assert team.resolve("bob") is not None  # spawn 即注册
    await handle.join()  # 等 bob 终止
    for _ in range(100):  # done_callback 异步触发 → 轮询注销
        if team.resolve("bob") is None:
            break
        await asyncio.sleep(0.01)
    assert team.resolve("bob") is None  # 自动注销


async def test_send_message_routes_and_sender_from_contextvar():
    """SendMessage 路由(teammate→mailbox / →lead.receive)+ sender 取自 _AGENT_NAME。"""
    from birdcode.agents.mailbox import _AGENT_NAME
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.send_message import SendMessageTool

    team = TeamManager()
    bob_mailbox: asyncio.Queue = asyncio.Queue()
    team.register("bob", bob_mailbox.put_nowait)
    lead_received: list = []
    team.register("lead", lambda m: lead_received.append(m))

    tool = SendMessageTool(team)
    # teammate 方向(default sender="lead")
    out = await tool.execute(to="bob", content="hi", summary="greet")
    assert "已发送给 bob" in out
    delivered = bob_mailbox.get_nowait()
    assert delivered.sender == "lead" and delivered.content == "hi" and delivered.summary == "greet"
    # lead 方向
    await tool.execute(to="lead", content="done")
    assert lead_received and lead_received[0].content == "done"
    # sender 取自 contextvar(set "alice")
    _AGENT_NAME.set("alice")
    await tool.execute(to="bob", content="from alice")
    delivered2 = bob_mailbox.get_nowait()
    assert delivered2.sender == "alice"
    # 未知 recipient
    out3 = await tool.execute(to="nobody", content="x")
    assert "未知" in out3


class _ToolCallsProvider:
    """轮1:发若干 tool call(name+args_dict)+ Done(tool_use);轮2:text+Done。

    驱动 teammate 经完整 executor 路径调任意工具(SendMessage / task 工具)。
    """

    def __init__(self, profile, calls):
        self.profile = profile
        self.calls = list(calls)  # [(tool_name, args_dict), ...]
        self._round = 0

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            if self._round == 0:
                self._round += 1
                for i, (name, args) in enumerate(self.calls):
                    yield ToolCallStart(id=f"tc{i}", name=name, args=json.dumps(args))
                yield Done(
                    usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="tool_use"
                )
            else:
                yield TextDelta(text="ok")
                yield Done(
                    usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn"
                )

        return _gen()


async def test_send_message_integration_via_tool(tmp_path, monkeypatch):
    """端到端:teammate(alice)经 SendMessage 工具发给 bob + lead。

    alice provider 发 SendMessage tool call → executor 执行 → TeamManager.resolve 路由 →
    bob mailbox / lead.receive 收到 MailboxMessage(sender="alice",_run_named 设的 _AGENT_NAME)。
    证明工具经完整 agent 执行路径(provider→executor→tool→deliver)投递成功。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.conversation import TurnController
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.send_message import SendMessageTool

    team = TeamManager()
    bob_mailbox: asyncio.Queue = asyncio.Queue()
    team.register("bob", bob_mailbox.put_nowait)

    async def _on_event(_e): ...
    async def _on_status(): ...
    lead_ctrl = TurnController(
        _HistoryRecordingProvider(profile=_cfg().providers["p"]),
        on_event=_on_event, on_status=_on_status,
    )
    team.register("lead", lead_ctrl.receive)

    parent_reg = ToolRegistry()  # alice 经 build_child_registry 继承 SendMessage
    parent_reg.register(SendMessageTool(team))

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _ToolCallsProvider(profile, [
            ("SendMessage", {"to": "bob", "content": "hi"}),
            ("SendMessage", {"to": "lead", "content": "done"}),
        ])

    monkeypatch.setattr(runner, "build_provider", _builder)

    alice_mailbox: asyncio.Queue = asyncio.Queue()
    alice_mailbox.put_nowait(_TEAMMATE_SHUTDOWN)  # alice 跑完轮1(2 SendMessage)→park→SHUTDOWN→终止
    r = SubagentRunner(
        defn=_defn(), prompt="alice 起步", description="d", tool_use_id="call_p1t5",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=parent_reg, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=alice_mailbox,
    )
    handle = await team.spawn("alice", r)
    await handle.join()

    # bob mailbox 收到 alice 的消息
    delivered_bob = bob_mailbox.get_nowait()
    assert delivered_bob.sender == "alice" and delivered_bob.content == "hi"
    # lead.receive 收到 → lead 后台跑一轮(轮询 history)
    for _ in range(100):
        if lead_ctrl.history:
            break
        await asyncio.sleep(0.01)
    assert lead_ctrl.history
    lead_text = lead_ctrl.history[-1].messages[0].content[0].text
    assert "[来自 alice]" in lead_text and "done" in lead_text


async def test_task_tools_integration_via_executor(tmp_path, monkeypatch):
    """端到端:teammate 经 task 工具(完整 executor 路径)TaskCreate+TaskUpdate,看板反映。

    bob 的 provider 发 TaskCreate+TaskUpdate tool call → executor 执行 → 看板 #1 =
    completed / assignee=bob / created_by=bob(_run_named 设的 _AGENT_NAME)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.task_tools import TaskCreateTool, TaskUpdateTool

    team = TeamManager()
    parent_reg = ToolRegistry()  # task 工具 → teammate 经 build_child_registry 继承
    parent_reg.register(TaskCreateTool(team))
    parent_reg.register(TaskUpdateTool(team))

    calls = [
        ("TaskCreate", {"title": "review auth", "assignee": "bob"}),
        ("TaskUpdate", {"id": "1", "status": "completed"}),
    ]

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _ToolCallsProvider(profile, calls)

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)  # bob 跑完轮1(2 tool call)→park→SHUTDOWN→终止
    r = SubagentRunner(
        defn=_defn(), prompt="bob 起步", description="d", tool_use_id="call_p2t3",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=parent_reg, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    handle = await team.spawn("bob", r)
    await handle.join()

    t = team.task_board.get("1")
    assert t is not None
    assert t.title == "review auth" and t.status == "completed" and t.assignee == "bob"
    assert t.created_by == "bob"  # teammate 的 _AGENT_NAME(_run_named 设)


async def test_spawn_name_reuse_does_not_orphan_second(tmp_path, monkeypatch):
    """#5 回归:两个 teammate 复用同名,先结束者不应注销仍存活的后注册者。

    旧 done-callback `lambda _t: unregister(name)` 按名删:bob1 先结束 → 抹掉 name→deliver,
    但此时 name 已被 bob2(后注册)覆盖 → bob1 误删 bob2(仍存活)→ bob2 静默孤立(消息再也投不到)。
    修复后按 deliver 身份校验:仅当 name 仍指向自己的 deliver 才注销。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()

    # bob1:mailbox 预置 SHUTDOWN → 跑完轮1 即终止(先结束)
    mb1: asyncio.Queue = asyncio.Queue()
    mb1.put_nowait(_TEAMMATE_SHUTDOWN)
    r1 = SubagentRunner(
        defn=_defn(), prompt="bob1", description="d", tool_use_id="call_reuse_1",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb1,
    )
    h1 = await team.spawn("bob", r1)

    # bob2:同名,后注册(覆盖 name→deliver)。mailbox 空 → park(存活)。
    mb2: asyncio.Queue = asyncio.Queue()
    r2 = SubagentRunner(
        defn=_defn(), prompt="bob2", description="d", tool_use_id="call_reuse_2",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb2,
    )
    h2 = await team.spawn("bob", r2)

    await h1.join()  # bob1 跑完轮1 → SHUTDOWN → 终止
    await asyncio.sleep(0.05)  # done-callback 在 task done 后异步触发,让一拍执行

    # bob2 仍存活、deliver 仍注册且功能正常(投递落到 mb2)。
    deliver = team.resolve("bob")
    assert deliver is not None, "bob2 的 deliver 被 bob1 的 done-callback 误删(静默孤立)"
    deliver(MailboxMessage(sender="x", to="bob", content="ping"))
    assert mb2.get_nowait().content == "ping"

    # 清理 bob2(免悬挂)
    mb2.put_nowait(_TEAMMATE_SHUTDOWN)
    await h2.join()


async def test_pull_claim_complete_e2e(tmp_path, monkeypatch):
    """T5/P3 端到端:teammate bob 经 pull 流程 List(pending)→Claim→Update(completed)。

    lead 预建 pending unassigned 任务;bob 经完整 executor 路径发三个 tool call:
    TaskList(pending) 看可抢的 → TaskClaim 原子领取 → TaskUpdate 标完成。看板最终
    #1 = completed + assignee=bob(claim 落 bob 名下)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.task_tools import TaskClaimTool, TaskListTool, TaskUpdateTool

    team = TeamManager()
    team.task_board.create(title="review auth", created_by="lead")  # #1 pending unassigned

    parent_reg = ToolRegistry()  # bob 经 build_child_registry 继承三个 task 工具
    parent_reg.register(TaskListTool(team))
    parent_reg.register(TaskClaimTool(team))
    parent_reg.register(TaskUpdateTool(team))

    calls = [
        ("TaskList", {"status": "pending"}),
        ("TaskClaim", {"id": "1"}),
        ("TaskUpdate", {"id": "1", "status": "completed"}),
    ]

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _ToolCallsProvider(profile, calls)

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)  # bob 跑完轮1(3 tool call)→park→SHUTDOWN→终止
    r = SubagentRunner(
        defn=_defn(), prompt="bob 起步", description="d", tool_use_id="call_p3e2e",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=parent_reg, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    handle = await team.spawn("bob", r)
    await handle.join()

    t = team.task_board.get("1")
    assert t is not None
    assert t.status == "completed"
    assert t.assignee == "bob"  # claim 把任务落到了 bob 名下(pull 成功)


async def test_dependency_unblock_claim_e2e(tmp_path, monkeypatch):
    """T5/P4 端到端:teammate bob 建依赖图 A→B、完成 A、B 自动解锁、claim B。

    bob 经完整 executor 路径发 4 个 tool call:TaskCreate(A) → TaskCreate(B,blocked_by=A)
    → TaskUpdate(A completed,B 自动解锁 pending) → TaskClaim(B)。最终 #2 = in_progress +
    assignee=bob(依赖图 + 自动解锁 + claim 尊重依赖全链路)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.registry import ToolRegistry
    from birdcode.tools.task_tools import TaskClaimTool, TaskCreateTool, TaskUpdateTool

    team = TeamManager()
    parent_reg = ToolRegistry()  # bob 经 build_child_registry 继承
    parent_reg.register(TaskCreateTool(team))
    parent_reg.register(TaskUpdateTool(team))
    parent_reg.register(TaskClaimTool(team))

    calls = [
        ("TaskCreate", {"title": "A"}),
        ("TaskCreate", {"title": "B", "blocked_by": ["1"]}),
        ("TaskUpdate", {"id": "1", "status": "completed"}),
        ("TaskClaim", {"id": "2"}),
    ]

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _ToolCallsProvider(profile, calls)

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()
    mailbox.put_nowait(_TEAMMATE_SHUTDOWN)
    r = SubagentRunner(
        defn=_defn(), prompt="bob 起步", description="d", tool_use_id="call_p4e2e",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=parent_reg, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    handle = await team.spawn("bob", r)
    await handle.join()

    t1 = team.task_board.get("1")
    t2 = team.task_board.get("2")
    assert t1.status == "completed"
    # B:被 claim 成功(依赖解锁后),状态 in_progress + 落 bob 名下
    assert t2.status == "in_progress" and t2.assignee == "bob"


async def test_team_manager_cancel_all(tmp_path, monkeypatch):
    """S1/#7:TeamManager 存 handle;cancel_all 批量硬取消所有 teammate。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    handles, metas = [], []
    for i, name in enumerate(["bob", "carol"]):
        mb: asyncio.Queue = asyncio.Queue()  # 空 → park
        r = SubagentRunner(
            defn=_defn(), prompt=name, description="d", tool_use_id=f"call_ca{i}",
            model_override="", spawn_depth=1, is_async=True,
            parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
            parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
            project_root=tmp_path, root=tmp_path, mailbox=mb,
        )
        metas.append(subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id))
        handles.append(await team.spawn(name, r))

    for mp in metas:
        assert await _wait_status(mp, "idle"), "teammate 未 park"

    team.cancel_all()
    for h in handles:
        with pytest.raises(asyncio.CancelledError):
            await h.join()


async def test_team_manager_shutdown_all(tmp_path, monkeypatch):
    """S1/#7:shutdown_all 批量干净终止(SHUTDOWN→completed,不抛)。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    handles, metas = [], []
    for i, name in enumerate(["dan", "eve"]):
        mb: asyncio.Queue = asyncio.Queue()
        r = SubagentRunner(
            defn=_defn(), prompt=name, description="d", tool_use_id=f"call_sd{i}",
            model_override="", spawn_depth=1, is_async=True,
            parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
            parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
            project_root=tmp_path, root=tmp_path, mailbox=mb,
        )
        metas.append(subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id))
        handles.append(await team.spawn(name, r))

    for mp in metas:
        assert await _wait_status(mp, "idle")

    team.shutdown_all()
    for h in handles:
        await h.join()  # SHUTDOWN 干净终止,不抛
    for mp in metas:
        m = read_subagent_meta(mp)
        assert m is not None and m.status == "completed"


async def test_teammate_passes_context_manager_one_shot_none(tmp_path, monkeypatch):
    """S3/#1:teammate(mailbox)传 ContextManager(防 history 跨轮累积爆);一次性(mailbox None)仍 None。

    teammate 长驻 history 跨轮累积 → O(n²) + ContextOverflow;接 ContextManager(store=None 纯
    内存)后 maybe_compact 压缩 + react 截断 overflow。一次性只一轮不爆,仍 None。
    """
    from birdcode.agents import runner

    captured: list = []
    real_run = runner.run_agent_loop

    async def _spy(**kw):
        captured.append(kw.get("context"))
        return await real_run(**kw)

    monkeypatch.setattr(runner, "run_agent_loop", _spy)

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    # teammate(mailbox 非空)→ context 非 None
    mb: asyncio.Queue = asyncio.Queue()
    mb.put_nowait(_TEAMMATE_SHUTDOWN)
    r = SubagentRunner(
        defn=_defn(), prompt="t", description="d", tool_use_id="call_s3t",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb,
    )
    await r.run()
    assert captured, "teammate 应至少调一次 run_agent_loop"
    assert captured[0] is not None, "teammate 应传 ContextManager(非 None)"

    # 一次性(mailbox None)→ context None
    captured.clear()
    r2 = SubagentRunner(
        defn=_defn(), prompt="o", description="d", tool_use_id="call_s3o",
        model_override="", spawn_depth=1, is_async=False,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path,
    )
    await r2.run()
    assert captured[0] is None, "一次性 subagent 应传 context=None(只一轮不爆)"


async def test_team_manager_reset(tmp_path, monkeypatch):
    """S5:reset() cancel 所有 teammate + 清 registry/board(/clear 用,team_mgr 引用不变)。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    handles = []
    for i, name in enumerate(["bob", "carol"]):
        mb: asyncio.Queue = asyncio.Queue()
        r = SubagentRunner(
            defn=_defn(), prompt=name, description="d", tool_use_id=f"call_rs{i}",
            model_override="", spawn_depth=1, is_async=True,
            parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
            parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
            project_root=tmp_path, root=tmp_path, mailbox=mb,
        )
        handles.append(await team.spawn(name, r))
    team.task_board.create(title="x")
    assert team.names() == ["bob", "carol"] and team._handles

    team.reset()

    assert team.names() == [] and not team._handles
    assert team.task_board.list() == []
    for h in handles:
        with pytest.raises(asyncio.CancelledError):
            await h.join()  # cancel 异步传播,join 等其终止


async def test_team_manager_join_all_waits_for_tasks(tmp_path):
    """F5 回归:cancel_all 后 join_all 把 teammate task 等到 done(其 finally 跑完 worktree cleanup)。

    on_unmount 用 join_all 让 runner.run() finally 内的 shielded cleanup 在 live loop 上跑完;
    不 await 则关 loop 砍断 cleanup → 留 git-locked worktree。此处验等待语义(isolation 默认
    None,不建真 worktree;cancel 在 task 首次调度前 pending → 不触达真实 LLM 调用):join_all
    返回时 handle 已 done,且 join_all 自身不抛(cancel 终态的 CancelledError 被吞)。
    """
    from birdcode.agents.teammate import TeamManager

    team = TeamManager()
    mb: asyncio.Queue = asyncio.Queue()
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_join",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb,
    )
    h = await team.spawn("bob", r)
    team.cancel_all()
    await team.join_all(timeout=2.0)  # 不抛(cancel 终态被吞)
    assert h.done()

    # 空 team → 立即返回(无 teammate 时退出路径不阻塞)
    await TeamManager().join_all(timeout=2.0)


def test_reset_preserves_lead_recipient():
    """F1 回归:reset() 保留 lead 注册(teammate→lead 通道在 /clear 后不断)。

    生产 App.on_mount 注册 lead=controller.receive;/clear 调 reset()。若 reset 随 registry
    一并清 lead,则 SendMessage(to=lead) 与 teammate 完成通知在 /clear 后全断(回归到 review
    #1 的死路)。lead deliver 跨 /clear 仍有效(同一 TurnController 实例,只重绑 _store)→ 保留。
    未注册过 lead 的 team,reset 后 names() 仍为空(test_team_manager_reset 已覆盖)。
    """
    from birdcode.agents.teammate import TeamManager

    team = TeamManager()
    received: list = []
    team.register("lead", lambda m: received.append(m))
    team.reset()
    deliver = team.resolve("lead")
    assert deliver is not None, "reset() 不应清掉 lead 注册"
    deliver(MailboxMessage(sender="bob", to="lead", content="hi"))
    assert received and received[0].content == "hi"


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_spawn_teammate_tool(tmp_path, monkeypatch):
    """S6:SpawnTeammateTool → 构造 mailbox runner + team_mgr.spawn;起 + 注册 + 重名拒 + 保留名拒。

    teammate 强制 worktree 隔离(F2)→ project_root 须为 git 仓库,故 _git_init(tmp_path)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.spawn_teammate_tool import SpawnTeammateTool

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)
    _git_init(tmp_path)  # F2:SpawnTeammate 需 git 仓库(worktree 隔离)

    team = TeamManager()
    tool = SpawnTeammateTool(
        team_mgr=team, defn=_defn(), cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, progress_cb=None,
    )
    out = await tool.execute(name="bob", prompt="做 X")
    assert "bob" in out and "bob" in team.names()

    # 重名 → 拒
    out2 = await tool.execute(name="bob", prompt="再次")
    assert "已存在" in out2

    # F12:保留名 stop/view → 拒(/agents 子命令首段冲突,经 /agents 永远寻不到)
    out3 = await tool.execute(name="stop", prompt="x")
    out4 = await tool.execute(name="view", prompt="y")
    assert "冲突" in out3 and "冲突" in out4
    assert "stop" not in team.names() and "view" not in team.names()

    # F11:'lead' 是系统保留名(主 session recipient 预注册占用)→ 拒,给清晰原因(非「已存在」)
    out5 = await tool.execute(name="lead", prompt="z")
    assert "保留名" in out5
    assert "lead" not in team.names()

    # #4:大小写不敏感保留名 + 入口 strip 空白(原裸 Field 让 'Lead'/' bob ' 滑过检查)。
    out_case = await tool.execute(name="Lead", prompt="x")  # 大小写撞 lead → 拒
    assert "保留名" in out_case and "Lead" not in team.names()
    out_ws = await tool.execute(name=" alice ", prompt="y")  # 首尾空白 → strip 后注册 "alice"
    assert "alice" in out_ws
    assert "alice" in team.names() and " alice " not in team.names()
    out_empty = await tool.execute(name="   ", prompt="z")  # 纯空白 → 拒
    assert "不能为空" in out_empty and not team.names().count(" alice ")

    # 清理:shutdown_all + 等 teammate 终止(worktree 经 finally 清理)
    team.shutdown_all()
    for h in list(team._handles.values()):
        await h.join()


async def test_spawn_teammate_requires_git(tmp_path, monkeypatch):
    """F2:project_root 非 git 仓库 → SpawnTeammate fast-fail(不 spawn 必崩的 teammate)。

    teammate 强制 worktree 隔离,create_worktree 需 .git;入口预检返友好错误。
    不跑 git,故无需 _HAS_GIT skip。
    """
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.spawn_teammate_tool import SpawnTeammateTool

    team = TeamManager()
    tool = SpawnTeammateTool(
        team_mgr=team, defn=_defn(), cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path,  # 裸 tmp_path,无 .git
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, progress_cb=None,
    )
    out = await tool.execute(name="bob", prompt="做 X")
    assert "git" in out and "错误" in out
    assert "bob" not in team.names()  # 未 spawn


@pytest.mark.skipif(not _HAS_GIT, reason="git not installed")
async def test_spawn_teammate_creates_isolated_worktrees(tmp_path, monkeypatch):
    """F2 修复回归:SpawnTeammate 强制 isolation=worktree → 每个 teammate 独立 worktree(互不串改)。

    旧实现 is_async 无 isolation → fork_async L5 拒所有非 bash 写 → 编码 teammate 废。修后两
    teammate 各起独立 worktree 目录(<repo>/.birdcode/worktrees/<agent_id>),路径互不相同。
    (「worktree 内写隔离、同名文件不串」由 test_agent_worktree_e2e 对 SubagentRunner 全覆盖,
    此处只验 SpawnTeammate 把 teammate 路由进了 worktree。)
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager
    from birdcode.tools.spawn_teammate_tool import SpawnTeammateTool

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)
    _git_init(tmp_path)

    team = TeamManager()
    tool = SpawnTeammateTool(
        team_mgr=team, defn=_defn(), cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, progress_cb=None,
    )
    # 清理放 finally(F12):中途断言失败也要 shutdown 两 teammate,免留 git-locked worktree。
    try:
        await tool.execute(name="alice", prompt="做 A")
        await tool.execute(name="bob", prompt="做 B")

        # 两 teammate 各自 worktree 异步创建 → 轮询直到 .birdcode/worktrees/ 下出现 2 个目录
        wt_root = tmp_path / ".birdcode" / "worktrees"
        for _ in range(100):
            if wt_root.exists() and len(list(wt_root.iterdir())) >= 2:
                break
            await asyncio.sleep(0.01)
        assert wt_root.exists(), "worktree 根目录未创建(isolation=worktree 未生效?)"
        names = sorted(p.name for p in wt_root.iterdir())
        assert len(names) == 2, f"两 teammate 应各有独立 worktree,实际: {names}"
        assert len(set(names)) == 2  # 两个不同 agent_id,互不串
    finally:
        # 清理:shutdown_all → teammate 终止 → finally 清 worktree
        team.shutdown_all()
        for h in list(team._handles.values()):
            await h.join()


async def test_team_manager_teammates_and_handle(tmp_path, monkeypatch):
    """S7:teammates()/handle() 公开接口;/agents UI 观察/打断/发消息用。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    mb: asyncio.Queue = asyncio.Queue()
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_th1",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb,
    )
    h = await team.spawn("bob", r)

    ts = team.teammates()
    assert len(ts) == 1 and ts[0][0] == "bob" and ts[0][1] is h
    assert team.handle("bob") is h
    assert team.handle("nobody") is None

    team.shutdown_all()
    await h.join()


async def test_handle_send_to_done_raises_shutdown_idempotent(tmp_path, monkeypatch):
    """#15:已结束 teammate send→raise(不静默丢);shutdown 幂等(不抛)。"""
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    mb: asyncio.Queue = asyncio.Queue()
    mb.put_nowait(_TEAMMATE_SHUTDOWN)  # 跑完首轮 → SHUTDOWN → completed
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_sd_done",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb,
    )
    h = await team.spawn("bob", r)
    await h.join()
    assert h.done()

    with pytest.raises(RuntimeError, match="已结束"):
        await h.send("post-mortem")
    h.shutdown()  # 幂等 no-op,不抛


async def test_teammate_park_stops_progress_tick(tmp_path, monkeypatch):
    """#9:teammate park 时 progress_tick 停(park 后 progress_cb 不再被调)。

    未修:tick 创建一次、仅 finally cancel → park(idle)时仍每秒报 phase=running + elapsed 增,
    与 meta=idle 矛盾、误导进度卡。修后:park cancel tick,wake recreate。
    """
    from birdcode.agents import runner
    from birdcode.agents.report import SubagentProgress

    recorded: list[SubagentProgress] = []

    async def _cb(p: SubagentProgress) -> None:
        recorded.append(p)

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()  # 空 → park
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_tick",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox, progress_cb=_cb,
    )
    task = asyncio.create_task(r.run())
    try:
        await asyncio.sleep(1.2)  # 首轮(<0.1s)→park;1.2s 让 tick 跑一次(若没停)
        n_at_park = len(recorded)
        await asyncio.sleep(1.3)  # 再等,tick 若跑会再调
        assert len(recorded) == n_at_park, "park 后 tick 应停,progress_cb 不再被调"
    finally:
        mailbox.put_nowait(_TEAMMATE_SHUTDOWN)
        await task


async def test_teammate_done_notifies_lead(tmp_path, monkeypatch):
    """#4:teammate 完成(非 cancel)→ 自动通知 lead(report 摘要注入 lead session)。

    lead 经 done-callback 收到 [teammate 完成] 消息,自动知晓 teammate 干完(cancel 不通知)。
    """
    from birdcode.agents import runner
    from birdcode.agents.teammate import TeamManager

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    team = TeamManager()
    lead_received: list = []
    team.register("lead", lambda m: lead_received.append(m))

    mb: asyncio.Queue = asyncio.Queue()
    mb.put_nowait(_TEAMMATE_SHUTDOWN)  # 跑首轮 → park → SHUTDOWN → done
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_notify",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mb,
    )
    h = await team.spawn("bob", r)
    await h.join()
    for _ in range(100):  # done-callback 异步触发
        if lead_received:
            break
        await asyncio.sleep(0.01)
    assert lead_received, "lead 应收到 teammate 完成通知"
    msg = lead_received[0]
    assert msg.sender == "bob" and msg.to == "lead"
    assert "完成" in msg.content


async def test_teammate_duration_excludes_park(tmp_path, monkeypatch):
    """#10:teammate duration_ms 不含 idle park(只计活跃段)。

    park 1s 期间不计入;duration 应远小于 park 时间(只首轮+轮2活跃,各<100ms)。
    """
    from birdcode.agents import runner

    def _builder(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryRecordingProvider(profile=profile)

    monkeypatch.setattr(runner, "build_provider", _builder)

    mailbox: asyncio.Queue = asyncio.Queue()  # 空 → park
    r = SubagentRunner(
        defn=_defn(), prompt="bob", description="d", tool_use_id="call_dur",
        model_override="", spawn_depth=1, is_async=True,
        parent_provider=_HistoryRecordingProvider(profile=_cfg().providers["p"]),
        parent_registry=None, parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path, mailbox=mailbox,
    )
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, r.agent_id)
    task = asyncio.create_task(r.run())
    try:
        for _ in range(100):  # 等首轮跑完 → park(idle)
            m = read_subagent_meta(meta_path)
            if m is not None and m.status == "idle":
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(1.0)  # park 1s(duration 若含 park 会 >1000ms)
        mailbox.put_nowait(MailboxMessage(sender="lead", to="bob", content="wake"))
        mailbox.put_nowait(_TEAMMATE_SHUTDOWN)
        report = await task
        # duration 应不含 park 1s → < 900ms(首轮+轮2活跃各<100ms)
        assert report.duration_ms < 900, f"duration 含 park? {report.duration_ms}ms"
    finally:
        if not task.done():
            mailbox.put_nowait(_TEAMMATE_SHUTDOWN)
            await task
