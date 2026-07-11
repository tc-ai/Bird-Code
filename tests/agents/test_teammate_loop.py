# tests/agents/test_teammate_loop.py
"""agent teams:长驻 teammate 循环(history 跨轮累积 + mailbox park/wake)。

teammate 跑第 1 轮→park→mailbox 唤醒跑第 2 轮;第 2 轮 history 含第 1 轮。
预载 mailbox=[第2轮 prompt, SHUTDOWN] → run() 确定性跑完(无需后台 task/时序)。
"""
import asyncio
import json

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage, ToolCallStart
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.mailbox import MailboxMessage
from birdcode.agents.runner import _TEAMMATE_SHUTDOWN, SubagentRunner
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.session.models import SessionContext
from birdcode.session.paths import subagent_meta_path
from birdcode.session.subagent_meta import read_subagent_meta


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
