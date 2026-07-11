# tests/agents/test_phase2_e2e.py
"""Phase2 端到端集成:真实 SubagentManager + TurnController + SessionStore(mock provider)。

覆盖四场景(主/子 provider 均 mock,build_provider 被 monkeypatch;store/controller/manager
与 _AgentTool→SubagentRunner 全链真实):
  1. 异步完成→唤醒:_AgentTool 异步派发 → 子 agent fake 立即 completed → manager._inject
     (enqueue + isTaskNotification user 行)→ notify_wake → 主 agent fake 消费。
  2. Esc 终止:派发 hang 子 agent → cancel_all → _live 清空 + cancelled 通知注入。
  3. /clear 转新会话:hang 子 agent 派发 → reopen 新 store + manager.rebind → 放行完成 →
     通知落【新】store(旧 store 无 isTaskNotification 行)。
  4. resume 重注入:构造 enqueue-only(crash 在 user 行前)的 store → reopen → 复刻
     _resume_pending_notifications(补投 user 行 + notify_wake)→ 主 agent 消费 + dequeue。

时序:派发后用 _wait_until_live_drained / _wait_until_idle 轮询终态(不靠固定 sleep),稳定可复现。
"""
from __future__ import annotations

import asyncio

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.agents import runner
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.manager import SubagentManager
from birdcode.agents.registry import AgentRegistry
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message, TurnController
from birdcode.session.codec import (
    find_pending_notifications,
    find_unresponded_notifications,
)
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore
from birdcode.tools.agent_tools import _AgentTool
from birdcode.tools.base import ToolOutput

# ---- mock providers ----


class _ChildProvider:
    """子 agent 流式桩。gate=None:立即 yield 文本 + Done(completed);gate 非空:阻塞到
    放行(gate.set)或被 cancel(CancelledError 经 gate.wait 上传,runner.run re-raise)。"""

    def __init__(self, profile, *, text="子 agent 完成了任务", gate=None):
        self.profile = profile
        self._text = text
        self._gate = gate

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            if self._gate is not None:
                await self._gate.wait()  # 挂起直到放行/取消
            yield TextDelta(text=self._text)
            yield Done(
                usage=TokenUsage(input_tokens=8, output_tokens=4), stop_reason="end_turn"
            )

        return _gen()


class _MainProvider:
    """主 agent 流式桩:固定 yield 一段响应文本 + Done(模拟主 agent 一轮回复)。"""

    def __init__(self, profile, *, text="已收到子 agent 通知,继续。"):
        self.profile = profile
        self._text = text

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text=self._text)
            yield Done(
                usage=TokenUsage(input_tokens=5, output_tokens=3), stop_reason="end_turn"
            )

        return _gen()


# ---- 驱动辅助 ----


async def _noop_event(_ev: object) -> None: ...


async def _noop_status() -> None: ...


def _cfg() -> AppConfig:
    return AppConfig(
        providers={
            "p": ProviderProfile(
                name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
            )
        },
        default="p",
    )


def _make_store(tmp_path, session_id: str) -> SessionStore:
    ctx = SessionContext(
        session_id=session_id, cwd=str(tmp_path), version="0.1.0", git_branch=None
    )
    return SessionStore(ctx, tmp_path, root=tmp_path)


def _registry() -> AgentRegistry:
    r = AgentRegistry()
    r.register(
        AgentDefinition(
            name="general-purpose", description="通用子 agent", system_prompt="SP",
            run_in_background=True,
        )
    )
    return r


def _make_agent_tool(store, cfg, mgr, parent_provider, tmp_path) -> _AgentTool:
    return _AgentTool(
        defn=_registry().resolve("general-purpose"),
        cfg=cfg,
        app=None,
        ctx=store.ctx,
        project_root=tmp_path,
        parent_provider=parent_provider,
        parent_registry=None,
        parent_gate=None,
        spawn_depth=0,
        subagent_mgr=mgr,
    )


def _patch_child_provider(monkeypatch, make) -> None:
    """monkeypatch runner.build_provider → 用 make(profile) 造子 agent provider(避免真实 API)。"""

    def _fake(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return make(profile)

    monkeypatch.setattr(runner, "build_provider", _fake)


def _queue_ops(rows: list[dict]) -> list[dict]:
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "queue-operation"]


def _notif_rows(rows: list[dict], agent_id: str | None = None) -> list[dict]:
    out = [
        r
        for r in rows
        if isinstance(r, dict) and r.get("type") == "user" and r.get("isTaskNotification")
    ]
    if agent_id is not None:
        out = [r for r in out if r.get("agentId") == agent_id]
    return out


async def _wait_until_idle(ctrl: TurnController, *, timeout: float = 2.0) -> None:
    """notify_wake 的 _drain 是后台 fire-and-forget;轮询 busy 直到 False。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while ctrl.busy:
        if loop.time() > deadline:
            raise TimeoutError("TurnController 未在限时内回到空闲")
        await asyncio.sleep(0.005)


async def _wait_until_live_drained(mgr: SubagentManager, *, timeout: float = 2.0) -> None:
    """轮询 manager._live 直到空(子 agent _supervise 收尾)。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while mgr._live:  # noqa: SLF001
        if loop.time() > deadline:
            raise TimeoutError("SubagentManager._live 未在限时内清空")
        await asyncio.sleep(0.005)


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch, tmp_path):
    """侧链/subagent-store 落 tmp(paths.default_root → tmp_path),不污染 ~/.birdcode。"""
    from birdcode.session import paths

    monkeypatch.setattr(paths, "default_root", lambda: tmp_path)


# ---- 场景 1:异步完成→唤醒消费 ----


async def test_e2e_async_complete_wakes_main(tmp_path, monkeypatch):
    cfg = _cfg()
    main_provider = _MainProvider(cfg.providers["p"])
    _patch_child_provider(monkeypatch, lambda profile: _ChildProvider(profile))

    store = _make_store(tmp_path, "e2e1")
    try:
        ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=store
        )
        mgr = SubagentManager(store=store, controller=ctrl)
        tool = _make_agent_tool(store, cfg, mgr, main_provider, tmp_path)

        out = await tool.execute(prompt="做 X")
        # 立即回执(不阻塞)
        assert isinstance(out, ToolOutput)
        assert "已派发" in out.text
        assert out.tool_use_result["status"] == "launched"
        assert out.tool_use_result["isAsync"] is True

        # 让 _supervise(子 agent completed + _inject)与 _drain(主 agent 消费)跑完
        await _wait_until_live_drained(mgr)
        await _wait_until_idle(ctrl)

        rows = store.mainline_rows()
        ops = _queue_ops(rows)
        enq = [o for o in ops if o.get("operation") == "enqueue"]
        deq = [o for o in ops if o.get("operation") == "dequeue"]
        # enqueue(completed) + dequeue(消费)各一
        assert len(enq) == 1
        assert enq[0].get("status") == "completed"
        assert len(deq) == 1
        # isTaskNotification user 行(_inject 落盘)
        assert len(_notif_rows(rows)) == 1
        # 主 agent 响应 assistant 行
        assert any(r.get("type") == "assistant" for r in rows)
        # 已消费:无待响应通知
        assert find_unresponded_notifications(rows) == []
    finally:
        store.close()


# ---- 场景 2:Esc 终止 → cancelled 通知注入 ----


async def test_e2e_cancel_all_injects_cancelled(tmp_path, monkeypatch):
    cfg = _cfg()
    main_provider = _MainProvider(cfg.providers["p"])
    gate = asyncio.Event()  # 子 agent 挂起直到我们 cancel
    _patch_child_provider(monkeypatch, lambda profile: _ChildProvider(profile, gate=gate))

    store = _make_store(tmp_path, "e2e2")
    try:
        ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=store
        )
        mgr = SubagentManager(store=store, controller=ctrl)
        tool = _make_agent_tool(store, cfg, mgr, main_provider, tmp_path)

        out = await tool.execute(prompt="挂起任务")
        assert isinstance(out, ToolOutput)
        # 让子 agent 进入 gate.wait()(确属 live)
        await asyncio.sleep(0.02)
        assert len(mgr._live) == 1  # noqa: SLF001

        mgr.cancel_all()  # Esc:取消所有在跑异步子 agent
        await _wait_until_live_drained(mgr)
        await _wait_until_idle(ctrl)

        # _live 已清空
        assert len(mgr._live) == 0  # noqa: SLF001
        rows = store.mainline_rows()
        ops = _queue_ops(rows)
        enq = [o for o in ops if o.get("operation") == "enqueue"]
        # cancelled 通知注入(enqueue status=cancelled)
        assert len(enq) == 1
        assert enq[0].get("status") == "cancelled"
        assert len(_notif_rows(rows)) == 1  # isTaskNotification user 行
    finally:
        store.close()


# ---- 场景 3:/clear 转新会话 → 通知落新 store ----


async def test_e2e_clear_reroutes_notification_to_new_store(tmp_path, monkeypatch):
    cfg = _cfg()
    main_provider = _MainProvider(cfg.providers["p"])
    gate = asyncio.Event()
    _patch_child_provider(monkeypatch, lambda profile: _ChildProvider(profile, gate=gate))

    old_store = _make_store(tmp_path, "clear1")
    new_store: SessionStore | None = None
    try:
        old_ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=old_store
        )
        mgr = SubagentManager(store=old_store, controller=old_ctrl)
        tool = _make_agent_tool(old_store, cfg, mgr, main_provider, tmp_path)

        out = await tool.execute(prompt="挂起任务")
        assert isinstance(out, ToolOutput)
        await asyncio.sleep(0.02)  # 让子 agent 进入 gate.wait()

        # /clear:reopen 新 store + manager.rebind 指向新 store/controller
        new_store = old_store.reopen("clear2")
        new_ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=new_store
        )
        mgr.rebind(store=new_store, controller=new_ctrl)

        gate.set()  # 放行子 agent 完成 → _inject 写入(当前绑定的)new_store
        await _wait_until_live_drained(mgr)
        await _wait_until_idle(new_ctrl)

        # 旧 store 无 isTaskNotification 行(launch 不写主线,_inject 走 new_store)
        assert _notif_rows(old_store.mainline_rows()) == []
        # 新 store 有:enqueue + 通知 user 行
        new_rows = new_store.mainline_rows()
        assert len(_notif_rows(new_rows)) == 1
        assert any(o.get("operation") == "enqueue" for o in _queue_ops(new_rows))
    finally:
        old_store.close()  # reopen 已关,再关幂等
        if new_store is not None:
            new_store.close()


# ---- 场景 4:resume 重注入(补投 + 唤醒消费)----


async def test_e2e_resume_reinjects_pending_notification(tmp_path, monkeypatch):  # noqa: ARG001
    """复刻 BirdApp._resume_pending_notifications 两轴:补投 → 重注入 → wake 消费。

    不实例化 BirdApp(需 Textual on_mount):直接在 reopen 的 store 上跑等价逻辑 + 真实
    TurnController,稳定验证 resume 重注入路径(crash 在 enqueue↔user 行之间 → 补投 → 消费)。
    """
    cfg = _cfg()
    main_provider = _MainProvider(cfg.providers["p"])

    # 会话 1:写一条 enqueue 后崩溃(无 user 行、无 dequeue)
    notif_text = (
        "<task-notification>\n子 agent 完成(crash 前已 enqueue)\n</task-notification>"
    )
    store1 = _make_store(tmp_path, "resume1")
    await store1.append_queue_operation(
        operation="enqueue",
        agent_id="a-resume",
        tool_use_id="tu-resume",
        status="completed",
    )
    store1.close()

    # 会话 2:reopen 同 session(模拟 resume 加载已有 jsonl)
    store2 = _make_store(tmp_path, "resume1")
    try:
        ctrl = TurnController(
            main_provider, on_event=_noop_event, on_status=_noop_status, store=store2
        )
        await ctrl.resume()  # 加载主线(空 turns,仅 enqueue 行)+ 设叶子 uuid 接续

        rows = store2.mainline_rows()
        # 投递补全轴:enqueue 无匹配 user 行 → pending
        pending = find_pending_notifications(rows)
        assert len(pending) == 1
        obj = pending[0]
        # 补投 isTaskNotification user 行。生产由 _reconstruct_subagent_notification 从侧链
        # 重读报告;此处直接给文本,聚焦验证 wake→消费→dequeue(重构来源见 test_resume_notifications)。
        await store2.append(
            Message(role="user", content=[TextBlock(text=notif_text)]),
            is_task_notification=True,
            agent_id=obj.get("agentId") or "",
        )
        # 重注入轴:已投递未响应 → 唤醒
        rows2 = store2.mainline_rows()
        assert len(find_unresponded_notifications(rows2)) == 1
        ctrl.notify_wake()
        await _wait_until_idle(ctrl)

        # 已消费:无待响应 + dequeue + assistant 落盘
        rows3 = store2.mainline_rows()
        assert find_unresponded_notifications(rows3) == []
        assert any(
            o.get("operation") == "dequeue" and o.get("agentId") == "a-resume"
            for o in _queue_ops(rows3)
        )
        assert any(r.get("type") == "assistant" for r in rows3)
    finally:
        store2.close()
