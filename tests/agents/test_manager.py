# tests/agents/test_manager.py
"""SubagentManager(异步子 agent 生命周期)单测:launch_async 注入 / cancel_all 取消 / rebind。

用 fake store/controller(无需真实 SessionStore):launch_async 后 sleep 让 _supervise
后台 task 跑完,再断言注入发生。controller.notify_wake 由 Task 8 实装于 TurnController,
此刻 fake controller 自带该方法即可。
"""

import asyncio

from birdcode.agents.definition import AgentDefinition
from birdcode.agents.manager import LaunchAck, SubagentManager
from birdcode.agents.runner import SubagentReport


class _FakeStore:
    def __init__(self) -> None:
        self.queue_ops: list[tuple[str, str, str]] = []
        self.appends: list[tuple[object, dict[str, object]]] = []

    async def append_queue_operation(
        self,
        *,
        operation: str,
        agent_id: str,
        tool_use_id: str,
        status: str,
    ) -> None:
        self.queue_ops.append((operation, agent_id, status))

    async def append(self, msg: object, **kw: object) -> None:
        self.appends.append((msg, kw))


class _FakeController:
    def __init__(self) -> None:
        self.wakes = 0

    def notify_wake(self) -> None:
        self.wakes += 1


class _DoneRunner:
    """run() 立即返回 completed report。"""

    agent_id = "sub-done"
    tool_use_id = "call_1"
    defn = AgentDefinition(name="general-purpose", description="d", system_prompt="SP")
    prompt = "做 X"
    description = "完成型子 agent"
    isolation = None  # 忠 SubagentRunner 接口

    async def run(self) -> SubagentReport:
        return SubagentReport(True, "结果", "completed", 10, 5, 1)


async def test_launch_async_injects_on_completion():
    store, ctrl = _FakeStore(), _FakeController()
    mgr = SubagentManager(store=store, controller=ctrl)
    ack = await mgr.launch_async(_DoneRunner())
    assert isinstance(ack, LaunchAck) and ack.agent_id == "sub-done"
    await asyncio.sleep(0.01)  # 让 _supervise task 跑完
    assert store.queue_ops[0][:3] == ("enqueue", "sub-done", "completed")  # enqueue 落盘
    assert any(kw.get("is_task_notification") for _, kw in store.appends)  # 通知 user 行
    assert ctrl.wakes == 1  # 唤醒


async def test_cancel_all_cancels_live():
    mgr = SubagentManager(store=_FakeStore(), controller=_FakeController())

    class _HangRunner:
        agent_id = "sub-hang"
        tool_use_id = "c"
        defn = _DoneRunner.defn
        prompt = "p"
        description = "挂起型子 agent"
        isolation = None

        async def run(self) -> SubagentReport:
            await asyncio.Event().wait()  # 永不自发完成

    await mgr.launch_async(_HangRunner())
    await asyncio.sleep(0.01)
    assert "sub-hang" in mgr._live
    mgr.cancel_all()
    await asyncio.sleep(0.01)
    assert "sub-hang" not in mgr._live  # cancelled → _supervise 收尾


def test_rebind_updates_targets():
    mgr = SubagentManager(store=_FakeStore(), controller=_FakeController())
    s2, c2 = _FakeStore(), _FakeController()
    mgr.rebind(store=s2, controller=c2)
    assert mgr._store is s2 and mgr._controller is c2


class _HangRunner:
    """run() 永不自发完成(等 cancel)。agent_id 可参数化(批量场景多实例)。"""

    tool_use_id = "c"
    defn = _DoneRunner.defn
    prompt = "p"
    description = "挂起型子 agent"
    isolation = None

    def __init__(self, aid: str) -> None:
        self.agent_id = aid

    async def run(self) -> SubagentReport:
        await asyncio.Event().wait()


async def test_supervise_cancel_branch_does_not_wake():
    """取消分支落盘 cancelled 通知但 wake=False(等 ESC 批量 wake 合并消费)。"""
    store, ctrl = _FakeStore(), _FakeController()
    mgr = SubagentManager(store=store, controller=ctrl)
    await mgr.launch_async(_HangRunner("sub-c"))
    await asyncio.sleep(0.01)
    mgr.cancel_all()
    await asyncio.sleep(0.05)  # _supervise 收尾(CancelledError → _inject wake=False)
    assert any(op == "enqueue" and st == "cancelled" for op, _, st in store.queue_ops)
    assert any(kw.get("is_task_notification") for _, kw in store.appends)
    assert ctrl.wakes == 0  # 取消分支不立即 wake


async def test_cancel_all_and_notify_wakes_once_for_multiple_cancelled():
    """多个 async 被 ESC 取消 → cancel_all_and_notify join 后只 wake 一次(批量合并)。

    regression:各 _supervise 取消分支各自 wake 时,先到的触发第一轮只消费部分 cancelled 通知
    = N 轮往返(ee6e8090)。现取消分支 wake=False,ESC 批量 wake 一次,_process_wake 一个
    turn 看到全部。
    """
    store, ctrl = _FakeStore(), _FakeController()
    mgr = SubagentManager(store=store, controller=ctrl)
    for aid in ("sub-a", "sub-b", "sub-c"):
        await mgr.launch_async(_HangRunner(aid))
    await asyncio.sleep(0.01)
    assert len(mgr._live) == 3
    mgr.cancel_all_and_notify()
    esc_task = mgr._esc_notify_task
    assert esc_task is not None
    await esc_task  # _join_and_notify:join 完 + 一次 wake
    cancelled_enqs = [op for op, _, st in store.queue_ops if op == "enqueue" and st == "cancelled"]
    assert len(cancelled_enqs) == 3  # 3 条 cancelled 通知各自落盘(保持 agentId 模型)
    assert ctrl.wakes == 1  # 但只 wake 一次(批量合并)
