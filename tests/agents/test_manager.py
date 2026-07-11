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
        self, *, operation: str, agent_id: str, tool_use_id: str, status: str,
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
