# src/birdcode/agents/manager.py
"""异步子 agent 生命周期管理:派生 → 监管 → 注入通知 → 重绑/取消。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from birdcode.agents.report import build_task_notification
from birdcode.agents.runner import SubagentReport, SubagentRunner
from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    # TurnController / SessionStore 仅作类型标注:运行时不 import,避免 manager ↔
    # conversation/session 重模块的循环/重载。真实 notify_wake 实装于 TurnController。
    from birdcode.conversation import TurnController
    from birdcode.session.store import SessionStore

log = get_logger("birdcode.agents.manager")


@dataclass
class LaunchAck:
    """launch_async 回执:agent_id + 给主 agent 的确认文本(内含 id,供其匹配完成通知)。"""

    agent_id: str
    ack_text: str


class SubagentManager:
    """后台跑异步子 agent,完成向主 jsonl 注入 <task-notification> 并唤醒主 agent。

    只持 store(_inject 落盘)+ controller(notify_wake)。runner 由 AgentTool 构造
    (带 ctx/project_root/progress_cb/tool_use_id),manager 只管 task 生命周期 + 注入。
    """

    def __init__(self, *, store: SessionStore, controller: TurnController) -> None:
        self._store = store
        self._controller = controller
        self._live: dict[str, asyncio.Task[None]] = {}

    async def launch_async(self, runner: SubagentRunner) -> LaunchAck:
        """派生后台 _supervise task 跟踪 runner,立即回执(不 await 子 agent 完成)。"""
        self._live[runner.agent_id] = asyncio.create_task(self._supervise(runner))
        return LaunchAck(
            agent_id=runner.agent_id,
            ack_text=(
                f"已派发异步子 agent {runner.description},本 agent 的 id 为 {runner.agent_id},"
                f"后续完成后会通知你,你可以根据 id 来进行匹配。"
            ),
        )

    async def _supervise(self, runner: SubagentRunner) -> None:
        """跑到终态(completed/error/cancelled)→ 注入通知。脱管后台:绝不 re-raise 杀主循环。"""
        try:
            report = await runner.run()
        except asyncio.CancelledError:
            report = SubagentReport(
                False, "被取消", "cancelled", duration_ms=0, tokens=0, tool_use_count=0
            )
            # 脱管后台:不 re-raise,写 cancelled 通知(不杀主循环)
        except Exception as exc:  # noqa: BLE001 - 子 agent 任意异常都转 error 通知,不冒泡
            report = SubagentReport(
                False, f"子 agent 异常: {exc}", "error", duration_ms=0, tokens=0, tool_use_count=0
            )
        finally:
            self._live.pop(runner.agent_id, None)
        try:
            await self._inject(runner, report)
        except Exception:  # noqa: BLE001 - 注入失败不杀 manager
            log.exception("SubagentManager._inject 失败(agent_id=%s)", runner.agent_id)

    async def _inject(self, runner: SubagentRunner, report: SubagentReport) -> None:
        """落盘 queue-operation(enqueue)+ task-notification user 行 + 唤醒主 agent。

        queue-operation 行自带 agentId/toolUseId/status(多异步辨识);紧接一条
        is_task_notification=True 的 user 行(载荷=<task-notification> 文本),供主 agent
        下一轮消费。最后 notify_wake() 让主 agent 的等待者醒来消费队列。
        """
        notification = build_task_notification(
            report, runner.defn, agent_id=runner.agent_id, prompt=runner.prompt,
            is_worktree=(runner.isolation == "worktree"),
        )
        await self._store.append_queue_operation(
            operation="enqueue",
            agent_id=runner.agent_id,
            tool_use_id=runner.tool_use_id,
            status=report.status,
        )
        msg = Message(role="user", content=[TextBlock(text=notification)])
        await self._store.append(msg, is_task_notification=True, agent_id=runner.agent_id)
        self._controller.notify_wake()

    def rebind(self, *, store: SessionStore, controller: TurnController) -> None:
        """重绑 store/controller(/resume 或 /compact 后接续新会话句柄)。"""
        self._store = store
        self._controller = controller

    def cancel_all(self) -> None:
        """取消所有在跑异步子 agent(不 await;各自 _supervise 收尾写 cancelled 通知)。"""
        for task in list(self._live.values()):
            task.cancel()
