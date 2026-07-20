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
        # ESC 批量 wake 收尾 task:持有引用防 GC;done 后清空(见 cancel_all_and_notify)。
        self._esc_notify_task: asyncio.Task[None] | None = None

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
        """跑到终态(completed/error/cancelled)→ 注入通知。脱管后台:绝不 re-raise 杀主循环。

        取消(CancelledError)分支落盘 wake=False:不立即 wake,等 ESC 批量 wake
        (cancel_all_and_notify)合并消费——否则多个 async 同时被取消时各自 wake,先到的触发
        第一轮只消费部分 = N 轮往返(见 ee6e8090)。正常完成仍即时 wake(wake=True)。
        on_unmount 取消由 begin_shutdown 抑制 wake,不走批量。
        """
        cancelled = False
        try:
            report = await runner.run()
        except asyncio.CancelledError:
            # B 方案:ESC 中断 async → cancelled 通知带 resume_agent 引导(模型 ask_user 后续跑)。
            report = SubagentReport(
                False,
                f"被用户中断(ESC)。若需恢复,先 ask_user 询问用户是否同意,"
                f"同意后 resume_agent(agent_id='{runner.agent_id}', direction=...) "
                f"异步续跑。勿擅自调用。",
                "cancelled",
                duration_ms=0,
                tokens=0,
                tool_use_count=0,
            )
            cancelled = True  # 脱管后台:不 re-raise,写 cancelled 通知(带引导,不杀主循环)
        except Exception as exc:  # noqa: BLE001 - 子 agent 任意异常都转 error 通知,不冒泡
            report = SubagentReport(
                False, f"子 agent 异常: {exc}", "error", duration_ms=0, tokens=0, tool_use_count=0
            )
        finally:
            self._live.pop(runner.agent_id, None)
        try:
            await self._inject(runner, report, wake=not cancelled)
        except Exception:  # noqa: BLE001 - 注入失败不杀 manager
            log.exception("SubagentManager._inject 失败(agent_id=%s)", runner.agent_id)

    async def _inject(
        self, runner: SubagentRunner, report: SubagentReport, *, wake: bool = True
    ) -> None:
        """落盘 queue-operation(enqueue)+ task-notification user 行 +(可选)唤醒主 agent。

        queue-operation 行自带 agentId/toolUseId/status(多异步辨识);紧接一条
        is_task_notification=True 的 user 行(载荷=<task-notification> 文本),供主 agent
        下一轮消费。wake=True 时 notify_wake() 让主 agent 醒来消费队列;wake=False 仅落盘
        (取消分支用:由 cancel_all_and_notify 收尾后批量 wake,合并消费多个 cancelled 通知)。
        """
        notification = build_task_notification(
            report,
            runner.defn,
            agent_id=runner.agent_id,
            prompt=runner.prompt,
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
        if wake:
            self._controller.notify_wake()

    def cancel_all_and_notify(self) -> None:
        """ESC:cancel 所有 async 子 agent + 收尾后一次 wake(合并消费所有 cancelled 通知)。

        各 _supervise 取消分支落盘 cancelled 通知但 wake=False;此处 cancel_all 后 join 等其
        落盘完,再一次 notify_wake——所有 cancelled 通知在同一次 _process_wake 前落盘,主 agent
        一个 turn 看到全部。否则各 _supervise 各自 wake,先到的触发第一轮只消费部分 = N 轮往返
        (见 ee6e8090)。on_unmount 不走此路(用 cancel_all + join_all,begin_shutdown 抑制 wake)。
        fire-and-get:持有 _esc_notify_task 防 GC;覆盖旧引用(前一轮 ESC 未完则 orphan,
        begin_shutdown 兜底)。join_all 超时未收尾的 agent 靠 resume 扫描兜底。
        """
        self.cancel_all()
        self._esc_notify_task = asyncio.create_task(self._join_and_notify())

    async def _join_and_notify(self) -> None:
        try:
            await self.join_all()
            self._controller.notify_wake()
        except Exception:  # noqa: BLE001 - best-effort,不杀 ESC 路径
            log.debug("ESC 批量 wake 收尾失败(best-effort)", exc_info=True)
        finally:
            self._esc_notify_task = None

    def has_live(self, agent_id: str) -> bool:
        """该 agent_id 是否有在跑的异步 task(幂等守卫:续跑前查,避免重复 launch)。

        _live dict 的公共只读访问器:替代外部对 manager._live 私有字段的直访(如 resume.py)。
        """
        return agent_id in self._live

    def rebind(self, *, store: SessionStore, controller: TurnController) -> None:
        """重绑 store/controller(/resume 或 /compact 后接续新会话句柄)。"""
        self._store = store
        self._controller = controller

    def cancel_all(self) -> None:
        """取消所有在跑异步子 agent(不 await;各自 _supervise 收尾写 cancelled 通知)。"""
        for task in list(self._live.values()):
            task.cancel()

    async def join_all(self, *, timeout: float = 5.0) -> None:
        """await 所有异步子 agent _supervise 终止(其 run() finally 内 worktree cleanup 跑完)。

        on_unmount 退出用:cancel_all 后调。不 await 则 app 关
        loop 砍断 _supervise → async worktree 子 agent 同样留 git-locked 残留(agent_id 随机,
        快速恢复救不了)。带总超时:cleanup 卡住不阻塞退出,超时退化为 best-effort。
        """
        tasks = list(self._live.values())  # 快照(_supervise finally 并发 pop 不影响迭代)
        if not tasks:
            return

        async def _drain(t: asyncio.Task[None]) -> None:
            try:
                await t
            except asyncio.CancelledError:
                pass  # cancel_all 触发的正常终态
            except Exception:  # noqa: BLE001 - 退出路径不因单个子 agent 异常中断其余清理
                log.debug("异步子 agent 清理 task 抛异常(已忽略,继续其余清理)", exc_info=True)

        try:
            await asyncio.wait_for(asyncio.gather(*[_drain(t) for t in tasks]), timeout=timeout)
        except TimeoutError:
            log.debug(
                "异步子 agent 清理未在 %.1fs 内全部完成,放弃等待(退化为 best-effort)", timeout
            )
        except asyncio.CancelledError:
            # #2:退出路径被 cancel(如 on_unmount 期 SIGINT)时不在 join 窗口抛出——让调用方
            # 继续后续 MCP/store 清理(join_all 契约:绝不因清理窗口传播取消)。store.close 由
            # on_unmount 的 finally 兜底,此处只需不阻断。
            log.debug("异步子 agent 清理被取消(退出路径,退化为 best-effort)")
