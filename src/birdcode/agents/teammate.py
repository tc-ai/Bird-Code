# src/birdcode/agents/teammate.py
"""agent teams:teammate spawn 入口 + handle(内部/测试用,不接 UI)。

最小形态:spawn 一个长驻 teammate(mailbox 模式 SubagentRunner)+ 起 background task +
返回 handle。handle 暴露 mailbox 投递(send/shutdown)+ 取消(cancel)/合并(join)。
后续加 TeamManager / 命名 registry / peer 寻址 / _live 托管(name 此处仅标签,
不登记;多 teammate 托管 + cancel-all 也留后续)。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from birdcode.agents.mailbox import _AGENT_NAME, MailboxMessage
from birdcode.agents.runner import _TEAMMATE_SHUTDOWN, SubagentReport, SubagentRunner
from birdcode.agents.task_board import TaskBoard
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.agents.teammate")


@dataclass
class TeammateHandle:
    """长驻 teammate 的控制句柄。name 不进 registry(仅标签)。"""

    name: str
    agent_id: str
    mailbox: asyncio.Queue
    _task: asyncio.Task

    async def send(self, prompt: str, *, sender: str = "lead") -> None:
        """poke:投一条消息(包成 MailboxMessage)进 mailbox,唤醒 park 中的 teammate 续跑。

        已结束(task.done)→ raise,不静默丢(/agents 发消息给死 teammate 应明确报错)。
        """
        if self._task.done():
            raise RuntimeError(f"teammate {self.name} 已结束,无法投递消息")
        self.mailbox.put_nowait(MailboxMessage(sender=sender, to=self.name, content=prompt))

    def shutdown(self) -> None:
        """干净终止:投 SHUTDOWN 哨兵,teammate 跑完当前轮后退出(report=completed)。

        已结束 → 幂等 no-op(不重复投、不抛)。
        """
        if self._task.done():
            return
        self.mailbox.put_nowait(_TEAMMATE_SHUTDOWN)

    def cancel(self) -> None:
        """硬取消(Esc 风格):CancelledError 打断,走 runner cancelled 分支(meta=cancelled)。"""
        self._task.cancel()

    def done(self) -> bool:
        return self._task.done()

    async def join(self) -> None:
        await self._task


async def spawn_teammate(*, name: str, runner: SubagentRunner) -> TeammateHandle:
    """启动长驻 teammate(runner 须带 mailbox),返回 handle。

    runner 由调用方完整构造(复用 SubagentRunner 现有入参);本函数只起 background task 并
    打包 handle。不接 SubagentManager._live(无 team 协调);TeamManager 接管多 teammate
    托管 + cancel-all + 命名 registry。
    """
    if runner.mailbox is None:
        raise ValueError("spawn_teammate 需要 mailbox 模式 runner(runner.mailbox 非空)")
    task = asyncio.create_task(_run_named(name, runner))  # _run_named 设 _AGENT_NAME=sender
    return TeammateHandle(name=name, agent_id=runner.agent_id, mailbox=runner.mailbox, _task=task)


async def _run_named(name: str, runner: SubagentRunner) -> SubagentReport:
    """teammate task body:设 _AGENT_NAME(供 SendMessage 取 sender)再跑 runner.run()。

    contextvar 在 task 内设 → 仅影响本 task(asyncio task 隔离 context),主 session/其他
    teammate 不受污染。CancelledError 经 await runner.run() 自然透传(与原 spawn 行为一致)。
    """
    _AGENT_NAME.set(name)
    return await runner.run()


class TeamManager:
    """team 管理:name→recipient deliver 注册表 + spawn 包装 + 自动注销。

    recipient 的 deliver = Callable[[MailboxMessage], None](teammate:其 mailbox.put_nowait;
    lead:其 receive)。不含任务表/协调/依赖(后续)。teammate task 结束时自动 unregister。
    """

    def __init__(self) -> None:
        self._recipients: dict[str, Callable[[MailboxMessage], None]] = {}
        self._handles: dict[str, TeammateHandle] = {}  # name→handle(cancel_all/shutdown_all 用)
        self.task_board: TaskBoard = TaskBoard()  # flat 共享看板(无 Lock,同步原子)
        # /clear(reset)递增的代次;spawn 时捕获,_on_done 据此丢弃跨会话的陈旧通知(F4)。
        self._generation: int = 0

    def register(self, name: str, deliver: Callable[[MailboxMessage], None]) -> None:
        self._recipients[name] = deliver

    def unregister(self, name: str) -> None:
        self._recipients.pop(name, None)

    def resolve(self, name: str) -> Callable[[MailboxMessage], None] | None:
        """返回 name 的 deliver 回调;未注册返回 None(SendMessage 据此判 unknown recipient)。"""
        return self._recipients.get(name)

    def names(self) -> list[str]:
        return list(self._recipients)

    def teammates(self) -> list[tuple[str, TeammateHandle]]:
        """列当前 teammate(name, handle),供 /agents UI 观察/打断/发消息。"""
        return list(self._handles.items())

    def handle(self, name: str) -> TeammateHandle | None:
        """取某 teammate 的 handle(interrupt/send 用);未注册 → None。"""
        return self._handles.get(name)

    async def spawn(self, name: str, runner: SubagentRunner) -> TeammateHandle:
        """spawn teammate + 注册其 mailbox.put_nowait 为 deliver;task 结束自动注销。

        注销按 deliver 身份校验:重名/复用 name 时,后注册者覆盖 name→deliver,先结束者的
        done-callback 若按名删,会把仍存活的后注册者从 registry 抹掉(静默孤立)。故仅当
        name 当前仍指向【本 teammate 的 deliver】才注销。
        """
        handle = await spawn_teammate(name=name, runner=runner)
        deliver = handle.mailbox.put_nowait  # deliver(msg)=mailbox.put_nowait(msg)
        self.register(name, deliver)
        self._handles[name] = handle
        gen = self._generation  # 捕获本代次:_on_done 据此丢弃跨 /clear 的陈旧通知(F4)

        def _on_done(_t: asyncio.Task[None]) -> None:  # noqa: ARG005 - _t 必需签名
            # 跨 /clear 陈旧:reset 已 bump 代次并 cancel+清 registry/handles;本回调是旧会话
            # teammate 的尾音(reset 前刚自然完成、回调滞留 call_soon 队列)→ 整体丢弃,免把旧
            # 会话的 [teammate 完成] 经「跨 /clear 仍有效的 lead deliver」投进新会话(F4)。
            if gen != self._generation:
                return
            # 身份校验避重名误删:仅当 name 仍指向本 teammate 的 deliver/handle 才清。
            if self._recipients.get(name) is deliver:
                self.unregister(name)
            if self._handles.get(name) is handle:
                self._handles.pop(name, None)
            # #4:teammate 完成(非 cancel)→ 通知 lead(report 摘要注入 lead session)。
            # lead 经此自动知晓 teammate 干完;cancel(用户打断/Esc)不通知(非正常完成)。
            if not _t.cancelled():
                try:
                    report = _t.result()
                    lead_deliver = self._recipients.get("lead")
                    if lead_deliver is not None and report is not None:
                        # 前缀随 report.status(F6):旧实现恒 "[teammate 完成]",worktree 建失败 /
                        # 护栏失败等 status='error' 的报告也被当「完成」误导 lead。cancelled 被
                        # 外层 _t.cancelled() 挡住不会到这;余下仅 completed/error,二元足矣。
                        tag = "完成" if report.status == "completed" else "出错"
                        lead_deliver(MailboxMessage(
                            sender=name, to="lead",
                            content=f"[teammate {tag}] {getattr(report, 'text', '')[:500]}",
                        ))
                except Exception:  # noqa: BLE001 - 通知失败不杀 cleanup 路径
                    pass

        handle._task.add_done_callback(_on_done)
        return handle

    def cancel_all(self) -> None:
        """硬取消所有 teammate(Esc 风格,CancelledError→meta cancelled)。action_interrupt 用。"""
        for handle in list(self._handles.values()):
            handle.cancel()

    def shutdown_all(self) -> None:
        """干净终止所有 teammate(SHUTDOWN 哨兵→跑完当前轮后 completed)。clear/退出用。"""
        for handle in list(self._handles.values()):
            handle.shutdown()

    async def join_all(self, *, timeout: float = 5.0) -> None:
        """await 所有 teammate task 终止(其 finally 内 shielded worktree cleanup 随之跑完)。

        on_unmount 退出用:cancel_all(或 shutdown_all)发终止信号后调本方法,把每个 teammate
        task 等到 done。run() 的 finally 已 best-effort await cleanup_subagent_worktree(git
        unlock/remove/branch-d,含 asyncio.shield),故 task 一旦 done 即等价 cleanup 已尽力跑完
        ——本方法只需 await 到 done。不 await 则 app 关 loop 时砍断 task → cleanup 中途留
        git-locked worktree;teammate 的 agent_id 随机,跨会话不复用,create_worktree 的快速恢复
        只救固定名的 CLI --worktree、救不了 teammate → 每次裸退出累积一批残留。

        带总超时:git 挂/超大仓库致 cleanup 卡住时不阻塞退出,超时退化为 fire-and-forget(等价
        原 best-effort,不更差)。CancelledError(cancel 正常终态)+ 单 task 异常都不中断其余。
        """
        handles = list(self._handles.values())  # 快照(_on_done 并发 pop 不影响迭代)
        if not handles:
            return

        async def _drain(h: TeammateHandle) -> None:
            try:
                await h.join()
            except asyncio.CancelledError:
                pass  # cancel_all/shutdown_all 触发的正常终态
            except Exception:  # noqa: BLE001 - 退出路径不因单个 teammate 异常中断其余清理
                log.debug("teammate 清理 task 抛异常(已忽略,继续其余清理)", exc_info=True)

        try:
            await asyncio.wait_for(
                asyncio.gather(*[_drain(h) for h in handles]), timeout=timeout
            )
        except TimeoutError:
            log.debug(
                "teammate 清理未在 %.1fs 内全部完成,放弃等待(退化为 best-effort)", timeout
            )
        except asyncio.CancelledError:
            # #2:退出路径被 cancel 时不在 join 窗口抛出——让调用方继续后续清理(join_all 契约:
            # 绝不因清理窗口传播取消)。store.close 由 on_unmount 的 finally 兜底。
            log.debug("teammate 清理被取消(退出路径,退化为 best-effort)")

    def reset(self) -> None:
        """新 session 重置(/clear 用):cancel 所有 teammate + 清 registry/board。

        工具持的 team_mgr 引用不变(只清内部状态),故 /clear 后无需重注册 team 工具。
        cancel_all 用 task.cancel(同步调度),teammate 异步收 CancelledError 终止。

        lead 的 deliver(controller.receive)跨 /clear 保留:App 全程用同一个
        TurnController 实例(/clear 只重绑其 _store,不换实例)→ bound method 仍有效。
        若随 registry 一并清掉,teammate→lead 通道会在每次 /clear 后断(SendMessage(to=lead)
        变未知 recipient、teammate 完成通知被丢)。spawn 的 teammate deliver 不保留(其 task
        已被 cancel_all 终止)。

        代次先 bump:本代 teammate 的 _on_done(reset 前刚自然完成、回调尚滞留 call_soon 队列的)
        见代次不符即整体丢弃,免把旧会话的 [teammate 完成] 经「跨 /clear 仍有效的 lead deliver」
        投进新会话(F4)。
        """
        self._generation += 1
        lead_deliver = self._recipients.get("lead")
        self.cancel_all()
        self._recipients.clear()
        self._handles.clear()
        self.task_board = TaskBoard()
        if lead_deliver is not None:
            self._recipients["lead"] = lead_deliver
