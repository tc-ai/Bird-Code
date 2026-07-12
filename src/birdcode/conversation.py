# src/birdcode/conversation.py
"""Conversation domain: messages, turns, the type-ahead queue, and the turn controller."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from birdcode.agent.provider import (
    Error,
    Interrupted,
    ProviderEvent,
    StreamingProvider,
    TokenUsage,
    TurnStart,
)
from birdcode.blocks import ContentBlock, TextBlock
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from birdcode.agent.context import ContextManager
    from birdcode.agents.mailbox import MailboxMessage
    from birdcode.memory.extractor import MemoryManager
    from birdcode.permission.rules import Rule
    from birdcode.session.store import SessionStore
    from birdcode.tools.executor import ToolExecutor
    from birdcode.ui.app import BirdApp

log = get_logger("birdcode.conversation")

EventCallback = Callable[[ProviderEvent], Awaitable[None]]
StatusCallback = Callable[[], Awaitable[None]]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]
    # synthetic=True:resume 末尾修复补的合成消息(占位 tool_result / 收尾 assistant)。
    # 标记后:UI 重放跳过(不弹固定气泡);持久化行带 synthetic 字段;计费/统计可识别。
    synthetic: bool = False
    # 行级 isTaskNotification 回填(子 agent 完成通知 user 行)。UI 重放据此跳过(不当
    # 用户气泡);不进 message_to_dict(绝不发给 LLM)。默认 False(普通 user/assistant 行)。
    is_task_notification: bool = False


@dataclass
class Turn:
    messages: list[Message] = field(default_factory=list)
    usage: TokenUsage | None = None
    interrupted: bool = False
    # wake 轮 provider 抛异常(超时/限流/5xx)时 _consume_wake 置 True;
    # _process_wake 据此跳过 dequeue(通知保持待响应、可重试)+ 不入 history(避免孤立 user turn)。
    failed: bool = False


class _WakeInput:
    """唤醒标记(入队项):主 agent 空闲或排队后消费 pending 异步子 agent 通知。

    与用户文本(str)共用同一队列,使 submit(用户输入)与 notify_wake(系统注入)走
    统一的 drain 通道,FIFO 互不抢占。
    """


# 模块级单例:enqueue 时入队此对象,_drain 用 isinstance(_WakeInput) 分派到 _process_wake。
_WAKE = _WakeInput()


class _PeerInput:
    """teammate→lead 的 peer 消息(入队项):包文本,与用户文本(str)区分。

    不计入 _user_count(状态栏「排队消息」只含用户文本;peer 消息不虚增)。
    _drain 用 isinstance(_PeerInput) 分派到 _process(unwrap .text)。
    """

    def __init__(self, text: str) -> None:
        self.text = text


class MessageQueue:
    """FIFO type-ahead/wake/peer 队列。

    项为用户文本(str)/唤醒标记(_WakeInput)/peer 消息(_PeerInput)。
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[str | _WakeInput | _PeerInput] = asyncio.Queue()
        # 状态栏「排队消息」计数只含用户文本(str);_WakeInput/_PeerInput 不计入,否则
        # 忙时若干异步子 agent 完成会虚增 ⌛N(误显为用户在排队输入)。
        self._user_count = 0

    def enqueue(self, item: str | _WakeInput | _PeerInput) -> None:
        self._q.put_nowait(item)
        if isinstance(item, str):
            self._user_count += 1

    def dequeue_nowait(self) -> str | _WakeInput | _PeerInput | None:
        try:
            item = self._q.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if isinstance(item, str):
            self._user_count -= 1
        return item

    @property
    def size(self) -> int:
        return self._user_count


class TurnController:
    """Drives turns: routes events to the UI, queues input while busy, supports interrupt.

    busy=True while a turn (or queued turns) are being processed. submit() enqueues
    when busy; on completion the queue drains FIFO, one independent turn per message.
    """

    def __init__(
        self,
        provider: StreamingProvider,
        *,
        on_event: EventCallback,
        on_status: StatusCallback,
        executor: ToolExecutor | None = None,
        app: BirdApp | None = None,
        store: SessionStore | None = None,
        context: ContextManager | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._provider = provider
        self._on_event = on_event
        self._on_status = on_status
        self._executor = executor
        self._app = app
        self._store = store
        self._context = context
        self._memory = memory
        self._extract_tasks: set[asyncio.Task[None]] = set()
        self._queue = MessageQueue()
        self.history: list[Turn] = []
        self.busy: bool = False
        self._stream_task: asyncio.Task[None] | None = None
        # notify_wake 空闲时后台起 _drain,持有其 task 引用(防 asyncio GC 中断执行)。
        self._drain_task: asyncio.Task[None] | None = None
        self._aborted: bool = False  # /clear 用：置 True 后本轮 CancelledError 不发 Interrupted
        # 退出态(on_unmount 置 True):notify_wake/receive 直接 return,不起新后台 drain。
        self._shutting_down: bool = False

    @property
    def queue_size(self) -> int:
        return self._queue.size

    async def submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._queue.enqueue(text)
        if self.busy:
            # 正忙:入队后由在跑的 _drain 消费(FIFO type-ahead)。
            await self._on_status()
            return
        # 空闲:本次 submit 驱动 drain 并 await —— 保留「submit 返回即本轮(+排队项)完成」
        # 契约(现有 submit/type-ahead/interrupt/abort 测试依赖此同步语义)。
        self.busy = True
        await self._drain()

    def notify_wake(self) -> None:
        """异步子 agent 完成注入通知后唤醒主 agent(系统注入,非用户输入)。

        空闲 → 后台起 _drain 并持有 task 引用(防 asyncio GC 中断);
        正忙 → _WAKE 入队,由在跑的 _drain 消费。
        与 submit 的关键区别:submit 空闲时 await drain(同步语义,调用方等到完成);
        wake 是后台注入,绝不阻塞调用方(manager._inject 在子 agent 监管 task 里同步调用)。

        退出态(on_unmount 置 _shutting_down):直接 return——免退出期起新后台 drain 跑
        run_agent_loop(#1)。通知行已由 _inject 落盘(若需),resume 时再消费;退出当下不跑 LLM 轮。
        """
        if self._shutting_down:
            return
        self._queue.enqueue(_WAKE)
        if not self.busy:
            self.busy = True
            # 持有 task 引用(防 GC 中断执行);done-callback 按身份清空(不覆盖更新的引用)。
            task = asyncio.create_task(self._drain())
            self._drain_task = task
            task.add_done_callback(self._on_drain_done)

    def receive(self, msg: MailboxMessage) -> None:
        """teammate 发来的消息:格式化 str 入队,空闲时后台起 _drain(不阻塞投递方,仿 notify_wake)。

        与 submit(用户输入,空闲时 await drain)和 notify_wake(系统注入)并列:teammate 消息
        走 str 分派(_drain→_process),与 _WakeInput(_process_wake)井水不犯河水。现有
        user/notify_wake/异步子 agent 通知流程全不变。

        退出态同 notify_wake:直接 return(免退出期起新后台 drain,#1)。
        """
        if self._shutting_down:
            return
        text = f"[来自 {msg.sender}]: {msg.content}"
        self._queue.enqueue(_PeerInput(text))  # peer 消息不计 _user_count(状态栏不虚增)
        if not self.busy:
            self.busy = True
            task = asyncio.create_task(self._drain())
            self._drain_task = task
            task.add_done_callback(self._on_drain_done)

    def _on_drain_done(self, task: asyncio.Task[None]) -> None:
        """后台 drain 完成回调:按身份清引用(防覆盖更新的 drain task)+ 记录未检索异常。"""
        if self._drain_task is task:
            self._drain_task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("background drain task failed", exc_info=exc)

    def begin_shutdown(self) -> None:
        """退出(on_unmount 最先调):抑制后续后台 drain + 取消已起的 orphan drain(#1)。

        notify_wake/receive 空闲时 create_task(_drain) 跑 run_agent_loop。退出收尾链
        (cancel_all→join_all→_supervise→_inject→notify_wake)若在退出期触发,会起一个
        on_unmount 既不 await 也不 cancel 的 orphan drain,在 MCP/store 的 await 期间跑 LLM 轮 +
        工具(Write/Bash)——d37798c 的 join_all 把它从偶发变成确定。置 _shutting_down 后
        notify_wake/receive 直接 return(不起新 drain);并取消已起的 _drain_task(退出前夕刚好
        有子 agent/teammate 完成触发的尾音)。busy 内联 drain(submit 驱动,_stream_task 链)是
        退出前既有行为、非本提交引入,不在此处置理。
        """
        self._shutting_down = True
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()

    def interrupt(self) -> None:
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()

    def set_provider(self, provider: StreamingProvider) -> None:
        """运行时切换后端（/profile）。_consume 每轮重新读 self._provider，下一轮生效。"""
        self._provider = provider

    async def resume(self) -> list[Turn]:
        """从 store 加载主线填 history,并设叶子 uuid(store 内部)使后续 append 接续。

        无 store → 返回空(未启用持久化)。链由 store 内部维护(_last_uuid)。
        """
        if self._store is None:
            return []
        turns = await self._store.load_mainline()
        self.history = turns
        return turns

    async def _on_message(
        self,
        msg: Message,
        *,
        usage: TokenUsage | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """agent_loop 每追加一条 Message(assistant / tool_result user)经此落盘。

        链由 store 内部维护(_last_uuid 自动挂);无 store 时空操作。
        """
        if self._store is None:
            return
        await self._store.append(
            msg,
            is_assistant=(msg.role == "assistant"),
            usage=usage,
            stop_reason=stop_reason,
        )

    def list_permissions(self) -> list[Rule]:
        """暴露权限规则(供 /permissions 查看)。"""
        return self._executor.list_permissions() if self._executor is not None else []

    def revoke_permission(self, rule_id: int) -> bool:
        """撤销一条权限规则(供 /permissions revoke <id>)。"""
        return self._executor.revoke_permission(rule_id) if self._executor is not None else False

    async def request_permission(
        self, tool_name: str, summary: str, path: str | None  # noqa: ARG002
    ) -> Literal["approve", "reject", "session"]:
        """executor/gate 经此请求用户确认;await 直到用户响应(Esc=Reject)。

        经 app 挂底部行内 PermissionPrompt;选择回调 set 进 Future;此处 await。
        stream task 被中断(Esc、或 /clear 取消)→ CancelledError 视为 Reject。
        path 已折进 summary(由 gate._summarize 生成),此处不再单独传。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def _on_result(result: object) -> None:
            if not future.done():
                future.set_result(str(result) if result is not None else "reject")

        assert self._app is not None  # BirdApp.on_mount 注入 self._app
        await self._app.mount_permission_prompt(tool_name, summary, _on_result)
        try:
            return await future  # type: ignore[return-value]
        except asyncio.CancelledError:
            # stream task 被中断 → 视为 Reject(中断正确传播为 Reject,不卡死)。
            return "reject"
        finally:
            self._app._dismiss_permission_prompt()  # noqa: SLF001 - 兜底卸载(正常选已在 choose 回调卸载)

    def abort(self) -> None:
        """/clear 用：取消当前流 + 清空 type-ahead 队列，且本轮不发 Interrupted。

        与 interrupt()（Esc，只停当前流、队列保留）的区别：abort 把排队消息也一并
        丢弃，并通过 _aborted 让 _process 的 CancelledError 路径静默（不挂中断行）。
        """
        self._aborted = True
        self.interrupt()
        while self._queue.dequeue_nowait() is not None:
            pass
        # 取消所有在跑/排队的记忆提取任务(/clear 期望干净状态)。
        for t in self._extract_tasks:
            t.cancel()
        self._extract_tasks.clear()

    async def _drain(self) -> None:
        """消费队列直到空:str → _process(用户输入),_WakeInput → _process_wake(通知响应)。

        busy 由调用方(submit/notify_wake)在起 drain 前置 True;finally 复位。type-ahead
        FIFO、Esc 中断、/clear abort 语义不变(_process/_consume 原样复用;abort 清空队列时
        _WakeInput 一并丢弃)。submit 空闲路径 await 本方法(同步语义);notify_wake 空闲路径
        create_task 本方法(后台),二者共享同一 drain 循环,绝不并发起两个 drain。

        _drain 跑在 lead 的 TurnController 内,恒以 "lead" 身份运行:receive 经 teammate task
        调用 create_task(_drain) 时,新 task 会复制投递方(teammate)的 context(_AGENT_NAME=
        teammate 名);不在本处钉回,lead 本轮(run_agent_loop 及其工具调用)会把 SendMessage/
        TaskCreate 的 sender 误归属到该 teammate。显式 set("lead") 只影响本 drain task 的 context
        副本(create_task 拷贝 / submit 的主任务本就是 lead),不污染原 teammate task。

        _AGENT_CWD 同源漏:worktree teammate 在 runner 内 set _AGENT_CWD=<worktree>,经同一
        create_task 拷贝进本 drain task;不钉回则 lead 本轮的 Git 系统提醒(_resolve_cwd 读
        _AGENT_CWD)会报 teammate worktree 的分支/脏状态,而非主仓。set(None) 钉回 → _resolve_cwd
        落 Path.cwd()(lead 主仓),同样只影响本 drain task 的 context 副本。
        """
        from birdcode.agent.system_prompt.env import _AGENT_CWD
        from birdcode.agents.mailbox import _AGENT_NAME

        _AGENT_NAME.set("lead")
        _AGENT_CWD.set(None)
        try:
            while True:
                item = self._queue.dequeue_nowait()
                if item is None:
                    break
                await self._on_status()
                if isinstance(item, _WakeInput):
                    await self._process_wake()
                elif isinstance(item, _PeerInput):
                    await self._process(item.text)
                else:
                    await self._process(item)
        finally:
            self.busy = False
            await self._on_status()

    async def _process(self, user_text: str) -> None:
        turn = Turn(messages=[Message(role="user", content=[TextBlock(text=user_text)])])
        # 每轮重置：上一轮 abort() 残留的 True 不应误吞本轮的 Esc 中断
        self._aborted = False
        # 落盘本轮 user 提问(链由 store 内部维护;无 store 空操作)。
        if self._store is not None:
            await self._store.append(turn.messages[0])
        self._stream_task = asyncio.create_task(self._consume(turn))
        try:
            await self._stream_task
        except asyncio.CancelledError:
            if not self._aborted:
                # 真正的中断（Esc）才标记并通知 UI；abort()（/clear）静默退出
                turn.interrupted = True
                await self._on_event(Interrupted())
        finally:
            self._stream_task = None
            # 被 abort()（/clear）取消的轮次不属于 history：既已取消，又随 /clear 一并清空。
            # 若无条件 append，它会复活成「孤立 user turn（无助手回复、interrupted=False）」，
            # 污染下一轮 LLM 上下文（history 即下一轮 stream 的输入）。Esc 中断
            # （_aborted=False）的轮次照常保留（history[-1].interrupted=True）。
            if not self._aborted:
                self.history.append(turn)
                self._maybe_extract(turn)

    def _maybe_extract(self, turn: Turn) -> None:
        """轮次干净结束后,fire-and-forget 触发记忆提取。

        - memory 未配置(mock/未启用)→ 跳过。
        - interrupted(Esc)→ 跳过(被打断的轮次通常无完整 assistant 回复)。
        - 多轮快速完成时任务排队(Lock 串行),不互相取消;abort/shutdown 才 cancel。
        """
        if self._memory is None or turn.interrupted:
            return
        task = asyncio.create_task(self._memory.extract(turn, self.history))
        self._extract_tasks.add(task)
        task.add_done_callback(self._extract_tasks.discard)

    async def _consume(self, turn: Turn) -> None:
        try:
            # 先开轮次：把本轮 user_text 随事件流带回 UI（修 type-ahead 下
            # _current_user_text 被后一条覆盖、用户气泡标错的竞态）。
            user_text = ""
            if turn.messages and turn.messages[0].content:
                first = turn.messages[0].content[0]
                if isinstance(first, TextBlock):
                    user_text = first.text
            await self._on_event(TurnStart(user_text=user_text))
            # 助手回复组装（含多轮 tool_use→tool_result 续跑）已下沉到 agent_loop；
            # 此处只把每个 ProviderEvent 原样转发给 UI。局部导入规避循环依赖
            # （agent_loop 反向 import conversation）。
            from birdcode.agent.agent_loop import run_agent_loop

            await run_agent_loop(
                provider=self._provider,
                turn=turn,
                history=self.history,
                executor=self._executor,
                emit=self._on_event,
                on_message=self._on_message,
                context=self._context,
            )
        except asyncio.CancelledError:
            raise  # 中断交给 _process 的 CancelledError 路径
        except Exception as exc:
            # 真实 provider 会抛异常（超时/限流/鉴权失败），而非 yield Error；在此
            # 转成 Error 事件，避免异常冒泡杀死 stream task、静默失败。
            log.exception("provider stream failed")
            try:
                await self._on_event(Error(message=str(exc) or exc.__class__.__name__))
            except Exception:
                # 回调自身就是故障源（如 UI 拆卸/异常）→ 不再经同一路径重发，否则
                # Error 重发会再次抛出、逃脱 _consume（_process 只接 CancelledError）
                # 到后台轮次任务（= 未检索异常）。降级为记录即可。
                log.error("on_event failed while reporting Error", exc_info=True)

    async def _process_wake(self) -> None:
        """消费「已投递未响应」的异步子 agent 完成通知,跑一轮主 agent 回复。

        关键不变量(避免重复落盘):通知 user 行由 manager._inject **已写入 store**,
        此处**绝不 append 通知 user 行**——只从 store 读出已落盘行 → 进 turn → 跑 loop
        (loop 仅经 _on_message 追加新 assistant/tool_result 行)→ 写 dequeue 标记消费。
        若误 append,通知会在主 jsonl 出现两次(一次 _inject、一次这里)。

        无 store / 无待响应通知 → no-op(dequeue 已消费的不会再唤醒)。
        """
        from birdcode.session.codec import find_unresponded_notifications

        if self._store is None:
            return
        unresponded = find_unresponded_notifications(self._store.mainline_rows())
        if not unresponded:
            return  # 已全部 dequeue / 无通知 → 不空跑一轮(不污染 history)
        turn = Turn(messages=[u.message for u in unresponded])
        # 每轮重置:上一轮 abort() 残留的 True 不应误吞本轮的 Esc 中断(对齐 _process)。
        self._aborted = False
        self._stream_task = asyncio.create_task(self._consume_wake(turn))
        try:
            await self._stream_task
        except asyncio.CancelledError:
            if not self._aborted:
                # 真正的中断（Esc）才标记并通知 UI；abort()（/clear）静默退出（对齐 _process）。
                turn.interrupted = True
                await self._on_event(Interrupted())
        finally:
            self._stream_task = None
            # turn.failed(wake 轮 provider 异常)→ 不消费通知:不写 dequeue(保持待响应、可重试)、
            # 不入 history(避免下轮上下文出现孤立 user turn)。Esc 中断(interrupted、_aborted=False)
            # 仍走原语义(保留 turn + 写 dequeue)。
            if not self._aborted and not turn.failed:
                self.history.append(turn)
                # 标记消费:为每条待响应通知写一条 dequeue(后续 find_unresponded 不再返回它们,
                # resume/再次 wake 不重复响应)。tool_use_id 留空(主 agent 消费,非工具回执)。
                for u in unresponded:
                    await self._store.append_queue_operation(
                        operation="dequeue",
                        agent_id=u.agent_id,
                        tool_use_id="",
                        status=u.status,
                    )
                self._maybe_extract(turn)

    async def _consume_wake(self, turn: Turn) -> None:
        """wake 轮的流式消费:与 _consume 几乎一致,区别——

        - TurnStart(user_text=""):系统注入(非用户刚打字),UI 不应把它当用户气泡。
        - 通知 user 行已在 turn.messages(来自 store),run_agent_loop 只追加新 assistant/
          tool_result 行并经 _on_message 落盘;不重复 append 通知行。
        """
        try:
            await self._on_event(TurnStart(user_text=""))
            from birdcode.agent.agent_loop import run_agent_loop

            await run_agent_loop(
                provider=self._provider,
                turn=turn,
                history=self.history,
                executor=self._executor,
                emit=self._on_event,
                on_message=self._on_message,
                context=self._context,
            )
        except asyncio.CancelledError:
            raise  # 中断交给 _process_wake 的 CancelledError 路径
        except Exception as exc:
            # 标记本轮失败 → _process_wake 据此跳过 dequeue(通知不被永久消费、下轮可重试)。
            turn.failed = True
            log.exception("wake turn provider stream failed")
            try:
                await self._on_event(Error(message=str(exc) or exc.__class__.__name__))
            except Exception:
                log.error("on_event failed in wake", exc_info=True)
