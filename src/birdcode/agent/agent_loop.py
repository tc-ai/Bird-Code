# src/birdcode/agent/agent_loop.py
"""多轮工具循环:stream → 遇 tool_use → executor 执行 → 回填 tool_result → 续 stream。

核心是一个清晰的 while。增量事件(TextDelta/ThinkingDelta)原样 emit 给 UI 做流式显示,
同时用 _AssistantAccumulator 拼成单个完整 block 进 history。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from birdcode.agent.context import ContextOverflow  # 运行期:raise / except
from birdcode.agent.provider import (
    CompactionEnd,
    CompactionStart,
    Done,
    EditDiff,
    Error,
    ProviderEvent,
    StreamingProvider,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
)
from birdcode.agent.provider import (
    ToolResult as ToolResultEvent,
)
from birdcode.blocks import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from birdcode.conversation import Message, Turn
from birdcode.tools.base import ToolResult
from birdcode.tools.executor import ToolExecutor
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    # 仅类型标注(参数 hint);运行期避免循环 import,context 模块只引用 blocks/conversation。
    from birdcode.agent.context import CompactionResult, ContextManager

log = get_logger("birdcode.agent.cache")

Emit = Callable[[ProviderEvent], Awaitable[None]]
OnMessage = Callable[..., Awaitable[None]]  # (msg, *, usage=None, stop_reason=None) -> None

# 护栏:防失控与死循环(见 run_agent_loop 末尾)。
_MAX_TOOL_ROUNDS = 200   # 轮次硬上限(200 宽松,不误杀复杂任务;到顶强制停防失控烧 token)
_MAX_SAME_FAILURES = 3   # 连续熔断:同一工具 + 相同参数连续失败 N 次→停(防模型死循环重试)
# Reactive(413)重试上限:overflow→react→重试最多 2 次,仍连续超限→emit Error(防死循环)。
_MAX_REACT_RETRIES = 2


def _canonical_input(inp: object) -> str:
    """工具参数规范字串(「相同参数」判定基准)。

    dict→sort_keys json(键序无关比较);其余→str。使同名同参调用签名一致,供连续失败计数。
    """
    if isinstance(inp, dict):
        return json.dumps(inp, sort_keys=True, ensure_ascii=False)
    return str(inp)


def _summarize_tool(tu: ToolUseBlock, output: str, ok: bool) -> str:
    """工具行简短摘要(给 UI 的 head,不是正文)。

    不能用 output[:200] —— Read 的 output 是带行号的文件正文,塞进 head 会导致:
    折叠态把文件前几行泄露在标题行、展开态 head 与 body 重复显示。改按工具语义给
    简短描述(Read→文件名·行数;Grep→命中数;Bash→命令…)。
    """
    if not ok:
        return output[:200]  # 错误信息直接给前 200 字,让用户看到失败原因
    inp = tu.input if isinstance(tu.input, dict) else {}
    if tu.name == "read_file":
        path = str(inp.get("file_path", ""))
        name = Path(path).name if path else ""
        lines = len(output.splitlines())
        return f"{name} · {lines} 行" if name else f"{lines} 行"
    if tu.name == "grep":
        return f"{len(output.splitlines())} 处命中" if output else "无命中"
    if tu.name == "glob":
        return f"{len(output.splitlines())} 个文件" if output else "无匹配"
    if tu.name in ("write_file", "edit_file"):
        path = str(inp.get("file_path", ""))
        return Path(path).name if path else tu.name
    if tu.name == "bash":
        cmd = str(inp.get("command", ""))[:60]
        return f"$ {cmd}" if cmd else tu.name
    return output.split("\n", 1)[0][:80] if output else tu.name


def _latest_usage(history: list[Turn], turn: Turn) -> int | None:
    """取最近一次 usage.input_tokens 作估算锚点(先看 turn.usage,再看 history 末尾)。

    锚点 = 上一次真实 API 回包的 input_tokens(精确,已含 system+tools+全历史)。
    ContextManager.maybe_compact 用它做增量估算基线(last_in);无锚点(None)→ 冷启动
    全量按 char 估。优先 turn.usage(本轮已 stream 过的中间态),否则 history[-1]。
    """
    u = turn.usage or (history[-1].usage if history else None)
    return u.input_tokens if u is not None and u.input_tokens else None


class _AssistantAccumulator:
    """把流式增量碎片累积成完整 content blocks。thinking 在 text/tool_use 之前。"""

    def __init__(self) -> None:
        self.blocks: list[ContentBlock] = []
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._signature: str = ""

    def add_text(self, text: str) -> None:
        self._text.append(text)

    def add_thinking(self, text: str, signature: str) -> None:
        if text:
            self._thinking.append(text)
        if signature:
            self._signature = signature

    def flush(self) -> None:
        if self._thinking:
            self.blocks.append(
                ThinkingBlock(text="".join(self._thinking), signature=self._signature)
            )
            self._thinking.clear()
            self._signature = ""
        if self._text:
            self.blocks.append(TextBlock(text="".join(self._text)))
            self._text.clear()


async def run_agent_loop(
    *,
    provider: StreamingProvider,
    turn: Turn,
    history: list[Turn],
    executor: ToolExecutor | None,
    emit: Emit,
    on_message: OnMessage | None = None,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
    max_same_failures: int = _MAX_SAME_FAILURES,
    context: ContextManager | None = None,
) -> None:
    """provider 需实现 stream(messages, *, history) -> AsyncIterator[ProviderEvent]。

    on_message:每追加一条 Message(assistant / tool_result user)经此回调落盘。

    context(Phase 1 安全网,可选):上下文管理器。每轮 stream 前调 maybe_compact
    评估压缩(超阈→摘要);provider 抛 ContextOverflow(413 / 400 'prompt is too long')
    时调 react 硬截断 history → 重试同轮 stream。None=未启用持久化 / 旧路径,行为不变
    (ContextOverflow 直接抛交上层)。详见 birdcode.agent.context.ContextManager。

    护栏(防失控 / 死循环):
    - max_tool_rounds:工具轮次硬上限(默认 200),到顶 emit Error 停止——防模型反复调
      工具不停、失控烧 token。200 宽松,不误杀复杂任务。
    - max_same_failures:同一工具 + 相同参数连续失败 N 次(默认 3)→ emit Error 熔断——
      防模型盯着一个失败调用死循环重试。任何成功或签名(工具名/参数)变化都重置计数,
      给模型换方法/换参数的机会。
    - _MAX_REACT_RETRIES:Reactive(413)重试上限(默认 2)。react 硬截断 history 后重试
      stream,若截了 tail 仍连续超限(估算误差 / 尾段仍太大)→ 不再 react、emit Error
      + return,防 react→continue→overflow 死循环。
    """
    rounds = 0
    last_fail_sig: tuple[str, str] | None = None  # (tool_name, canonical_input)
    fail_streak = 0
    react_retries = 0  # Reactive(413)重试计数:每轮 turn 进入时 reset;> _MAX_REACT_RETRIES 则停

    async def _on_compact_activity(phase: str, reason: str, cres: object) -> None:
        """maybe_compact 的活动回调 → emit CompactionStart/End(UI 转圈/灰色行,reason 分色)。

        reason ∈ {"auto","snip"}(maybe_compact 的 _compact / snip 路径);manual /compact 不经事件流。
        """
        if phase == "start":
            await emit(CompactionStart(reason=reason))
            return
        summary = ""
        fell_back = False
        if cres is not None:
            c = cast("CompactionResult", cres)
            summary = (
                f"已折叠旧工具结果 {c.pre_tokens}→{c.post_tokens} token"
                if reason == "snip"
                else f"已压缩 {c.pre_tokens}→{c.post_tokens} token"
            )
            fell_back = c.fell_back
        await emit(CompactionEnd(reason=reason, summary=summary, fell_back=fell_back))

    while True:
        # 上下文管理(Phase 1 安全网):每轮 stream 前评估压缩(超阈→Autocompact 摘要)。
        # 无 context(未启用持久化 / 旧路径)→ 跳过,行为完全不变。压缩路径任何异常都
        # catch + log,绝不抛 Traceback 杀主循环。
        if context is not None:
            try:
                await context.maybe_compact(
                    history=history,
                    current=turn.messages,
                    last_in=_latest_usage(history, turn),
                    on_activity=_on_compact_activity,
                )
            except Exception:  # noqa: BLE001 - 压缩失败不杀主循环
                log.exception("maybe_compact 失败,跳过本轮压缩")
        acc = _AssistantAccumulator()
        tool_uses: list[ToolUseBlock] = []
        stop_reason: str | None = None
        try:
            async for ev in provider.stream(list(turn.messages), history=history):
                await emit(ev)
                if isinstance(ev, TextDelta):
                    acc.add_text(ev.text)
                elif isinstance(ev, ThinkingDelta):
                    acc.add_thinking(ev.text, ev.signature)
                elif isinstance(ev, ToolCallStart):
                    acc.flush()
                    try:
                        inp = json.loads(ev.args or "{}")
                    except json.JSONDecodeError:
                        inp = {}
                    tu = ToolUseBlock(id=ev.id, name=ev.name, input=inp)
                    tool_uses.append(tu)
                    acc.blocks.append(tu)
                elif isinstance(ev, Done):
                    stop_reason = ev.stop_reason
                    if ev.usage is not None:
                        turn.usage = ev.usage
                        u = ev.usage
                        log.info(
                            "[cache] in=%d out=%d read=%d create=%d cost=%.4f",
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_tokens,
                            u.cache_creation_tokens,
                            u.cost_usd,
                        )
        except ContextOverflow:
            # 413 / 超限:react 硬截断 history → 重试同轮 stream。
            # 关键不变量:413 在吐 token 前发生,故 _AssistantAccumulator 是空的、
            # turn.messages 也还没 append assistant 消息(append 在下面 acc.flush() 之后)
            # → continue 重试是干净的,不会重复 append / 误留半截 assistant。
            # 无 context(未启用持久化)→ 原样抛交上层(行为与旧路径完全一致)。
            if context is None:
                raise
            react_retries += 1
            if react_retries > _MAX_REACT_RETRIES:
                # 截了 tail 仍连续超限(估算误差 / 尾段仍太大)→ 不再 react,防死循环。
                # 提示用户 /clear 开新会话或手动 /compact 后重试。
                await emit(
                    Error(
                        message=(
                            "上下文连续超限,已重试压缩上限仍无法继续。"
                            "建议 /clear 开新会话,或手动 /compact 后重试。"
                        )
                    )
                )
                return
            try:
                await context.react(history=history)
            except Exception:  # noqa: BLE001 - react 自身失败无救,原样抛
                log.exception("react 硬截断失败")
                raise
            continue  # 重试本轮 stream(不 increment rounds;maybe_compact 会再调一次)
        acc.flush()
        assistant_msg = Message(role="assistant", content=acc.blocks)
        turn.messages.append(assistant_msg)
        if on_message is not None:
            await on_message(assistant_msg, usage=turn.usage, stop_reason=stop_reason)

        # 有 tool_use 就执行(不论 stop_reason)——避免 provider 异常发了 ToolCallStart 但
        # stop_reason != tool_use 时,留下无 tool_result 的 tool_use 导致下轮 API 报 400。
        if not tool_uses:
            return

        # ESC 中断以 CancelledError 穿透 execute_batch(executor 收窄 except 后 serial/parallel
        # 两分支都透传)。若不兜底,assistant(tool_use) 会在 history 里悬空、无对应 tool_result,
        # 下一轮 API 报 400。故捕获后为所有 tool_use 合成 error result 回填,再原样 raise 交
        # _process 标 interrupted + 发 Interrupted。本路径不 emit ToolResult——"⚡ 中断" UI 由
        # Interrupted 事件接管(app.py 只收拾没收到 ToolResult 的转圈行),与现有 UX/测试一致。
        cancelled: asyncio.CancelledError | None = None
        results: list[ToolResult]
        try:
            results = (
                await executor.execute_batch(tool_uses)
                if executor is not None
                else [
                    ToolResult(tu.id, ok=False, error_message="未配置工具执行器")
                    for tu in tool_uses
                ]
            )
        except asyncio.CancelledError as exc:
            cancelled = exc
            # 已完成的工具保留真实结果(executor.last_partial,仅串行工具):避免已落盘的
            # write/edit 被当"未执行"→ 模型下轮重试造成重复写入。未完成的(cancellation 落点及
            # 之后)回填 error,
            # 但**不说"未执行"**——bash 等命令非幂等、ESC 杀进程组杀在中间已改部分文件(文件系统
            # 不回滚),谎称"未执行"会致模型盲目重试 rm/mv/>>/sed 造成损害。故文案提示"可能已部分
            # 完成,请核查",让模型 read_file/ls/git status 重建认知再补救。悬空兜底不变(每个
            # tool_use 仍配 tool_result,不致下轮 API 400)。
            partial = executor.last_partial() if executor is not None else {}
            results = [
                partial.get(tu.id)
                or ToolResult(
                    tu.id,
                    ok=False,
                    error_message=(
                        "用户中断(Esc):执行可能在完成前被终止,已改/已删的文件不会回滚。"
                        "重试前请先核查实际状态(read_file / ls / git status)。"
                    ),
                )
                for tu in tool_uses
            ]

        result_blocks: list[ContentBlock] = []
        for tu, r in zip(tool_uses, results, strict=True):
            if r.ok:
                content = r.llm_content or r.output  # 双轨:优先 llm_content(超阈占位)
            else:
                content = r.error_message or r.output
            result_blocks.append(
                ToolResultBlock(
                    tool_use_id=tu.id,
                    content=content,
                    is_error=not r.ok,
                )
            )
            if cancelled is None:  # 正常完成才 emit;中断路径的 UI 由 Interrupted 事件接管
                summary = _summarize_tool(tu, r.output or r.error_message, r.ok)
                await emit(
                    ToolResultEvent(
                        id=tu.id,
                        ok=r.ok,
                        summary=summary,
                        output=r.output or r.error_message,  # UI 仍用 output(截断版)
                        truncated=r.truncated,
                    )
                )
                if r.ok and r.diff is not None:
                    await emit(EditDiff(id=tu.id, old=r.diff[0], new=r.diff[1]))
        result_msg = Message(role="user", content=result_blocks)
        turn.messages.append(result_msg)
        if on_message is not None:
            await on_message(result_msg)
        # 下一轮 stream 自动带上 assistant(tool_use)+ user(tool_result)

        if cancelled is not None:
            raise cancelled

        # —— 护栏(仅正常完成路径;中断已 raise)——
        # 1) 连续同一失败签名熔断:同一工具 + 相同参数连续失败 N 次 → 停(防死循环重试)。
        #    任何成功 / 签名变化都重置 streak(模型换了方法或参数,给机会)。
        for tu, r in zip(tool_uses, results, strict=True):
            if r.ok:
                last_fail_sig = None
                fail_streak = 0
            else:
                sig = (tu.name, _canonical_input(tu.input))
                if sig == last_fail_sig:
                    fail_streak += 1
                else:
                    last_fail_sig = sig
                    fail_streak = 1
        if fail_streak >= max_same_failures:
            assert last_fail_sig is not None  # fail_streak>=1 必有失败签名
            await emit(
                Error(
                    message=(
                        f"连续 {fail_streak} 次同一工具调用失败"
                        f"(工具={last_fail_sig[0]},相同参数),已停止以防死循环。"
                        "请核查工具参数 / 权限,或换一种方式后重试。"
                    )
                )
            )
            return
        # 2) 轮次硬上限:防模型反复调工具不停、失控烧 token。到顶强制停 + 提示。
        rounds += 1
        if rounds >= max_tool_rounds:
            await emit(
                Error(
                    message=(
                        f"已达最大工具轮次({max_tool_rounds}),已停止以防失控。"
                        "若任务未完成,请拆分或重新提问继续。"
                    )
                )
            )
            return
