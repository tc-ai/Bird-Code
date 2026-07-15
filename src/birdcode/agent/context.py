# src/birdcode/agent/context.py
"""上下文管理(Phase 1 安全网):token 估算 + Autocompact + Reactive + 持久化边界。

ContextManager 注入 run_agent_loop:每轮 stream 前评估并按需压缩 history(原地 + 落边界行);
base_llm 拦上下文超限抛 ContextOverflow,agent_loop 捕获 → react 硬截断 → 重试。
设计见 docs/superpowers/specs/2026-07-06-context-management-design.md。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from birdcode.blocks import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    # 仅类型标注用(运行期无循环 import 风险)。
    from birdcode.agent.provider import StreamingProvider
    from birdcode.config.schema import AppConfig
    from birdcode.session.store import SessionStore

log = get_logger("birdcode.context")


class ContextOverflow(Exception):
    """上下文超限(413 / 400 + "prompt is too long")。agent_loop 捕获 → react。"""


@dataclass
class CompactionResult:
    """一次压缩的结果(供 UI 提示 / 日志)。"""

    trigger: str  # "manual" | "auto" | "reactive"(摘要熔断时保留调用方 trigger + fell_back=True)
    pre_tokens: int
    post_tokens: int
    boundary_uuid: str
    summary_uuid: str
    fell_back: bool  # 是否走了硬截断兜底(摘要失败/超限应急)


# 压缩/精简活动回调:(phase, reason, result)。phase ∈ {"start","end"};
# reason ∈ {"auto","snip"}(manual/reactive 不经此回调)。start 时 result=None;
# end 时 result 为本次结果(可能 None:prefix 为空等提前返回不会触发本回调;正常结束非 None)。
# 供 agent_loop 桥接成 CompactionStart/CompactionEnd 事件 → UI 转圈/灰色行(reason 分色)。
CompactActivity = Callable[[str, str, "CompactionResult | None"], Awaitable[None]]


def _is_cjk(ch: str) -> bool:
    """宽泛 CJK 判定(中日韩统一表意 + 兼容区)。"""
    return ord(ch) > 0x2E80


class TokenEstimator:
    """近似 token 估算(不引精确 tokenizer)。

    - 锚点模式(有 last_in):estimate = last_in + char_to_token(delta)。
      last_in = 最近一次 API usage.input_tokens(精确,已含 system+tools+全历史)。
    - 冷启动(无 last_in):全量按 char 估 char_to_token(total) ≈ total/3 混合。
    char→token:ascii/4 + cjk/1.5(保守偏估约 15%,早触发更安全)。
    """

    @staticmethod
    def char_to_token(text: str) -> int:
        """字符 → token 估算:ascii 按 4、CJK 按 1.5 分开计(保守偏估)。"""
        ascii_chars = 0
        cjk_chars = 0
        for ch in text:
            if _is_cjk(ch):
                cjk_chars += 1
            else:
                ascii_chars += 1
        return int(ascii_chars / 4 + cjk_chars / 1.5)

    def _messages_chars(self, messages: list[Message]) -> str:
        """把消息块的文本拼成一段(估算用,非 API payload)。"""
        parts: list[str] = []
        for m in messages:
            for b in m.content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
                elif isinstance(b, ThinkingBlock):
                    parts.append(b.text)
                elif isinstance(b, ToolUseBlock):
                    # input 的字符串形式也占 token
                    parts.append(json.dumps(b.input, ensure_ascii=False))
                elif isinstance(b, ToolResultBlock):
                    parts.append(b.content)
        return "".join(parts)

    def estimate(
        self,
        *,
        history: list[Turn],
        current: list[Message],
        last_in: int | None,
    ) -> int:
        if last_in is not None:
            # 锚点:history 已含在 last_in,只加 current 增量
            return last_in + self.char_to_token(self._messages_chars(current))
        # 冷启动:全量按 char 估
        all_chars = self._messages_chars([m for t in history for m in t.messages] + current)
        return self.char_to_token(all_chars)


# summary 输出里只保留 <summary>…</summary>;<analysis> 草稿块丢弃。DOTALL 跨行匹配。
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
# summary 连续失败(异常 / 空 / 无 <summary> 标签)达此次 → 熔断 → 硬截断兜底。
_MAX_SUMMARY_FAILURES = 3
# 摘要重试指数退避基数(秒):attempt 0→0.5, 1→1.0;仅在还会重试时睡。避免瞬时错误
# (429/超时)连续 3 次后不可逆硬截断。
_BACKOFF_BASE = 0.5

# —— Tier1 snip(确定性清旧 tool_result,零 LLM 成本推迟全量摘要)——
# 触发阈值 = autocompact_threshold × 此比例(接近但未到全量摘要)。
_SNIP_THRESHOLD_RATIO = 0.85
# 保留最近 N 个 tool_result 不清(仿 Claude Code Tier1,留近期上下文)。
_SNIP_KEEP = 5
# 占位符:替换旧 tool_result content。read_history / 重跑工具可取回原文。
_SNIP_PLACEHOLDER = (
    "[此工具结果已折叠以节省上下文 — 需要完整内容请重新运行工具或使用 read_history]"
)


class ContextManager:
    """每轮 stream 前评估并按需压缩 history;413 应急硬截断。

    provider: 持有 complete() 跑无工具摘要(复用 active profile 同模型,质量优先)。
    store: 持久化边界行 + 摘要行;None=纯内存路径(单测/未启用持久化)。
    cfg: 阈值 / 尾段预算 / 摘要预留。
    estimator: token 估算(默认 TokenEstimator())。

    主入口:
      - maybe_compact:每轮 stream 前调;超阈值 → Autocompact(摘要)。
      - compact_now:/compact 手动;无视阈值。
      - react:413 应急;直接硬截断到尾段(不调 LLM)。
    所有异常 catch + log,绝不杀主循环。
    """

    def __init__(
        self,
        provider: StreamingProvider,
        store: SessionStore | None,
        cfg: AppConfig,
        *,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._cfg = cfg
        self._est = estimator or TokenEstimator()
        # 刚构造(resume 首估)/ 刚压缩 / 刚 react 后,last_in 陈旧(= 截断前的大 input),
        # 直接用会误触发 Autocompact。置 True → 下次 maybe_compact 走冷启动估一次、消费清零。
        self._stale_anchor = True

    def set_provider(self, provider: StreamingProvider) -> None:
        """运行时换 provider(/profile 切后端时同步),否则摘要 complete() 仍走旧 profile。"""
        self._provider = provider

    # —— 纯方法(可单测,无 IO)——

    def _split_prefix_tail(self, history: list[Turn]) -> tuple[list[Turn], list[Turn]]:
        """按 Turn 边界切 [prefix(待摘要), tail(原样保留)]。

        tail:从最新 Turn 往回收,收满 compact_tail_budget token 为止,最少保 1 个 Turn。
        切点必在 Turn 边界 → 绝不拆开同一 Turn 内的 tool_use↔tool_result 对(否则 API 400)。

        注意:token 计算必走 estimator._messages_chars(已正确处理 text/thinking/tool_use/
        tool_result 全块型);手写 comprehension 会因 ToolResultBlock 无 .text 崩。
        """
        budget = self._cfg.compact_tail_budget
        tail: list[Turn] = []
        acc = 0
        for turn in reversed(history):
            t_tokens = self._est.char_to_token(self._est._messages_chars(turn.messages))
            if tail and acc + t_tokens > budget:
                break
            tail.append(turn)
            acc += t_tokens
        tail.reverse()
        cut = len(history) - len(tail)
        return history[:cut], tail

    def _extract_summary(self, text: str) -> str | None:
        """从 LLM 输出截 <summary>…</summary>;<analysis> 丢弃。无标签 / 空 → None(计失败)。"""
        m = _SUMMARY_RE.search(text)
        if m is None:
            return None
        out = m.group(1).strip()
        return out or None

    def _summary_system(self) -> str:
        """9 节结构化摘要引导语 + 硬约束(不调工具、不脑补代码)。"""
        return (
            "你在为一个 AI 编码 agent 生成本轮对话的结构化摘要,用于上下文压缩后续接。\n"
            "严格规则:\n"
            "1. 不要调用任何工具。\n"
            "2. 先写 <analysis>…</analysis> 草稿块梳理思路(会被丢弃),再写 "
            "<summary>…</summary> 正式摘要(只保留这个)。\n"
            "3. 摘要按 9 节结构:\n"
            "① 主要请求和意图：用户到底想做什么\n"
            "② 关键技术概念：讨论过的重要技术点\n"
            "③ 文件和代码段：涉及哪些文件，关键代码片段要保留\n"
            "④ 错误和修复：遇到了什么错，怎么解决的\n"
            "⑤ 问题解决过程：解决问题的思路和方法\n"
            "⑥ 所有用户消息：用户说过的所有非工具结果的话（原文保留！）\n"
            "⑦ 待办任务：还没完成的事\n"
            "⑧ 当前工作：最近在做什么（要最详细）\n"
            "⑨ 可能的下一步：接下来打算做什么\n"
            "4. 结尾硬约束:需要文件细节请提示重新 read_file,勿凭摘要脑补代码。"
        )

    def _build_summary_message(self, summary_text: str) -> Message:
        """构造摘要 user 行(续接 prompt + 摘要正文 + 重读/历史提示)。"""
        return Message(
            role="user",
            content=[
                TextBlock(
                    text=(
                        "This session is continued from a previous conversation "
                        "that was compacted.\n"
                        f"Summary:\n{summary_text}\n\n"
                        "(需要文件细节请重新 read_file,勿凭摘要脑补代码。"
                        "需要历史对话原文时调 read_history 分页读取。)"
                    )
                )
            ],
        )

    # —— 持久化(有 store 才做;否则纯内存)——

    async def _persist_compaction(
        self,
        *,
        trigger: str,
        pre_tokens: int,
        post_tokens: int,
        summary_text: str | None,
        logical_parent_uuid: str | None,
        tail_turns: list[Turn],
    ) -> tuple[str, str]:
        """原子写【边界行 + 摘要行 + 重写尾段】,返回 (boundary_uuid, summary_uuid)。

        修 tail-drop + 非原子双写 + 幻影 uuid 三问题(设计偏差):
        - 尾段原样重写到边界之后 → resume 时 split_live_segment 取「边界起」即含尾段,
          装回[摘要 + 尾段],不再丢尾段。
        - boundary/summary/tail 拼一段一次 flush → 不会「只有边界没摘要」。
        - store=None 或落盘失败 → 返回 ("",""),不持久化、不返幻影 uuid。
        summary_text=None(硬截断)→ 摘要行替换为简短截断说明。
        """
        if self._store is None:
            return "", ""
        summary_msg = self._build_summary_messages(summary_text)
        result = await self._store.append_compaction(
            subtype="compact_boundary",
            logical_parent_uuid=logical_parent_uuid,
            content="Conversation compacted",
            compact_metadata={
                "trigger": trigger,
                "preTokens": pre_tokens,
                "postTokens": post_tokens,
                # 尾段轮数:read_history 据此跳过边界前【最后 N 轮】(= 尾段原始副本,已在
                # live 上下文),只读真正被摘要替代、已丢失的 prefix,避免头几页与当前上下文重复。
                "preservedTurnCount": len(tail_turns),
            },
            summary_msg=summary_msg,
            tail_turns=tail_turns,
        )
        return result if result is not None else ("", "")

    def _build_summary_messages(self, summary_text: str | None) -> Message:
        """摘要 user 行:成功用真摘要,硬截断(None)用截断说明。"""
        if summary_text is None:
            return Message(
                role="user",
                content=[
                    TextBlock(
                        text=(
                            "(上下文超限/摘要失败,已应急硬截断,前文未摘要。需要细节请重新读文件;"
                            "如需历史对话原文,调用 read_history 分页读取。)"
                        )
                    )
                ],
            )
        return self._build_summary_message(summary_text)

    def _tail_head_or_none(self, prefix: list[Turn]) -> str | None:
        """压缩前最后一条 uuid(供 boundary 的 logicalParentUuid 续链)。

        Phase 1 简化:用 store._last_uuid(压缩尚未发生时它指向当前 history 末条)。
        无 store / 无 prefix → None。tail_uuids 精确集合 Phase 2 再做。
        """
        if self._store is None or not prefix:
            return None
        return getattr(self._store, "_last_uuid", None)

    # —— 主入口 ——

    async def maybe_compact(
        self,
        *,
        history: list[Turn],
        current: list[Message],
        last_in: int | None,
        on_activity: CompactActivity | None = None,
    ) -> CompactionResult | None:
        """每轮 stream 前调。超阈值 → Autocompact(摘要);未超 → None。

        摘要连续失败 _MAX_SUMMARY_FAILURES 次 → 硬截断兜底(保留调用方 trigger、fell_back=True)。

        on_activity:压缩真正开始/结束时回调(供 UI 转圈)。仅在实际摘要路径触发;
        未超阈值直接 return None 时不触发。
        """
        # 刚压缩/react/resume 后 last_in 陈旧(= 截断前的大 input),直接用会误触发。
        # _stale_anchor 置位时改走冷启动估(消费后清零,下一轮回归真 last_in)。
        last_in_eff: int | None = None if self._stale_anchor else last_in
        self._stale_anchor = False
        est = self._est.estimate(history=history, current=current, last_in=last_in_eff)
        if est >= self._cfg.autocompact_threshold:
            return await self._compact(
                history, trigger="auto", pre_tokens=est, on_activity=on_activity
            )
        # 接近但未到阈值 → Tier1 snip(确定性清旧 tool_result content,零 LLM 成本推迟摘要)。
        if est >= self._tier1_threshold():
            return await self._snip(
                history, current=current, pre_tokens=est, on_activity=on_activity
            )
        return None

    def _tier1_threshold(self) -> int:
        """Tier1 snip 触发阈值 = autocompact_threshold × _SNIP_THRESHOLD_RATIO。

        派生自 autocompact_threshold(基于 context_window 的 property)→ 自动随配置变。
        """
        return int(self._cfg.autocompact_threshold * _SNIP_THRESHOLD_RATIO)

    def _snip_old_tool_results(self, history: list[Turn]) -> int:
        """原地(幂等)把旧 ToolResultBlock.content 替换为占位符,保留最近 _SNIP_KEEP 个。

        按 block 反向计数(并行 tool_result 同一消息可有多个)。返回被替换的 block 数。
        幂等:已占位的 block 仍是 str、被重收集、替换同文案 → net 不变。
        安全:ToolResultBlock.content 是普通可变 str(blocks.py);normalize 孤儿剥离只看 id,
        改 content 不破坏 tool_use↔tool_result 配对;codec 只校验 content 是 str。
        """
        results: list[ToolResultBlock] = [
            b
            for turn in history
            for m in turn.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        if len(results) <= _SNIP_KEEP:
            return 0
        to_replace = results[: len(results) - _SNIP_KEEP]
        for b in to_replace:
            b.content = _SNIP_PLACEHOLDER
        return len(to_replace)

    async def _snip(
        self,
        history: list[Turn],
        *,
        current: list[Message],
        pre_tokens: int,
        on_activity: CompactActivity | None = None,
    ) -> CompactionResult | None:
        """Tier1 snip:清旧 tool_result content(留最近 _SNIP_KEEP 个),零 LLM 成本推迟摘要。

        无 tool_result 可清(replaced==0)→ None(不误触发 UI)。snip 后置 _stale_anchor
        (旧 last_in 反映 snip 前大输入,下轮冷启动估)+ 冷启动重估;若仍 ≥ autocompact_threshold
        (清得不够)→ 转全量 _compact(trigger="auto",走其 auto UI,不双 spinner)。
        否则返回 CompactionResult(trigger="snip")并触发 on_activity(UI 灰色行)。
        """
        replaced = self._snip_old_tool_results(history)
        if replaced == 0:
            return None
        self._stale_anchor = True
        post = self._est.estimate(history=history, current=current, last_in=None)
        if post >= self._cfg.autocompact_threshold:
            return await self._compact(
                history, trigger="auto", pre_tokens=post, on_activity=on_activity
            )
        result = CompactionResult(
            trigger="snip",
            pre_tokens=pre_tokens,
            post_tokens=post,
            boundary_uuid="",
            summary_uuid="",
            fell_back=False,
        )
        if on_activity is not None:
            await on_activity("start", "snip", None)
            await on_activity("end", "snip", result)
        return result

    async def compact_now(
        self, *, history: list[Turn], trigger: str = "manual"
    ) -> CompactionResult | None:
        """/compact 手动:无视阈值走 Autocompact(含摘要)。"""
        est = self._est.estimate(history=history, current=[], last_in=None)
        return await self._compact(history, trigger=trigger, pre_tokens=est)

    async def react(self, *, history: list[Turn]) -> CompactionResult:
        """413 应急:直接硬截断到尾段(**不调 LLM**),写 trigger=reactive 边界行。

        热路径要快且必成功——不等摘要。fell_back 恒 True。
        """
        return await self._hard_truncate(history, trigger="reactive")

    async def _compact(
        self,
        history: list[Turn],
        *,
        trigger: str,
        pre_tokens: int,
        on_activity: CompactActivity | None = None,
    ) -> CompactionResult | None:
        """Autocompact 主体:切 prefix/tail → 试摘要(最多 3 次,指数退避)→ 成功则原地替换。

        prefix 为空(没东西可摘要)→ 返回 None(不触发 on_activity)。
        摘要连续失败 _MAX_SUMMARY_FAILURES 次 → 走 _hard_truncate 兜底(fell_back=True)。
        on_activity:真正开始摘要时触发 "start",结束时(含熔断兜底)触发 "end" + result。
        """
        prefix, tail = self._split_prefix_tail(history)
        if not prefix:  # 尾段即全部(单 turn 巨大 / 会话小)→ 没东西可摘要
            log.info("compact:无可摘要的前段(尾段即全部),跳过")
            return None  # 不写假边界;react 走 _hard_truncate、不经此分支

        if on_activity is not None:
            await on_activity("start", trigger, None)
        result: CompactionResult | None = None
        try:
            summary_text: str | None = None
            for attempt in range(_MAX_SUMMARY_FAILURES):
                try:
                    flat_prefix = [m for t in prefix for m in t.messages]
                    raw = await self._provider.summarize_with_prefix(
                        prefix=flat_prefix,
                        instruction=self._summary_system(),
                        max_tokens=self._cfg.compact_summary_reserve,
                    )
                    summary_text = self._extract_summary(raw)
                    if summary_text is not None:
                        break
                    # 提取失败给诊断(可能 reserve 不足致 </summary> 未闭合,或模型未产标签)
                    log.warning(
                        "summary 提取失败(无 <summary> 标签;可能 reserve 过小)"
                    )
                except Exception:  # noqa: BLE001 - 摘要失败不杀主循环,计数熔断
                    log.warning(
                        "summary 调用失败,将重试(达 %d 次熔断)",
                        _MAX_SUMMARY_FAILURES,
                        exc_info=True,
                    )
                # 重试前指数退避(仅当还会重试),避免瞬时错误(429/超时)连续 3 次后硬截断
                if summary_text is None and attempt < _MAX_SUMMARY_FAILURES - 1:
                    await asyncio.sleep(_BACKOFF_BASE * (2**attempt))

            if summary_text is None:
                log.warning("summary 连续失败 %d 次,熔断 → 硬截断兜底", _MAX_SUMMARY_FAILURES)
                # 保留调用方 trigger(manual/auto),用 fell_back=True 标识走了兜底
                result = await self._hard_truncate(history, trigger=trigger)
                return result

            # 成功:持久化(含重写尾段)+ 原地替换 history = [摘要Turn, *tail]
            logical_parent = self._tail_head_or_none(prefix)
            post_tokens = self._est.estimate(history=tail, current=[], last_in=None)
            boundary_uuid, summary_uuid = await self._persist_compaction(
                trigger=trigger,
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                summary_text=summary_text,
                logical_parent_uuid=logical_parent,
                tail_turns=tail,
            )
            summary_turn = Turn(messages=[self._build_summary_message(summary_text)])
            history[:] = [summary_turn, *tail]  # 原地替换(调用方 list 可见)
            self._stale_anchor = True  # history 刚变,旧 last_in 失效 → 下次冷启动估
            result = CompactionResult(
                trigger=trigger,
                pre_tokens=pre_tokens,
                post_tokens=self._est.estimate(history=history, current=[], last_in=None),
                boundary_uuid=boundary_uuid,
                summary_uuid=summary_uuid,
                fell_back=False,
            )
            return result
        finally:
            if on_activity is not None:
                await on_activity("end", trigger, result)

    async def _hard_truncate(self, history: list[Turn], *, trigger: str) -> CompactionResult:
        """兜底:丢掉 tail 之前的全部(不调 LLM、必成功),history 原地替换为 tail。

        trigger=reactive(413 应急)/ 或调用方 trigger(摘要熔断时透传 manual|auto,fell_back=True)。
        摘要行写截断说明(不调 LLM)。fell_back 恒 True。
        """
        prefix, tail = self._split_prefix_tail(history)
        pre = self._est.estimate(history=history, current=[], last_in=None)
        logical_parent = self._tail_head_or_none(prefix)
        post_tokens = self._est.estimate(history=tail, current=[], last_in=None)
        boundary_uuid, summary_uuid = await self._persist_compaction(
            trigger=trigger,
            pre_tokens=pre,
            post_tokens=post_tokens,
            summary_text=None,
            logical_parent_uuid=logical_parent,
            tail_turns=tail,
        )
        history[:] = list(tail)  # 原地替换(调用方 list 可见)
        self._stale_anchor = True  # history 刚截断,旧 last_in 失效 → 下次冷启动估
        log.warning(
            "上下文硬截断(trigger=%s):%d → %d turns",
            trigger,
            len(prefix) + len(tail),
            len(tail),
        )
        return CompactionResult(
            trigger=trigger,
            pre_tokens=pre,
            post_tokens=post_tokens,
            boundary_uuid=boundary_uuid,
            summary_uuid=summary_uuid,
            fell_back=True,
        )


# ---- /context 命令:上下文占用快照 ----


@dataclass
class ContextUsage:
    """/context 用的上下文占用快照。

    各类目 token 均为**估算**(char→token,非精确 tokenizer)。last_measured 是最近一次
    API 回包的真实 usage.input_tokens(含 system+tools+全历史、但无法拆类目),作参考脚注。
    mcp_tools 是 tools 的子集(mcp__ 前缀那部分),不额外计入 used(避免重复)。
    """

    window: int
    autocompact_threshold: int
    system_prompt: int
    tools: int  # 全部工具 schema(含 MCP)
    tool_count: int
    mcp_tools: int  # MCP 子集(mcp__ 前缀),是 tools 的一部分,不计入 used
    mcp_tool_count: int
    messages: int
    last_measured: int | None = None

    @property
    def used(self) -> int:
        """已用 = system + tools(含 MCP) + messages。"""
        return self.system_prompt + self.tools + self.messages

    @property
    def free(self) -> int:
        return max(0, self.window - self.used)


def estimate_context_usage(
    *,
    window: int,
    autocompact_threshold: int,
    system_text: str,
    tools: list[dict[str, object]],
    mcp_tools: list[dict[str, object]],
    history: list[Turn],
    last_measured: int | None = None,
) -> ContextUsage:
    """/context 用:分类目 token 估算(system_prompt / tools / messages)。

    有 last_measured(API 实测全量)→ messages 反推,使 Used 锚定真实(与 maybe_compact
    的 last_in 同源),消除「char 粗估虚高/虚低」误导:面板到阈值即真触发。无 last_measured
    (冷启动)→ 全量按 char 估(老逻辑)。system_prompt/tools/mcp_tools 始终 char 估,作结构参考。
    """
    est = TokenEstimator()
    sys_est = est.char_to_token(system_text)
    tools_est = est.char_to_token(json.dumps(tools, ensure_ascii=False))
    if last_measured is not None:
        # 实测全量已含 system+tools+全历史;messages 反推保证分类目加和 = Used = 实测。
        # last_measured < 估算开销(char 估偏高 / 小会话)→ floor 0,Used 回退估算(小、无妨)。
        messages_est = max(0, last_measured - sys_est - tools_est)
    else:
        messages_est = est.estimate(history=history, current=[], last_in=None)
    return ContextUsage(
        window=window,
        autocompact_threshold=autocompact_threshold,
        system_prompt=sys_est,
        tools=tools_est,
        tool_count=len(tools),
        mcp_tools=est.char_to_token(json.dumps(mcp_tools, ensure_ascii=False)),
        mcp_tool_count=len(mcp_tools),
        messages=messages_est,
        last_measured=last_measured,
    )
