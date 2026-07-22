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
# 每次 snip 至少要清这么多 token 才值当(改 content 会打掉 prompt cache,清太少不划算)。
# 未超阈 + 可清总量 < 此值 → 不动,留给将来自然压缩(对齐 CC clear_at_least)。
_SNIP_CLEAR_AT_LEAST = 2000
# 占位符:替换旧 tool_result content。read_history / 重跑工具可取回原文。
_SNIP_PLACEHOLDER = "[此工具结果已折叠以节省上下文 — 需要完整内容请重新运行工具或使用 read_history]"


@dataclass
class SnipCandidate:
    """snip 决策用的 tool_result 快照(瞬时:每次 _snip 从 history 现场建,用完丢)。

    size/replaced 从 block.content 派生(单一真相源,无跨轮存储);block 是原地改
    content 的反向引用。
    """

    tool_use_id: str
    size: int  # len(block.content):当前占用 context 的真实大小(非原始 size)
    replaced: bool  # block.content == _SNIP_PLACEHOLDER
    block: ToolResultBlock


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
        enable_snip: bool = True,
    ) -> None:
        self._provider = provider
        self._store = store
        self._cfg = cfg
        self._est = estimator or TokenEstimator()
        self._enable_snip = enable_snip
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
        # enable_snip=False(子 agent 只要全量压缩)→ 跳过 snip,留给 _compact 撞阈时一次性摘要。
        if self._enable_snip and est >= self._tier1_threshold():
            return await self._snip(history, current=current, on_activity=on_activity)
        return None

    def _tier1_threshold(self) -> int:
        """Tier1 snip 触发阈值 = autocompact_threshold × _SNIP_THRESHOLD_RATIO。

        派生自 autocompact_threshold(基于 context_window 的 property)→ 自动随配置变。
        下界 0:小窗口 + 默认 reserve/margin 可使 autocompact_threshold 为负,无下界则
        est >= tier1 恒真 → 每轮误进 snip(虽 replaced==0 时仍 no-op,但白算一遍)。
        """
        return max(int(self._cfg.autocompact_threshold * _SNIP_THRESHOLD_RATIO), 0)

    def _build_candidates(self, history: list[Turn]) -> list[SnipCandidate]:
        """遍历 history,为每个 ToolResultBlock 建 SnipCandidate(瞬时派生)。

        返回顺序 = history 顺序(供 _protected_ids 按 recency 补足用)。ToolUseBlock 不产生候选
        (snip 只改 tool_result content,无需 id→name 配对)。
        """
        out: list[SnipCandidate] = []
        for turn in history:
            for m in turn.messages:
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        out.append(
                            SnipCandidate(
                                tool_use_id=b.tool_use_id,
                                size=len(b.content),
                                replaced=b.content == _SNIP_PLACEHOLDER,
                                block=b,
                            )
                        )
        return out

    def _protected_ids(self, history: list[Turn]) -> set[str]:
        """保护集 = 最近完整 turn(history[-1])全部 tool_result + 按 recency 补足 _SNIP_KEEP。

        history[-1] 是最近一个完整 turn(流结束才 append,见 conversation.py);其 tool_result
        是模型除当前轮外最新鲜上下文,必保。不足 _SNIP_KEEP 时按 recency(倒序)补足。
        当前在跑的 turn 在 current、不在 history、天然不被 snip。
        """
        protected: set[str] = set()
        if not history:
            return protected
        for m in history[-1].messages:
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    protected.add(b.tool_use_id)
        if len(protected) < _SNIP_KEEP:
            for turn in reversed(history[:-1]):
                for m in reversed(turn.messages):
                    for b in reversed(m.content):
                        if isinstance(b, ToolResultBlock) and b.tool_use_id not in protected:
                            protected.add(b.tool_use_id)
                            if len(protected) >= _SNIP_KEEP:
                                return protected
        return protected

    async def _snip(
        self,
        history: list[Turn],
        *,
        current: list[Message],
        on_activity: CompactActivity | None = None,
    ) -> CompactionResult | None:
        """Tier2 snip:按当前内联 size 优先清最大的 tool_result(保护集),零 LLM 成本推迟摘要。

        保护集 = 最近一个完整 turn(history[-1])的 tool_result + 按 recency 补足 _SNIP_KEEP。
        瞬时派生 SnipCandidate(单一真相源 block.content)。pre-check:max_clearable <
        max(deficit, _SNIP_CLEAR_AT_LEAST) → deficit>0 时 _compact(超阈救不了)/ 否则 None
        (未超阈不值当)。否则贪心清最大的,停于 est_now < 阈值 且 cleared ≥ 最小清理量。
        消除旧 restore 路径(pre-check 失败即直转 _compact,未创建占位 → 摘要看原文)。
        无可清(snippable 空)→ None(幂等:不误触发 UI/_stale_anchor)。
        """
        candidates = self._build_candidates(history)
        protected = self._protected_ids(history)
        snippable = [c for c in candidates if not c.replaced and c.tool_use_id not in protected]
        if not snippable:
            return None
        snippable.sort(key=lambda c: c.size, reverse=True)  # 大的优先清

        # pre/post 均 cold 估(同口径,避免 anchored pre + cold post 混口径致 UI post>pre)。
        pre_cold = self._est.estimate(history=history, current=current, last_in=None)
        # 每块清掉后净减 token = char_to_token(原 content) − char_to_token(占位符)。
        # 与 estimator 同口径(都 char_to_token 扫真实 char),只扫 snippable,免二次全量估。
        ph_tokens = self._est.char_to_token(_SNIP_PLACEHOLDER)
        clearable_tokens = [
            max(0, self._est.char_to_token(c.block.content) - ph_tokens) for c in snippable
        ]
        max_clearable = sum(clearable_tokens)
        deficit = pre_cold - self._cfg.autocompact_threshold
        required = max(deficit, _SNIP_CLEAR_AT_LEAST)

        if max_clearable < required:
            if deficit > 0:
                # cold 估实际超阈但 snip 清不够 → 直接全量摘要(不占位、不 restore)。
                return await self._compact(
                    history, trigger="auto", pre_tokens=pre_cold, on_activity=on_activity
                )
            return None  # 未超阈 + 清不够最小量 → 不动,留给将来自然压缩

        # 贪心:清最大的,直到 est_now < 阈值 且 cleared ≥ 最小清理量(pre-check 保证可达)。
        cleared = 0
        for idx, c in enumerate(snippable):
            # 占位符比原文还长的小块(clearable≤0)清了反涨 context → 跳过(不占位、不计 cleared)。
            # 刀锋情形(deficit==max_clearable,清完大块恰好压到阈值、==而非 <)下,无此守卫会继续
            # 把尾部小块换成更长的占位符 → 原文丢失、post 可超 pre。大块在 sort 前部,此处理论上
            # 只遍历到尾部小块(全小块时 max_clearable=0 < required,已被 pre-check 分流)。
            if clearable_tokens[idx] <= 0:
                continue
            if (
                pre_cold - cleared < self._cfg.autocompact_threshold
                and cleared >= _SNIP_CLEAR_AT_LEAST
            ):
                break
            c.block.content = _SNIP_PLACEHOLDER
            cleared += clearable_tokens[idx]

        self._stale_anchor = True  # 旧 last_in 反映 snip 前大输入,下轮冷启动估
        # post 由 cleared 推算(char_to_token 对拼接可加,误差仅每块一次 int 截断的微量噪声;clearable
        # 均 max(0,·) → cleared≥0 → post≤pre_cold,UI 不会 post>pre,且与 pre 同 cold 口径)。省一次
        # 全量 O(history) cold 估。
        post = max(0, pre_cold - cleared)
        result = CompactionResult(
            trigger="snip",
            pre_tokens=pre_cold,
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
            # prefix 在重试循环内不变,提出循环外避免重复 rebuild。
            flat_prefix = [m for t in prefix for m in t.messages]
            for attempt in range(_MAX_SUMMARY_FAILURES):
                try:
                    raw = await self._provider.summarize_with_prefix(
                        prefix=flat_prefix,
                        instruction=self._summary_system(),
                        max_tokens=self._cfg.compact_summary_reserve,
                    )
                    summary_text = self._extract_summary(raw)
                    if summary_text is not None:
                        break
                    # 提取失败给诊断(可能 reserve 不足致 </summary> 未闭合,或模型未产标签)
                    log.warning("summary 提取失败(无 <summary> 标签;可能 reserve 过小)")
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
