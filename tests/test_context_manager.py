# tests/test_context_manager.py
"""T8: ContextManager 核心 —— tail 切割 + Autocompact + 摘要提取 + 熔断 + Reactive + /compact。

store=None 走纯内存路径(不落盘),其余逻辑(history 原地替换、CompactionResult)照常,
使单测无需真实 store。asyncio_mode=auto,直接 await。
"""

from birdcode.agent.context import ContextManager, TokenEstimator
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig
from birdcode.conversation import Message, Turn


def _turn(text: str) -> Turn:
    return Turn(messages=[Message(role="user", content=[TextBlock(text=text)])])


class _SummaryProvider:
    """假 provider:complete 返回带 <analysis>/<summary> 的文本。

    fail_times>0 时前 N 次抛错(测熔断);成功时回 <summary> 包裹文本。
    仅实现 complete(ContextManager 只用它),无 stream。
    """

    def __init__(self, summary_text: str = "THE SUMMARY", *, fail_times: int = 0) -> None:
        self._summary = summary_text
        self._fail_times = fail_times
        self._calls = 0

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("summary LLM boom")
        return f"<analysis>draft</analysis><summary>{self._summary}</summary>"

    async def summarize_with_prefix(
        self,
        *,
        prefix: list,
        instruction: str,
        max_tokens: int,  # noqa: ANN001
    ) -> str:
        # 与 complete 同语义(共享 _calls + 失败注入),供 _compact 新调用路径(复用 prefix 缓存)测试。
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("summary LLM boom")
        return f"<analysis>draft</analysis><summary>{self._summary}</summary>"


def _make_cfg(**overrides: object) -> AppConfig:
    base: dict[str, object] = {
        "providers": {
            "p": {"protocol": "anthropic", "model": "m", "base_url": "u", "api_key": "k"}
        },
        "default": "p",
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def _cm(provider=None, **overrides: object) -> ContextManager:
    return ContextManager(
        provider=provider or _SummaryProvider(),
        store=None,
        cfg=_make_cfg(**overrides),
        estimator=TokenEstimator(),
    )


# —— 纯方法 ——


def test_split_tail_respects_turn_boundary_and_budget():
    """tail 切割:从最新 Turn 往回收满 budget 停;切点在 Turn 边界(不拆 tool pair)。"""
    cm = _cm()
    turns = [_turn("a"), _turn("b" * 200_000), _turn("c")]  # 中间 turn 巨大
    prefix, tail = cm._split_prefix_tail(turns)
    # tail 至少含最后一个 turn;prefix 含前面的;切点必在 turn 边界
    assert tail[-1] is turns[-1]
    assert all(t in turns for t in prefix + tail)
    # tail 总 token ≤ budget,或 tail 恰为唯一(最少保 1)的最后一个 turn
    tail_tokens = sum(cm._est.char_to_token(cm._est._messages_chars(t.messages)) for t in tail)
    assert tail_tokens <= cm._cfg.compact_tail_budget or len(tail) == 1


def test_summary_extraction_drops_analysis_keeps_summary():
    cm = _cm()
    out = cm._extract_summary("<analysis>scratch</analysis><summary>KEEP-ME</summary>")
    assert out is not None
    assert "KEEP-ME" in out
    assert "scratch" not in out


def test_summary_extraction_returns_none_when_no_summary_tag():
    cm = _cm()
    assert cm._extract_summary("no tags here") is None


# —— maybe_compact ——


async def test_maybe_compact_skips_when_under_threshold():
    cm = _cm()
    turns = [_turn("small")]
    res = await cm.maybe_compact(history=turns, current=[], last_in=1000)  # 1000 < 167_000
    assert res is None
    assert len(turns) == 1  # 未改


async def test_maybe_compact_triggers_and_replaces_history_with_summary_plus_tail():
    cm = _cm(provider=_SummaryProvider(summary_text="MY-SUMMARY"))
    turns = [_turn("q" * 200_000) for _ in range(5)]  # 远超阈值
    res = await cm.maybe_compact(history=turns, current=[], last_in=300_000)
    assert res is not None
    assert res.trigger == "auto"
    assert res.fell_back is False
    # history 原地替换:首条应是摘要 turn(其首条消息文本含 MY-SUMMARY)
    first_texts = [b.text for b in turns[0].messages[0].content if hasattr(b, "text")]
    assert any("MY-SUMMARY" in t for t in first_texts)


# —— 熔断 ——


async def test_circuit_breaker_3_failures_falls_back_to_hard_truncate():
    """summary 连续失败 3 次 → 熔断 → 硬截断(不调 LLM 出摘要),fell_back=True。"""
    p = _SummaryProvider(fail_times=3)  # 前 3 次全失败
    cm = _cm(provider=p)
    turns = [_turn("q" * 200_000) for _ in range(5)]
    res = await cm.maybe_compact(history=turns, current=[], last_in=300_000)
    assert res is not None
    assert res.fell_back is True
    assert res.trigger == "auto"  # F13:保留调用方 trigger(maybe_compact=auto),用 fell_back 标识兜底
    # history 缩到 tail(无摘要 turn)
    assert len(turns) < 5
    assert p._calls == 3  # 恰好试了 3 次


async def test_no_spurious_compact_after_compaction_stale_last_in():
    """F6:压缩后 last_in 陈旧(大),maybe_compact 不应对刚压缩的小 history 再触发。

    _stale_anchor 让压缩后第一次 maybe_compact 走冷启动估 → 小 history < 阈值 → None。
    (未修时:用陈旧大 last_in → est ≥ 阈值 → 又压一次,叠冗余边界。)
    """
    p = _SummaryProvider(summary_text="SUM")
    cm = _cm(provider=p)
    turns = [_turn("q" * 200_000) for _ in range(5)]  # 远超阈值
    first = await cm.maybe_compact(history=turns, current=[], last_in=300_000)
    assert first is not None and len(turns) < 5  # 真超阈,压缩了
    # 第二次:传陈旧的大 last_in(模拟 tail.turn.usage 仍是截断前的大值)
    second = await cm.maybe_compact(history=turns, current=[], last_in=300_000)
    assert second is None  # F6:_stale_anchor → 冷启动估 → 不再误触发


# —— Reactive(413 应急)——


async def test_react_hard_truncates_without_llm():
    """react:直接硬截断到尾段,不调 complete。"""
    p = _SummaryProvider()
    cm = _cm(provider=p)
    turns = [_turn(f"q{i}" * 50_000) for i in range(6)]
    res = await cm.react(history=turns)
    assert res.trigger == "reactive"
    assert res.fell_back is True
    assert p._calls == 0  # 未调 LLM
    assert len(turns) < 6  # 缩了


# —— 手动 /compact ——


async def test_compact_now_manual_ignores_threshold():
    cm = _cm(provider=_SummaryProvider(summary_text="MANUAL"))
    # 多 turn 使 prefix 非空,走摘要路径
    turns = [_turn(f"x{i}" * 50_000) for i in range(4)]
    res = await cm.compact_now(history=turns, trigger="manual")
    assert res is not None
    assert res.trigger == "manual"


# —— 工具结果块不崩 _messages_chars(token 计算路径)——


def test_split_prefix_tail_handles_tool_result_blocks():
    """_messages_chars 必须处理 ToolResultBlock(无 .text 属性),不能崩。"""
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    turns = [
        Turn(
            messages=[
                Message(role="user", content=[TextBlock(text="real question")]),
                Message(role="assistant", content=[]),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="t1", content="result blob" * 1000)],
                ),
            ]
        ),
        Turn(messages=[Message(role="user", content=[TextBlock(text="tail q")])]),
    ]
    prefix, tail = cm._split_prefix_tail(turns)
    assert tail[-1] is turns[-1]
    assert all(t in turns for t in prefix + tail)


def test_build_candidates_pairs_size_replaced_block():
    """_build_candidates:每个 ToolResultBlock → SnipCandidate(size=len(content)、replaced、
    block 反引);ToolUseBlock 不产生候选(只统计 tool_result)。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock, ToolUseBlock

    cm = _cm()
    turns = [
        Turn(
            messages=[
                Message(role="assistant", content=[ToolUseBlock(id="tu1", name="bash", input={})]),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="tu1", content="BIG" * 100),
                        ToolResultBlock(tool_use_id="tu2", content=_SNIP_PLACEHOLDER),  # 已占位
                    ],
                ),
            ]
        ),
    ]
    cands = cm._build_candidates(turns)
    assert len(cands) == 2  # 仅 2 个 tool_result;tool_use 不产生候选
    by_id = {c.tool_use_id: c for c in cands}
    assert by_id["tu1"].size == len("BIG" * 100)
    assert by_id["tu1"].replaced is False
    assert by_id["tu1"].block.tool_use_id == "tu1"
    assert by_id["tu2"].replaced is True


# —— Tier2 snip(size 优先 + 保护最近完整 turn + 最小清理量)——


def _results_turn(*contents: str, start: int = 0) -> Turn:
    """构造一个含若干 tool_result 的 turn(id 从 start 起)。"""
    from birdcode.blocks import ToolResultBlock

    return Turn(
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id=f"t{start + i}", content=c)
                    for i, c in enumerate(contents)
                ],
            )
        ]
    )


async def test_snip_clears_largest_first_and_protects_last_turn():
    """size 优先:old turn 里最大的先占位;recent turn(history[-1])的 tool_result 全保护不动。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # 6 块:t1=y*50000 最大;尾部 4 块 filler 被 _SNIP_KEEP 补足保护 → t0/t1 snippable。
    old = _results_turn("x" * 1000, "y" * 50000, "z" * 5000, "a" * 10, "a" * 10, "a" * 10, start=0)
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "snip"
    blocks_old = old.messages[0].content
    blocks_recent = recent.messages[0].content
    # 最大块(t1=y*50000)被占位;recent 的块保留
    assert blocks_old[1].content == _SNIP_PLACEHOLDER
    assert blocks_recent[0].content == "w" * 1000


async def test_snip_clears_only_largest_then_stops_at_clear_at_least():
    """size 优先(硬):多个可清候选里,清完最大块即满足 _SNIP_CLEAR_AT_LEAST → 立即停,
    次大/小块保留。证明贪心按 size 降序、停在最小清理量(非全清)。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # old:t0(BIG)>t1(MID)>t2(SMALL) 为最旧 3 块(保护窗外、可清);
    # t3-t6 filler(最新 4 块)与 recent 一起填满保护集 5 → t0/t1/t2 snippable。
    old = _results_turn(
        "B" * 50000,
        "M" * 10000,
        "S" * 2000,
        "f" * 10,
        "f" * 10,
        "f" * 10,
        "f" * 10,
        start=0,
    )
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "snip"
    bo = old.messages[0].content
    assert bo[0].content == _SNIP_PLACEHOLDER  # BIG(最大)被清
    assert bo[1].content == "M" * 10000  # MID 保留(已满足最小量 → 停)
    assert bo[2].content == "S" * 2000  # SMALL 保留
    assert recent.messages[0].content[0].content == "w" * 1000  # 保护集未动


async def test_snip_skips_blocks_smaller_than_placeholder():
    """占位符比原文长的小 tool_result(clearable≤0)不清,绝不 post>pre、原文不丢。

    刀锋构造:deficit == 最大可清块的 net → 清完大块恰好压到阈值(== 非 <),贪心循环会
    继续走向尾部 tiny 候选。无 clearable≤0 守卫时 tiny 被换成更长的占位符 → 原文丢失,
    且 tiny 足够多时 post 可超 pre;有守卫时跳过 → tiny 原文保留、post<pre。
    """
    from birdcode.agent.context import _SNIP_CLEAR_AT_LEAST, _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock

    est = TokenEstimator()
    ph_tokens = est.char_to_token(_SNIP_PLACEHOLDER)
    big = "B" * 24_000
    big_clearable = est.char_to_token(big) - ph_tokens
    assert big_clearable >= _SNIP_CLEAR_AT_LEAST  # 大块单独够最小清理量
    # old:big + 200 个 tiny(最旧 → 在保护集外、snippable)+ 4 filler(被保护集补足)。
    # 保护集 = recent(1) + 按 recency 补到 _SNIP_KEEP(再取 4 个 old 最新的 filler)。
    old = _results_turn(big, *["a"] * 200, "f" * 10, "f" * 10, "f" * 10, "f" * 10, start=0)
    recent = _results_turn("w" * 60_000, start=10)  # 受保护,撑大 pre_cold
    turns = [old, recent]
    pre_cold = est.estimate(history=turns, current=[], last_in=None)
    # threshold = pre_cold - big_clearable → deficit 恰 = big_clearable(清完大块 == 阈值,刀锋)。
    threshold = pre_cold - big_clearable
    cm = _cm(
        context_window=threshold + 1024,
        compact_summary_reserve=1024,
        compact_safety_margin=0,
    )
    res = await cm._snip(turns, current=[])
    assert res is not None and res.trigger == "snip"
    assert res.pre_tokens >= res.post_tokens  # 关键不变量:绝不涨
    # tiny 原文保留(200 个均未被占位)
    contents = [
        b.content
        for t in turns
        for m in t.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert contents.count("a") == 200


async def test_snip_idempotent_excludes_already_placeholdered():
    """已占位的候选排除:二次进入(其它块仍可清时)不再重复占位同一块。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # t0/t1 大;尾部 4 块 filler 被补足保护 → t0/t1 snippable(每轮清 1 个,二次仍可清另一个)。
    old = _results_turn("x" * 50000, "y" * 50000, "a" * 10, "a" * 10, "a" * 10, "a" * 10, start=0)
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    first = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert first is not None and first.trigger == "snip"
    placeholdered = [
        b.tool_use_id for b in old.messages[0].content if b.content == _SNIP_PLACEHOLDER
    ]
    assert len(placeholdered) >= 1
    # 二次:已占位的不再动(仍可能清下一块,但不会把占位块再"清"一次)
    cm._stale_anchor = False
    second = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    # 已占位块二次后仍是占位符(未被改回/重复处理)
    assert all(
        b.content == _SNIP_PLACEHOLDER
        for b in old.messages[0].content
        if b.tool_use_id in placeholdered
    )
    if second is None:
        # 若二次无可清(snippable 空)→ None,合理
        pass


async def test_snip_below_clear_at_least_under_threshold_returns_none():
    """未超阈 + 可清总量 < _SNIP_CLEAR_AT_LEAST → 不动(None),留给将来自然压缩。"""
    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # old turn 单块极小(总可清 << _SNIP_CLEAR_AT_LEAST);recent 保护自己
    old = _results_turn("x" * 100, start=0)
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is None  # 不值当,不动


async def test_snip_pre_post_same_cold_estimator():
    """snip 的 pre/post 均 cold 估(同口径)→ pre >= post。"""
    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    old = _results_turn(*["x" * 5000 for _ in range(6)], start=0)
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "snip"
    assert res.pre_tokens >= res.post_tokens


async def test_snip_over_threshold_insufficient_goes_to_compact_sees_originals():
    """cold 估超阈(deficit>0)但 snip 清不够 → 直接 _compact;摘要在未创建任何占位前看到
    原文(证明消除 restore:旧路径先占位再恢复,新路径根本没占位)。

    用 _Recording 抓 summarize_with_prefix 的 prefix(= 待摘要段,含 tool_result)。
    """
    from birdcode.agent.context import _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock

    class _Recording(_SummaryProvider):
        def __init__(self) -> None:
            super().__init__(summary_text="SUM")
            self.seen_prefix: list | None = None

        async def summarize_with_prefix(
            self, *, prefix: list[Message], instruction: str, max_tokens: int
        ) -> str:
            self.seen_prefix = prefix
            return await super().summarize_with_prefix(
                prefix=prefix, instruction=instruction, max_tokens=max_tokens
            )

    p = _Recording()
    cm = _cm(provider=p)
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # 5 turn 大 text(cold 估超阈)+ 每 turn 2 小 tool_result(snip 清不动 deficit)。
    # history[-1]=最后大 turn → 其 2 result 必保;按 recency 补足到 _SNIP_KEEP 后,
    # 最旧若干 result snippable,但总可清 << deficit → pre-check 失败 → _compact。
    turns = [
        Turn(
            messages=[
                Message(role="user", content=[TextBlock(text="q" * 150_000)]),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=f"t{i}a", content="ORIGINAL-RESULT" + "r" * 500
                        ),
                        ToolResultBlock(
                            tool_use_id=f"t{i}b", content="ORIGINAL-RESULT" + "r" * 500
                        ),
                    ],
                ),
            ]
        )
        for i in range(5)
    ]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "auto"
    assert p._calls >= 1  # 调了 LLM 摘要
    # 摘要看到的 prefix 含原文 tool_result,不含占位符(新路径未创建任何占位)
    assert p.seen_prefix is not None
    seen = [b.content for m in p.seen_prefix for b in m.content if isinstance(b, ToolResultBlock)]
    assert any("ORIGINAL-RESULT" in (c or "") for c in seen)
    assert not any(c == _SNIP_PLACEHOLDER for c in seen)


async def test_snip_no_tool_results_returns_none():
    """tier1 但 history 无 tool_result → 无可清 → None(不误触发 UI)。"""
    cm = _cm()
    cm._stale_anchor = False
    turns = [_turn("q" * 100)]
    res = await cm.maybe_compact(history=turns, current=[], last_in=cm._tier1_threshold() + 100)
    assert res is None


async def test_snip_fires_on_activity_reason():
    """snip 触发 on_activity(reason="snip");无可清时(None)不触发。"""
    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # 8 块:尾部 4 块被补足保护 → t0-t3 snippable(共可清 >> _SNIP_CLEAR_AT_LEAST)。
    old = _results_turn(*["x" * 5000 for _ in range(8)], start=0)
    recent = _results_turn("w" * 1000, start=10)
    turns = [old, recent]
    events: list[tuple[str, str]] = []

    async def on_activity(phase: str, reason: str, result: object) -> None:
        events.append((phase, reason))

    res = await cm.maybe_compact(
        history=turns, current=[], last_in=tier1 + 100, on_activity=on_activity
    )
    assert res is not None and res.trigger == "snip"
    assert ("start", "snip") in events and ("end", "snip") in events


# —— Task 5:保护集 _protected_ids(最近完整 turn + 补足 _SNIP_KEEP)——


def test_protected_ids_keeps_whole_last_turn_and_topups_to_keep():
    """保护集:① history[-1] 全部 tool_result;② 不足 _SNIP_KEEP 按 recency 补足。"""
    from birdcode.agent.context import _SNIP_KEEP
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    # old turn:3 个;recent turn:2 个(< _SNIP_KEEP=5)
    old = Turn(
        messages=[
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id=f"o{i}", content="x") for i in range(3)],
            )
        ]
    )
    recent = Turn(
        messages=[
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id=f"r{i}", content="x") for i in range(2)],
            )
        ]
    )
    protected = cm._protected_ids([old, recent])
    # recent 全保(2)+ 从 old 补到 5(再取 3 个 old)= 5
    assert {"r0", "r1"} <= protected
    assert len(protected) == _SNIP_KEEP


def test_protected_ids_keeps_all_when_last_turn_exceeds_keep():
    """history[-1] 的 tool_result 多于 _SNIP_KEEP → 全保(不截断)。"""
    from birdcode.agent.context import _SNIP_KEEP
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    recent = Turn(
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id=f"r{i}", content="x") for i in range(_SNIP_KEEP + 3)
                ],
            )
        ]
    )
    protected = cm._protected_ids([recent])
    assert len(protected) == _SNIP_KEEP + 3  # 全保


def test_protected_ids_empty_history():
    cm = _cm()
    assert cm._protected_ids([]) == set()
