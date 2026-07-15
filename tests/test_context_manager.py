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
        self, *, prefix: list, instruction: str, max_tokens: int  # noqa: ANN001
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
        Turn(messages=[
            Message(role="user", content=[TextBlock(text="real question")]),
            Message(role="assistant", content=[]),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="t1", content="result blob" * 1000)],
            ),
        ]),
        Turn(messages=[Message(role="user", content=[TextBlock(text="tail q")])]),
    ]
    prefix, tail = cm._split_prefix_tail(turns)
    assert tail[-1] is turns[-1]
    assert all(t in turns for t in prefix + tail)


# —— Tier1 snip(清旧 tool_result,零 LLM 成本推迟摘要)——


def test_snip_old_tool_results_keeps_last_5_and_is_idempotent():
    """保留最近 _SNIP_KEEP 个,其余置占位符;再跑不变(幂等)。"""
    from birdcode.agent.context import _SNIP_KEEP, _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    blocks = [ToolResultBlock(tool_use_id=f"t{i}", content=f"result-{i}" * 10) for i in range(8)]
    turn = Turn(messages=[Message(role="user", content=blocks)])
    assert cm._snip_old_tool_results([turn]) == 8 - _SNIP_KEEP  # 清 3 个
    assert all(b.content == _SNIP_PLACEHOLDER for b in blocks[: 8 - _SNIP_KEEP])  # 前 3 占位
    keep = blocks[8 - _SNIP_KEEP :]
    assert all(b.content == f"result-{i}" * 10 for i, b in enumerate(keep, start=8 - _SNIP_KEEP))
    snap = [b.content for b in blocks]
    # 幂等:已占位的二次调用无实际工作 → 返回 0(不误触发 UI/_stale_anchor)。
    assert cm._snip_old_tool_results([turn]) == 0
    assert [b.content for b in blocks] == snap  # 内容不变


def test_snip_old_tool_results_returns_zero_when_no_tool_results():
    cm = _cm()
    turn = Turn(messages=[Message(role="user", content=[TextBlock(text="only text")])])
    assert cm._snip_old_tool_results([turn]) == 0


async def test_maybe_compact_tier1_snips_and_sets_stale():
    """est ∈ [tier1, autocompact) → snip(占位旧 tool_result)+ _stale_anchor=True + trigger=snip。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    cm._stale_anchor = False  # 让 last_in 生效(est = last_in + current 增量)
    blocks = [ToolResultBlock(tool_use_id=f"t{i}", content="x" * 1000) for i in range(8)]
    turns = [Turn(messages=[Message(role="user", content=blocks)])]
    tier1 = cm._tier1_threshold()
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "snip"
    assert cm._stale_anchor is True
    assert any(b.content == _SNIP_PLACEHOLDER for b in blocks)  # 有被占位


async def test_maybe_compact_below_tier1_no_op():
    cm = _cm()
    cm._stale_anchor = False
    turns = [_turn("small")]
    res = await cm.maybe_compact(history=turns, current=[], last_in=1000)  # < tier1
    assert res is None


async def test_maybe_compact_tier1_no_tool_results_returns_none():
    """est ∈ tier1 但 history 无 tool_result → snip 无东西可清 → None(不误触发 UI)。"""
    cm = _cm()
    cm._stale_anchor = False
    turns = [_turn("q" * 100)]
    res = await cm.maybe_compact(
        history=turns, current=[], last_in=cm._tier1_threshold() + 100
    )
    assert res is None


async def test_maybe_compact_tier1_snip_insufficient_falls_to_compact():
    """snip 清完仍 ≥ autocompact(巨大 text 清不动)→ 转 _compact(trigger=auto)。"""
    from birdcode.blocks import ToolResultBlock

    p = _SummaryProvider(summary_text="SUM")
    cm = _cm(provider=p)
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    # 多 turn(prefix 非空):每 turn 巨大 text(snip 清不动)+ 2 tool_result(共 10,snip 清 5)
    turns = [
        Turn(messages=[
            Message(role="user", content=[TextBlock(text="q" * 150_000)]),
            Message(role="user", content=[
                ToolResultBlock(tool_use_id=f"t{i}a", content="r" * 500),
                ToolResultBlock(tool_use_id=f"t{i}b", content="r" * 500),
            ]),
        ])
        for i in range(5)
    ]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "auto"  # snip 不够 → 转 _compact
    assert p._calls >= 1  # 调了 LLM 摘要


async def test_maybe_compact_tier1_snip_on_activity_reason():
    """snip 触发 on_activity,reason="snip"(UI 据此分灰色)。"""
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    blocks = [ToolResultBlock(tool_use_id=f"t{i}", content="x" * 1000) for i in range(8)]
    turns = [Turn(messages=[Message(role="user", content=blocks)])]
    events: list[tuple[str, str]] = []

    async def on_activity(phase: str, reason: str, result: object) -> None:
        events.append((phase, reason))

    res = await cm.maybe_compact(
        history=turns, current=[], last_in=tier1 + 100, on_activity=on_activity
    )
    assert res is not None and res.trigger == "snip"
    assert ("start", "snip") in events
    assert ("end", "snip") in events


# —— snip 修复:xhigh review #3/#4/#5 ——


async def test_tier1_snip_idempotent_no_spurious_activity():
    """#3:二次进入 tier1(tool_result 已全占位)→ _snip_targets 过滤 → 无实际工作 →
    不重复触发 on_activity / 不重置 _stale_anchor(修每轮 spurious snip 事件)。"""
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    blocks = [ToolResultBlock(tool_use_id=f"t{i}", content="x" * 1000) for i in range(8)]
    turns = [Turn(messages=[Message(role="user", content=blocks)])]
    events: list[tuple[str, str]] = []

    async def on_activity(phase: str, reason: str, result: object) -> None:
        events.append((phase, reason))

    first = await cm.maybe_compact(
        history=turns, current=[], last_in=tier1 + 100, on_activity=on_activity
    )
    assert first is not None and first.trigger == "snip"
    assert ("start", "snip") in events and ("end", "snip") in events
    # 二次:blocks 已占位 → _snip_targets 返回空 → None,不触发 activity
    events.clear()
    cm._stale_anchor = False  # 重置以再次走 last_in 进 tier1(隔离 _stale_anchor 干扰)
    second = await cm.maybe_compact(
        history=turns, current=[], last_in=tier1 + 100, on_activity=on_activity
    )
    assert second is None
    assert events == []  # 无 spurious activity


async def test_tier1_snip_escalate_restores_original_tool_results():
    """#4:snip 清完仍 ≥ 阈值 → 升级 _compact 前**恢复原文 tool_result**(否则摘要看到占位符,
    丢前文细节——旧路径 _render_prefix_for_summary 带 2000-char 预览保留的细节)。"""
    from birdcode.agent.context import _SNIP_PLACEHOLDER
    from birdcode.blocks import ToolResultBlock

    class _RecordingProvider(_SummaryProvider):
        def __init__(self) -> None:
            super().__init__(summary_text="SUM")
            self.seen_prefix: list | None = None

        async def summarize_with_prefix(  # noqa: ANN001
            self, *, prefix: list, instruction: str, max_tokens: int
        ) -> str:
            self.seen_prefix = prefix
            return await super().summarize_with_prefix(
                prefix=prefix, instruction=instruction, max_tokens=max_tokens
            )

    p = _RecordingProvider()
    cm = _cm(provider=p)
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    turns = [
        Turn(messages=[
            Message(role="user", content=[TextBlock(text="q" * 150_000)]),
            Message(role="user", content=[
                ToolResultBlock(tool_use_id=f"t{i}a", content="ORIGINAL-RESULT" + "r" * 500),
                ToolResultBlock(tool_use_id=f"t{i}b", content="ORIGINAL-RESULT" + "r" * 500),
            ]),
        ])
        for i in range(5)
    ]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "auto"  # snip 不够 → 升级 _compact
    assert p._calls >= 1
    # 摘要看到的 prefix 含原文 tool_result,不含占位符
    assert p.seen_prefix is not None
    seen = [
        b.content
        for m in p.seen_prefix
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert any("ORIGINAL-RESULT" in (c or "") for c in seen)
    assert not any(c == _SNIP_PLACEHOLDER for c in seen)


async def test_tier1_snip_pre_post_same_cold_estimator():
    """#5:snip 的 pre/post 均 cold 估(同口径)→ pre >= post(修 anchored pre + cold post
    混口径致 UI 显示 post>pre 的误导)。"""
    from birdcode.blocks import ToolResultBlock

    cm = _cm()
    cm._stale_anchor = False
    tier1 = cm._tier1_threshold()
    blocks = [ToolResultBlock(tool_use_id=f"t{i}", content="x" * 5000) for i in range(8)]
    turns = [Turn(messages=[Message(role="user", content=blocks)])]
    res = await cm.maybe_compact(history=turns, current=[], last_in=tier1 + 100)
    assert res is not None and res.trigger == "snip"
    assert res.pre_tokens >= res.post_tokens  # 同 cold 口径,清后必 <= 清前
