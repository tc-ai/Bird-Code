# tests/session/test_e2e_compaction.py
"""T11 端到端:压缩 → 落边界行 → resume 只装[摘要+尾段];413 → react 硬截断。

串起 codec + SessionStore + ContextManager 全链:真实 SessionStore(tmp_path 落盘)+
mock summary provider(complete 返回 <summary>),验证「写边界行 → resume 只装摘要」
的完整链路(codec + store + ContextManager 串联)。asyncio_mode=auto,直接 await。

注意:计划原文 compact_summary_reserve=1000 / compact_tail_budget=500 违反 schema
ge=1024 约束,已提到下限 1024(不影响测试意图——仍远小于 turn 体量,触发压缩)。
"""
import json
from pathlib import Path

from birdcode.agent.context import ContextManager
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig
from birdcode.conversation import Message, Turn
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


class _SummaryProv:
    """假 provider:complete 返回 <analysis>+<summary>(供 ContextManager 提取摘要)。"""

    async def complete(
        self, system: str, user: str, *, max_tokens: int, model: str | None = None
    ) -> str:
        return "<analysis>x</analysis><summary>压缩摘要:用户在做 X</summary>"

    async def summarize_with_prefix(
        self, *, prefix: list, instruction: str, max_tokens: int  # noqa: ANN001
    ) -> str:
        return "<analysis>x</analysis><summary>压缩摘要:用户在做 X</summary>"


def _ctx() -> SessionContext:
    return SessionContext(session_id="e2e", cwd=".", version="1", git_branch=None)


def _make_store(tmp_path: Path) -> SessionStore:
    """真实落盘 store(tmp_path),与 test_e2e_resume 同风格。"""
    return SessionStore(_ctx(), tmp_path, root=tmp_path)


def _cfg(**overrides: object) -> AppConfig:
    base: dict[str, object] = {
        "providers": {
            "p": {"protocol": "anthropic", "model": "m", "base_url": "u", "api_key": "k"}
        },
        "default": "p",
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


async def test_e2e_compact_then_resume_truncated(tmp_path):
    """超阈 history → compact_now 落[边界+摘要+重写尾段] → resume 装回[摘要+尾段],旧 bulk 被截。

    用可区分的 3 个大 turn(a/b/c,各 ~50000 token,远超阈值 8476);compact_now 手动触发
    Autocompact(摘要成功路径)。close → 重开 store 模拟跨进程 resume:split_live_segment
    切到最后边界起 = [边界+摘要+重写尾段 c] → 摘要在、尾段 c 在、旧 bulk(a)被摘要替代不在。
    (修 tail-drop:尾段必须重写到边界之后,resume 才装得回。)
    """
    store = _make_store(tmp_path)
    # 小窗口 + 最小允许的 reserve/margin/tail(受 schema ge=1024 / ge=0 约束)
    cfg = _cfg(
        context_window=10_000,
        compact_summary_reserve=1024,  # ge=1024(计划原文 1000 不合法,提到下限)
        compact_safety_margin=500,  # ge=0
        compact_tail_budget=1024,  # ge=1024(计划原文 500 不合法,提到下限)
    )
    cm = ContextManager(_SummaryProv(), store, cfg)

    # 3 个可区分的大 turn(各 ~50000 token,远超阈值 10000-1024-500=8476)
    turns = [
        Turn(messages=[Message(role="user", content=[TextBlock(text="a" * 200_000)])]),
        Turn(messages=[Message(role="user", content=[TextBlock(text="b" * 200_000)])]),
        Turn(messages=[Message(role="user", content=[TextBlock(text="c" * 200_000)])]),
    ]
    # 落盘(模拟已持久化的对话历史)
    for t in turns:
        for m in t.messages:
            await store.append(m)

    # 手动触发压缩(无视阈值,走 Autocompact 摘要路径)
    res = await cm.compact_now(history=turns, trigger="manual")
    assert res is not None
    assert not res.fell_back  # 摘要成功,未走硬截断兜底

    # resume:close 旧 store、重开新 store 模拟跨进程 load_mainline
    store.close()
    s2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    loaded = await s2.load_mainline()
    s2.close()
    texts = [
        b.text for t in loaded for m in t.messages for b in m.content if hasattr(b, "text")
    ]
    assert any("压缩摘要" in t for t in texts)  # 摘要文本在
    assert not any("a" * 100 in t for t in texts)  # 旧 bulk(a,被摘要替代)不在
    assert any("c" * 100 in t for t in texts)  # 尾段(最后 1 turn = c)原样保留、resume 装回


async def test_e2e_reactive_on_overflow(tmp_path):
    """413 应急:react 硬截断 history 到尾段,store 落 trigger=reactive 边界行,fell_back=True。

    react 不调 LLM(热路径要快且必成功),只切 tail + 写边界。history 原地缩短。
    """
    store = _make_store(tmp_path)
    cfg = _cfg()  # 默认阈值,react 不看阈值
    cm = ContextManager(_SummaryProv(), store, cfg)

    turns = [
        Turn(messages=[Message(role="user", content=[TextBlock(text="q" * 100_000)])])
        for _ in range(5)
    ]
    for t in turns:
        for m in t.messages:
            await store.append(m)

    res = await cm.react(history=turns)
    assert res.trigger == "reactive"
    assert res.fell_back  # 硬截断恒 fell_back
    assert len(turns) < 5  # history 被原地截短(默认 tail_budget=30000,1 turn=25000 token → tail=1)
    store.close()


def _rows(store: SessionStore) -> list[dict]:
    return [
        json.loads(ln)
        for ln in store._jsonl.read_text(encoding="utf-8").splitlines()  # noqa: SLF001
        if ln.strip()
    ]


async def test_append_compaction_writes_boundary_summary_tail_in_order(tmp_path):
    """append_compaction 一次写[边界 + 摘要 + 重写尾段],顺序、链式 parent、_last_uuid 推进正确。"""
    store = _make_store(tmp_path)
    summary_msg = Message(role="user", content=[TextBlock(text="<summary>S</summary>")])
    tail = [
        Turn(messages=[Message(role="user", content=[TextBlock(text="tail-q")])]),
        Turn(messages=[Message(role="assistant", content=[TextBlock(text="tail-a")])]),
    ]
    res = await store.append_compaction(
        subtype="compact_boundary",
        logical_parent_uuid="prev-uuid",
        content="Conversation compacted",
        compact_metadata={"trigger": "manual"},
        summary_msg=summary_msg,
        tail_turns=tail,
    )
    assert res is not None
    boundary_uuid, summary_uuid = res
    rows = _rows(store)
    # 顺序:boundary(system,parentUuid=null)→ summary(isCompactSummary)→ 2 条尾段
    assert rows[0]["subtype"] == "compact_boundary"
    assert rows[0]["parentUuid"] is None
    assert rows[0]["logicalParentUuid"] == "prev-uuid"
    assert rows[0]["uuid"] == boundary_uuid
    assert rows[1].get("isCompactSummary") is True
    assert rows[1]["parentUuid"] == boundary_uuid
    assert rows[1]["uuid"] == summary_uuid
    # 尾段从 summary 链式挂父、顺序保留
    assert rows[2]["message"]["content"][0]["text"] == "tail-q"
    assert rows[2]["parentUuid"] == summary_uuid
    assert rows[3]["message"]["content"][0]["text"] == "tail-a"
    assert rows[3]["parentUuid"] == rows[2]["uuid"]
    assert store._last_uuid == rows[3]["uuid"]  # 推进到末条尾段行
    store.close()


async def test_append_compaction_failure_returns_none_no_partial(tmp_path):
    """落盘失败(句柄已关)→ 返 None,不推进 _last_uuid、不写半截(原子,F2/F3)。"""
    store = _make_store(tmp_path)
    baseline = await store.append(  # 先写基线消息、设 _last_uuid
        Message(role="user", content=[TextBlock(text="base")])
    )
    assert store._last_uuid == baseline
    store._fh.close()  # noqa: SLF001 - 关句柄 → 后续 write 抛 ValueError
    res = await store.append_compaction(
        subtype="compact_boundary",
        logical_parent_uuid=None,
        content="x",
        compact_metadata={},
        summary_msg=Message(role="user", content=[TextBlock(text="s")]),
        tail_turns=[],
    )
    assert res is None  # 不返幻影 uuid
    assert store._last_uuid == baseline  # 未推进
    rows = _rows(store)
    assert len(rows) == 1  # 无任何 compaction 行(没写半截边界)
    assert rows[0]["message"]["content"][0]["text"] == "base"


async def test_append_compaction_non_serializable_returns_none_no_raise(tmp_path):
    """CR #7(补全):compact_metadata 含不可 JSON 序列化对象 → blob 构造的 json.dumps
    抛 TypeError。

    旧实现 blob = "".join(json.dumps(row)...) 在 try **之外**(原仅兜 write/flush 的
    OSError/ValueError/TypeError),序列化 TypeError 逃出 except,经
    context._persist_compaction(无 try)杀主循环——CR #7「不杀主循环」对序列化路径是
    死保护。修后 blob 构造进 try,与落盘失败同处理:返 None、不推进 _last_uuid、不写半截。
    (当前唯一生产调用方 context.py 只传 str/int,故为 latent;本测试锁定契约。)
    """
    store = _make_store(tmp_path)
    baseline = await store.append(  # 基线消息 + 设 _last_uuid
        Message(role="user", content=[TextBlock(text="base")])
    )
    assert store._last_uuid == baseline
    res = await store.append_compaction(
        subtype="compact_boundary",
        logical_parent_uuid=None,
        content="x",
        compact_metadata={"bad": object()},  # 不可 JSON 序列化 → json.dumps TypeError
        summary_msg=Message(role="user", content=[TextBlock(text="s")]),
        tail_turns=[],
    )
    assert res is None  # 不返幻影 uuid、不抛
    assert store._last_uuid == baseline  # 未推进
    rows = _rows(store)
    assert len(rows) == 1  # 无 compaction 行(没写半截边界)
    assert rows[0]["message"]["content"][0]["text"] == "base"
    store.close()
