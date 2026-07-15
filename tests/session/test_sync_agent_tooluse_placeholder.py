"""sync 中断 tool_use 占位:主会话 resume 时,待续跑 sync agent tool_use 的合成
tool_result content 改为「中断,待续跑」(真结果经 resume_agent inline 返回)。

关联方案(关键设计点):主会话 sync agent 的 tool_use_id 是 LLM 生成(如 toolu_xxx),
runner 内部 tool_use_id="(sync-agent)" 只进子 agent meta/queue-op,不进主 jsonl。
中断期 tool_result 未落盘 → toolUseResult 也不可见。故「待续跑 sync agent tool_use」
靠【tool_use.name == 非终态 sync meta.agent_type】配对(sync 阻塞,同时刻仅一个),
再由 codec._mark_pending_sync_agent_tooluses 改合成 tool_result 的 content。
"""
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.session import codec, paths
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.store import SessionStore
from birdcode.session.subagent_meta import write_subagent_meta


def _ctx(sid="s1"):
    return SessionContext(session_id=sid, cwd="C:/proj", version="0.1.0", git_branch="main")


def _dangling_sync_agent_turns() -> list[Turn]:
    """末尾 assistant 含 sync agent tool_use、无 result → repair 补合成 error tool_result。

    tool_use.name='general-purpose'(默认 agent 类型,与 _AgentTool 注册名一致)。
    """
    turns = [
        Turn(messages=[
            Message(role="user", content=[TextBlock(text="go")]),
            Message(
                role="assistant",
                content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                      input={"prompt": "do x"})],
            ),
        ])
    ]
    codec.repair_trailing_edge(turns)
    return turns


def _find_tool_result(turns: list[Turn], tool_use_id: str) -> ToolResultBlock:
    return next(
        b for t in turns for m in t.messages for b in m.content
        if isinstance(b, ToolResultBlock) and b.tool_use_id == tool_use_id
    )


# ---- codec._mark_pending_sync_agent_tooluses:纯函数 ----


def test_mark_pending_changes_content_for_sync_agent_type():
    """命中:tool_use.name ∈ sync_agent_types → 合成 tool_result content 含「中断/续跑」。"""
    turns = _dangling_sync_agent_turns()
    n = codec._mark_pending_sync_agent_tooluses(turns, {"general-purpose"})
    assert n == 1
    tr = _find_tool_result(turns, "toolu_1")
    assert "中断" in tr.content and "续跑" in tr.content


def test_mark_pending_noop_when_type_not_matched():
    """未命中:sync_agent_types 不含 tool_use.name → content 保持 repair 默认。

    区分关键词:「续跑」仅占位有;repair 默认含「中断/请重新发起」但无「续跑」。
    """
    turns = _dangling_sync_agent_turns()
    n = codec._mark_pending_sync_agent_tooluses(turns, {"some-other-agent"})
    assert n == 0
    tr = _find_tool_result(turns, "toolu_1")
    assert "续跑" not in tr.content
    assert "请重新发起" in tr.content  # repair 默认文案未被动


def test_mark_pending_noop_when_empty_types():
    """空类型集(= 无待续跑 sync meta)→ 不改任何 content。"""
    turns = _dangling_sync_agent_turns()
    assert codec._mark_pending_sync_agent_tooluses(turns, set()) == 0
    tr = _find_tool_result(turns, "toolu_1")
    assert "续跑" not in tr.content
    assert "请重新发起" in tr.content


def test_mark_pending_does_not_touch_non_synthetic_tool_results():
    """只改 synthetic=True 的 tool_result(repair 补的);真实 tool_result 不被动。"""
    turns = [
        Turn(messages=[
            Message(role="user", content=[TextBlock(text="go")]),
            Message(
                role="assistant",
                content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                      input={"prompt": "do x"})],
            ),
            # 真实 tool_result(synthetic=False)——不该被改
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="toolu_1", content="real result")],
            ),
        ])
    ]
    n = codec._mark_pending_sync_agent_tooluses(turns, {"general-purpose"})
    assert n == 0
    tr = _find_tool_result(turns, "toolu_1")
    assert tr.content == "real result"


# ---- 端到端:SessionStore.load_mainline 扫 meta → 占位 ----


async def test_load_mainline_marks_pending_sync_agent_when_meta_nonterminal(tmp_path):
    """主 jsonl 末尾悬空 sync agent tool_use + 非终态 sync meta → load_mainline 占位。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="go")]))
    await store.append(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                  input={"prompt": "do x"})],
        ),
        is_assistant=True,
    )
    store.close()
    # 写非终态 sync meta(agent_type 匹配 tool_use.name)
    write_subagent_meta(
        paths.subagent_meta_path(tmp_path, "s1", tmp_path, "a9"),
        SubagentMeta(
            agent_id="a9", agent_type="general-purpose", tool_use_id="(sync-agent)",
            is_async=False, status="running",
        ),
    )
    # resume:重开 store,load_mainline
    store2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    tr = _find_tool_result(turns, "toolu_1")
    assert "中断" in tr.content and "续跑" in tr.content


async def test_load_mainline_keeps_default_when_no_pending_meta(tmp_path):
    """无非终态 sync meta → 占位不触发,content 保持 repair 默认「请重新发起」。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="go")]))
    await store.append(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                  input={"prompt": "do x"})],
        ),
        is_assistant=True,
    )
    store.close()
    # 不写 meta(无非终态 sync 子 agent)
    store2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    tr = _find_tool_result(turns, "toolu_1")
    assert "续跑" not in tr.content  # 无待续跑 sync meta → 不触发占位
    assert "请重新发起" in tr.content


async def test_load_mainline_ignores_terminal_sync_meta(tmp_path):
    """终态 meta(completed)不可续 → 不视为待续跑,content 保持默认。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="go")]))
    await store.append(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                  input={"prompt": "do x"})],
        ),
        is_assistant=True,
    )
    store.close()
    # 写终态 sync meta(completed 不在 RESUMABLE_STATUSES)
    write_subagent_meta(
        paths.subagent_meta_path(tmp_path, "s1", tmp_path, "a9"),
        SubagentMeta(
            agent_id="a9", agent_type="general-purpose", tool_use_id="(sync-agent)",
            is_async=False, status="completed",
        ),
    )
    store2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    tr = _find_tool_result(turns, "toolu_1")
    assert "续跑" not in tr.content  # 终态 meta 不可续 → 不触发占位
    assert "请重新发起" in tr.content


async def test_load_mainline_ignores_async_pending_meta(tmp_path):
    """async meta(即使非终态)不触发 sync 占位——sync/async 占位语义不同。"""
    store = SessionStore(_ctx(), tmp_path, root=tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="go")]))
    await store.append(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="toolu_1", name="general-purpose",
                                  input={"prompt": "do x"})],
        ),
        is_assistant=True,
    )
    store.close()
    # 写非终态但 is_async=True 的 meta(async 走 notification 路径,不走 sync 占位)
    write_subagent_meta(
        paths.subagent_meta_path(tmp_path, "s1", tmp_path, "a9"),
        SubagentMeta(
            agent_id="a9", agent_type="general-purpose", tool_use_id="(async-agent)",
            is_async=True, status="running",
        ),
    )
    store2 = SessionStore(_ctx(), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    tr = _find_tool_result(turns, "toolu_1")
    assert "续跑" not in tr.content  # async 不走 sync 占位
