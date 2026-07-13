# tests/session/test_store.py
import json
import os
from pathlib import Path

import pytest

from birdcode.agent.provider import TokenUsage
from birdcode.blocks import TextBlock
from birdcode.conversation import Message
from birdcode.session import paths
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


def _ctx(sid="s1"):
    return SessionContext(session_id=sid, cwd="C:/proj", version="0.1.0", git_branch="main")


def _jsonl(tmp_path, sid="s1"):
    """store 路径用 encode_cwd(project_root),project_root=tmp_path(测试注入)。"""
    return tmp_path / paths.encode_cwd(tmp_path) / f"{sid}.jsonl"


def _store(tmp_path: Path, sid="s1") -> SessionStore:
    return SessionStore(_ctx(sid), tmp_path, root=tmp_path)


async def test_append_then_load_mainline_roundtrip(tmp_path):
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答")]),
        is_assistant=True, usage=TokenUsage(input_tokens=3), stop_reason="end_turn",
    )
    store.close()

    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    assert len(turns) == 1
    assert len(turns[0].messages) == 2
    assert turns[0].messages[0].content[0].text == "问"
    assert turns[0].messages[1].content[0].text == "答"
    assert turns[0].usage is not None and turns[0].usage.input_tokens == 3


async def test_append_chains_parent_uuid(tmp_path):
    """连续 append 自动挂 parentUuid 链(链状态封装在 store)。"""
    store = _store(tmp_path)
    u1 = await store.append(Message(role="user", content=[TextBlock(text="a")]))
    a1 = await store.append(
        Message(role="assistant", content=[TextBlock(text="b")]), is_assistant=True,
    )
    assert u1 is not None
    assert a1 is not None
    # 读文件验证链
    store.close()
    rows = [
        json.loads(ln)
        for ln in _jsonl(tmp_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows[0]["parentUuid"] is None
    assert rows[1]["parentUuid"] == rows[0]["uuid"]


async def test_load_skips_corrupt_line(tmp_path):
    """半截坏行跳过,不致命(append-only 崩溃安全)。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    # 配一条 assistant 收尾,避免 F3 把“无 assistant 跟随”的末尾 user 误判为 orphan。
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True
    )
    store.close()
    # 追加一行坏数据
    jf = _jsonl(tmp_path)
    with jf.open("a", encoding="utf-8") as f:
        f.write("{这不是 json\n")
    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()  # 不抛
    assert len(turns) == 1 and turns[0].messages[0].content[0].text == "问"


async def test_load_empty_returns_empty(tmp_path):
    store = _store(tmp_path, "nope")
    assert await store.load_mainline() == []


async def test_persist_tool_output_over_threshold(tmp_path):
    store = _store(tmp_path)
    big = "x" * (32 * 1024 + 100)
    info = await store.persist_tool_output("call_1", big)
    assert info is not None
    assert info.size == len(big)
    assert "<persisted-output>" in info.llm_content
    assert "tool-results" in info.path
    assert Path(info.path).read_text(encoding="utf-8") == big
    store.close()


async def test_persist_tool_output_always_persists_when_called(tmp_path):
    """store 是 dumb sink:阈值在 executor(per-tool),store 被调即写,不再按 size 拒。

    旧 32K 守卫会拒绝低阈值工具(grep 20K)在 20K~32K 的落盘请求;移除后 store 只管写。
    """
    store = _store(tmp_path)
    info = await store.persist_tool_output("call_1", "小输出")
    assert info is not None  # 即便小输出也写(dumb sink)
    assert Path(info.path).read_text(encoding="utf-8") == "小输出"
    store.close()


async def test_persist_placeholder_contains_preview(tmp_path):
    store = _store(tmp_path)
    big = "HEADDATA\n" + "y" * (32 * 1024) + "\nTAIL"
    info = await store.persist_tool_output("call_1", big)
    assert info is not None
    assert "HEADDATA" in info.llm_content  # 预览含开头
    store.close()


async def test_persist_size_is_bytes_not_chars(tmp_path):
    """PersistInfo.size 是字节(UTF-8),非字符数——使占位 'KB' 展示准确。

    CJK 字符 UTF-8 占 3 字节;size 须为字节数,否则占位 'KB' 对中文低估约 3 倍。
    阈值仍是字符(max_result_chars,executor 比较 len(str));此处只管落盘文件大小。
    """
    store = _store(tmp_path)
    info = await store.persist_tool_output("c1", "中" * 100)  # 100 字符 = 300 字节
    assert info is not None
    assert info.size == 300  # 字节,非 100 字符
    assert "(0.3KB)" in info.llm_content  # 300/1024 ≈ 0.3KB
    store.close()


async def test_list_sessions_orders_by_mtime_desc(tmp_path):
    """list_sessions 按 mtime 倒序,first_user/turn_count 来自 sidecar(append 时写)。"""
    s1 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s1.append(Message(role="user", content=[TextBlock(text="第一会话提问")]))
    await s1.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    s1.close()
    s2 = SessionStore(_ctx("s2"), tmp_path, root=tmp_path)
    await s2.append(Message(role="user", content=[TextBlock(text="第二会话提问")]))
    s2.close()
    # 让 s2 的 mtime 更新(s2 后写)
    later = s2._jsonl.stat().st_mtime + 5  # noqa: SLF001
    os.utime(s2._jsonl, (later, later))  # noqa: SLF001

    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert [s.session_id for s in summaries] == ["s2", "s1"]
    assert summaries[0].first_user_summary == "第二会话提问"
    assert summaries[1].turn_count == 1


# ---- sidecar 索引(<sid>.meta):/sessions 快 + 可读 ----


def _meta(tmp_path, sid="s1"):
    return _jsonl(tmp_path, sid).with_suffix(".meta")


async def test_append_user_text_writes_sidecar(tmp_path):
    """append 一条 user 文本消息 → 写 <sid>.meta(turn_count=1, first_user=文本)。"""
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="第一次提问")]))
    store.close()
    meta = _meta(tmp_path)
    assert meta.exists()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["turn_count"] == 1
    assert data["first_user"] == "第一次提问"


async def test_append_tool_result_does_not_bump_sidecar(tmp_path):
    """tool_result 回填(user role 但无 text block)不计入 turn_count。"""
    from birdcode.blocks import ToolResultBlock

    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    await store.append(
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="ok")])
    )
    store.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["turn_count"] == 1  # 只算提问,不算 tool_result


async def test_second_prompt_increments_count_keeps_first(tmp_path):
    """第二轮提问:turn_count=2,first_user 仍是首条。"""
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="第一问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    await store.append(Message(role="user", content=[TextBlock(text="第二问")]))
    store.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["turn_count"] == 2
    assert data["first_user"] == "第一问"


async def test_list_sessions_reads_sidecar_without_scanning(tmp_path):
    """list_sessions 从 sidecar 取摘要,不扫 jsonl。

    jsonl 内容损坏但 sidecar 在 → 仍返回 sidecar 值(证明没扫 jsonl)。
    """
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="原始提问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    store.close()
    # 弄坏 jsonl 内容(sidecar 仍在)
    _jsonl(tmp_path).write_text("这不是 json 了\n", encoding="utf-8")
    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert len(summaries) == 1
    assert summaries[0].first_user_summary == "原始提问"
    assert summaries[0].turn_count == 1


async def test_list_sessions_backfills_missing_sidecar(tmp_path):
    """无 sidecar 的老会话 → list_sessions 扫一次 jsonl 补建 sidecar + 返回值。"""
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="老会话提问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    store.close()
    meta = _meta(tmp_path)
    meta.unlink()  # 模拟老会话:无 sidecar
    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert summaries[0].first_user_summary == "老会话提问"
    assert summaries[0].turn_count == 1
    assert meta.exists()  # backfill 已写 sidecar,下次 O(1)


async def test_list_sessions_corrupt_sidecar_falls_back_to_scan(tmp_path):
    """sidecar 损坏(非 JSON)→ 当作缺失,lazy backfill 重扫 jsonl 重写 sidecar。"""
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="真提问")]))
    store.close()
    _meta(tmp_path).write_text("{坏 json", encoding="utf-8")
    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert summaries[0].first_user_summary == "真提问"
    # 已被 backfill 重写为合法 JSON
    assert json.loads(_meta(tmp_path).read_text(encoding="utf-8"))["turn_count"] == 1


async def test_resume_backfills_meta_from_history(tmp_path):
    """resume 老会话(无 sidecar)→ load_mainline 从 turns 算准 first_user/turn_count 落 meta。"""
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="历史第一问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    await store.append(Message(role="user", content=[TextBlock(text="历史第二问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答2")]), is_assistant=True
    )
    store.close()
    _meta(tmp_path).unlink()  # 删 sidecar,模拟升级前的老会话

    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await store2.load_mainline()
    store2.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["first_user"] == "历史第一问"
    assert data["turn_count"] == 2


async def test_resume_then_continue_meta_accurate(tmp_path):
    """resume 老会话(2 轮、无 sidecar)→ 续聊 1 轮 → turn_count=3、first_user 不被覆盖。

    无 load_mainline backfill 时:_update_meta_on_prompt 从空 meta 起 → count=1、
    first_user=新提问(错)。backfill 从历史 turns 算准后续聊,故 count=3、first_user 不变。
    """
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="历史第一问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    await store.append(Message(role="user", content=[TextBlock(text="历史第二问")]))
    store.close()
    _meta(tmp_path).unlink()  # 模拟老会话无 sidecar

    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await store2.load_mainline()  # backfill meta → turn_count=2
    await store2.append(Message(role="user", content=[TextBlock(text="新第三问")]))  # 续聊
    store2.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["turn_count"] == 3  # 2 历史 + 1 新
    assert data["first_user"] == "历史第一问"  # 不被新提问覆盖


async def test_append_io_failure_does_not_raise(tmp_path):
    """IO 失败只 log,不杀主循环(契 CLAUDE.md §7)。"""
    store = _store(tmp_path)
    store._fh.close()  # noqa: SLF001 - 强制关闭句柄制造写失败
    # 不应抛异常(关闭后 write 抛 ValueError,store 应捕获降级)
    await store.append(Message(role="user", content=[TextBlock(text="x")]))


async def test_output_sink_delegates_to_store(tmp_path):
    store = _store(tmp_path)
    sink = store.as_output_sink()
    big = "z" * (32 * 1024 + 1)
    info = await sink.persist("call_9", big)
    assert info is not None and info.size == len(big)
    store.close()


# ---- F4:reopen 异常安全 ----


async def test_reopen_keeps_old_handle_open_when_new_init_fails(tmp_path, monkeypatch):
    """F4:新 __init__ 抛异常时,旧 store 句柄仍开 → 后续 append 行被持久化(不只“不抛”)。

    旧实现先 self.close() 再构造新 store;新 __init__ 抛异常时旧 fh 已关,
    调用方持已关 store。改:先构造新、成功后再 close 旧。
    """
    store = _store(tmp_path)
    orig_init = SessionStore.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN002, ANN003
        # reopen 内部那次 SessionStore(...) 构造直接抛(此时旧 fh 不应已被 close)
        raise RuntimeError("simulated init failure")

    # 仅 patch reopen 内部那次构造用的 __init__;store 自身已用原 __init__ 建好。
    monkeypatch.setattr(SessionStore, "__init__", patched_init)
    try:
        with pytest.raises(RuntimeError):
            store.reopen("s2")
    finally:
        monkeypatch.setattr(SessionStore, "__init__", orig_init)
    # 旧 store 句柄仍开 → 写入应成功落盘(不只是“不抛”)
    await store.append(Message(role="user", content=[TextBlock(text="after-fail")]))
    store.close()
    rows = [
        json.loads(ln)
        for ln in _jsonl(tmp_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(
        r.get("message", {}).get("content", [{}])[0].get("text") == "after-fail"
        for r in rows
    )


# ---- F7:非 dict JSON 行 ----


async def test_load_mainline_skips_non_dict_json_line(tmp_path):
    """F7:有效 JSON 但非 dict(如 list/str/int)行 → 跳过,不抛 AttributeError。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    # 配 assistant 收尾,避免 F3 把“无 assistant 跟随”的末尾 user 误判为 orphan。
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True
    )
    store.close()
    jf = _jsonl(tmp_path)
    with jf.open("a", encoding="utf-8") as f:
        f.write("[1, 2, 3]\n")  # 有效 JSON,但非 dict
        f.write('"str行"\n')
        f.write("42\n")
    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()  # 不抛
    assert len(turns) == 1 and turns[0].messages[0].content[0].text == "问"
    store2.close()


# ---- F8:文件序兜底(反向追溯遇缺失 parent 不丢更早行)----


async def test_load_mainline_tolerates_missing_parent_uuid(tmp_path):
    """F8:某行 parentUuid 指向不存在的 uuid → 文件序兜底,返回全部主线行(不丢更早行)。

    单链场景(无 fork/子agent)下,文件序 = 主线序,反向追溯无必要。
    """
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="第一问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="第一答")]), is_assistant=True
    )
    await store.append(Message(role="user", content=[TextBlock(text="第二问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="第二答")]), is_assistant=True
    )
    store.close()
    # 故意把多行的 parentUuid 改成不存在的 uuid(模拟链断裂)
    jf = _jsonl(tmp_path)
    rows = [json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows[1]["parentUuid"] = "nonexistent-uuid"
    rows[2]["parentUuid"] = "another-missing"
    jf.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    store2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    # 文件序下 2 轮,更早行不丢
    assert len(turns) == 2
    assert turns[0].messages[0].content[0].text == "第一问"
    assert turns[1].messages[0].content[0].text == "第二问"


# ---- F13:list_sessions 过滤空会话 ----


async def test_list_sessions_skips_empty_jsonl(tmp_path):
    """F13:0 字节 jsonl 不被 list_sessions 收录。

    __init__ 已不再创建空文件(惰性 open),故手工 touch 一个,且造在 store 构造之后
    (免被 _purge_empty_sessions 清掉),紧接 list_sessions 前造 → 验证 list 层的排除。
    """
    # 一个有内容的对照会话
    s1 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s1.append(Message(role="user", content=[TextBlock(text="提问")]))
    s1.close()
    d = tmp_path / paths.encode_cwd(tmp_path)
    (d / "empty.jsonl").touch()  # 0 字节文件(造在 list_sessions 前,不被 purge 清)
    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert [s.session_id for s in summaries] == ["s1"]  # empty 被跳过


async def test_empty_session_creates_no_jsonl(tmp_path):
    """启动即退出、无任何内容 → 不落 jsonl(不残留 0kb 文件)。

    构造后未 append 前、close 之后,文件均不应存在(惰性 open:首次写入才建文件)。
    """
    store = SessionStore(_ctx("e1"), tmp_path, root=tmp_path)
    assert not store._jsonl.exists()  # noqa: SLF001 - 未写入,无文件
    store.close()
    assert not store._jsonl.exists()  # noqa: SLF001 - close 后仍无文件


async def test_close_deletes_zero_byte_jsonl(tmp_path):
    """兜底:若本会话 jsonl 为 0 字节(异常遗留),close 时删掉,不留 0kb。"""
    store = SessionStore(_ctx("z1"), tmp_path, root=tmp_path)
    # 手工造一个 0 字节同名文件(模拟异常遗留/旧版占位),不经 append。
    store._jsonl.parent.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
    store._jsonl.touch()  # noqa: SLF001
    assert store._jsonl.exists()  # noqa: SLF001
    store.close()
    assert not store._jsonl.exists()  # noqa: SLF001 - close 删空文件


async def test_purge_empty_sessions_removes_legacy_zero_byte(tmp_path):
    """遗留的 0kb jsonl 在新 store 构造时被清扫(_purge_empty_sessions)。"""
    d = tmp_path / paths.encode_cwd(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    legacy = d / "stale.jsonl"
    legacy.touch()  # 遗留 0kb
    # 构造任意一个 store → 触发清扫本项目目录
    s = SessionStore(_ctx("x"), tmp_path, root=tmp_path)
    s.close()
    assert not legacy.exists()


# ---- #2: 修复持久化,再 resume 稳定 ----


async def test_repair_persisted_survives_reresume(tmp_path):
    """#2:崩溃留下悬空 tool_use → resume1 持久化修复 → 追加新轮 → resume2 不再触发
    修复(合成消息已落盘),tool_use 已配对、主线连续。

    旧实现(只内存修复):resume1 后 append 把悬空 tool_use 埋到中段,resume2 仅查末尾
    会漏掉 → tool_use 无配对 tool_result → 下轮 API 400。
    """
    from birdcode.blocks import ToolResultBlock, ToolUseBlock

    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    await store.append(
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="echo", input={})]),
        is_assistant=True,
    )
    store.close()  # 模拟崩溃:tool_use 已落盘、tool_result 未落盘

    # resume1:修复(补 tool_result + 合成 assistant)并持久化
    s1 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns1 = await s1.load_mainline()
    s1.close()
    assert turns1[-1].messages[-1].role == "assistant"  # 合成 assistant 收尾

    # resume 后继续聊一轮(推进 _last_uuid 到叶子,再 append)
    s1b = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s1b.load_mainline()
    await s1b.append(Message(role="user", content=[TextBlock(text="第二问")]))
    await s1b.append(
        Message(role="assistant", content=[TextBlock(text="第二答")]), is_assistant=True
    )
    s1b.close()

    # resume2:不再触发修复;2 轮、c1 已配对、连续
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns2 = await s2.load_mainline()
    s2.close()
    assert len(turns2) == 2
    msgs0 = turns2[0].messages
    use_ids = {b.id for m in msgs0 for b in m.content if isinstance(b, ToolUseBlock)}
    res_ids = {
        b.tool_use_id for m in msgs0 for b in m.content if isinstance(b, ToolResultBlock)
    }
    assert use_ids <= res_ids  # c1 已被合成 tool_result 配对
    assert msgs0[-1].role == "assistant"
    assert turns2[1].messages[0].content[0].text == "第二问"


# ---- Task 5: compaction 边界行 + 摘要行 + load_mainline 边界截断 ----


def _read_jsonl(store: SessionStore) -> list[dict]:
    """读 store 的 jsonl,splitlines 解析(便于校验落盘字段)。"""
    return [
        json.loads(ln)
        for ln in store._jsonl.read_text(encoding="utf-8").splitlines()  # noqa: SLF001
        if ln.strip()
    ]


async def test_append_system_compact_writes_null_parent_and_logical(tmp_path):
    """边界行:parentUuid=null、logicalParentUuid 续链、compactMetadata 落盘。"""
    store = _store(tmp_path)
    # 先 append 一条普通消息,拿到它的 uuid 作为 logical_parent
    last = await store.append(Message(role="user", content=[TextBlock(text="hi")]))
    buuid = await store.append_system_compact(
        subtype="compact_boundary",
        logical_parent_uuid=last,
        content="Conversation compacted",
        compact_metadata={"trigger": "manual", "preTokens": 100, "postTokens": 50},
    )
    rows = _read_jsonl(store)
    boundary = [r for r in rows if r.get("subtype") == "compact_boundary"][0]
    assert boundary["parentUuid"] is None
    assert boundary["logicalParentUuid"] == last
    assert boundary["compactMetadata"]["trigger"] == "manual"
    # store 内部 _last_uuid 推进到边界行(后续摘要行 append 接续)
    assert store._last_uuid == buuid  # noqa: SLF001
    store.close()


async def test_append_compact_summary_parents_boundary(tmp_path):
    """摘要 user 行:parentUuid = 边界行 uuid,带 isCompactSummary=True。"""
    store = _store(tmp_path)
    buuid = await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="x", compact_metadata={},
    )
    suuid = await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="SUMMARY")])
    )
    rows = _read_jsonl(store)
    summary = [r for r in rows if r.get("isCompactSummary")][0]
    assert summary["parentUuid"] == buuid
    assert summary["isCompactSummary"] is True
    assert store._last_uuid == suuid  # noqa: SLF001
    store.close()


async def test_load_mainline_truncates_at_last_boundary(tmp_path):
    """resume:边界前 bulk 不装入,只得[摘要+尾段]。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="OLD-BULK")]))
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="c", compact_metadata={},
    )
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="THE-SUMMARY")])
    )
    await store.append(Message(role="user", content=[TextBlock(text="tail-q")]))
    store.close()

    # resume:重新打开,触发 load_mainline
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await s2.load_mainline()
    s2.close()
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "THE-SUMMARY" in texts and "tail-q" in texts
    assert "OLD-BULK" not in texts  # 边界前 bulk 被截断


async def test_load_mainline_keeps_only_last_boundary_after_multiple_compactions(tmp_path):
    """多次 compaction:resume 只装[最后摘要 + 尾段];更早边界+bulk 全丢。"""
    store = _store(tmp_path)
    # 第一次压缩前的 bulk
    await store.append(Message(role="user", content=[TextBlock(text="BULK-1")]))
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="c1", compact_metadata={},
    )
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="SUMMARY-1")])
    )
    # 第二次压缩前的中段(在第一次摘要后、第二次边界前)
    await store.append(Message(role="user", content=[TextBlock(text="MID")]))
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="c2", compact_metadata={},
    )
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="SUMMARY-2")])
    )
    # 尾段
    await store.append(Message(role="user", content=[TextBlock(text="tail")]))
    store.close()

    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await s2.load_mainline()
    s2.close()
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "SUMMARY-2" in texts and "tail" in texts
    # 早期 bulk / 中段 / 第一条摘要全丢
    assert "BULK-1" not in texts
    assert "MID" not in texts
    assert "SUMMARY-1" not in texts


async def test_load_mainline_no_boundary_loads_all(tmp_path):
    """无边界行 → 行为不变,全量装载(现状兼容)。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True
    )
    store.close()

    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await s2.load_mainline()
    s2.close()
    assert len(turns) == 1
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "问" in texts and "答" in texts


async def test_append_system_compact_io_failure_does_not_raise(tmp_path):
    """IO 失败只 log,不杀主循环(契 CLAUDE.md §7)。"""
    store = _store(tmp_path)
    store._fh.close()  # noqa: SLF001 - 强制关闭句柄制造写失败
    # 不应抛异常
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="x", compact_metadata={},
    )
    store.close()


async def test_append_compact_summary_io_failure_does_not_raise(tmp_path):
    """摘要行 IO 失败只 log,不杀主循环。"""
    store = _store(tmp_path)
    store._fh.close()  # noqa: SLF001 - 强制关闭句柄制造写失败
    # 不应抛异常
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="S")])
    )
    store.close()


async def test_load_mainline_after_resume_extends_correctly(tmp_path):
    """resume 截断后续 append:_last_uuid 接到尾段叶子,新提问挂链不串入老 bulk。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="OLD")]))
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="c", compact_metadata={},
    )
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="SUM")])
    )
    store.close()

    # resume 后续聊
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s2.load_mainline()
    await s2.append(Message(role="user", content=[TextBlock(text="new-q")]))
    s2.close()

    # 再 resume:_last_uuid 接到 new-q,old bulk 仍被边界截断
    s3 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    turns = await s3.load_mainline()
    s3.close()
    texts = [b.text for t in turns for m in t.messages for b in m.content if hasattr(b, "text")]
    assert "SUM" in texts and "new-q" in texts
    assert "OLD" not in texts


# ---- Task 6: append 透传新参 + append_queue_operation ----


def _read_jsonl_lines(tmp_path, sid="s1"):
    return [
        json.loads(ln)
        for ln in _jsonl(tmp_path, sid).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


async def test_append_threads_tool_use_results_and_source(tmp_path):
    """append 透传 tool_use_results + source_tool_assistant_uuid → 落盘顶层字段。"""
    from birdcode.blocks import ToolResultBlock

    store = _store(tmp_path)
    await store.append(
        Message(role="assistant", content=[TextBlock(text="派生")]), is_assistant=True,
    )
    await store.append(
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_x", content="done")]),
        tool_use_results={"call_x": {"agentId": "a1", "status": "completed"}},
        source_tool_assistant_uuid="asst-uuid",
    )
    store.close()
    rows = _read_jsonl_lines(tmp_path)
    result_line = [r for r in rows if r.get("toolUseResult")][0]
    assert result_line["toolUseResult"] == {"call_x": {"agentId": "a1", "status": "completed"}}
    assert result_line["sourceToolAssistantUUID"] == "asst-uuid"


async def test_append_task_notification_skips_sidecar(tmp_path):
    """isTaskNotification 注入行不计入 turn_count(append 侧)。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="真提问")]))
    await store.append(
        Message(role="user", content=[TextBlock(text="<task-notification>x</task-notification>")]),
        is_task_notification=True, agent_id="a1",
    )
    store.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["turn_count"] == 1  # 注入行不算


async def test_append_queue_operation_writes_line(tmp_path):
    """append_queue_operation 落 queue-operation 事件行;对消息 DAG 透明(不推进 _last_uuid)。

    事件行无 uuid/parentUuid/content(通知正文由紧随的 isTaskNotification user 行承载)。
    """
    store = _store(tmp_path)
    prev_uuid = await store.append(Message(role="user", content=[TextBlock(text="q")]))
    await store.append_queue_operation(
        operation="enqueue", agent_id="a1", tool_use_id="call_x",
        status="completed",
    )
    assert store._last_uuid == prev_uuid  # noqa: SLF001 - 事件行不进链
    store.close()
    rows = _read_jsonl_lines(tmp_path)
    q_line = [r for r in rows if r.get("type") == "queue-operation"][0]
    assert q_line["operation"] == "enqueue"
    assert q_line["agentId"] == "a1"
    assert "uuid" not in q_line and "parentUuid" not in q_line  # 事件行不挂 DAG
    assert "content" not in q_line  # 通知正文不在此行


# ---- Task 7: sidecar 行级计数排除 isTaskNotification / isCompactSummary ----


async def test_scan_excludes_task_notification_and_compact_summary(tmp_path):
    """list_sessions lazy backfill:isTaskNotification / isCompactSummary 行不计入。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="真提问")]))
    await store.append(
        Message(role="user", content=[TextBlock(text="<task-notification>x</task-notification>")]),
        is_task_notification=True, agent_id="a1",
    )
    store.close()
    # 手动塞一条 isCompactSummary 行(模拟压缩摘要)
    jf = _jsonl(tmp_path)
    compact_line = {
        "type": "user", "isCompactSummary": True,
        "message": {"role": "user", "content": [{"type": "text", "text": "压缩摘要"}]},
        "uuid": "x", "sessionId": "s1", "timestamp": "t",
    }
    with jf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(compact_line, ensure_ascii=False) + "\n")
    _meta(tmp_path).unlink()  # 强制 lazy backfill
    summaries = SessionStore.list_sessions(tmp_path, tmp_path)
    assert summaries[0].turn_count == 1  # 只算真提问
    assert summaries[0].first_user_summary == "真提问"


async def test_load_mainline_backfill_excludes_flagged_rows(tmp_path):
    """load_mainline backfill:isTaskNotification 行不计入 turn_count。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="第一问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    await store.append(
        Message(role="user", content=[TextBlock(text="<task-notification>x</task-notification>")]),
        is_task_notification=True, agent_id="a1",
    )
    store.close()
    _meta(tmp_path).unlink()  # 触发 load_mainline backfill
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s2.load_mainline()
    s2.close()
    data = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))
    assert data["turn_count"] == 1  # 注入行不算
    assert data["first_user"] == "第一问"


# ---- CR 修复回归测试(#2/#3/#5/#9)----


def test_count_handles_null_message_content():
    """CR #3: message.content=null(坏行)不杀 _count(防御回归)。

    旧 _first_user_and_turn_count_from_turns 经 message_from_dict 的 isinstance 检查防御;
    T7 改行级 _count 后丢了这个防御,content=null → for b in None → TypeError 杀 resume。
    """
    from birdcode.session.store import _count_first_user_and_turns_from_rows

    rows = [
        {"type": "user", "message": {"role": "user", "content": None}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "好"}]}},
    ]
    first_user, turn_count = _count_first_user_and_turns_from_rows(rows)  # 不抛
    assert turn_count == 1
    assert first_user == "好"


def test_count_excludes_sidechain_rows():
    """CR #9: isSidechain 行(若泄露进主 jsonl)不计入 turn_count(lazy backfill 防御)。"""
    from birdcode.session.store import _count_first_user_and_turns_from_rows

    rows = [
        {
            "type": "user", "isSidechain": True,
            "message": {"role": "user", "content": [{"type": "text", "text": "侧链"}]},
        },
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "主线"}]},
        },
    ]
    first_user, turn_count = _count_first_user_and_turns_from_rows(rows)
    assert turn_count == 1
    assert first_user == "主线"


async def test_append_queue_operation_invalid_status_does_not_raise(tmp_path):
    """CR #2: 非法 status(混淆 AgentToolUseResult.status 的 launched/running 与
    QueueOperationLine.status 的 completed/error/cancelled)→ encode 抛 ValidationError
    被 try 捕获,不杀主循环。旧实现 encode 在 try 外 → 崩溃。
    """
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="q")]))
    await store.append_queue_operation(  # 不抛
        operation="enqueue", agent_id="a1", tool_use_id="call_x",
        status="running",
    )
    store.close()


async def test_load_mainline_does_not_overwrite_existing_sidecar(tmp_path):
    """CR #5: sidecar 已存在时,resume 不用 live 段计数覆盖(否则压缩后永久少计)。

    旧实现:load_mainline 无条件用 split 后的 live 段重写 sidecar → 压缩后 live 段只剩
    [摘要+尾段],turn_count 被覆盖成尾段轮次,丢失压缩前累计的真实提问数。
    """
    store = _store(tmp_path, "s1")
    await store.append(Message(role="user", content=[TextBlock(text="第一问")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True)
    await store.append(Message(role="user", content=[TextBlock(text="第二问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答2")]), is_assistant=True
    )
    # 压缩:边界 + 摘要(都不更新 sidecar)
    await store.append_system_compact(
        subtype="compact_boundary", logical_parent_uuid=None, content="c", compact_metadata={},
    )
    await store.append_compact_summary(
        Message(role="user", content=[TextBlock(text="SUMMARY")])
    )
    store.close()
    # 压缩前 sidecar = 2 轮真实提问
    assert json.loads(_meta(tmp_path).read_text(encoding="utf-8"))["turn_count"] == 2

    # resume:live 段只有[摘要](isCompactSummary 排除)→ 计数 0;旧实现会覆盖成 0
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    await s2.load_mainline()
    s2.close()
    # 新实现:sidecar 已存在,不覆盖 → 仍是 2
    assert json.loads(_meta(tmp_path).read_text(encoding="utf-8"))["turn_count"] == 2


# ---- Phase2 F8: mainline_rows() 访问器 ----


async def test_mainline_rows_returns_mainline_only(tmp_path):
    """mainline_rows() 返回主线行(跳过 sidechain/空/坏行),内容正确。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="问")]))
    await store.append(
        Message(role="assistant", content=[TextBlock(text="答")]), is_assistant=True
    )
    store.close()
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    rows = s2.mainline_rows()
    s2.close()
    assert len(rows) == 2
    assert [r["type"] for r in rows] == ["user", "assistant"]
    assert rows[0]["message"]["content"][0]["text"] == "问"
    assert rows[1]["message"]["content"][0]["text"] == "答"


async def test_mainline_rows_skips_sidechain(tmp_path):
    """mainline_rows 跳过 isSidechain 行(与 _read_mainline_rows 一致)。"""
    store = _store(tmp_path)
    await store.append(Message(role="user", content=[TextBlock(text="主线")]))
    store.close()
    # 手动塞一条 isSidechain 行(模拟侧链泄露进主 jsonl)
    jf = _jsonl(tmp_path)
    sidechain_line = {
        "type": "user", "isSidechain": True,
        "message": {"role": "user", "content": [{"type": "text", "text": "侧链"}]},
        "uuid": "sc1", "sessionId": "s1", "timestamp": "t",
    }
    with jf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sidechain_line, ensure_ascii=False) + "\n")
    s2 = SessionStore(_ctx("s1"), tmp_path, root=tmp_path)
    rows = s2.mainline_rows()
    s2.close()
    assert len(rows) == 1
    assert rows[0]["message"]["content"][0]["text"] == "主线"


def test_mainline_rows_empty_when_no_file(tmp_path):
    """无 jsonl 文件时返回空列表(不抛)。"""
    store = _store(tmp_path, "never-exists")
    assert store.mainline_rows() == []
    store.close()
