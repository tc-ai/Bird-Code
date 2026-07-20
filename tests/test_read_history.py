# tests/test_read_history.py
"""read_history 工具测试:从当前会话 jsonl 读「被压缩掉的历史对话原文」。

覆盖:
- 未注入 jsonl(未启用持久化)→ 友好错误
- 无压缩边界 → 提示尚未压缩
- 有边界:读压缩点之前的轮次,【最新→最早】展示,offset/limit 分页
- offset 越界 → 错误提示
- 边界之后的(摘要/尾段 = live 上下文)不返回
- 多次压缩:以【最近一次】压缩点为基点
- tool_result 渲染截断
- 只读注入的当前会话路径(结构上不读其它会话)
"""

import json
from pathlib import Path

from birdcode.tools.read_history import ReadHistoryTool

_TS = "2026-07-06T00:00:00.000Z"


def _u(text: str, uuid: str, parent: str | None) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _a(text: str, uuid: str, parent: str | None) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _assistant_tool_use(uuid: str, parent: str | None, tu_id: str = "tu1") -> dict:
    """assistant 行只含一个 tool_use 块。"""
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tu_id, "name": "grep", "input": {"pattern": "x"}},
            ],
        },
    }


def _user_tool_result(uuid: str, parent: str | None, content: str, tu_id: str = "tu1") -> dict:
    """user 行只含一个 tool_result 块(回填,无 text → 不开新 Turn)。"""
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tu_id, "content": content}],
        },
    }


def _boundary(uuid: str, logical_parent: str, *, preserved: int | None = None) -> dict:
    meta: dict = {"trigger": "auto", "preTokens": 167000, "postTokens": 48000}
    if preserved is not None:
        meta["preservedTurnCount"] = preserved
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "parentUuid": None,
        "logicalParentUuid": logical_parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "content": "Conversation compacted",
        "compactMetadata": meta,
    }


def _summary(uuid: str, parent: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": _TS,
        "isSidechain": False,
        "isCompactSummary": True,
        "message": {"role": "user", "content": [{"type": "text", "text": "SUMMARY..."}]},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


async def test_not_enabled_returns_error() -> None:
    tool = ReadHistoryTool()  # 未 set_jsonl
    out = await tool.execute()
    assert "<error>" in out and "未启用" in out


async def test_no_boundary_means_not_compacted(tmp_path: Path) -> None:
    f = tmp_path / "s1.jsonl"
    _write_jsonl(f, [_u("hello", "1", None), _a("hi", "2", "1")])
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute()
    assert "尚未压缩" in out


async def test_reads_compressed_newest_first_with_pagination(tmp_path: Path) -> None:
    """3 轮被压缩(T1=alpha 最早 → T3=gamma 最新),offset=0/limit=2 → gamma,beta;alpha 不含。"""
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("alpha", "1", None),
        _a("A1", "2", "1"),
        _u("beta", "3", "2"),
        _a("A2", "4", "3"),
        _u("gamma", "5", "4"),
        _a("A3", "6", "5"),
        _boundary("b1", "6"),
        _summary("s1", "b1"),
        _u("tail-live", "7", "s1"),
        _a("tail-resp", "8", "7"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)

    out = await tool.execute(offset=0, limit=2)
    assert "共 3 轮" in out
    assert "gamma" in out and "beta" in out
    assert "alpha" not in out  # limit=2 只取最新两轮
    # 最新→最早:gamma 段在 beta 段之前
    assert out.index("gamma") < out.index("beta")
    # 边界之后的 live 内容(摘要/尾段)不在历史输出里
    assert "tail-live" not in out and "SUMMARY" not in out
    # 翻页提示:更早的历史用 offset=2
    assert "offset=2" in out


async def test_pagination_to_older_window(tmp_path: Path) -> None:
    """offset=2/limit=2 → 只剩 T1(alpha);提示可往更新翻。"""
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("alpha", "1", None),
        _a("A1", "2", "1"),
        _u("beta", "3", "2"),
        _a("A2", "4", "3"),
        _u("gamma", "5", "4"),
        _a("A3", "6", "5"),
        _boundary("b1", "6"),
        _summary("s1", "b1"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)

    out = await tool.execute(offset=2, limit=2)
    assert "alpha" in out
    assert "beta" not in out and "gamma" not in out
    # offset>0 → 提示往更新(更接近压缩点)翻
    assert "更新" in out and "offset=0" in out


async def test_offset_out_of_range_errors(tmp_path: Path) -> None:
    f = tmp_path / "s1.jsonl"
    _write_jsonl(
        f,
        [_u("alpha", "1", None), _a("A1", "2", "1"), _boundary("b1", "2"), _summary("s1", "b1")],
    )
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute(offset=5, limit=2)
    assert "<error>" in out and "offset 5 超出" in out
    assert "offset=0" in out  # hint 引导从头读


async def test_most_recent_boundary_is_anchor(tmp_path: Path) -> None:
    """两次压缩:以【最近一次】boundary 为基点 —— 它之前的可读,它之后的 tail 不可读。"""
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("old", "1", None),
        _a("A1", "2", "1"),
        _boundary("b1", "2"),
        _summary("sum1", "b1"),
        _u("mid", "3", "sum1"),
        _a("A2", "4", "3"),
        _boundary("b2", "4"),
        _summary("sum2", "b2"),
        _u("tail-live", "5", "sum2"),
        _a("A3", "6", "5"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute(offset=0, limit=20)
    # b2 之前的 mid/old 可读;b2 之后的 tail-live 不可读;更早的摘要被过滤不展示
    assert "mid" in out and "old" in out
    assert "tail-live" not in out
    assert "SUMMARY" not in out  # 更早的压缩摘要不计入、不展示
    assert "共 2 轮" in out  # 只剩 old + mid 两轮真实对话(摘要不占名额)


async def test_compact_summaries_do_not_count_toward_limit(tmp_path: Path) -> None:
    """更早的压缩摘要(isCompactSummary)不是 user-assistant 对 → 不占 limit 名额、不展示。"""
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("t1", "1", None),
        _a("a1", "2", "1"),
        _boundary("b1", "2"),
        _summary("sum1", "b1"),
        _u("t2", "3", "sum1"),
        _a("a2", "4", "3"),
        _boundary("b2", "4"),
        _summary("sum2", "b2"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    # 只剩 t1/t2 两轮真实对话(中间的 sum1 摘要被过滤);limit=1 最新优先 → 只返回 t2
    out = await tool.execute(offset=0, limit=1)
    assert "t2" in out
    assert "t1" not in out
    assert "SUMMARY" not in out
    assert "共 2 轮" in out  # 摘要不计入 total


async def test_tail_skipped_when_preserved_turn_count_set(tmp_path: Path) -> None:
    """边界 compactMetadata.preservedTurnCount=尾段轮数 → read_history 跳过尾段原始副本,
    只返回真正丢失的 prefix,不与当前上下文(live 尾段)重复。"""
    f = tmp_path / "s1.jsonl"
    rows = [
        # prefix(被摘要、已丢失):alpha, beta
        _u("alpha", "1", None),
        _a("A1", "2", "1"),
        _u("beta", "3", "2"),
        _a("A2", "4", "3"),
        # 尾段(保留进 live、又重写到边界后):gamma —— 边界前这份是原始副本,应跳过
        _u("gamma", "5", "4"),
        _a("A3", "6", "5"),
        _boundary("b1", "6", preserved=1),  # 尾段 1 轮 = gamma
        _summary("s1", "b1"),
        # 边界之后:重写的尾段(= live)
        _u("gamma-live", "7", "s1"),
        _a("A3-live", "8", "7"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)

    out = await tool.execute(offset=0, limit=10)
    assert "alpha" in out and "beta" in out  # prefix 返回
    assert "gamma" not in out  # 尾段原始副本已跳过,不与 live 重复
    assert "gamma-live" not in out  # 边界之后的 live 也不读
    assert "共 2 轮" in out  # 只数 prefix(alpha,beta)
    assert "尾段最近 1 轮已在当前上下文,已自动跳过" in out


async def test_old_boundary_without_preserved_reads_all(tmp_path: Path) -> None:
    """老边界(无 preservedTurnCount,本修复前的会话)→ N=0 退回旧行为:读全部边界前轮次。

    锁「不回归」:旧会话不会因缺字段出错,只是头几页可能与上下文重复(已知、可接受)。
    """
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("alpha", "1", None),
        _a("A1", "2", "1"),
        _u("gamma", "5", "4"),
        _a("A3", "6", "5"),
        _boundary("b1", "6"),  # 无 preserved → N=0
        _summary("s1", "b1"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute(offset=0, limit=10)
    # N=0 → 全读(含尾段原始副本 gamma),旧行为
    assert "alpha" in out and "gamma" in out
    assert "共 2 轮" in out
    assert "已自动跳过" not in out  # 无跳过提示


async def test_tool_result_block_truncated(tmp_path: Path) -> None:
    """单条 tool_result 超 _MAX_BLOCK_CHARS → 渲染带 (截断) 标记。"""
    from birdcode.tools.read_history import _MAX_BLOCK_CHARS

    huge = "X" * (_MAX_BLOCK_CHARS * 3)
    f = tmp_path / "s1.jsonl"
    rows = [
        _u("run grep", "1", None),
        _assistant_tool_use("2", "1"),
        _user_tool_result("3", "2", huge),
        _a("done", "4", "3"),
        _boundary("b1", "4"),
        _summary("s1", "b1"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute(offset=0, limit=5)
    assert "[tool_result" in out and "(截断)" in out
    # 完整原文不应整段出现(被截断)
    assert huge not in out


async def test_only_reads_injected_session_path(tmp_path: Path) -> None:
    """结构保证:只读 set_jsonl 注入的当前会话,不读同目录其它会话 jsonl。"""
    other = tmp_path / "other.jsonl"
    _write_jsonl(
        other,
        [
            _u("secret-other-session", "1", None),
            _a("A", "2", "1"),
            _boundary("b1", "2"),
            _summary("s1", "b1"),
        ],
    )
    cur = tmp_path / "current.jsonl"
    _write_jsonl(
        cur,
        [
            _u("current-msg", "1", None),
            _a("A", "2", "1"),
            _boundary("b1", "2"),
            _summary("s1", "b1"),
        ],
    )
    tool = ReadHistoryTool()
    tool.set_jsonl(cur)  # 只注入当前会话
    out = await tool.execute(offset=0, limit=5)
    assert "current-msg" in out
    assert "secret-other-session" not in out  # 其它会话绝不读


def test_description_and_name() -> None:
    d = ReadHistoryTool.description
    assert ReadHistoryTool.name == "read_history"
    assert "历史" in d and "压缩点" in d and "offset" in d


def test_limit_field_default_matches_constant() -> None:
    """模型省略 limit 时,Pydantic Field default 决定实际值(executor 走 model_dump 传入)。

    Field default 必须与 _DEFAULT_LIMIT 一致 —— 否则生产默认值会偏离常量/description/execute
    形参(曾发生 Field=20 而常量=3 的分叉)。本测即锁此不变式。
    """
    from birdcode.tools.read_history import _DEFAULT_LIMIT, _ReadHistoryInput

    assert _ReadHistoryInput.model_validate({"offset": 0}).limit == _DEFAULT_LIMIT


async def test_text_and_tool_use_args_truncated(tmp_path: Path) -> None:
    """TextBlock 与 ToolUseBlock 入参超 _MAX_BLOCK_CHARS → 渲染截断,不撑爆单块。

    防御:大段粘贴(text)、write_file/edit_file 的 content(tool_use 入参)无界渲染会
    挤掉同窗其余轮次。两块都应带 (截断) 标记。
    """
    from birdcode.tools.read_history import _MAX_BLOCK_CHARS

    huge_text = "T" * (_MAX_BLOCK_CHARS * 3)
    huge_content = "A" * (_MAX_BLOCK_CHARS * 3)
    f = tmp_path / "s1.jsonl"
    rows = [
        {
            "type": "user",
            "uuid": "1",
            "parentUuid": None,
            "sessionId": "s1",
            "timestamp": _TS,
            "isSidechain": False,
            "message": {"role": "user", "content": [{"type": "text", "text": huge_text}]},
        },
        {
            "type": "assistant",
            "uuid": "2",
            "parentUuid": "1",
            "sessionId": "s1",
            "timestamp": _TS,
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "write_file",
                        "input": {"file_path": "x", "content": huge_content},
                    },
                ],
            },
        },
        _boundary("b1", "2"),
        _summary("s1", "b1"),
    ]
    _write_jsonl(f, rows)
    tool = ReadHistoryTool()
    tool.set_jsonl(f)
    out = await tool.execute(offset=0, limit=5)
    assert "[user (截断)]" in out
    assert "[tool_use write_file (截断)]" in out
    assert huge_text not in out  # 完整原文不出现
