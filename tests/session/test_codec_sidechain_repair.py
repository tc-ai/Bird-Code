# tests/session/test_codec_sidechain_repair.py
"""Task 3:repair_trailing_edge 参数化 + 侧链加载测试。

覆盖:
- 默认 mainline 行为严格不变(不传 error_message_fn → 原默认文案)。
- 传 conservative_sidechain_error_message → 决策 B 文案(带工具名 + "结果未知" +
  "勿盲目重复"),is_error=True。
- load_sidechain_turns:读侧链 jsonl → split/decode/repair(保守文案),末尾恒以
  assistant 收尾、tool_use 全配对。
- _read_jsonl_rows:跳空行/坏 JSON/非 dict/半截行,文件不存在返回 []。
"""
from __future__ import annotations

import json
from pathlib import Path

from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.session.codec import (
    _read_jsonl_rows,
    conservative_sidechain_error_message,
    load_sidechain_turns,
    repair_trailing_edge,
)


def _turn_with_unpaired_tool_use() -> list[Turn]:
    # 末尾 assistant 含 tool_use、无 tool_result(子 agent 死在工具执行中途)
    return [
        Turn(
            messages=[
                Message(role="user", content=[TextBlock(text="do thing")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="toolu_1", name="bash", input={"cmd": "git commit"})],
                ),
            ]
        )
    ]


# ---- 默认 mainline 行为不变(向后兼容红线)----


def test_default_repair_message_unchanged_for_mainline():
    """不传 error_message_fn → 产出与原默认完全一致的文案与 is_error。"""
    turns = _turn_with_unpaired_tool_use()
    repair_trailing_edge(turns)  # 不传 fn → 用原默认文案
    synth = turns[-1].messages[-2]  # 倒数第二:补的 error tool_result(user);末尾是合成 assistant
    assert synth.role == "user"
    tr = next(b for b in synth.content if isinstance(b, ToolResultBlock))
    assert tr.content == "工具调用因会话中断未完成,请重新发起"
    assert tr.is_error is True


def test_default_repair_returns_synthetic_messages():
    """默认调用仍返回追加的合成 Message 列表(向后兼容:供 store 持久化)。"""
    turns = _turn_with_unpaired_tool_use()
    synth = repair_trailing_edge(turns)
    # 悬空补 error tool_result(user) + 末尾补 assistant → 至少 2 条合成
    assert len(synth) >= 1
    assert all(m.synthetic for m in synth)


# ---- 决策 B 文案(保守 + 工具名)----


def test_sidechain_repair_uses_conservative_message_with_tool_name():
    """传 conservative_sidechain_error_message → 文案带工具名 + 保守措辞。"""
    turns = _turn_with_unpaired_tool_use()
    repair_trailing_edge(turns, error_message_fn=conservative_sidechain_error_message)
    synth = turns[-1].messages[-2]
    assert synth.role == "user"
    tr = next(b for b in synth.content if isinstance(b, ToolResultBlock))
    assert "bash" in tr.content  # 带工具名
    assert "结果未知" in tr.content  # 保守措辞
    assert "勿盲目重复" in tr.content  # 警告副作用
    assert tr.is_error is True


def test_conservative_message_function_directly():
    """conservative_sidechain_error_message 直接调用:带工具名、含关键短语。"""
    tu = ToolUseBlock(id="t1", name="write_file", input={"path": "a.py"})
    msg = conservative_sidechain_error_message(tu)
    assert "write_file" in msg
    assert "结果未知" in msg
    assert "勿盲目重复" in msg


def test_conservative_message_falls_back_when_name_empty():
    """tu.name 为空字符串 → 兜底为"工具"(不抛、不出现空洞)。"""
    tu = ToolUseBlock(id="t1", name="", input={})
    msg = conservative_sidechain_error_message(tu)
    assert "工具" in msg
    assert "结果未知" in msg


def test_custom_fn_does_not_break_trailing_assistant():
    """传 fn 时,末尾仍补合成 assistant(主线恒以 assistant 收尾)。"""
    turns = _turn_with_unpaired_tool_use()
    repair_trailing_edge(turns, error_message_fn=conservative_sidechain_error_message)
    assert turns[-1].messages[-1].role == "assistant"


# ---- _read_jsonl_rows(容错半截行/坏 JSON)----


def test_read_jsonl_rows_skips_blank_and_bad_lines(tmp_path: Path):
    """空行/坏 JSON/非 dict 跳过;半截行(mid-flush 断电)不抛。"""
    p = tmp_path / "sc.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "idx": 1}),
                "",  # 空行
                "{not json",  # 坏 JSON(半截行)
                json.dumps([1, 2, 3]),  # 非 dict
                json.dumps({"type": "assistant", "idx": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = _read_jsonl_rows(p)
    assert len(rows) == 2
    assert rows[0]["idx"] == 1
    assert rows[1]["idx"] == 2


def test_read_jsonl_rows_missing_file_returns_empty(tmp_path: Path):
    """文件不存在 → 返回 [](不抛)。"""
    assert _read_jsonl_rows(tmp_path / "nope.jsonl") == []


# ---- load_sidechain_turns(端到端:读侧链 → repair 保守文案)----


def test_load_sidechain_turns_repairs_with_conservative_message(tmp_path: Path):
    """侧链末尾 unpaired tool_use → load 后补保守 error tool_result + assistant。"""
    p = tmp_path / "sc.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "do thing"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "toolu_1", "name": "bash",
                                 "input": {"cmd": "git commit"}}
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    turns = load_sidechain_turns(p)
    assert len(turns) >= 1
    msgs = turns[-1].messages
    # 末尾恒以 assistant 收尾
    assert msgs[-1].role == "assistant"
    # 倒数第二:保守 error tool_result(带工具名)
    tr_msg = msgs[-2]
    assert tr_msg.role == "user"
    tr = next(b for b in tr_msg.content if isinstance(b, ToolResultBlock))
    assert tr.is_error is True
    assert "bash" in tr.content
    assert "结果未知" in tr.content


def test_load_sidechain_turns_missing_file_returns_empty(tmp_path: Path):
    """侧链文件不存在 → [](不抛,供 Task 4 容错:无侧链 = 空历史)。"""
    assert load_sidechain_turns(tmp_path / "nope.jsonl") == []


def test_load_sidechain_turns_already_paired_no_repair(tmp_path: Path):
    """侧链末尾已配对(完整 Turn)→ 不追加保守 error,末尾保持原 assistant。"""
    p = tmp_path / "sc.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": [{"type": "text", "text": "q"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": "c1", "name": "echo", "input": {}}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    turns = load_sidechain_turns(p)
    msgs = turns[-1].messages
    # 末尾仍是原 assistant(text),不追加保守 error
    assert msgs[-1].role == "assistant"
    assert isinstance(msgs[-1].content[0], TextBlock)
    assert len(msgs) == 4
