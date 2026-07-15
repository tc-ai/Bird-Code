# tests/session/test_sidechain_partial_line.py
"""Task 15:侧链半截行容错(断电 mid-flush)回归测试。

锁定行为:断电 mid-flush 在侧链 jsonl 末尾留下半截 JSON 行时,``load_sidechain_turns``
必须——
- 不抛异常(读路径恒成功,不让 resume 因半截行崩溃);
- 跳过半截行(不污染返回的 turns,不出现残缺/拼错的 block);
- 保留半截行之前的所有完整行(至少含完整那条 turn)。

T3 已在 ``_read_jsonl_rows`` 用 ``except json.JSONDecodeError: continue`` 保证此行为;
本测试把它钉死成回归红线,防后续重构(如改用 streaming reader / 裸 json.loads)破坏。

覆盖:
- 末行半截(无尾换行,最贴近真实 mid-flush 形态)→ 跳过、完整 turn 保留。
- 多条完整行 + 末尾半截 → 全部完整行保留。
- 半截行内容不渗入 turns(content 不被污染)。
- 仅半截行(无完整行)→ [] 而非抛错。
- 末行半截 + 带尾换行(另一种坏 JSON 形态)亦跳过。
"""
from __future__ import annotations

import json
from pathlib import Path

from birdcode.blocks import TextBlock
from birdcode.session.codec import load_sidechain_turns


def _good_user_line(text: str = "原任务") -> str:
    """一条完整、可正常解码的侧链 user 行(isSidechain=true)。"""
    return json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
            "uuid": "u1",
            "sessionId": "s",
            "timestamp": "2026-07-15T00:00:00Z",
            "isSidechain": True,
            "agentId": "sub-1",
        },
        ensure_ascii=False,
    )


# 断电 mid-flush 的典型形态:写了 JSON 前缀就被截断(无尾换行,parser 必失败)。
_PARTIAL_ASSISTANT = '{"type":"assistant","message":{"role":"assistant",'


def test_partial_trailing_line_is_skipped(tmp_path: Path):
    """brief Step 1 主用例:完整行 + 末尾半截行 → 不抛、至少返回完整那条 turn。"""
    p = tmp_path / "agent-sub-1.jsonl"
    good = _good_user_line("原任务") + "\n"
    p.write_text(good + _PARTIAL_ASSISTANT, encoding="utf-8")  # 末行半截,无尾换行
    turns = load_sidechain_turns(p)  # 不抛,半截行跳过
    assert len(turns) >= 1  # 至少含完整那条


def test_partial_trailing_line_preserves_good_content(tmp_path: Path):
    """半截行不污染 history:完整 user turn 的 text 原样保留,不出现残缺 assistant block。"""
    p = tmp_path / "agent-sub-1.jsonl"
    p.write_text(_good_user_line("原任务") + "\n" + _PARTIAL_ASSISTANT, encoding="utf-8")
    turns = load_sidechain_turns(p)
    assert len(turns) == 1
    msgs = turns[0].messages
    # 完整 user 行被解码,文本原样保留
    user_texts = [
        b.text for m in msgs for b in m.content if isinstance(b, TextBlock)
    ]
    assert "原任务" in user_texts
    # 半截 assistant 行被 _read_jsonl_rows 丢弃 → decode 看不到它 → 不产生残缺 block。
    # repair 会因末尾以 user 收尾而补一条合成 assistant(固定文案「已自动补全收尾」),
    # 该合成文本与半截行原文(`{"role":"assistant"...`)无任何重叠 —— 用此断言钉死:
    # 若半截行未被跳过,其原文不会以 TextBlock 形式渗入 history。
    for m in msgs:
        if m.role == "assistant":
            for b in m.content:
                if isinstance(b, TextBlock):
                    assert "role" not in b.text
                    assert "assistant" not in b.text


def test_multiple_good_lines_then_partial_keeps_all_good(tmp_path: Path):
    """多条完整行 + 末尾半截 → 所有完整行保留(半截只丢自己,不连累前序)。"""
    p = tmp_path / "agent-sub-1.jsonl"
    p.write_text(
        _good_user_line("第一问") + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "第一答"}],
                },
            },
            ensure_ascii=False,
        )
        + "\n"
        + _good_user_line("第二问") + "\n"
        + _PARTIAL_ASSISTANT,
        encoding="utf-8",
    )
    turns = load_sidechain_turns(p)
    # 两轮完整提问(user-question 起 Turn)都应保留
    all_texts = [
        b.text for t in turns for m in t.messages for b in m.content
        if isinstance(b, TextBlock)
    ]
    assert "第一问" in all_texts
    assert "第一答" in all_texts
    assert "第二问" in all_texts


def test_only_partial_line_returns_empty(tmp_path: Path):
    """文件只有半截行(无任何完整行)→ [](不抛,调用方见空历史即容错)。"""
    p = tmp_path / "agent-sub-1.jsonl"
    p.write_text(_PARTIAL_ASSISTANT, encoding="utf-8")
    assert load_sidechain_turns(p) == []


def test_partial_line_with_trailing_newline_also_skipped(tmp_path: Path):
    """半截行 + 尾换行(另一种坏 JSON 形态:flush 写了换行但 JSON 体仍残缺)亦跳过。"""
    p = tmp_path / "agent-sub-1.jsonl"
    p.write_text(
        _good_user_line("原任务") + "\n" + _PARTIAL_ASSISTANT + "\n",
        encoding="utf-8",
    )
    turns = load_sidechain_turns(p)
    assert len(turns) >= 1
