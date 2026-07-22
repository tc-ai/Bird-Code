# tests/test_context_usage.py
"""/context 命令的纯逻辑测试:token 估算 + 报告渲染。

覆盖 estimate_context_usage(分类目估算 + 派生 used/free + 工具计数)
与 format_context_report(进度条 / 阈值标记 / MCP 子项 / 实测脚注)。
"""

import json

from birdcode.agent.context import ContextUsage, TokenEstimator, estimate_context_usage
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn
from birdcode.ui.pure import _fmt_tokens, format_context_report

# ---- _fmt_tokens ----


def test_fmt_tokens_small_is_int():
    assert _fmt_tokens(0) == "0"
    assert _fmt_tokens(500) == "500"
    assert _fmt_tokens(999) == "999"


def test_fmt_tokens_thousands_with_one_decimal():
    assert _fmt_tokens(1000) == "1.0k"
    assert _fmt_tokens(1500) == "1.5k"
    assert _fmt_tokens(128000) == "128.0k"


def test_fmt_tokens_millions_as_lowercase_m():
    # 百万级 → 小写 m:整数百万 "1m"、非整数 "1.2m";k 舍入到 1000 升级 m(杜绝 1000.0k)
    assert _fmt_tokens(999_999) == "1m"
    assert _fmt_tokens(1_000_000) == "1m"
    assert _fmt_tokens(1_234_567) == "1.2m"


# ---- estimate_context_usage ----


def test_estimate_categories_and_derived():
    est = TokenEstimator()
    sys_text = "system prompt text here"
    tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}]
    mcp = [{"name": "mcp__srv__t", "description": "mcp tool", "input_schema": {"type": "object"}}]
    history = [Turn(messages=[Message(role="user", content=[TextBlock(text="hello world")])])]
    u = estimate_context_usage(
        window=100_000,
        autocompact_threshold=67_000,
        system_text=sys_text,
        tools=tools,
        mcp_tools=mcp,
        history=history,
    )
    assert u.system_prompt == est.char_to_token(sys_text)
    assert u.tools == est.char_to_token(json.dumps(tools, ensure_ascii=False))
    assert u.mcp_tools == est.char_to_token(json.dumps(mcp, ensure_ascii=False))
    assert u.tool_count == 1 and u.mcp_tool_count == 1
    assert u.messages == est.estimate(history=history, current=[], last_in=None)
    # used = system + tools(含 MCP) + messages;MCP 是 tools 子集,不重复加
    assert u.used == u.system_prompt + u.tools + u.messages
    assert u.free == 100_000 - u.used
    assert u.last_measured is None


def test_estimate_carries_last_measured():
    u = estimate_context_usage(
        window=1000,
        autocompact_threshold=500,
        system_text="x",
        tools=[],
        mcp_tools=[],
        history=[],
        last_measured=99_999,
    )
    assert u.last_measured == 99_999


# ---- B:有 API 实测时,Used 锚定真实全量(与 maybe_compact 同源,不再误导)----


def test_used_anchors_to_last_measured_when_present():
    """有 last_measured(API 实测全量)→ Used 锚定真实,messages 反推。

    消除「面板 char 粗估 109k、真实才 23k」的误导:Used = 实测值,到 95k 即真触发。
    """
    u = estimate_context_usage(
        window=128_000,
        autocompact_threshold=95_000,
        system_text="abcdefghij",  # 10 ascii → char_to_token = 2
        tools=[],
        mcp_tools=[],
        history=[Turn(messages=[Message(role="user", content=[TextBlock(text="hi")])])],
        last_measured=50_000,
    )
    assert u.used == 50_000  # 锚定实测,不是 char 粗估
    assert u.messages == 50_000 - u.system_prompt - u.tools  # 反推,使分类目加和 = Used


def test_messages_floors_zero_when_measured_below_overhead():
    """实测值小于估算的 system+tools 开销(deepseek 省 token、char 估偏高)→ messages=0,
    Used 回退为估算开销(小会话,离阈值远,偏高无妨)。"""
    u = estimate_context_usage(
        window=128_000,
        autocompact_threshold=95_000,
        system_text="a" * 400_000,  # → char_to_token = 100_000(远超 last_measured)
        tools=[],
        mcp_tools=[],
        history=[Turn(messages=[Message(role="user", content=[TextBlock(text="hello world")])])],
        last_measured=100,
    )
    assert u.messages == 0  # max(0, 100 - 100000 - 0);冷启动估会是 2,锚定后归 0
    assert u.used == u.system_prompt + u.tools  # 回退估算,不报负


def test_free_clamps_at_zero_when_over_window():
    u = estimate_context_usage(
        window=1000,
        autocompact_threshold=500,
        system_text="x" * 200_000,  # 远超 window
        tools=[],
        mcp_tools=[],
        history=[],
    )
    assert u.used > 1000
    assert u.free == 0  # max(0, window-used)


# ---- format_context_report ----


def _usage(**over: object) -> ContextUsage:
    base: dict[str, object] = dict(
        window=100_000,
        autocompact_threshold=67_000,
        system_prompt=2_000,
        tools=5_000,
        tool_count=8,
        mcp_tools=1_500,
        mcp_tool_count=3,
        messages=55_000,
        last_measured=56_000,
    )
    base.update(over)
    return ContextUsage(**base)  # type: ignore[arg-type]


def test_report_has_core_parts():
    r = format_context_report(_usage(), "deepseek-v4-pro")
    # 头部:模型 + 已用 + 窗口 + 占比
    assert "deepseek-v4-pro" in r
    assert "62.0k" in r  # used = 2000+5000+55000 = 62000
    assert "100.0k" in r  # window
    assert "(62.0%)" in r
    # 进度条 + 阈值标记
    assert "[" in r and "]" in r
    assert "█" in r and "░" in r
    assert "▲ autocompact" in r
    # 分类目
    assert "System prompt" in r
    assert "Tools (8)" in r
    assert "└ MCP (3)" in r  # MCP 子项(tool_count>0 时显示)
    assert "Messages" in r
    assert "Used" in r and "Free" in r
    # 阈值 + 实测脚注
    assert "67.0k" in r  # autocompact_threshold
    assert "上次 API 实测 input=56.0k" in r


def test_report_hides_mcp_row_when_zero():
    r = format_context_report(_usage(mcp_tool_count=0, mcp_tools=0), "m")
    assert "└ MCP" not in r


def test_report_no_measurement_note_when_none():
    r = format_context_report(_usage(last_measured=None), "m")
    assert "尚无 API 实测" in r
    assert "上次 API 实测" not in r


def test_report_marker_inside_filled_when_over_threshold():
    # used > threshold:进度条已越过阈值标记位置
    r = format_context_report(_usage(messages=90_000), "m")  # used=97000 > 67000
    assert "(97.0%)" in r
    assert "▲ autocompact" in r  # 标记仍在(落在 filled 区内)
