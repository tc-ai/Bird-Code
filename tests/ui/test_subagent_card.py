# tests/ui/test_subagent_card.py
from birdcode.agents.report import SubagentProgress
from birdcode.ui.widgets.subagent_card import SubagentCard


def test_format_duration():
    assert SubagentCard._format_duration(0) == "0s"
    assert SubagentCard._format_duration(73_000) == "1m 13s"
    assert SubagentCard._format_duration(125_000) == "2m 5s"


def test_format_tokens():
    # < 1000 原样;>= 1000 一位小数 k
    assert SubagentCard._format_tokens(0) == "0"
    assert SubagentCard._format_tokens(999) == "999"
    assert SubagentCard._format_tokens(12300) == "12.3k"


def test_card_renders_progress():
    card = SubagentCard()
    card.update_progress(SubagentProgress(
        agent_id="sub-1", description="探索代码库", elapsed_ms=73000,
        input_tokens=12300, tool_use_count=4, phase="running",
    ))
    markup = card.render() if hasattr(card, "render") else str(card._renderable)
    assert "探索代码库" in markup
    assert "12.3k" in markup  # 格式化 tokens(非原始 12300)
    assert "1m 13s" in markup  # 格式化时长
