# tests/test_status_bar.py
from birdcode.agent.provider import TokenUsage
from birdcode.ui.widgets.status_bar import _fmt_tokens, format_status


def test_status_basic_unicode():
    s = format_status(
        model="mock",
        mode="suggest",
        cwd="~/pkg",
        usage=TokenUsage(input_tokens=3, output_tokens=5, cost_usd=0.0002),
        queue_size=0,
        busy=False,
        unicode_supported=True,
    )
    assert "birdcode" in s and "mock" in s and "~/pkg" in s
    assert "↑3" in s and "↓5" in s and "$0.0002" in s
    assert "·" in s and "⌛" not in s and "working" not in s


def test_status_ascii_and_queue():
    s = format_status(
        model="mock",
        mode="suggest",
        cwd="p",
        usage=None,
        queue_size=2,
        busy=True,
        unicode_supported=False,
    )
    assert "|" in s and "#" in s and "working" in s
    assert "tok" not in s  # 无 usage 不显示


def test_status_shows_cache_segment():
    s = format_status(
        model="m",
        mode="suggest",
        cwd="p",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=5,
            cache_read_tokens=8500,
            cost_usd=0.01,
        ),
        queue_size=0,
        busy=False,
        unicode_supported=True,
    )
    assert "↑100" in s
    assert "cache↑8.5k" in s  # k 缩写 + 输入侧箭头
    assert "$0.0100" in s


def test_status_ascii_arrows_when_unicode_unsupported():
    s = format_status(
        model="m",
        mode="suggest",
        cwd="p",
        usage=TokenUsage(input_tokens=3, output_tokens=5, cost_usd=0.0002),
        queue_size=0,
        busy=False,
        unicode_supported=False,
    )
    assert "^3" in s and "v5" in s  # ASCII 回退


def test_fmt_tokens_boundaries():
    assert _fmt_tokens(0) == "0"
    assert _fmt_tokens(999) == "999"
    assert _fmt_tokens(1000) == "1.0k"
    assert _fmt_tokens(1500) == "1.5k"
    assert _fmt_tokens(8500) == "8.5k"
    assert _fmt_tokens(999_999) == "1.0M"  # 跨数量级取整不退化成 1000.0k
    assert _fmt_tokens(1_000_000) == "1.0M"
    assert _fmt_tokens(1_234_567) == "1.2M"


def test_status_creation_labeled_and_combined():
    # 仅 creation(read=0):必须带 new 标签,与 input ↑ 区分
    s_only_create = format_status(
        model="m", mode="suggest", cwd="p",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=5,
            cache_creation_tokens=120,
            cost_usd=0.01,
        ),
        queue_size=0, busy=False, unicode_supported=True,
    )
    assert "new↑120" in s_only_create

    # read + creation 同时存在
    s_both = format_status(
        model="m", mode="suggest", cwd="p",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=5,
            cache_read_tokens=8500,
            cache_creation_tokens=120,
            cost_usd=0.01,
        ),
        queue_size=0, busy=False, unicode_supported=True,
    )
    assert "cache↑8.5k" in s_both
    assert "new↑120" in s_both
