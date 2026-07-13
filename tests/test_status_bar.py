# tests/test_status_bar.py
from birdcode.ui.widgets.status_bar import _fmt_tokens, format_status


def test_status_mode_default_green():
    s = format_status(mode="default", unicode_supported=True)
    assert "[#9ece6a]default[/]" in s  # 默认模式绿色(hex truecolor)
    assert "(shift+tab cycle)" in s  # 模式旁的切档提示
    assert "\\+Enter 换行" in s  # 快捷键提示
    assert "回车发送" not in s and "/help 查看" not in s  # 旧提示已移除
    assert "Shift+C" not in s  # Shift+C 复制提示已去掉
    assert "·" in s  # unicode 分隔


def test_status_mode_plan_green():
    s = format_status(mode="plan", unicode_supported=True)
    assert "[#9ece6a]plan[/]" in s  # 最安全→绿


def test_status_mode_accept_edits_orange():
    s = format_status(mode="accept-edits", unicode_supported=True)
    assert "[#ff9e64]accept edits[/]" in s  # 醒目→橙(hex,非命名色)


def test_status_mode_bypass_red():
    s = format_status(mode="bypass", unicode_supported=True)
    assert "[#f7768e]bypass[/]" in s  # 最危险→红


def test_status_ascii_separator_and_hints():
    s = format_status(mode="default", unicode_supported=False)
    assert "|" in s  # ASCII 分隔
    assert "·" not in s
    assert "\\+Enter" in s  # 提示
    assert "Shift+C" not in s  # 已去掉


def test_status_unknown_mode_falls_back_green():
    s = format_status(mode="weird", unicode_supported=True)
    assert "[#9ece6a]weird[/]" in s  # 未知 mode 退化原值 + 绿


def test_status_no_usage_model_cwd():
    # 用量/模型/路径已移到顶端 banner;状态行不应再出现这些。
    s = format_status(mode="default", unicode_supported=True)
    assert "mock" not in s
    assert "$" not in s
    assert "↑" not in s and "↓" not in s


def test_status_shows_queue_count_when_queued():
    # type-ahead:忙时输入的消息排队 → 状态行显示 ⌛N(仅 queue_size>0 时)。
    s = format_status(mode="default", unicode_supported=True, queue_size=2)
    assert "⌛2" in s


def test_status_no_queue_indicator_when_zero():
    s = format_status(mode="default", unicode_supported=True, queue_size=0)
    assert "⌛" not in s


def test_status_queue_ascii_fallback():
    s = format_status(mode="default", unicode_supported=False, queue_size=1)
    assert "#1" in s


def test_fmt_tokens_boundaries():
    assert _fmt_tokens(0) == "0"
    assert _fmt_tokens(999) == "999"
    assert _fmt_tokens(1000) == "1.0k"
    assert _fmt_tokens(1500) == "1.5k"
    assert _fmt_tokens(8500) == "8.5k"
    assert _fmt_tokens(999_999) == "1.0M"  # 跨数量级取整不退化成 1000.0k
    assert _fmt_tokens(1_000_000) == "1.0M"
    assert _fmt_tokens(1_234_567) == "1.2M"
