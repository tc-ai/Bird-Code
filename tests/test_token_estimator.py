from birdcode.agent.context import TokenEstimator
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, Turn


def _turn_user(text: str) -> Turn:
    return Turn(messages=[Message(role="user", content=[TextBlock(text=text)])])


def test_cold_start_char_estimate_no_anchor():
    """无锚点:全量按 char 估(ascii/3 近似,这里只校验单调性与量级)。"""
    est = TokenEstimator()
    short = est.estimate(
        history=[], current=[Message(role="user", content=[TextBlock(text="hello")])], last_in=None
    )
    long = est.estimate(
        history=[],
        current=[Message(role="user", content=[TextBlock(text="hello " * 1000)])],
        last_in=None,
    )
    assert short > 0
    assert long > short * 100  # 长文本明显更大


def test_anchored_mode_uses_last_in_plus_delta():
    """有锚点:estimate = last_in + delta(新增消息的 char 估)。"""
    est = TokenEstimator()
    # last_in=100000;新增一条短 user 消息
    val = est.estimate(
        history=[],
        current=[Message(role="user", content=[TextBlock(text="a" * 40)])],
        last_in=100_000,
    )
    # ascii 40 chars / 4 = 10 token 增量
    assert val == 100_000 + 10


def test_cjk_counts_heavier_than_ascii():
    """CJK 1.5 chars/token 比 ascii 4 chars/token 重,同样字符数 CJK 估更高。"""
    est = TokenEstimator()
    ascii_est = est.char_to_token("a" * 60)  # 60/4=15
    cjk_est = est.char_to_token("中" * 60)  # 60/1.5=40
    assert cjk_est > ascii_est
    assert ascii_est == 15
    assert cjk_est == 40


def test_mixed_cjk_ascii():
    est = TokenEstimator()
    # 12 ascii + 6 cjk = 12/4 + 6/1.5 = 3 + 4 = 7
    assert est.char_to_token("hello world!" + "六七八九十一") == 7


def test_anchored_ignores_history_content_when_last_in_present():
    """锚点模式:last_in 已含历史;estimate 只加 delta(当前消息),不重复算 history。"""
    est = TokenEstimator()
    big_history = [_turn_user("x" * 10000)]
    val = est.estimate(
        history=big_history,
        current=[Message(role="user", content=[TextBlock(text="hi")])],
        last_in=50_000,
    )
    # 只加 "hi" 的增量(2 ascii / 4 = 0.5 → 向上取整或 floor,实现定)
    assert val >= 50_000 and val <= 50_001
