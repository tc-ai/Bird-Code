"""直接测试 pricing 叶子模块的 estimate_cost。"""

from pytest import approx

from birdcode.agent.pricing import estimate_cost
from birdcode.agent.provider import TokenUsage


def test_estimate_cost_known_model():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # claude-sonnet-4-5: 3.0 in + 15.0 out per 1M
    assert estimate_cost("claude-sonnet-4-5", u) == approx(18.0)
    # deepseek-reasoner: 0.55 in + 2.19 out per 1M
    assert estimate_cost("deepseek-reasoner", u) == approx(2.74)


def test_estimate_cost_unknown_model_returns_zero():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("some-unknown-model", u) == 0.0


def test_estimate_cost_zero_tokens():
    u = TokenUsage(input_tokens=0, output_tokens=0)
    assert estimate_cost("claude-sonnet-4-5", u) == 0.0


def test_estimate_cost_case_insensitive_prefix():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert estimate_cost("CLAUDE-SONNET-4-5-20250929", u) == approx(3.0)


# --- 缓存 token 计费(Anthropic 少计 / OpenAI 多计 的修复)---
# Anthropic:input_tokens 已排除缓存;cache_read 0.1×、cache_creation 1.25×(均对 input 价)。
def test_estimate_cost_anthropic_cache_read_priced():
    # 纯缓存读轮(无新鲜输入):旧实现返回 0(少计)。修复后 1M × 0.1× × 3.0 = 0.3。
    u = TokenUsage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    assert estimate_cost("claude-sonnet-4-5", u) == approx(0.3)


def test_estimate_cost_anthropic_cache_creation_priced():
    # 冷启动写缓存:1M × 1.25× × 3.0 = 3.75(偏差最大的那块)。
    u = TokenUsage(input_tokens=0, output_tokens=0, cache_creation_tokens=1_000_000)
    assert estimate_cost("claude-sonnet-4-5", u) == approx(3.75)


def test_estimate_cost_anthropic_mixed_round():
    # fresh 1M@3.0 + read 1M@0.3 + create 1M@3.75 + out 1M@15.0 = 22.05
    u = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    assert estimate_cost("claude-sonnet-4-5", u) == approx(22.05)


# OpenAI:prompt_tokens 含 cached_tokens → 先扣除按新鲜计费,缓存部分按 0.5×。
def test_estimate_cost_openai_subtracts_and_discounts_cache():
    # gpt-4o input 2.5/1M。fresh=200k@2.5 + cached=800k@0.5×2.5=1.25 → 0.5 + 1.0 = 1.5。
    # 旧实现把 1M 全按 2.5 计 = 2.5(多计)。修复后 1.5。
    u = TokenUsage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=800_000)
    assert estimate_cost("gpt-4o", u) == approx(1.5)


def test_estimate_cost_no_cache_tokens_unchanged_for_both_families():
    """cache=0 时两 family 与旧实现逐位一致(向后兼容回归)。"""
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("claude-sonnet-4-5", u) == approx(18.0)
    assert estimate_cost("gpt-4o", u) == approx(12.5)  # 2.5 + 10.0
