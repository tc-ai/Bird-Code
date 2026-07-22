"""Best-effort token cost estimation. 价目表手工维护（USD/百万 token），未知模型返回 0。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdcode.agent.provider import TokenUsage

# (input_per_1M, output_per_1M)，best-effort，手动维护
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


def _lookup_price(model: str) -> tuple[float, float] | None:
    m = model.lower()
    for prefix, price in _PRICE_TABLE.items():
        if m.startswith(prefix):
            return price
    return None


def _provider_family(model: str) -> str:
    """按模型名前缀判 provider 族,决定缓存 token 的计费语义与折扣。

    - anthropic(claude*):input_tokens 已归一为全量(含缓存);fresh = 全量 − read − create,
      read 0.1×、create 1.25×(均对 input 价)。
    - openai(gpt*):prompt_tokens 含 cached_tokens,需先扣除;read 0.5×、无 creation 概念。
    - other(deepseek 等):缓存计费规则不一且未核实,走 legacy(忽略缓存),不臆测以免引入新偏差。
    """
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt"):
        return "openai"
    return "other"


def estimate_cost(model: str, usage: TokenUsage) -> float:
    price = _lookup_price(model)
    if price is None:
        return 0.0
    in_per_1m, out_per_1m = price
    family = _provider_family(model)
    if family == "openai":
        # prompt_tokens 含 cached_tokens:扣除按新鲜价计,缓存命中部分按 0.5×(避免多计)。
        fresh_input = max(0, usage.input_tokens - usage.cache_read_tokens)
        read_mult = 0.5
        create_mult = 0.0
    elif family == "anthropic":
        # input_tokens 已归一为全量(含缓存,见 anthropic_provider message_stop);fresh = 全量 −
        # read − create,读 0.1×、写 1.25×。不扣则缓存部分被按新鲜价重复计费(多计)。
        fresh_input = max(
            0, usage.input_tokens - usage.cache_read_tokens - usage.cache_creation_tokens
        )
        read_mult = 0.1
        create_mult = 1.25
    else:
        # deepseek 等:规则未核实,保留旧逻辑(忽略缓存),已知对走 OpenAI 兼容 API 的会有残留多计。
        return (
            usage.input_tokens / 1_000_000 * in_per_1m
            + usage.output_tokens / 1_000_000 * out_per_1m
        )
    return (
        fresh_input / 1_000_000 * in_per_1m
        + usage.cache_read_tokens / 1_000_000 * in_per_1m * read_mult
        + usage.cache_creation_tokens / 1_000_000 * in_per_1m * create_mult
        + usage.output_tokens / 1_000_000 * out_per_1m
    )
