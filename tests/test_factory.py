import pytest

from birdcode.agent.anthropic_provider import AnthropicProvider
from birdcode.agent.factory import build_provider, estimate_cost
from birdcode.agent.openai_provider import OpenAIProvider
from birdcode.agent.provider import TokenUsage
from birdcode.config.schema import AppConfig, ProviderProfile


def _cfg(prof: ProviderProfile) -> AppConfig:
    return AppConfig(providers={prof.name: prof}, default=prof.name)


def test_factory_anthropic():
    prof = ProviderProfile(
        name="c", protocol="anthropic", model="claude-sonnet-4-5", base_url="u", api_key="k"
    )
    assert isinstance(build_provider(prof, _cfg(prof)), AnthropicProvider)


def test_factory_openai():
    prof = ProviderProfile(
        name="d", protocol="openai", model="deepseek-reasoner", base_url="u", api_key="k"
    )
    assert isinstance(build_provider(prof, _cfg(prof)), OpenAIProvider)


def test_factory_unknown_protocol_raises():
    prof = ProviderProfile(name="x", protocol="openai", model="m", base_url="u", api_key="k")
    prof.protocol = "gemini"  # 绕过 schema 走 factory 分支
    with pytest.raises(ValueError, match="protocol"):
        build_provider(prof, _cfg(prof))


def test_estimate_cost_known_and_unknown():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("claude-sonnet-4-5", u) > 0
    assert estimate_cost("deepseek-reasoner", u) > 0
    assert estimate_cost("some-unknown-model", u) == 0.0


# ---- stage2: build_provider 透传 registry ----


def test_factory_threads_registry_into_anthropic():
    from birdcode.tools.registry import ToolRegistry

    prof = ProviderProfile(
        name="c", protocol="anthropic", model="claude-sonnet-4-5", base_url="u", api_key="k"
    )
    cfg = AppConfig(providers={prof.name: prof}, default=prof.name)
    reg = ToolRegistry()
    p = build_provider(prof, cfg, registry=reg)
    assert isinstance(p, AnthropicProvider)
    assert p._registry is reg  # noqa: SLF001


def test_factory_threads_registry_into_openai():
    from birdcode.tools.registry import ToolRegistry

    prof = ProviderProfile(
        name="d", protocol="openai", model="deepseek-reasoner", base_url="u", api_key="k"
    )
    cfg = AppConfig(providers={prof.name: prof}, default=prof.name)
    reg = ToolRegistry()
    p = build_provider(prof, cfg, registry=reg)
    assert isinstance(p, OpenAIProvider)
    assert p._registry is reg  # noqa: SLF001


def test_factory_registry_defaults_none():
    prof = ProviderProfile(
        name="c", protocol="anthropic", model="claude-sonnet-4-5", base_url="u", api_key="k"
    )
    cfg = AppConfig(providers={prof.name: prof}, default=prof.name)
    p = build_provider(prof, cfg)
    assert p._registry is None  # noqa: SLF001
