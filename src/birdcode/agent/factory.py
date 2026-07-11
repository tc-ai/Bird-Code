"""provider 工厂。

best-effort cost 估算已抽离到 birdcode.agent.pricing；本模块仅 re-export 保留兼容。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from birdcode.agent.anthropic_provider import AnthropicProvider
from birdcode.agent.openai_provider import OpenAIProvider
from birdcode.agent.pricing import estimate_cost  # noqa: F401  — re-exported for compatibility
from birdcode.agent.provider import StreamingProvider

if TYPE_CHECKING:
    from birdcode.config.schema import AppConfig, ProviderProfile
    from birdcode.tools.registry import ToolRegistry


def build_provider(
    profile: ProviderProfile,
    app: AppConfig,
    *,
    registry: ToolRegistry | None = None,
    mcp_instructions: Callable[[], dict[str, str]] | None = None,
    system_override: str | None = None,
) -> StreamingProvider:
    """构造真实 provider。registry 非空时透传——provider 据此生成 tools= 参数。

    mcp_instructions 非空时透传——provider 据此把 server instructions 拼进稳定 system block。
    system_override 非空时子 agent persona 前置到 system(见 _BaseLLMProvider._system_text)。
    MockProvider 不经此构造(它在 cli/app.py 直接建,不持有 registry)。
    """
    if profile.protocol == "anthropic":
        return AnthropicProvider(
            profile, app, registry=registry,
            mcp_instructions=mcp_instructions, system_override=system_override,
        )
    if profile.protocol == "openai":
        return OpenAIProvider(
            profile, app, registry=registry,
            mcp_instructions=mcp_instructions, system_override=system_override,
        )
    raise ValueError(f"未知 protocol: {profile.protocol!r}")
