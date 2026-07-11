import pytest
from pydantic import ValidationError

from birdcode.config.schema import (
    AppConfig,
    McpHttpServer,
    McpStdioServer,
    ProviderProfile,
    ThinkingConfig,
)


def test_profile_requires_protocol_model_base_url_api_key():
    with pytest.raises(ValidationError):
        ProviderProfile(protocol="openai", model="x")  # 缺 base_url/api_key


def test_thinking_budget_min_1024():
    with pytest.raises(ValidationError):
        ThinkingConfig(budget_tokens=512)
    assert ThinkingConfig(budget_tokens=4096).budget_tokens == 4096


def test_appconfig_defaults():
    cfg = AppConfig(
        providers={
            "p": ProviderProfile(
                protocol="openai",
                model="gpt-4o",
                base_url="https://api.openai.com/v1",
                api_key="sk-",
            )
        }
    )
    assert cfg.default is None
    assert cfg.max_tokens == 8192
    assert cfg.system_prompt == ""  # 默认空:走 system_prompt 包拼装


def test_protocol_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        ProviderProfile(
            protocol="gemini",
            model="x",
            base_url="u",
            api_key="k",
        )


def test_mcp_servers_default_empty():
    cfg = AppConfig(providers={})
    assert cfg.mcp_servers == {}


def test_stdio_server_schema():
    cfg = AppConfig(
        providers={},
        mcp_servers={"fs": McpStdioServer(type="stdio", command="npx", args=["-y", "@mcp/fs"])},
    )
    srv = cfg.mcp_servers["fs"]
    assert isinstance(srv, McpStdioServer)
    assert srv.command == "npx"
    assert srv.env == {}


def test_http_server_schema():
    cfg = AppConfig(
        providers={},
        mcp_servers={
            "grafana": McpHttpServer(
                type="streamable_http",
                url="https://x/mcp",
                headers={"Authorization": "Bearer t"},
            )
        },
    )
    srv = cfg.mcp_servers["grafana"]
    assert isinstance(srv, McpHttpServer)
    assert srv.url == "https://x/mcp"


def test_mcp_server_discriminated_from_raw_dict():
    cfg = AppConfig.model_validate(
        {
            "providers": {},
            "mcp_servers": {
                "fs": {"type": "stdio", "command": "npx", "args": ["-y", "@mcp/fs"]},
                "grafana": {"type": "streamable_http", "url": "https://x/mcp"},
            },
        }
    )
    assert isinstance(cfg.mcp_servers["fs"], McpStdioServer)
    assert isinstance(cfg.mcp_servers["grafana"], McpHttpServer)
