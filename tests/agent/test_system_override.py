# tests/agent/test_system_override.py
from birdcode.agent.base_llm import _BaseLLMProvider
from birdcode.config.schema import AppConfig, ProviderProfile


def _app(project_instructions="PROJECT-RULES", system_prompt="") -> AppConfig:
    return AppConfig(
        providers={
            "p": ProviderProfile(
                name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
            )
        },
        default="p",
        system_prompt=system_prompt,
        project_instructions=project_instructions,
    )


def test_no_override_keeps_project_instructions():
    p = _BaseLLMProvider(
        ProviderProfile(
            name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
        ),
        _app(),
    )
    text = p._system_text()
    assert "PROJECT-RULES" in text


def test_override_prepends_persona_and_keeps_project():
    p = _BaseLLMProvider(
        ProviderProfile(
            name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
        ),
        _app(),
        system_override="PERSONA",
    )
    text = p._system_text()
    assert "PERSONA" in text
    assert "PROJECT-RULES" in text  # 未被逃生口旁路
    assert text.index("PERSONA") < text.index("PROJECT-RULES")


def test_profile_property_exposed():
    prof = ProviderProfile(
        name="p", protocol="anthropic", model="m", base_url="http://x", api_key="k"
    )
    p = _BaseLLMProvider(prof, _app())
    assert p.profile is prof
