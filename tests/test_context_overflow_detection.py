# tests/test_context_overflow_detection.py
"""T6: base_llm 拦上下文超限(413 / 400 + 'prompt is too long')→ raise ContextOverflow。

Anthropic 历史用 400 BadRequestError 报「prompt is too long」,只认 413 会漏检;
故双兜底:status==413 OR (status==400 AND 关键词命中)。
"""

import asyncio

import pytest

from birdcode.agent.base_llm import _BaseLLMProvider
from birdcode.agent.context import ContextOverflow
from birdcode.config.schema import AppConfig, ProviderProfile


class _FakeStatus(Exception):
    """模拟 SDK status error:status_code + message。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_detects_413():
    assert _BaseLLMProvider._is_context_overflow(_FakeStatus(413, "Request Entity Too Large"))


def test_detects_400_prompt_too_long_case_insensitive():
    assert _BaseLLMProvider._is_context_overflow(
        _FakeStatus(400, "prompt is too long: 250000 > 200000")
    )
    assert _BaseLLMProvider._is_context_overflow(
        _FakeStatus(400, "Requested 210000. Maximum Context Length is 200000")
    )


def test_does_not_detect_unrelated_400():
    assert not _BaseLLMProvider._is_context_overflow(
        _FakeStatus(400, "invalid_request_error: bad tool_use id")
    )


def test_does_not_detect_500():
    assert not _BaseLLMProvider._is_context_overflow(_FakeStatus(500, "internal error"))


def test_stream_raises_context_overflow_on_413():
    """stream 遇 413 → raise ContextOverflow(而非 yield Error)。"""

    class _P(_BaseLLMProvider):
        def _convert(self, messages):  # noqa: ANN001, ANN202 — 测试桩
            return []

        async def _open_stream(self, payload, **kwargs):  # noqa: ANN001, ANN202
            raise _FakeStatus(413, "too large")

        def _translate(self, raw):  # noqa: ANN001, ANN202
            return iter(())

    profile = ProviderProfile(protocol="anthropic", model="m", base_url="u", api_key="k")
    app = AppConfig.model_validate(
        {
            "providers": {
                "p": {"protocol": "anthropic", "model": "m", "base_url": "u", "api_key": "k"}
            },
            "default": "p",
        }
    )
    p = _P(profile, app)

    async def go() -> None:
        async for _ev in p.stream([], history=[]):
            pass

    with pytest.raises(ContextOverflow):
        asyncio.run(go())
