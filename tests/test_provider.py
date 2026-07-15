# tests/test_provider.py
import pytest
from pydantic import TypeAdapter

from birdcode.agent.provider import ProviderEvent


def test_token_usage_defaults():
    from birdcode.agent.provider import TokenUsage

    u = TokenUsage()
    assert u.input_tokens == 0 and u.cost_usd == 0.0


def test_event_discrimination_text():
    from birdcode.agent.provider import ProviderEvent, TextDelta

    ev = TypeAdapter(ProviderEvent).validate_python({"type": "text", "text": "hi"})
    assert isinstance(ev, TextDelta) and ev.text == "hi"


def test_event_discrimination_thinking():
    from birdcode.agent.provider import ProviderEvent, ThinkingDelta

    ev = TypeAdapter(ProviderEvent).validate_python({"type": "thinking", "text": "hm"})
    assert isinstance(ev, ThinkingDelta) and ev.text == "hm"


def test_event_discrimination_tool_result():
    from birdcode.agent.provider import ProviderEvent, ToolResult

    ev = TypeAdapter(ProviderEvent).validate_python(
        {"type": "tool_result", "id": "t1", "ok": False, "summary": "boom"}
    )
    assert isinstance(ev, ToolResult) and ev.ok is False and ev.summary == "boom"


def test_streaming_provider_is_protocol():
    from birdcode.agent.provider import StreamingProvider

    assert hasattr(StreamingProvider, "stream")


@pytest.mark.asyncio
async def test_protocol_can_be_implemented():
    from birdcode.agent.provider import Done, StreamingProvider, TextDelta

    class P:
        async def stream(self, messages, *, history):
            yield TextDelta(text="x")
            yield Done()

        async def complete(self, system, user, *, max_tokens, model=None):
            return "ok"

        async def summarize_with_prefix(self, *, prefix, instruction, max_tokens):
            return "ok"

    assert isinstance(P(), StreamingProvider)


def test_done_has_stop_reason_default_none():
    from birdcode.agent.provider import Done, TokenUsage

    d = Done(usage=TokenUsage())
    assert d.stop_reason is None


def test_done_accepts_tool_use_stop_reason():
    from birdcode.agent.provider import Done

    d = Done(usage=None, stop_reason="tool_use")
    assert d.stop_reason == "tool_use"


def test_thinkingdelta_has_signature_default_empty():
    from birdcode.agent.provider import ThinkingDelta

    t = ThinkingDelta(text="hm")
    assert t.signature == ""
    t2 = ThinkingDelta(text="", signature="sig")
    assert t2.signature == "sig"


# ---- stage2 polish(§7): ToolResult 事件加 truncated ----


def test_tool_result_event_has_truncated_default_false():
    from birdcode.agent.provider import ToolResult

    ev = ToolResult(id="t1", ok=True, summary="ok")
    assert ev.truncated is False


def test_tool_result_event_accepts_truncated_true():
    from birdcode.agent.provider import ToolResult

    ev = ToolResult(id="t1", ok=True, summary="...[已截断]...", truncated=True)
    assert ev.truncated is True


def test_tool_result_event_discrimination_with_truncated():
    ev = TypeAdapter(ProviderEvent).validate_python(
        {"type": "tool_result", "id": "t1", "ok": True, "summary": "s", "truncated": True}
    )
    from birdcode.agent.provider import ToolResult

    assert isinstance(ev, ToolResult) and ev.truncated is True


def test_token_usage_cache_field_defaults_zero():
    from birdcode.agent.provider import TokenUsage

    u = TokenUsage()
    assert u.cache_read_tokens == 0
    assert u.cache_creation_tokens == 0


def test_token_usage_accepts_cache_fields():
    from birdcode.agent.provider import TokenUsage

    u = TokenUsage(input_tokens=10, cache_read_tokens=500, cache_creation_tokens=20)
    assert u.cache_read_tokens == 500 and u.cache_creation_tokens == 20
