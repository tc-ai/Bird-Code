import json

import pytest

from birdcode.agent.anthropic_provider import AnthropicProvider
from birdcode.agent.provider import Done, ThinkingDelta, ToolCallStart
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig, ProviderProfile, ThinkingConfig
from birdcode.conversation import Message


@pytest.fixture(autouse=True)
def _stub_reminder(monkeypatch):
    """stream 测试默认打桩 build_system_reminder,避免真实 git 子进程。"""
    from birdcode.agent import base_llm

    async def _default(**kwargs) -> str:
        return "<system-reminder>stub</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _default)


def _app(budget=None):
    prof = ProviderProfile(
        name="c",
        protocol="anthropic",
        model="claude-sonnet-4-5",
        base_url="https://api.anthropic.com",
        api_key="k",
        thinking=ThinkingConfig(budget_tokens=budget) if budget else None,
    )
    return AppConfig(providers={"c": prof}, system_prompt="SYS"), prof


def _ev(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


async def _stream(seq):
    for x in seq:
        yield x


class FakeMessages:
    def __init__(self, events):
        self.events = events
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _stream(self.events)


class FakeAnthropic:
    def __init__(self, events):
        self.messages = FakeMessages(events)


@pytest.mark.asyncio
async def test_translates_thinking_text_usage():
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=10))),
        _ev(type="content_block_delta", delta=_ev(type="thinking_delta", thinking="hm ")),
        _ev(type="content_block_delta", delta=_ev(type="text_delta", text="hi")),
        _ev(type="message_delta", usage=_ev(output_tokens=5)),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    types = [e.type for e in out]
    assert "thinking" in types and "text" in types
    assert isinstance(out[-1], Done)
    assert out[-1].usage is not None
    assert out[-1].usage.input_tokens == 10 and out[-1].usage.output_tokens == 5
    assert out[-1].usage.cost_usd > 0  # claude-sonnet-4-5 已接入 estimate_cost


@pytest.mark.asyncio
async def test_system_param_and_thinking_kwarg_when_configured():
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app(budget=2048)
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    kw = client.messages.last_kwargs
    assert kw["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert kw["stream"] is True
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2048}


@pytest.mark.asyncio
async def test_no_thinking_kwarg_when_disabled():
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert "thinking" not in client.messages.last_kwargs


@pytest.mark.asyncio
async def test_translates_tool_use_and_signature_and_stop_reason():
    """content_block_start(tool_use) + input_json_delta + signature_delta +
    message_delta(stop_reason=tool_use) → ToolCallStart(完整 JSON) + Done(tool_use)。"""
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="content_block_delta", delta=_ev(type="signature_delta", signature="SIG")),
        _ev(
            type="content_block_start",
            index=1,
            content_block=_ev(type="tool_use", id="tu_1", name="echo", input={}),
        ),
        _ev(
            type="content_block_delta",
            index=1,
            delta=_ev(type="input_json_delta", partial_json='{"text": "hello'),
        ),
        _ev(
            type="content_block_delta",
            index=1,
            delta=_ev(type="input_json_delta", partial_json='"}'),
        ),
        _ev(type="content_block_stop", index=1),
        _ev(type="message_delta", delta=_ev(stop_reason="tool_use"), usage=_ev(output_tokens=2)),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    user_msg = [Message(role="user", content=[TextBlock(text="x")])]
    out = [e async for e in p.stream(user_msg, history=[])]

    sig_ev = [e for e in out if isinstance(e, ThinkingDelta) and e.signature]
    assert sig_ev and sig_ev[-1].signature == "SIG"

    tc = next(e for e in out if isinstance(e, ToolCallStart))
    assert tc.id == "tu_1" and tc.name == "echo"
    assert json.loads(tc.args) == {"text": "hello"}

    done = out[-1]
    assert isinstance(done, Done) and done.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_convert_emits_thinking_block_with_signature_in_payload():
    """_convert 输出含 thinking(signature)且在 tool_use 之前。"""
    from birdcode.blocks import ThinkingBlock, ToolUseBlock

    app, prof = _app(budget=2048)
    p = AnthropicProvider(prof, app, client=FakeAnthropic([]))
    msgs = [
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="hm", signature="SIG"),
                ToolUseBlock(id="t1", name="echo", input={"text": "a"}),
            ],
        )
    ]
    payload = p._convert(msgs)
    blocks = payload[0]["content"]
    types = [b["type"] for b in blocks]
    assert types.index("thinking") < types.index("tool_use")
    assert blocks[0]["signature"] == "SIG"
    assert blocks[1]["input"] == {"text": "a"}


# ---- stage2: registry → tools= 参数 ----


@pytest.mark.asyncio
async def test_registry_emits_tools_and_tool_choice_when_given():
    """provider 持有 registry 时,_open_stream 的 kwargs 含 tools + tool_choice=auto。"""
    from birdcode.tools.read_tool import ReadTool
    from birdcode.tools.registry import ToolRegistry

    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    reg = ToolRegistry()
    reg.register(ReadTool())
    p = AnthropicProvider(prof, app, client=client, registry=reg)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    kw = client.messages.last_kwargs
    assert "tools" in kw
    assert kw["tool_choice"] == {"type": "auto"}
    # tools 含 read_file 的 schema
    names = [t["name"] for t in kw["tools"]]
    assert "read_file" in names
    assert kw["tools"][0]["input_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_no_tools_kwarg_when_registry_absent():
    """registry=None(默认)时,_open_stream 不加 tools / tool_choice。"""
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    kw = client.messages.last_kwargs
    assert "tools" not in kw
    assert "tool_choice" not in kw


@pytest.mark.asyncio
async def test_usage_parses_cache_read_and_creation_tokens():
    """message_start 携带 cache_read/creation_input_tokens → usage 正确解析。"""
    events = [
        _ev(
            type="message_start",
            message=_ev(
                usage=_ev(
                    input_tokens=10,
                    cache_read_input_tokens=4096,
                    cache_creation_input_tokens=128,
                )
            ),
        ),
        _ev(type="message_delta", usage=_ev(output_tokens=5)),
        _ev(type="message_stop"),
    ]
    app, prof = _app()
    client = FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=client)
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    u = out[-1].usage
    assert u.cache_read_tokens == 4096
    assert u.cache_creation_tokens == 128
