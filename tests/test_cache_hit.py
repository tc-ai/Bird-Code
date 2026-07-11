"""缓存策略自动化验证(不打真实 API):

1. system block 两次调用字节一致 → 满足 Anthropic 前缀缓存前提。
2. 响应携带 cache_read_input_tokens 时 usage.cache_read_tokens 精确解析。
真实命中率定性评估见 scripts/cache_eval.py(多轮/长对话场景也延后到那里)。
"""

from types import SimpleNamespace

import pytest

from birdcode.agent.anthropic_provider import AnthropicProvider
from birdcode.agent.provider import Done
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message


@pytest.fixture(autouse=True)
def _stub_reminder(monkeypatch):
    """stream 测试默认打桩 build_system_reminder,避免真实 git 子进程。"""
    from birdcode.agent import base_llm

    async def _default(**kwargs) -> str:
        return "<system-reminder>stub</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _default)


def _ev(**kw):
    return SimpleNamespace(**kw)


async def _stream(seq):
    for x in seq:
        yield x


class _FakeMessages:
    def __init__(self, events):
        self.events = events
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _stream(self.events)


class _FakeAnthropic:
    def __init__(self, events):
        self.messages = _FakeMessages(events)


def _app_prof():
    prof = ProviderProfile(
        name="c",
        protocol="anthropic",
        model="claude-sonnet-4-5",
        base_url="https://api.anthropic.com",
        api_key="k",
    )
    return AppConfig(providers={"c": prof}), prof  # system_prompt 默认空 → 走模块拼装


@pytest.mark.asyncio
async def test_system_block_identical_across_calls():
    """两次 stream(不同 user 文本),system block 字节一致 → 可缓存前提成立;
    且 block 结构正确(带 cache_control、文本非空)。"""
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app_prof()
    c1 = _FakeAnthropic(events)
    p1 = AnthropicProvider(prof, app, client=c1)
    _ = [e async for e in p1.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    c2 = _FakeAnthropic(events)
    p2 = AnthropicProvider(prof, app, client=c2)
    _ = [e async for e in p2.stream([Message(role="user", content=[TextBlock("y")])], history=[])]
    sys1 = c1.messages.last_kwargs["system"]
    sys2 = c2.messages.last_kwargs["system"]
    assert sys1 == sys2  # 跨调用字节一致
    # 结构守卫:带 cache_control、文本非空(防止退化成空前缀也通过一致性检查)
    assert sys1[0]["cache_control"] == {"type": "ephemeral"}
    assert len(sys1[0]["text"]) > 100


@pytest.mark.asyncio
async def test_cache_read_input_tokens_parsed_into_usage():
    """响应携带 cache_read_input_tokens=4096 → usage.cache_read_tokens 精确等于 4096。"""
    events = [
        _ev(
            type="message_start",
            message=_ev(usage=_ev(input_tokens=10, cache_read_input_tokens=4096)),
        ),
        _ev(type="message_stop"),
    ]
    app, prof = _app_prof()
    p = AnthropicProvider(prof, app, client=_FakeAnthropic(events))
    out = [
        e async for e in p.stream([Message(role="user", content=[TextBlock("again")])], history=[])
    ]
    done = next(e for e in out if isinstance(e, Done))
    assert done.usage.cache_read_tokens == 4096
    assert done.usage.cache_creation_tokens == 0  # 未提供 → 默认 0
