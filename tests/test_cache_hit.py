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
    def __init__(self, events, *, nonstream_msg=None):
        self.events = events
        self.last_kwargs = None
        self._nonstream_msg = nonstream_msg

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        # 非流式(summarize_with_prefix)返回 message 对象;流式返回 async gen。
        if not kwargs.get("stream", True) and self._nonstream_msg is not None:
            return self._nonstream_msg
        return _stream(self.events)


class _FakeAnthropic:
    def __init__(self, events, *, nonstream_msg=None):
        self.messages = _FakeMessages(events, nonstream_msg=nonstream_msg)


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


@pytest.mark.asyncio
async def test_messages_cache_breakpoint_at_prior_end():
    """history 非空:_open_stream 在 prior 末条(msgs[prior_len-1])最后 block 加 cache_control;
    current 部分不加 → 稳定 prior 每轮命中、current/reminder 为增量。"""
    from birdcode.conversation import Turn

    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app_prof()
    c = _FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=c)
    history = [Turn(messages=[
        Message(role="user", content=[TextBlock(text="past question")]),
        Message(role="assistant", content=[TextBlock(text="past answer")]),
    ])]
    _ = [
        e async for e in p.stream(
            [Message(role="user", content=[TextBlock(text="now")])], history=history
        )
    ]
    msgs = c.messages.last_kwargs["messages"]
    assert len(msgs) == 3  # prior(user, assistant) + current(user)
    # prior 末条(msgs[1] assistant)最后 block 有断点
    assert msgs[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    # current(msgs[2] user)无断点
    assert "cache_control" not in msgs[2]["content"][-1]


@pytest.mark.asyncio
async def test_no_messages_breakpoint_when_no_history():
    """history=[] → prior_len=0 → messages 无 cache 断点(只 system/tools 有)。"""
    events = [
        _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
        _ev(type="message_stop"),
    ]
    app, prof = _app_prof()
    c = _FakeAnthropic(events)
    p = AnthropicProvider(prof, app, client=c)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    for m in c.messages.last_kwargs["messages"]:
        for b in m["content"]:
            assert "cache_control" not in b


@pytest.mark.asyncio
async def test_summarize_with_prefix_reuses_main_system_tools_none_and_breakpoint():
    """summarize_with_prefix:复用主 system(非摘要模板)+tools(cached)+tool_choice=none;
    prior 末 block 有 cache 断点;末条 user == instruction(落断点之后)。"""
    class _Reg:
        def to_anthropic_tools(self, *, cached=False):  # noqa: ANN001, ANN202
            return [{"type": "custom", "name": "f", "input_schema": {},
                     "cache_control": {"type": "ephemeral"}}]

    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="<summary>S</summary>")])
    app, prof = _app_prof()
    c = _FakeAnthropic([], nonstream_msg=msg)
    p = AnthropicProvider(prof, app, client=c, registry=_Reg())
    prefix = [
        Message(role="user", content=[TextBlock(text="q1")]),
        Message(role="assistant", content=[TextBlock(text="a1")]),
    ]
    out = await p.summarize_with_prefix(prefix=prefix, instruction="SUMM-INSTR", max_tokens=1000)
    kw = c.messages.last_kwargs
    assert "SUMM-INSTR" not in kw["system"][0]["text"]  # 复用主 system(非摘要模板)
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["tool_choice"] == {"type": "none"}
    assert kw["tools"][-1].get("cache_control") == {"type": "ephemeral"}
    msgs = kw["messages"]
    assert msgs[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}  # prior 末断点
    assert "cache_control" not in msgs[2]["content"][-1]  # instruction 无断点
    assert msgs[2]["content"][-1]["text"] == "SUMM-INSTR"
    assert out == "<summary>S</summary>"


@pytest.mark.asyncio
async def test_summarize_with_prefix_mirrors_thinking_and_bumps_max_tokens():
    """profile 带 thinking → 镜像 thinking(enabled)+ max_tokens > budget(防 prefix 含
    ThinkingBlock 时 API 400;且留 text 空间防 </summary> 被截断 → 提取失败)。"""
    from birdcode.config.schema import ThinkingConfig

    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="<summary>S</summary>")])
    prof = ProviderProfile(
        name="c", protocol="anthropic", model="claude-sonnet-4-5",
        base_url="u", api_key="k", thinking=ThinkingConfig(budget_tokens=8000),
    )
    app = AppConfig(providers={"c": prof})
    c = _FakeAnthropic([], nonstream_msg=msg)
    p = AnthropicProvider(prof, app, client=c)
    await p.summarize_with_prefix(prefix=[], instruction="I", max_tokens=2000)
    kw = c.messages.last_kwargs
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert kw["max_tokens"] >= 8000 + 4096  # 兜底:留 text 空间
