import json
from types import SimpleNamespace

import pytest

from birdcode.agent.openai_provider import OpenAIProvider
from birdcode.agent.provider import Done, TextDelta, ThinkingDelta, ToolCallStart
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message


@pytest.fixture(autouse=True)
def _stub_reminder(monkeypatch):
    """stream 测试默认打桩 build_system_reminder,避免真实 git 子进程。"""
    from birdcode.agent import base_llm

    async def _default(**kwargs) -> str:
        return "<system-reminder>stub</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _default)


def _app(effort=None):
    prof = ProviderProfile(
        name="d",
        protocol="openai",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        api_key="k",
        reasoning_effort=effort,
    )
    return AppConfig(providers={"d": prof}, system_prompt="SYS"), prof


def _U(pin, cout):
    from types import SimpleNamespace

    return SimpleNamespace(prompt_tokens=pin, completion_tokens=cout)


def _chunk(content=None, reasoning=None, usage=None):
    from types import SimpleNamespace

    choices = []
    if content is not None or reasoning is not None:
        delta = SimpleNamespace(content=content)
        if reasoning is not None:
            delta.reasoning_content = reasoning
        choices = [SimpleNamespace(delta=delta)]
    return SimpleNamespace(choices=choices, usage=usage)


async def _stream(seq):
    for x in seq:
        yield x


class FakeCompletions:
    def __init__(self, chunks, nonstream_resp=None):
        self.chunks = chunks
        self.last_kwargs = None
        self._nonstream_resp = nonstream_resp

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        # 非流式(summarize_with_prefix, stream=False)返回 message 对象;流式返回 async gen。
        if not kwargs.get("stream", True) and self._nonstream_resp is not None:
            return self._nonstream_resp
        return _stream(self.chunks)


class FakeChat:
    def __init__(self, chunks, nonstream_resp=None):
        self.completions = FakeCompletions(chunks, nonstream_resp)


class FakeOpenAI:
    def __init__(self, chunks, nonstream_resp=None):
        self.chat = FakeChat(chunks, nonstream_resp)


@pytest.mark.asyncio
async def test_reasoning_and_text_and_usage():
    chunks = [_chunk(reasoning="hm "), _chunk(content="hi"), _chunk(usage=_U(7, 3))]
    app, prof = _app()
    client = FakeOpenAI(chunks)
    p = OpenAIProvider(prof, app, client=client)
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    types = [e.type for e in out]
    assert "thinking" in types and "text" in types
    assert isinstance(out[-1], Done)
    assert out[-1].usage.input_tokens == 7 and out[-1].usage.output_tokens == 3
    assert out[-1].usage.cost_usd > 0  # deepseek-reasoner 已接入 estimate_cost


@pytest.mark.asyncio
async def test_text_only_when_no_reasoning_field():
    chunks = [_chunk(content="hi"), _chunk(usage=_U(1, 1))]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert not any(isinstance(e, ThinkingDelta) for e in out)
    assert any(isinstance(e, TextDelta) for e in out)


@pytest.mark.asyncio
async def test_system_message_prepended_and_max_completion_tokens():
    chunks = [_chunk(usage=_U(1, 1))]
    app, prof = _app(effort="medium")
    client = FakeOpenAI(chunks)
    p = OpenAIProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    kw = client.chat.completions.last_kwargs
    msgs = kw["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert kw["stream"] is True
    assert kw["stream_options"] == {"include_usage": True}
    assert kw["max_completion_tokens"] == app.max_tokens
    assert kw["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_reasoning_effort_omitted_when_none():
    chunks = [_chunk(usage=_U(1, 1))]
    app, prof = _app()
    client = FakeOpenAI(chunks)
    p = OpenAIProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert "reasoning_effort" not in client.chat.completions.last_kwargs


def _tc_chunk(index=0, id=None, name=None, arguments=None, finish_reason=None):
    """构造一个带 delta.tool_calls 的 chunk,模拟 OpenAI/DeepSeek 流式工具调用。"""
    function = SimpleNamespace(
        name=name, arguments=arguments
    )  # name/arguments 任一可为 None(分片只带 arguments)
    tool_call = SimpleNamespace(index=index, id=id, function=function)
    delta = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_translates_tool_calls_and_finish_reason():
    """delta.tool_calls 流式(id+name 首片,arguments 分片)→ 拼装成 ToolCallStart;
    finish_reason=tool_calls → Done(stop_reason=tool_use)。"""
    chunks = [
        _tc_chunk(index=0, id="call_1", name="echo", arguments='{"text": "'),
        _tc_chunk(index=0, arguments='hi"}'),
        _tc_chunk(finish_reason="tool_calls"),
        _chunk(usage=_U(1, 2)),
    ]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    tc = next(e for e in out if isinstance(e, ToolCallStart))
    assert tc.id == "call_1" and tc.name == "echo"
    assert json.loads(tc.args) == {"text": "hi"}
    done = out[-1]
    assert isinstance(done, Done) and done.stop_reason == "tool_use"
    assert done.usage.input_tokens == 1 and done.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_convert_maps_tool_use_and_tool_result_to_openai_style():
    """_convert:assistant 的 ToolUseBlock → tool_calls;ToolResultBlock → 独立 role=tool 消息。"""
    chunks = [_chunk(usage=_U(1, 1))]
    app, prof = _app()
    client = FakeOpenAI(chunks)
    p = OpenAIProvider(prof, app, client=client)
    msgs = [
        Message(
            role="assistant",
            content=[
                TextBlock("ok"),
                ToolUseBlock(id="call_1", name="echo", input={"text": "hi"}),
            ],
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="hi")]),
    ]
    _ = [e async for e in p.stream(msgs, history=[])]
    sent = client.chat.completions.last_kwargs["messages"]
    # [0]=system, [1]=assistant(text+tool_calls), [2]=tool result
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == "ok"
    assert sent[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": json.dumps({"text": "hi"})},
        }
    ]
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": "hi"}


# ---- stage2: registry → tools= 参数 ----


@pytest.mark.asyncio
async def test_registry_emits_tools_when_given():
    """provider 持有 registry 时,_open_stream 的 kwargs 含 tools(OpenAI function 列表)。"""
    from birdcode.tools.read_tool import ReadTool
    from birdcode.tools.registry import ToolRegistry

    chunks = [_chunk(usage=_U(1, 1))]
    app, prof = _app()
    client = FakeOpenAI(chunks)
    reg = ToolRegistry()
    reg.register(ReadTool())
    p = OpenAIProvider(prof, app, client=client, registry=reg)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    kw = client.chat.completions.last_kwargs
    assert "tools" in kw
    assert kw["tools"][0]["type"] == "function"
    assert kw["tools"][0]["function"]["name"] == "read_file"
    assert kw["tools"][0]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_no_tools_kwarg_when_registry_absent():
    chunks = [_chunk(usage=_U(1, 1))]
    app, prof = _app()
    client = FakeOpenAI(chunks)
    p = OpenAIProvider(prof, app, client=client)
    _ = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert "tools" not in client.chat.completions.last_kwargs


# ---- stage2 polish(§7): 无 trailing usage 时钩子补发 Done ----


@pytest.mark.asyncio
async def test_done_emitted_on_finish_reason_without_trailing_usage():
    """无 trailing usage chunk 时,流正常结束的 _finish_stream 钩子必须补发 Done——
    否则 UI 收不到 Done 卡 spinner。"""
    finish_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")],
        usage=None,
    )
    chunks = [_chunk(content="hi"), finish_chunk]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    # 必须有 Done 收尾(尽管没有 trailing usage)
    assert isinstance(out[-1], Done)
    # 无 usage 时 usage 为 None(stop_reason 仍透传)
    assert out[-1].usage is None
    assert out[-1].stop_reason == "end_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "expected_stop"),
    [
        ("stop", "end_turn"),  # 自然完成
        ("tool_calls", "tool_use"),  # 调工具
        ("length", "max_tokens"),  # max_tokens 截断(修复前被错映射成 end_turn)
        ("content_filter", "content_filter"),  # 内容过滤(修复前被错映射成 end_turn)
    ],
)
async def test_finish_reason_mapped_to_canonical_stop_reason(finish_reason, expected_stop):
    """OpenAI finish_reason → Anthropic 规范 stop_reason(跨 provider 统一)。

    修复前 `else "end_turn"` 把 length/content_filter 兜进 end_turn → 截断/过滤的响应被续跑
    交叉校验误判"自然完成"、注入截断文本。修复后显式映射;仅 stop/未知 → end_turn。
    """
    finish_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=finish_reason)],
        usage=None,
    )
    chunks = [_chunk(content="hi"), finish_chunk]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert isinstance(out[-1], Done)
    assert out[-1].stop_reason == expected_stop


@pytest.mark.asyncio
async def test_usage_parses_cached_tokens():
    """trailing usage 携带 prompt_tokens_details.cached_tokens → cache_read_tokens。"""
    details = SimpleNamespace(cached_tokens=800)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5, prompt_tokens_details=details)
    chunks = [_chunk(usage=usage)]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert out[-1].usage.cache_read_tokens == 800
    assert out[-1].usage.cache_creation_tokens == 0  # OpenAI 无 creation 概念


@pytest.mark.asyncio
async def test_usage_without_details_defaults_zero():
    """无 prompt_tokens_details → cache_read_tokens=0(向后兼容旧响应)。"""
    chunks = [_chunk(usage=_U(100, 5))]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert out[-1].usage.cache_read_tokens == 0


# ---- input_tokens 口径归一(deepseek 缓存语义)----
# 实测(deepseek-v4-pro 长会话):trailing usage 的 prompt_tokens 只报未命中(miss),
# 命中部分在 prompt_tokens_details.cached_tokens。BirdCode 到处把 input_tokens 当
# "完整请求大小"用(maybe_compact 的 last_in、/context 脚注),必须归一为真实全量,
# 否则 autocompact 永不触发、面板脚注误导。OpenAI 标准 prompt_tokens 已含 cached。


@pytest.mark.asyncio
async def test_usage_full_input_under_deepseek_cache_semantics():
    """deepseek:prompt_tokens(=miss)< cached(=hit)→ input_tokens = miss + hit(真实全量)。

    复刻实测 log:in=497 read=121856 → 真实全量 122353(原代码只存 497 → autocompact 失灵)。
    """
    details = SimpleNamespace(cached_tokens=121856)
    usage = SimpleNamespace(
        prompt_tokens=497, completion_tokens=1584, prompt_tokens_details=details
    )
    chunks = [_chunk(usage=usage)]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert out[-1].usage.input_tokens == 122353  # 497 + 121856
    assert out[-1].usage.cache_read_tokens == 121856


@pytest.mark.asyncio
async def test_usage_input_not_double_counted_under_openai_semantics():
    """OpenAI:prompt_tokens 已含 cached(prompt ≥ cached)→ input_tokens = prompt_tokens,不重复加。"""
    details = SimpleNamespace(cached_tokens=800)
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=5, prompt_tokens_details=details)
    chunks = [_chunk(usage=usage)]
    app, prof = _app()
    p = OpenAIProvider(prof, app, client=FakeOpenAI(chunks))
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert out[-1].usage.input_tokens == 1000  # 不是 1800
    assert out[-1].usage.cache_read_tokens == 800


# ---- summarize_with_prefix(#10:reasoning 摘要留 text 空间)----


def _nonstream_resp(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.mark.asyncio
async def test_summarize_bumps_max_tokens_for_reasoning():
    """#10:reasoning_effort set → summarize_with_prefix 提升 max_completion_tokens
    (留 reasoning 之后的 text 空间,防 </summary> 被截断 → 提取失败 → 重试 → 硬截断)。"""
    app, prof = _app(effort="medium")
    client = FakeOpenAI([], nonstream_resp=_nonstream_resp("<summary>S</summary>"))
    p = OpenAIProvider(prof, app, client=client)
    await p.summarize_with_prefix(prefix=[], instruction="I", max_tokens=2000)
    kw = client.chat.completions.last_kwargs
    assert kw["stream"] is False
    assert kw["max_completion_tokens"] == 2000 + 4096  # reasoning → +margin
    assert kw["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_summarize_no_margin_when_no_reasoning():
    """无 reasoning_effort → max_completion_tokens 不提升(与 _open_stream 一致)。"""
    app, prof = _app()  # effort=None
    client = FakeOpenAI([], nonstream_resp=_nonstream_resp("<summary>S</summary>"))
    p = OpenAIProvider(prof, app, client=client)
    await p.summarize_with_prefix(prefix=[], instruction="I", max_tokens=2000)
    kw = client.chat.completions.last_kwargs
    assert kw["max_completion_tokens"] == 2000
    assert "reasoning_effort" not in kw
