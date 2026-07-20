# tests/test_provider_complete.py
"""T7: provider.complete() 无工具一次性补全(摘要专用)。

覆盖:
- MockProvider.complete(确定性,<summary> 包裹)。
- AnthropicProvider.complete(messages.create 非流式,拼回 text blocks)。
- OpenAIProvider.complete(chat.completions.create 非流式,取 choices[0].message.content)。
"""

from types import SimpleNamespace

from birdcode.agent.anthropic_provider import AnthropicProvider
from birdcode.agent.mock_provider import MockProvider
from birdcode.agent.openai_provider import OpenAIProvider
from birdcode.config.schema import AppConfig, ProviderProfile

# —— MockProvider ——


async def test_mock_complete_wraps_summary_tags():
    p = MockProvider()
    out = await p.complete(system="be a summarizer", user="blah", max_tokens=1000)
    assert "<summary>" in out and "</summary>" in out


async def test_mock_complete_echoes_user_for_analysis_extraction():
    """Mock 把 user 内容塞进 <summary>,供 ContextManager 的提取逻辑测试。"""
    p = MockProvider()
    out = await p.complete(system="s", user="SECRET-CONTENT", max_tokens=1000)
    assert "SECRET-CONTENT" in out


# —— AnthropicProvider(假 client,非流式 create)——


class _CompleteMessages:
    """模拟 Anthropic messages.create 非流式返回:resp.content 是 text block 列表。"""

    def __init__(self, resp) -> None:
        self._resp = resp
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class _CompleteAnthropic:
    def __init__(self, resp) -> None:
        self.messages = _CompleteMessages(resp)


def _anth_app():
    prof = ProviderProfile(
        name="c",
        protocol="anthropic",
        model="claude-sonnet-4-5",
        base_url="https://api.anthropic.com",
        api_key="k",
    )
    return AppConfig(providers={"c": prof}, system_prompt="SYS"), prof


async def test_anthropic_complete_joins_text_blocks_and_omits_tools():
    """非流式 messages.create:拼回 content 里 text blocks,不发 stream/tools/tool_choice。"""
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="HELLO "),
            SimpleNamespace(type="text", text="WORLD"),
            SimpleNamespace(type="thinking", thinking="scratch"),  # 非 text,应被忽略
        ]
    )
    app, prof = _anth_app()
    client = _CompleteAnthropic(resp)
    p = AnthropicProvider(prof, app, client=client)
    out = await p.complete(system="be concise", user="summarize me", max_tokens=2048)
    assert out == "HELLO WORLD"
    kw = client.messages.last_kwargs
    assert kw["stream"] is False  # 非流式
    assert kw["max_tokens"] == 2048
    assert kw["system"] == "be concise"
    assert kw["messages"] == [{"role": "user", "content": "summarize me"}]
    # 摘要专用:不注入 tools / tool_choice(即便 provider 持有 registry)
    assert "tools" not in kw
    assert "tool_choice" not in kw


async def test_anthropic_complete_returns_empty_when_no_text_blocks():
    """resp.content 无 text block → 返回空串(不崩)。"""
    resp = SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="x", name="n", input={})])
    app, prof = _anth_app()
    p = AnthropicProvider(prof, app, client=_CompleteAnthropic(resp))
    assert await p.complete(system="s", user="u", max_tokens=100) == ""


# —— OpenAIProvider(假 client,非流式 create)——


class _CompleteCompletions:
    def __init__(self, resp) -> None:
        self._resp = resp
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class _CompleteChat:
    def __init__(self, resp) -> None:
        self.completions = _CompleteCompletions(resp)


class _CompleteOpenAI:
    def __init__(self, resp) -> None:
        self.chat = _CompleteChat(resp)


def _oai_app():
    prof = ProviderProfile(
        name="d",
        protocol="openai",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        api_key="k",
    )
    return AppConfig(providers={"d": prof}, system_prompt="SYS"), prof


async def test_openai_complete_returns_message_content():
    """非流式 chat.completions.create:取 choices[0].message.content。"""
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="THE ANSWER"))])
    app, prof = _oai_app()
    client = _CompleteOpenAI(resp)
    p = OpenAIProvider(prof, app, client=client)
    out = await p.complete(system="sys", user="ask", max_tokens=512)
    assert out == "THE ANSWER"
    kw = client.chat.completions.last_kwargs
    assert kw["model"] == "deepseek-reasoner"
    assert kw["max_completion_tokens"] == 512  # 与 _open_stream 一致(max_completion_tokens)
    assert kw["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ask"},
    ]
    assert "stream" not in kw or kw["stream"] is False  # 非流式
    assert "tools" not in kw  # 摘要专用


async def test_openai_complete_empty_when_no_choices():
    """无 choices → 返回空串(不崩)。"""
    resp = SimpleNamespace(choices=[])
    app, prof = _oai_app()
    p = OpenAIProvider(prof, app, client=_CompleteOpenAI(resp))
    assert await p.complete(system="s", user="u", max_tokens=100) == ""
