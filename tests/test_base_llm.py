import httpx
import pytest
from anthropic import (
    APIStatusError as AnthropicStatus,
)
from anthropic import (
    AuthenticationError as AnthropicAuth,
)
from anthropic import (
    BadRequestError as AnthropicBadRequest,
)
from anthropic import (
    RateLimitError as AnthropicRateLimit,
)
from openai import (
    APIStatusError as OpenAIStatus,
)
from openai import (
    AuthenticationError as OpenAIAuth,
)
from openai import (
    BadRequestError as OpenAIBadRequest,
)
from openai import (
    RateLimitError as OpenAIRateLimit,
)

from birdcode.agent.base_llm import _BaseLLMProvider
from birdcode.agent.provider import Done, Error, TextDelta
from birdcode.blocks import TextBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message


@pytest.fixture(autouse=True)
def _stub_reminder(monkeypatch):
    """所有 stream 测试默认打桩 build_system_reminder,避免真实 git 子进程(提速 + hermetic)。
    需要特定 reminder 内容的测试(_inject_reminder 的 3 个新测试)自行 monkeypatch 覆盖。"""
    from birdcode.agent import base_llm

    async def _default(**kwargs) -> str:
        return "<system-reminder>stub</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _default)


class Transient(Exception):
    """测试用「可重试」标记异常。"""


class Permanent(Exception):
    """测试用「永久」标记异常。"""


def _profile() -> ProviderProfile:
    return ProviderProfile(protocol="openai", model="m", base_url="u", api_key="k")


def _app() -> AppConfig:
    return AppConfig(providers={"p": _profile()})


class _TestableProvider(_BaseLLMProvider):
    """open_stream 按脚本执行；translate 把 raw 透传为已构造事件。"""

    def __init__(self, app, script):
        super().__init__(_profile(), app, max_retries=2, backoff_base=0.0)
        self._script = list(script)
        self._calls = 0

    def _convert(self, messages):
        return [
            {
                "role": m.role,
                "content": "".join(b.text for b in m.content if isinstance(b, TextBlock)),
            }
            for m in messages
        ]

    async def _open_stream(self, payload, **kwargs):
        self._calls += 1
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # 已是 async iterable

    def _translate(self, raw):
        yield raw  # 测试直接塞 ProviderEvent

    def _is_retryable(self, exc):
        return isinstance(exc, Transient)

    def _is_permanent(self, exc):
        return isinstance(exc, Permanent)


async def _aiter(seq):
    """迭代 seq；遇到 BaseException 实例则 *抛出*（而非 yield），以模拟流中断。"""
    for x in seq:
        if isinstance(x, BaseException):
            raise x
        yield x


@pytest.mark.asyncio
async def test_happy_path_emits_events_and_done():
    p = _TestableProvider(_app(), [_aiter([TextDelta(text="hi"), Done(usage=None)])])
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert isinstance(out[-1], Done)
    assert any(isinstance(e, TextDelta) for e in out)


@pytest.mark.asyncio
async def test_retry_transient_before_first_token():
    p = _TestableProvider(
        _app(),
        [
            Transient("429"),
            _aiter([TextDelta(text="ok"), Done(usage=None)]),
        ],
    )
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert p._calls == 2
    assert any(isinstance(e, TextDelta) and e.text == "ok" for e in out)


@pytest.mark.asyncio
async def test_no_retry_after_first_token():
    p = _TestableProvider(
        _app(),
        [_aiter([TextDelta(text="partial"), Transient("mid-stream")])],
    )
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert p._calls == 1  # 首token后断流不重试
    assert any(isinstance(e, TextDelta) and e.text == "partial" for e in out)
    assert isinstance(out[-1], Error)


@pytest.mark.asyncio
async def test_permanent_error_yields_error_no_retry():
    p = _TestableProvider(_app(), [Permanent("401")])
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert p._calls == 1
    assert isinstance(out[-1], Error)


@pytest.mark.asyncio
async def test_retry_exhausted_yields_error():
    p = _TestableProvider(_app(), [Transient("429"), Transient("429"), Transient("429")])
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert p._calls == 3  # 初次 + 2 次重试
    assert isinstance(out[-1], Error)


# —— Fix 2：用真实 SDK 异常验证「生产」分类逻辑（不覆盖 _is_retryable/_is_permanent）——


class _RealPredProvider(_BaseLLMProvider):
    """复用真实分类谓词；仅提供 _open_stream/_translate 占位实现以支持流级测试。"""

    def __init__(self, app, *, script=None, model="claude-sonnet-4-5"):
        prof = ProviderProfile(protocol="anthropic", model=model, base_url="u", api_key="k")
        super().__init__(prof, app, max_retries=2, backoff_base=0.0)
        self._script = list(script or [])
        self._calls = 0

    def _convert(self, messages):
        return [
            {
                "role": m.role,
                "content": "".join(b.text for b in m.content if isinstance(b, TextBlock)),
            }
            for m in messages
        ]

    async def _open_stream(self, payload, **kwargs):
        self._calls += 1
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return _aiter([Done(usage=None)])

    def _translate(self, raw):
        yield raw


def _anthropic(exc_cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status_code=status, request=req)
    return exc_cls(message="x", response=resp, body=None)


def _openai(exc_cls, status):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req)
    return exc_cls(message="x", response=resp, body=None)


@pytest.fixture()
def _real_provider():
    return _RealPredProvider(_app())


@pytest.mark.parametrize(
    "exc",
    [
        _anthropic(AnthropicRateLimit, 429),
        _openai(OpenAIRateLimit, 429),
        _anthropic(AnthropicStatus, 500),
        _anthropic(AnthropicStatus, 503),
        _openai(OpenAIStatus, 500),
        _openai(OpenAIStatus, 503),
    ],
)
def test_real_retryable_classification(_real_provider, exc):
    assert _real_provider._is_retryable(exc) is True
    assert _real_provider._is_permanent(exc) is False


@pytest.mark.parametrize(
    "exc",
    [
        _anthropic(AnthropicBadRequest, 400),
        _anthropic(AnthropicAuth, 401),
        _openai(OpenAIBadRequest, 400),
        _openai(OpenAIAuth, 401),
    ],
)
def test_real_permanent_classification(_real_provider, exc):
    assert _real_provider._is_retryable(exc) is False
    assert _real_provider._is_permanent(exc) is True


@pytest.mark.asyncio
async def test_stream_retries_on_real_429_then_succeeds():
    """端到端：_open_stream 抛 RateLimitError(429) 两次后成功 → 重试后成功，调用 3 次。"""
    p = _RealPredProvider(
        _app(),
        script=[
            _anthropic(AnthropicRateLimit, 429),
            _openai(OpenAIRateLimit, 429),
            _aiter([TextDelta(text="ok"), Done(usage=None)]),
        ],
    )
    out = [e async for e in p.stream([Message(role="user", content=[TextBlock("x")])], history=[])]
    assert p._calls == 3
    assert any(isinstance(e, TextDelta) and e.text == "ok" for e in out)
    assert isinstance(out[-1], Done)


# ---- stage2: provider 持有 registry ----


def test_base_provider_stores_registry_when_given():
    """_BaseLLMProvider.__init__ 接收 registry 并存 self._registry。"""
    from birdcode.tools.registry import ToolRegistry

    reg = ToolRegistry()

    class _Concrete(_BaseLLMProvider):
        def _convert(self, messages):
            return []

        async def _open_stream(self, payload, **kwargs):
            return iter(())  # 不在测试中调用

        def _translate(self, raw):
            return iter(())

    p = _Concrete(_profile(), _app(), registry=reg)
    assert p._registry is reg  # noqa: SLF001


def test_base_provider_registry_defaults_none():
    class _Concrete(_BaseLLMProvider):
        def _convert(self, messages):
            return []

        async def _open_stream(self, payload, **kwargs):
            return iter(())

        def _translate(self, raw):
            return iter(())

    p = _Concrete(_profile(), _app())
    assert p._registry is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_inject_reminder_appends_to_first_user(monkeypatch):
    from birdcode.agent import base_llm

    async def _fake(**kwargs) -> str:
        return "<system-reminder>R</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _fake)
    p = _TestableProvider(_app(), [_aiter([Done(usage=None)])])
    # 首条 user 含两段文本:reminder 必须追加到「末尾」而非前置(recency 位置)
    original = [Message(role="user", content=[TextBlock("x"), TextBlock("y")])]
    injected = await p._inject_reminder(list(original))
    assert injected[0].content[0].text == "x"  # 原内容在前
    assert injected[0].content[1].text == "y"
    assert injected[0].content[-1].text == "<system-reminder>R</system-reminder>"  # reminder 在末尾
    assert original[0].content == [TextBlock("x"), TextBlock("y")]  # 未 mutate 原始


@pytest.mark.asyncio
async def test_inject_reminder_skips_when_first_not_user(monkeypatch):
    from birdcode.agent import base_llm

    async def _fake(**kwargs) -> str:
        return "<system-reminder>R</system-reminder>"

    monkeypatch.setattr(base_llm, "build_system_reminder", _fake)
    p = _TestableProvider(_app(), [_aiter([Done(usage=None)])])
    msgs = [Message(role="assistant", content=[TextBlock("a")])]
    injected = await p._inject_reminder(list(msgs))
    assert injected[0].content == [TextBlock("a")]  # 首条非 user → 不注入


@pytest.mark.asyncio
async def test_inject_reminder_empty_skips(monkeypatch):
    from birdcode.agent import base_llm

    async def _fake(**kwargs) -> str:
        return ""

    monkeypatch.setattr(base_llm, "build_system_reminder", _fake)
    p = _TestableProvider(_app(), [_aiter([Done(usage=None)])])
    msgs = [Message(role="user", content=[TextBlock("x")])]
    injected = await p._inject_reminder(list(msgs))
    assert injected[0].content == [TextBlock("x")]
