"""记忆系统集成:provider model 覆盖、build_system_text、TurnController 触发。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.agent.openai_provider import OpenAIProvider
from birdcode.agent.system_prompt import build_system_reminder, build_system_text
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import TurnController


def _profile(**over):
    base = dict(protocol="openai", model="main-model", base_url="http://x", api_key="k")
    base.update(over)
    return ProviderProfile(**base)


def _cfg(profile):
    return AppConfig(providers={"p": profile}, default="p")


# —— schema 字段 ——


def test_provider_profile_hak_model_defaults_none():
    p = _profile()
    assert p.hak_model is None


def test_provider_profile_accepts_hak_model():
    p = _profile(hak_model="haiku-mini")
    assert p.hak_model == "haiku-mini"


def test_app_config_has_project_root_default_none():
    cfg = _cfg(_profile())
    assert cfg.project_root is None


# —— complete() model 覆盖（OpenAI 路径，注入 fake client）——


class _FakeOpenAIResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeOpenAIClient:
    """记录 chat.completions.create 的 kwargs,返回固定响应。"""

    def __init__(self, content: str = "ok") -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeOpenAIResponse(self._content)


@pytest.mark.asyncio
async def test_complete_uses_profile_model_by_default():
    client = _FakeOpenAIClient()
    profile = _profile(model="main-model")
    provider = OpenAIProvider(profile, _cfg(profile), client=client)
    await provider.complete("sys", "u", max_tokens=10)
    assert client.calls[0]["model"] == "main-model"


@pytest.mark.asyncio
async def test_complete_model_override_wins():
    client = _FakeOpenAIClient()
    profile = _profile(model="main-model")
    provider = OpenAIProvider(profile, _cfg(profile), client=client)
    await provider.complete("sys", "u", max_tokens=10, model="haiku-mini")
    assert client.calls[0]["model"] == "haiku-mini"


# —— 记忆注入:走 <system-reminder>(build_system_reminder),不走 build_system_text ——


def test_build_system_text_does_not_include_memory():
    # 记忆块已移出静态 system prompt(改走每轮 reminder):build_system_text 不含记忆
    t = build_system_text(project_instructions="BIRDCODE_MD")
    assert "BIRDCODE_MD" in t
    assert "① 身份" in t


@pytest.mark.asyncio
async def test_build_system_reminder_includes_memory():
    r = await build_system_reminder(memory="MEMORY_IDX")
    assert "MEMORY_IDX" in r
    assert r.startswith("<system-reminder>")


@pytest.mark.asyncio
async def test_build_system_reminder_memory_optional():
    r = await build_system_reminder()
    assert r.startswith("<system-reminder>")  # 无记忆也正常(仍含日期/git)


def test_build_system_text_override_bypasses():
    t = build_system_text(override="ONLY")
    assert t == "ONLY"


# —— _memory_index 每轮现读:写入后下一次即见(本会话内及时更新) ——


def test_memory_index_reads_fresh_after_write(tmp_path, monkeypatch):
    from pathlib import Path

    from birdcode.memory.paths import project_memory_dir
    from birdcode.memory.store import write_memory

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # 隔离用户级目录
    cfg = _cfg(_profile())
    cfg.project_root = tmp_path
    provider = OpenAIProvider(_profile(), cfg, client=_FakeOpenAIClient())

    assert provider._memory_index() == ""  # 初始无记忆

    write_memory(
        project_memory_dir(tmp_path),
        name="ci",
        description="用 GH Actions",
        type_="project",
        body="b",
    )
    # 写入后下一次现读即见 —— 本会话内及时更新(无需重启)
    idx = provider._memory_index()  # noqa: SLF001
    assert "用 GH Actions" in idx


# —— TurnController 接入:fire-and-forget 提取 + abort cancel ——


class RecordingMemory:
    """假 MemoryManager:记录 extract 调用;extract 立即返回(无锁无 LLM)。"""

    def __init__(self):
        self.calls: list = []  # list[Turn]

    async def extract(self, turn, history):
        self.calls.append(turn)


def _make_controller(memory=None, provider=None):
    async def _noop_event(_ev):
        pass

    async def _noop_status():
        pass

    return TurnController(
        provider or MockProvider(),
        on_event=_noop_event,
        on_status=_noop_status,
        memory=memory,
    )


@pytest.mark.asyncio
async def test_controller_triggers_extract_on_clean_turn():
    mem = RecordingMemory()
    ctrl = _make_controller(memory=mem)
    # MockProvider 两阶段流(user→tool_use→tool_result→end_turn)会在 _process 内跑完
    await ctrl.submit("hello world")
    # 提取是 fire-and-forget;await 一拍让 task 跑完
    for t in list(ctrl._extract_tasks):
        await t
    assert len(mem.calls) == 1


@pytest.mark.asyncio
async def test_controller_no_memory_does_not_crash():
    ctrl = _make_controller(memory=None)
    await ctrl.submit("hello")
    assert ctrl._extract_tasks == set()


@pytest.mark.asyncio
async def test_controller_aborts_cancel_extract_task():
    mem = RecordingMemory()
    ctrl = _make_controller(memory=mem)

    # 手动塞一个 pending task 模拟在跑的提取
    async def _hang():
        await asyncio.sleep(100)

    # 持有任务引用:abort() 会清空 _extract_tasks,直接断言引用上的 cancelled 状态
    # (而非遍历已空的集合——那会 vacuously true,测不出真取消)。
    task = asyncio.create_task(_hang())
    ctrl._extract_tasks.add(task)
    ctrl.abort()
    await asyncio.sleep(0)  # 让 cancel 传递到任务
    assert task.cancelled()  # 真正被取消
    assert ctrl._extract_tasks == set()  # 集合也清空了
