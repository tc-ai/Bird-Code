"""记忆提取测试:fake provider 返回固定 JSON,验证解析/落盘/去重/路由/串行。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.memory.extractor import MemoryManager
from birdcode.memory.paths import project_memory_dir, user_memory_dir
from birdcode.memory.store import list_memories


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


class FakeProvider:
    """记录 complete 参数,返回预设 JSON。"""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def complete(self, system, user, *, max_tokens, model=None):
        self.calls.append({"system": system, "model": model, "max_tokens": max_tokens})
        return self._payload


def _turn(user_text="帮我改成 any", assistant_text="好的"):
    return Turn(messages=[
        Message(role="user", content=[TextBlock(text=user_text)]),
        Message(role="assistant", content=[TextBlock(text=assistant_text)]),
    ])


def _op_json(ops):
    return json.dumps({"operations": ops}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_extract_create_lands_in_correct_dir(tmp_path, home):
    payload = _op_json([{
        "action": "create", "type": "feedback",
        "name": "prefers-any", "description": "用 any", "body": "用户要 any",
    }])
    mgr = MemoryManager(FakeProvider(payload), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    metas = list_memories(user_memory_dir())
    assert [m.name for m in metas] == ["prefers-any"]


@pytest.mark.asyncio
async def test_extract_project_type_goes_project_dir(tmp_path, home):
    payload = _op_json([{
        "action": "create", "type": "project",
        "name": "ci-tool", "description": "用 GH Actions", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(payload), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    assert [m.name for m in list_memories(project_memory_dir(tmp_path))] == ["ci-tool"]
    assert list_memories(user_memory_dir()) == []


@pytest.mark.asyncio
async def test_extract_update_overwrites(tmp_path, home):
    # 先 create
    p1 = _op_json([{
        "action": "create", "type": "user", "name": "n", "description": "v1", "body": "old",
    }])
    mgr = MemoryManager(FakeProvider(p1), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    # 再 update
    p2 = _op_json([{
        "action": "update", "type": "user", "name": "n", "description": "v2", "body": "new",
    }])
    mgr2 = MemoryManager(FakeProvider(p2), project_root=tmp_path, extract_model="haiku")
    await mgr2.extract(_turn(), [])
    metas = list_memories(user_memory_dir())
    assert len(metas) == 1
    assert metas[0].description == "v2"


@pytest.mark.asyncio
async def test_extract_create_same_name_dedups_to_update(tmp_path, home):
    p1 = _op_json([{
        "action": "create", "type": "user", "name": "n", "description": "v1", "body": "a",
    }])
    p2 = _op_json([{
        "action": "create", "type": "user", "name": "n", "description": "v2", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(p1), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    mgr2 = MemoryManager(FakeProvider(p2), project_root=tmp_path, extract_model="haiku")
    await mgr2.extract(_turn(), [])
    assert len(list_memories(user_memory_dir())) == 1  # 不重复创建


@pytest.mark.asyncio
async def test_extract_delete_removes(tmp_path, home):
    p1 = _op_json([{
        "action": "create", "type": "user", "name": "n", "description": "d", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(p1), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    p2 = _op_json([{"action": "delete", "name": "n"}])
    mgr2 = MemoryManager(FakeProvider(p2), project_root=tmp_path, extract_model="haiku")
    await mgr2.extract(_turn(), [])
    assert list_memories(user_memory_dir()) == []


@pytest.mark.asyncio
async def test_extract_empty_operations_does_nothing(tmp_path, home):
    mgr = MemoryManager(FakeProvider(_op_json([])), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    assert list_memories(user_memory_dir()) == []
    assert list_memories(project_memory_dir(tmp_path)) == []


@pytest.mark.asyncio
async def test_extract_malformed_json_skipped(tmp_path, home):
    mgr = MemoryManager(FakeProvider("这不是 JSON"), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])  # 不抛
    assert list_memories(user_memory_dir()) == []


@pytest.mark.asyncio
async def test_extract_passes_extract_model_to_complete(tmp_path, home):
    payload = _op_json([])
    provider = FakeProvider(payload)
    mgr = MemoryManager(provider, project_root=tmp_path, extract_model="haiku-mini")
    await mgr.extract(_turn(), [])
    assert provider.calls[0]["model"] == "haiku-mini"


@pytest.mark.asyncio
async def test_extract_serializes_via_lock(tmp_path, home, monkeypatch):
    """并发两次 extract,Lock 保证串行(用计数验证不重入)。"""
    payload = _op_json([{
        "action": "create", "type": "user", "name": "n", "description": "d", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(payload), project_root=tmp_path, extract_model="haiku")
    await asyncio.gather(mgr.extract(_turn(), []), mgr.extract(_turn(), []))
    # 两次都跑完,最终一个文件(create 同名幂等)
    assert len(list_memories(user_memory_dir())) == 1


@pytest.mark.asyncio
async def test_extract_skips_malformed_op_keeps_good_ones(tmp_path, home):
    payload = _op_json([
        {"action": "create", "type": "feedback", "name": "good", "description": "d", "body": "b"},
        # 下一行 action 拼错,应被跳过
        {"action": "craete", "type": "user", "name": "bad", "description": "d", "body": "b"},
    ])
    mgr = MemoryManager(FakeProvider(payload), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    # 好 op 落地,坏 op 被跳过,不波及整批
    assert [m.name for m in list_memories(user_memory_dir())] == ["good"]


# —— F2(回退):跨作用域同名记忆合法共存,不互相删 ——


@pytest.mark.asyncio
async def test_extract_cross_scope_same_name_coexists(tmp_path, home):
    # user/project 是独立命名空间:同名记忆可合法共存(个人偏好 vs 项目事实)。
    # create/update 不应删另一目录的同名文件(否则静默销毁跨作用域合法记忆)。
    p_user = _op_json([{
        "action": "create", "type": "user",
        "name": "db-url", "description": "个人偏好", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(p_user), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    assert [m.name for m in list_memories(user_memory_dir())] == ["db-url"]

    p_proj = _op_json([{
        "action": "create", "type": "project",
        "name": "db-url", "description": "项目事实", "body": "b2",
    }])
    mgr2 = MemoryManager(FakeProvider(p_proj), project_root=tmp_path, extract_model="haiku")
    await mgr2.extract(_turn(), [])
    # 两者都在(独立命名空间),没被互相删
    assert [m.name for m in list_memories(user_memory_dir())] == ["db-url"]
    assert [m.name for m in list_memories(project_memory_dir(tmp_path))] == ["db-url"]


@pytest.mark.asyncio
async def test_extract_update_rename_removes_old_file(tmp_path, home):
    # 先建 old-name
    p1 = _op_json([{
        "action": "create", "type": "project",
        "name": "old-name", "description": "d1", "body": "b",
    }])
    mgr = MemoryManager(FakeProvider(p1), project_root=tmp_path, extract_model="haiku")
    await mgr.extract(_turn(), [])
    assert [m.name for m in list_memories(project_memory_dir(tmp_path))] == ["old-name"]

    # update 改名 old-name → new-name:删旧文件、写新文件(不留孤儿)
    p2 = _op_json([{
        "action": "update", "type": "project",
        "name": "old-name", "new_name": "new-name",
        "description": "d2", "body": "b2",
    }])
    mgr2 = MemoryManager(FakeProvider(p2), project_root=tmp_path, extract_model="haiku")
    await mgr2.extract(_turn(), [])
    assert [m.name for m in list_memories(project_memory_dir(tmp_path))] == ["new-name"]
    assert not (project_memory_dir(tmp_path) / "old-name.md").exists()  # 旧文件已删


# —— F9/F10:_render_turn 截断 + 工具输出 ——


def test_render_turn_truncates_single_oversized_block(tmp_path):
    # 单个 120KB TextBlock 不能击穿 8000 上限(先按剩余预算截断本段再加)
    mgr = MemoryManager(FakeProvider(_op_json([])), project_root=tmp_path, extract_model="m")
    big = Turn(messages=[Message(role="user", content=[TextBlock(text="X" * 120_000)])])
    rendered = mgr._render_turn(big)  # noqa: SLF001
    assert len(rendered) < 10_000
    assert "截断" in rendered


def test_render_turn_includes_tool_result(tmp_path):
    # 工具输出到达提取器(否则最有价值的 project/reference 信号全丢)
    mgr = MemoryManager(FakeProvider(_op_json([])), project_root=tmp_path, extract_model="m")
    turn = Turn(messages=[
        Message(role="user", content=[TextBlock(text="查一下版本")]),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="1", name="bash", input={"command": "psql --version"})],
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="1", content="PostgreSQL 15.3")]),
    ])
    rendered = mgr._render_turn(turn)  # noqa: SLF001
    assert "PostgreSQL 15.3" in rendered
    assert "tool_result" in rendered
    assert "bash" in rendered  # 工具名带上(否则看不出结果出自哪个工具)


# —— F1:set_provider(/profile 重连)——


@pytest.mark.asyncio
async def test_set_provider_reroutes_to_new_provider_and_model(tmp_path, home):
    p1 = FakeProvider(_op_json([]))
    mgr = MemoryManager(p1, project_root=tmp_path, extract_model="old-model")
    await mgr.extract(_turn(), [])
    assert p1.calls[0]["model"] == "old-model"
    # /profile 切换:set_provider 换 provider + 模型,后续 extract 走新 provider
    p2 = FakeProvider(_op_json([]))
    mgr.set_provider(p2, extract_model="new-model")
    await mgr.extract(_turn(), [])
    assert p2.calls[0]["model"] == "new-model"
