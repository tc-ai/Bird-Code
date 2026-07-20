from pathlib import Path

import pytest

from birdcode.session.jsonutil import write_json_atomic
from birdcode.session.models import SubagentMeta
from birdcode.session.subagent_meta import (
    list_subagent_metas,
    read_subagent_meta,
    write_subagent_meta,
)


def test_write_read_roundtrip(tmp_path):
    m = SubagentMeta(agent_id="a1", tool_use_id="call_x", status="launched")
    p = tmp_path / "agent-a1.meta.json"
    write_subagent_meta(p, m)
    assert p.exists()
    back = read_subagent_meta(p)
    assert back is not None
    assert back.agent_id == "a1" and back.status == "launched"


def test_read_missing_returns_none(tmp_path):
    assert read_subagent_meta(tmp_path / "nope.meta.json") is None


def test_read_corrupt_returns_none(tmp_path):
    p = tmp_path / "agent-a1.meta.json"
    p.write_text("{坏 json", encoding="utf-8")
    assert read_subagent_meta(p) is None


def test_list_subagent_metas(tmp_path):
    write_subagent_meta(
        tmp_path / "agent-a1.meta.json",
        SubagentMeta(agent_id="a1", tool_use_id="c1"),
    )
    write_subagent_meta(
        tmp_path / "agent-a2.meta.json",
        SubagentMeta(agent_id="a2", tool_use_id="c2"),
    )
    # 非 meta 文件不被收录
    (tmp_path / "agent-a1.jsonl").write_text("{}", encoding="utf-8")
    metas = list_subagent_metas(tmp_path)
    assert {m.agent_id for m in metas} == {"a1", "a2"}


def test_list_subagent_metas_sorted(tmp_path):
    """list_subagent_metas 按 agent_id 确定序:glob 顺序随文件系统而异(ext4 哈希序),
    排序保证 reminder 行 / UI 列表 / 测试断言稳定(防 review Finding 11)。"""
    # 故意反向写入,验证顺序不依赖写入序
    for aid in ("sub-3", "sub-1", "sub-2"):
        write_subagent_meta(
            tmp_path / f"agent-{aid}.meta.json",
            SubagentMeta(agent_id=aid, tool_use_id=f"c-{aid}"),
        )
    metas = list_subagent_metas(tmp_path)
    assert [m.agent_id for m in metas] == ["sub-1", "sub-2", "sub-3"]


def test_read_non_utf8_returns_none(tmp_path):
    """CR #4: 非 UTF-8 字节(崩溃转储/外部编辑)→ None,不杀 /agents 视图。

    UnicodeDecodeError 是 ValueError 子类但非 JSONDecodeError/OSError;旧 except
    (OSError, JSONDecodeError) 漏掉,read_text 抛出传播,违反 docstring「缺/坏 → None」。
    """
    p = tmp_path / "agent-a1.meta.json"
    p.write_bytes(b"\xff\xfe\x00\x01bad bytes")
    assert read_subagent_meta(p) is None


def test_write_json_atomic_preserves_target_on_failure(tmp_path, monkeypatch):
    """边界 C:tmp 写失败时 target 原文件不被破坏(原子写核心保证)。

    模拟崩溃在 tmp.write_text 中途(patch Path.write_text 抛 OSError)。write_json_atomic
    写 tmp(非 target),tmp 失败 → os.replace 未执行 → target 原内容保留,杜绝半截 JSON
    让 read_subagent_meta 返 None、子 agent 永久悬挂。对比旧 path.write_text 直接覆盖:
    写 target 中途崩溃会留半截 target。
    """
    p = tmp_path / "agent-a1.meta.json"
    original = '{"agentId": "a1", "status": "old"}'
    p.write_text(original, encoding="utf-8")  # patch 前用真 write 预置

    def _raise(self, *a, **kw):
        raise OSError("crash mid-write")

    monkeypatch.setattr(Path, "write_text", _raise)
    with pytest.raises(OSError):
        write_json_atomic(p, '{"agentId": "a1", "status": "new"}')
    assert p.read_text(encoding="utf-8") == original  # target 未被破坏


def test_write_json_atomic_uses_replace(tmp_path, monkeypatch):
    """边界 C:write_json_atomic 走 temp+os.replace 原子模式(tmp 名为 <name>.tmp)。"""
    replaced: list[tuple] = []
    monkeypatch.setattr(
        "birdcode.session.jsonutil.os.replace",
        lambda src, dst: replaced.append((src, dst)),
    )
    p = tmp_path / "agent-a1.meta.json"
    write_json_atomic(p, '{"agentId": "a1"}')
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == p
    assert src.name == p.name + ".tmp"
