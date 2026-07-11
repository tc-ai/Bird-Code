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


def test_read_non_utf8_returns_none(tmp_path):
    """CR #4: 非 UTF-8 字节(崩溃转储/外部编辑)→ None,不杀 /agents 视图。

    UnicodeDecodeError 是 ValueError 子类但非 JSONDecodeError/OSError;旧 except
    (OSError, JSONDecodeError) 漏掉,read_text 抛出传播,违反 docstring「缺/坏 → None」。
    """
    p = tmp_path / "agent-a1.meta.json"
    p.write_bytes(b"\xff\xfe\x00\x01bad bytes")
    assert read_subagent_meta(p) is None
