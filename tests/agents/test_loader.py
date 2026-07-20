from pathlib import Path

from birdcode.agents.definition import AgentSource
from birdcode.agents.loader import load_md_agents


def test_parse_full(tmp_path: Path):
    (tmp_path / "explore.md").write_text(
        "---\n"
        "name: explore\n"
        "description: 只读搜索\n"
        "disallowed_tools:\n  - Write\n  - Edit\n"
        "model: haiku\n"
        "---\n"
        "你是一个只读 agent。\n",
        encoding="utf-8",
    )
    defs = load_md_agents(tmp_path, AgentSource.USER)
    assert len(defs) == 1
    d = defs[0]
    assert d.name == "explore"
    assert d.description == "只读搜索"
    assert d.disallowed_tools == ("Write", "Edit")
    assert d.model == "haiku"
    assert d.source is AgentSource.USER
    assert d.system_prompt.strip() == "你是一个只读 agent。"
    assert d.path is not None


def test_parse_scalar_disallowed_tools(tmp_path: Path):
    """YAML 标量 disallowed_tools: Write → 单元素元组,不逐字符(否则黑名单静默失效)。"""
    (tmp_path / "ro.md").write_text(
        "---\nname: ro\ndescription: d\ndisallowed_tools: Write\n---\nSP\n",
        encoding="utf-8",
    )
    d = load_md_agents(tmp_path, AgentSource.USER)[0]
    assert d.disallowed_tools == ("Write",)


def test_parse_bool_scalar_disallowed_tools(tmp_path: Path):
    """YAML 布尔标量(`disallowed_tools: yes`→True)→ 降级空黑名单,不抛。

    非 list/str 标量在旧实现会让 tuple(str(t) for t in disallowed) 抛 TypeError、且位于
    _parse_one 的 try 之外 → 经 build_agent_registry 崩 on_mount 启动。契约要求坏值跳过。
    """
    (tmp_path / "bad.md").write_text(
        "---\nname: bad\ndescription: d\ndisallowed_tools: yes\n---\nSP\n",
        encoding="utf-8",
    )
    d = load_md_agents(tmp_path, AgentSource.USER)[0]
    assert d.disallowed_tools == ()


def test_parse_int_scalar_disallowed_tools(tmp_path: Path):
    """整数标量(`disallowed_tools: 3`→int)同样降级空,不抛。"""
    (tmp_path / "bad.md").write_text(
        "---\nname: bad\ndescription: d\ndisallowed_tools: 3\n---\nSP\n",
        encoding="utf-8",
    )
    d = load_md_agents(tmp_path, AgentSource.USER)[0]
    assert d.disallowed_tools == ()


def test_name_required(tmp_path: Path):
    """D10:name 必填(不再取 stem);缺 → 跳过。"""
    (tmp_path / "plan.md").write_text("---\ndescription: d\n---\nSP\n", encoding="utf-8")
    assert load_md_agents(tmp_path, AgentSource.PROJECT) == []


def test_description_required(tmp_path: Path):
    """D10:description 必填;缺 → 跳过。"""
    (tmp_path / "plan.md").write_text("---\nname: plan\n---\nSP\n", encoding="utf-8")
    assert load_md_agents(tmp_path, AgentSource.PROJECT) == []


def test_kind_parallel_safe_optional(tmp_path: Path):
    """D3/D4:kind/parallel_safe 可选覆盖默认。"""
    (tmp_path / "ro.md").write_text(
        "---\nname: ro\ndescription: d\nkind: read\nparallel_safe: true\n---\nSP\n",
        encoding="utf-8",
    )
    d = load_md_agents(tmp_path, AgentSource.USER)[0]
    assert d.kind == "read"
    assert d.parallel_safe is True


def test_kind_parallel_safe_default_and_bad(tmp_path: Path):
    """未声明 → write/False;坏值(kind 非法 / parallel_safe 非 bool)→ 降级 write/False。"""
    (tmp_path / "a.md").write_text("---\nname: a\ndescription: d\n---\nSP\n", encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "---\nname: b\ndescription: d\nkind: maybe\nparallel_safe: maybe\n---\nSP\n",
        encoding="utf-8",
    )
    defs = {d.name: d for d in load_md_agents(tmp_path, AgentSource.USER)}
    assert defs["a"].kind == "write" and defs["a"].parallel_safe is False  # 默认
    assert defs["b"].kind == "write"  # "maybe" 非法 → 降级
    assert defs["b"].parallel_safe is False  # "maybe"(str) 非 bool → 降级
    assert defs["a"].run_in_background is False  # 未声明 → 默认同步


def test_run_in_background_optional(tmp_path: Path):
    """run_in_background 可选覆盖默认(缺省同步);坏值降级 False。"""
    (tmp_path / "async.md").write_text(
        "---\nname: async\ndescription: d\nrun_in_background: true\n---\nSP\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.md").write_text(
        "---\nname: bad\ndescription: d\nrun_in_background: maybe\n---\nSP\n",
        encoding="utf-8",
    )
    defs = {d.name: d for d in load_md_agents(tmp_path, AgentSource.USER)}
    assert defs["async"].run_in_background is True
    assert defs["bad"].run_in_background is False  # "maybe"(str) 非 bool → 降级


def test_unknown_fields_ignored(tmp_path: Path):
    (tmp_path / "x.md").write_text(
        "---\nname: x\ncolor: red\nfoo: bar\ndescription: d\n---\nSP\n", encoding="utf-8"
    )
    d = load_md_agents(tmp_path, AgentSource.BUILTIN)[0]
    assert d.description == "d"


def test_malformed_frontmatter_skipped(tmp_path: Path):
    (tmp_path / "bad.md").write_text("no frontmatter at all", encoding="utf-8")
    assert load_md_agents(tmp_path, AgentSource.USER) == []


def test_empty_dir(tmp_path: Path):
    assert load_md_agents(tmp_path, AgentSource.USER) == []
