from dataclasses import FrozenInstanceError

import pytest

from birdcode.agents.definition import AgentDefinition, AgentSource, split_frontmatter


def test_defaults():
    d = AgentDefinition(name="explore", description="d", system_prompt="SP")
    assert d.disallowed_tools == ()
    assert d.model == ""
    assert d.source is AgentSource.BUILTIN
    assert d.path is None
    assert d.kind == "write"  # 默认保守:能力未知 → write + 串行
    assert d.parallel_safe is False
    assert d.run_in_background is False  # 默认同步


def test_frozen():
    d = AgentDefinition(name="x", description="d", system_prompt="SP")
    with pytest.raises(FrozenInstanceError):
        d.name = "y"  # type: ignore[misc]


def test_agent_source_values():
    assert AgentSource.BUILTIN.value == "builtin"
    assert AgentSource.PROJECT.value == "project"
    assert AgentSource.USER.value == "user"


# ---------------------------------------------------------------------------
# split_frontmatter — 仅在列 0 的独立 --- 上切分(防值内/正文内 --- 误切)
# ---------------------------------------------------------------------------


def test_split_frontmatter_basic():
    raw = "---\nname: foo\ndescription: bar\n---\nbody line\n"
    fm, body = split_frontmatter(raw)
    assert fm == "name: foo\ndescription: bar\n"
    assert body == "body line\n"


def test_split_frontmatter_value_containing_separator():
    """值里的 --- 子串(`description: a---b`)不被误切:frontmatter 完整、正文不污染。

    回归:旧 ``raw.split('---', 2)`` 会把第二次切分落进 frontmatter → 描述截断成 'a'、
    正文被 prepend 'b'。本断言钉死修复。
    """
    raw = "---\nname: foo\ndescription: a---b\n---\nbody\n"
    fm, body = split_frontmatter(raw)
    import yaml

    meta = yaml.safe_load(fm)
    assert meta["description"] == "a---b"  # 完整保留,未被截断
    assert body.strip() == "body"  # 正文未被污染


def test_split_frontmatter_indented_separator_in_block_scalar_not_closed():
    """块标量里缩进的 --- 是内容,不是闭标记(列 0 才算)。"""
    raw = "---\nname: foo\ndescription: |\n  line1\n  ---\n  line3\n---\nbody\n"
    fm, body = split_frontmatter(raw)
    # 闭标记是末尾列 0 的 ---;缩进的 --- 不算
    assert "line3" in fm
    assert body.strip() == "body"


def test_split_frontmatter_body_with_separator_line():
    """正文里出现的列 0 --- 也不影响首闭标记定位(取首个列 0 ---)。"""
    raw = "---\nname: foo\n---\nintro\n---\nmore\n"
    fm, body = split_frontmatter(raw)
    assert fm == "name: foo\n"
    assert body == "intro\n---\nmore\n"


def test_split_frontmatter_no_frontmatter():
    assert split_frontmatter("no frontmatter here") is None
    assert split_frontmatter("") is None


def test_split_frontmatter_unclosed_returns_none():
    """有开标记但无列 0 闭标记 → None(不把整文当 frontmatter)。"""
    assert split_frontmatter("---\nname: foo\n") is None
