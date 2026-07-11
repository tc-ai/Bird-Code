"""记忆文件存储(frontmatter + slugify + 解析/序列化 + CRUD + 索引重建)测试。"""
from __future__ import annotations

import pytest

from birdcode.memory.store import (
    INDEX_FILE,
    MemoryFrontmatter,
    delete_memory,
    list_memories,
    memory_exists,
    parse_memory,
    read_index,
    rebuild_index,
    serialize_memory,
    slugify,
    write_memory,
)

# —— slugify ——


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Prefers Any Over Interface", "prefers-any-over-interface"),
        ("用 any 替代 interface{}", "any-interface"),  # 非 [a-z0-9] 折成 -
        ("  spaced  out  ", "spaced-out"),
        ("---leading-trailing---", "leading-trailing"),
        ("UPPER_CASE_NAME", "upper-case-name"),
        ("!!!", "untitled"),  # 全非法 → untitled
        ("", "untitled"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


# —— serialize + parse 往返 ——


def test_serialize_then_parse_roundtrip():
    text = serialize_memory(
        name="prefers-any",
        description="用户偏好 any 而非 interface{}",
        type_="feedback",
        body="用户明确要求用 any。\n\n**Why:** 更简洁",
    )
    parsed = parse_memory(text)
    assert parsed is not None
    fm, body = parsed
    assert isinstance(fm, MemoryFrontmatter)
    assert fm.name == "prefers-any"
    assert fm.description == "用户偏好 any 而非 interface{}"
    assert fm.type == "feedback"
    assert body == "用户明确要求用 any。\n\n**Why:** 更简洁"


def test_serialize_has_frontmatter_delimiters():
    text = serialize_memory(name="x", description="d", type_="user", body="b")
    assert text.startswith("---\n")
    assert "\n---\n" in text


# —— parse 防御 ——


def test_parse_rejects_missing_frontmatter():
    assert parse_memory("just body, no frontmatter") is None


def test_parse_rejects_unclosed_frontmatter():
    assert parse_memory("---\nname: x\nthis never closes") is None


def test_parse_rejects_invalid_type():
    text = "---\nname: x\ndescription: d\ntype: secret\n---\nbody"
    assert parse_memory(text) is None


# —— CRUD: write / delete / exists ——


def test_write_memory_creates_file_with_slug_name(tmp_path):
    path = write_memory(
        tmp_path,
        name="Prefers Any",
        description="d",
        type_="feedback",
        body="b",
    )
    assert path == tmp_path / "prefers-any.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    fm, body = parse_memory(text)  # parse_memory 已由 Task 2 导入
    assert fm.name == "Prefers Any"
    assert body == "b"


def test_write_memory_creates_missing_dir(tmp_path):
    target_dir = tmp_path / "nested" / "memory"
    write_memory(target_dir, name="x", description="d", type_="user", body="b")
    assert (target_dir / "x.md").is_file()


def test_write_memory_overwrites_existing(tmp_path):
    write_memory(tmp_path, name="x", description="v1", type_="user", body="old")
    write_memory(tmp_path, name="x", description="v2", type_="user", body="new")
    text = (tmp_path / "x.md").read_text(encoding="utf-8")
    fm, body = parse_memory(text)
    assert fm.description == "v2"
    assert body == "new"


def test_write_memory_is_atomic_no_tmp_left(tmp_path):
    write_memory(tmp_path, name="x", description="d", type_="user", body="b")
    # 成功后不应残留 .tmp
    assert not list(tmp_path.glob("*.tmp"))


def test_memory_exists_after_write(tmp_path):
    assert memory_exists(tmp_path, "Some Name") is False
    write_memory(tmp_path, name="Some Name", description="d", type_="user", body="b")
    assert memory_exists(tmp_path, "Some Name") is True


def test_delete_memory_returns_true_when_existed(tmp_path):
    write_memory(tmp_path, name="x", description="d", type_="user", body="b")
    assert delete_memory(tmp_path, "x") is True
    assert not (tmp_path / "x.md").exists()


def test_delete_memory_returns_false_when_missing(tmp_path):
    assert delete_memory(tmp_path, "nope") is False


# —— list_memories / rebuild_index / read_index ——


def test_list_memories_returns_parsed_metas_sorted_by_name(tmp_path):
    write_memory(tmp_path, name="beta", description="b", type_="user", body="x")
    write_memory(tmp_path, name="alpha", description="a", type_="feedback", body="y")
    metas = list_memories(tmp_path)
    assert [m.name for m in metas] == ["alpha", "beta"]
    assert metas[0].type == "feedback"
    assert metas[0].path == tmp_path / "alpha.md"


def test_list_memories_skips_index_file(tmp_path):
    write_memory(tmp_path, name="x", description="d", type_="user", body="b")
    (tmp_path / INDEX_FILE).write_text("- [x](x.md)", encoding="utf-8")
    metas = list_memories(tmp_path)
    assert [m.name for m in metas] == ["x"]  # MEMORY.md 不算


def test_list_memories_skips_malformed(tmp_path):
    write_memory(tmp_path, name="good", description="d", type_="user", body="b")
    (tmp_path / "bad.md").write_text("no frontmatter here", encoding="utf-8")
    metas = list_memories(tmp_path)
    assert [m.name for m in metas] == ["good"]


def test_list_memories_missing_dir_returns_empty(tmp_path):
    assert list_memories(tmp_path / "nope") == []


def test_rebuild_index_writes_pointer_lines_sorted(tmp_path):
    write_memory(tmp_path, name="beta", description="贝塔说明", type_="user", body="x")
    write_memory(tmp_path, name="alpha", description="阿尔法", type_="feedback", body="y")
    rebuild_index(tmp_path)
    idx = (tmp_path / INDEX_FILE).read_text(encoding="utf-8")
    assert idx == "- [阿尔法](alpha.md)\n- [贝塔说明](beta.md)\n"


def test_rebuild_index_empty_when_no_memories(tmp_path):
    rebuild_index(tmp_path)
    assert (tmp_path / INDEX_FILE).read_text(encoding="utf-8") == ""


def test_rebuild_index_skips_when_dir_absent_and_empty(tmp_path):
    # 目录不存在且无记忆 → 不创建目录
    rebuild_index(tmp_path / "nope")
    assert not (tmp_path / "nope").exists()


def test_read_index_returns_empty_when_missing(tmp_path):
    assert read_index(tmp_path) == ""


def test_read_index_returns_contents(tmp_path):
    (tmp_path / INDEX_FILE).write_text("- [x](x.md)\n", encoding="utf-8")
    assert read_index(tmp_path) == "- [x](x.md)\n"


def test_parse_survives_triple_dash_in_values():
    # name/description 含 --- 子串:行级解析不能误判为 frontmatter 边界
    text = serialize_memory(
        name="prefers---any",
        description="用 any---不用 interface",
        type_="feedback",
        body="正文",
    )
    parsed = parse_memory(text)
    assert parsed is not None
    fm, body = parsed
    assert fm.name == "prefers---any"
    assert fm.description == "用 any---不用 interface"
    assert body == "正文"


# —— frontmatter 解析鲁棒性(F4 多行标量 / F5 BOM)——


def test_parse_survives_newline_triple_dash_in_description():
    # description 含 \n---\n:yaml 发出缩进续行 '  ---',列0 判定不误判为边界
    text = serialize_memory(
        name="x", description="line1\n---\nline2", type_="feedback", body="b"
    )
    parsed = parse_memory(text)
    assert parsed is not None
    fm, body = parsed
    assert fm.description == "line1\n---\nline2"
    assert body == "b"


def test_parse_handles_utf8_bom_prefix():
    # win32 Notepad 另存 UTF-8 BOM:首行 ﻿---,str.strip() 不去 BOM
    text = "﻿---\nname: x\ndescription: d\ntype: user\n---\nbody"
    parsed = parse_memory(text)
    assert parsed is not None
    fm, _ = parsed
    assert fm.name == "x"


def test_list_memories_reads_bom_file(tmp_path):
    (tmp_path / "bom.md").write_text(
        "﻿---\nname: bom\ndescription: 带BOM\ntype: user\n---\n正文",
        encoding="utf-8",
    )
    metas = list_memories(tmp_path)
    assert [m.name for m in metas] == ["bom"]


def test_list_memories_skips_non_utf8_file(tmp_path):
    # 非 UTF-8(win32 GBK/GB18030 等编辑器默认)→ 跳过该文件、不崩、继续解析好的
    (tmp_path / "bad.md").write_bytes(b"\x80\x81 not utf8\n---\nname: bad\n")
    (tmp_path / "good.md").write_text(
        "---\nname: good\ndescription: d\ntype: user\n---\nb", encoding="utf-8"
    )
    metas = list_memories(tmp_path)  # 不抛 UnicodeDecodeError
    assert [m.name for m in metas] == ["good"]
