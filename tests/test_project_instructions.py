"""项目指令(BirdCode.md)加载与 @include 安全护栏的测试。

覆盖:基础加载、项目级>用户级优先级排序、@include 递归解析(相对目录/带空格/缺失)、
环路防护(A→B→A、自包含)、菱形去重、路径穿越拦截(../ 与绝对路径)、用户级边界限 ~/.birdcode、
嵌套深度上限、fenced code block 围栏感知、单文件体积上限、build_system_text 拼装行为与缓存稳定性。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.agent.system_prompt import build_system_text
from birdcode.agent.system_prompt.instructions import (
    MAX_INCLUDE_DEPTH,
    load_project_instructions,
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    """把 Path.home 指向临时目录,隔离用户级 ~/.birdcode。"""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", lambda: h)
    return h


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# —— 基础加载与优先级 ——


def test_no_files_returns_empty(tmp_path, home):
    assert load_project_instructions(tmp_path) == ""


def test_loads_project_level(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "用 4 空格缩进。")
    out = load_project_instructions(tmp_path)
    assert "用 4 空格缩进。" in out
    assert "项目级" in out


def test_loads_user_level(tmp_path, home):
    _write(home / ".birdcode" / "BirdCode.md", "全局用中文回复。")
    out = load_project_instructions(tmp_path)
    assert "全局用中文回复。" in out
    assert "用户级" in out


def test_project_level_before_user_level(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "PROJECT_MARKER")
    _write(home / ".birdcode" / "BirdCode.md", "USER_MARKER")
    out = load_project_instructions(tmp_path)
    assert out.index("PROJECT_MARKER") < out.index("USER_MARKER")


def test_empty_file_emits_no_block(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "   \n  \n")
    _write(home / ".birdcode" / "BirdCode.md", "USER_ONLY")
    out = load_project_instructions(tmp_path)
    assert "项目级" not in out
    assert "USER_ONLY" in out


# —— @include 解析 ——


def test_include_basic(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "头部\n@include rules.md\n尾部")
    _write(tmp_path / "rules.md", "规则A")
    out = load_project_instructions(tmp_path)
    assert "头部" in out
    assert "规则A" in out
    assert "尾部" in out
    assert out.index("头部") < out.index("规则A") < out.index("尾部")


def test_include_relative_to_including_file_dir(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "@include sub/a.md")
    _write(tmp_path / "sub" / "a.md", "A\n@include b.md")  # 相对 sub/ 解析
    _write(tmp_path / "sub" / "b.md", "DEEP")
    out = load_project_instructions(tmp_path)
    assert "DEEP" in out


def test_include_quoted_path_with_spaces(tmp_path, home):
    _write(tmp_path / "BirdCode.md", '@include "my rules.md"')
    _write(tmp_path / "my rules.md", "QUOTED_OK")
    out = load_project_instructions(tmp_path)
    assert "QUOTED_OK" in out


def test_include_missing_file_marker(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "@include nope.md")
    out = load_project_instructions(tmp_path)
    assert "未找到" in out


def test_include_directory_auto_finds_birdcode_md(tmp_path, home):
    # @include <目录> → 自动找 <目录>/BirdCode.md 并引入
    (tmp_path / "subdir").mkdir()
    _write(tmp_path / "subdir" / "BirdCode.md", "FROM_SUBDIR")
    _write(tmp_path / "BirdCode.md", "@include subdir")
    out = load_project_instructions(tmp_path)
    assert "FROM_SUBDIR" in out


def test_include_directory_without_birdcode_md_gives_up(tmp_path, home):
    # @include <目录> 但目录下无 BirdCode.md → 放弃(留未找到标记,不崩)
    (tmp_path / "emptydir").mkdir()
    _write(tmp_path / "BirdCode.md", "@include emptydir\n正文")
    out = load_project_instructions(tmp_path)
    assert "正文" in out
    assert "未找到" in out  # emptydir/BirdCode.md 不存在
    assert "emptydir" in out  # 标记里带出解析后的路径,便于排查


# —— 安全护栏 ——


def test_cycle_a_b_a(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "@include a.md")
    _write(tmp_path / "a.md", "A\n@include b.md")
    _write(tmp_path / "b.md", "B\n@include a.md")
    out = load_project_instructions(tmp_path)  # 不应死循环
    assert "A" in out and "B" in out
    assert "循环" in out or "重复" in out


def test_self_include_is_cycle(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "X\n@include BirdCode.md")
    out = load_project_instructions(tmp_path)
    assert out.count("X") == 1  # 只展开一次


def test_diamond_dedup(tmp_path, home):
    _write(tmp_path / "BirdCode.md", "@include b.md\n@include c.md")
    _write(tmp_path / "b.md", "B\n@include d.md")
    _write(tmp_path / "c.md", "C\n@include d.md")
    _write(tmp_path / "d.md", "D_CONTENT")
    out = load_project_instructions(tmp_path)
    assert out.count("D_CONTENT") == 1


def test_traversal_escape_rejected(tmp_path, home):
    secret = tmp_path.parent / "secret.md"
    _write(secret, "TOP_SECRET")
    _write(tmp_path / "BirdCode.md", "@include ../secret.md")
    out = load_project_instructions(tmp_path)
    assert "TOP_SECRET" not in out
    assert "越界" in out


def test_absolute_path_escape_rejected(tmp_path, home):
    outside = tmp_path.parent / "outside_abs.md"
    _write(outside, "ABS_SECRET")
    _write(tmp_path / "BirdCode.md", f"@include {outside}")
    out = load_project_instructions(tmp_path)
    assert "ABS_SECRET" not in out
    assert "越界" in out


def test_user_level_confined_to_birdcode_dir(tmp_path, home):
    # 用户级 include 不能逃出 ~/.birdcode
    _write(home / "Documents" / "rules.md", "HOME_DOC")
    _write(home / ".birdcode" / "BirdCode.md", "@include ../Documents/rules.md")
    out = load_project_instructions(tmp_path)
    assert "HOME_DOC" not in out
    assert "越界" in out


def test_depth_cap(tmp_path, home):
    # 构造一条深度超过 MAX_INCLUDE_DEPTH 的链,尾部应被跳过
    _write(tmp_path / "BirdCode.md", "L0\n@include d1.md")
    for i in range(1, MAX_INCLUDE_DEPTH + 1):  # d1..d5,各自 include 下一层
        _write(tmp_path / f"d{i}.md", f"L{i}\n@include d{i + 1}.md")
    _write(tmp_path / f"d{MAX_INCLUDE_DEPTH + 1}.md", "TOO_DEEP")
    out = load_project_instructions(tmp_path)
    assert "TOO_DEEP" not in out
    assert "嵌套过深" in out


def test_fence_awareness(tmp_path, home):
    body = "正文\n```text\n@include notreal.md\n```\n结尾"
    _write(tmp_path / "BirdCode.md", body)
    out = load_project_instructions(tmp_path)
    # 围栏内的 @include 不应被展开(notreal.md 不存在,若展开会出现"未找到")
    assert "未找到" not in out
    assert "@include notreal.md" in out  # 原样保留


def test_per_file_size_cap(tmp_path, home, monkeypatch):
    from birdcode.agent.system_prompt import instructions as mod

    monkeypatch.setattr(mod, "MAX_FILE_BYTES", 64)
    _write(tmp_path / "BirdCode.md", "@include big.md")
    _write(tmp_path / "big.md", "X" * 200)
    out = load_project_instructions(tmp_path)
    assert "单文件超限" in out


# —— build_system_text 集成 ——


def test_build_system_text_prepends_project_instructions():
    t = build_system_text(project_instructions="MY_RULES")
    assert t.startswith("MY_RULES")
    assert "① 身份" in t  # 七模块仍在
    assert t.index("MY_RULES") < t.index("① 身份")


def test_build_system_text_override_bypasses_instructions():
    t = build_system_text(override="ONLY", project_instructions="MY_RULES")
    assert t == "ONLY"


def test_build_system_text_empty_instructions_unchanged():
    assert build_system_text() == build_system_text(project_instructions="")
    assert build_system_text() == build_system_text(project_instructions=None)


def test_build_system_text_cache_stability():
    a = build_system_text(project_instructions="X", mcp_instructions={"s": "instr"})
    b = build_system_text(project_instructions="X", mcp_instructions={"s": "instr"})
    assert a == b
