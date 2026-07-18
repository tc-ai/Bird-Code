import shutil
from pathlib import Path

import pytest

from birdcode.tools.base import ToolOutput
from birdcode.tools.grep_tool import GrepTool

_HAS_RG = shutil.which("rg") is not None


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("FOO = 2\nbar = 3\n", encoding="utf-8")
    (tmp_path / "c.md").write_text("# FOO heading\n", encoding="utf-8")
    return tmp_path


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_files_with_matches_default(sample_tree: Path) -> None:
    out = await GrepTool().execute(pattern="foo", path=str(sample_tree))
    # 默认 output_mode=files_with_matches → 每行一个文件路径
    files = sorted(Path(ln).name for ln in out.splitlines() if ln.strip())
    assert "a.py" in files


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_content_mode_with_line_numbers(sample_tree: Path) -> None:
    out = await GrepTool().execute(
        pattern="FOO",
        path=str(sample_tree),
        output_mode="content",
        line_number=True,
        glob="*.py",
    )
    assert "b.py" in out and "FOO" in out  # content 模式带文件名+行


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_ignore_case(sample_tree: Path) -> None:
    out = await GrepTool().execute(
        pattern="foo",
        path=str(sample_tree),
        ignore_case=True,
        output_mode="files_with_matches",
    )
    files = [ln for ln in out.splitlines() if ln.strip()]
    # foo 既命中 a.py(foo),ignore_case 也命中 b.py(FOO)、c.md(FOO)
    assert len(files) >= 2


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_count_mode(sample_tree: Path) -> None:
    out = await GrepTool().execute(
        pattern="FOO", path=str(sample_tree), output_mode="count", ignore_case=True
    )
    # count 模式:每文件一行「路径:次数」
    assert ":" in out  # 至少一处「file:count」


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
async def test_grep_no_match_returns_error_hint(sample_tree: Path) -> None:
    out = await GrepTool().execute(pattern="zzzznotfound", path=str(sample_tree))
    assert "<error>" in out and "无匹配" in out
    assert "<hint>" in out


async def test_grep_rg_missing_returns_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """rg 未安装 → 返回安装提示词(不抛异常)。通过 monkeypatch _which_rg 模拟。

    本用例不依赖真实 rg(无论机器装没装都跑):_which_rg 返回 None → 走「未安装」分支。
    """

    def _which_none() -> str | None:
        return None

    monkeypatch.setattr("birdcode.tools.grep_tool._which_rg", _which_none)
    out = await GrepTool().execute(pattern="x", path=".")
    assert "<error>" in out and "ripgrep 未安装" in out
    assert "<hint>" in out and "安装" in out


async def test_grep_no_match_via_exitcode1(monkeypatch: pytest.MonkeyPatch) -> None:
    """rg 退出码 1 = 无匹配(正常,非错误)→ 返回无匹配提示词。模拟,不依赖真实 rg。"""

    def _which_ok() -> str | None:
        return "/fake/rg"

    async def _fake_run(args, *, cwd):  # type: ignore[no-untyped-def]
        return 1, "", ""  # exit 1, 空输出

    monkeypatch.setattr("birdcode.tools.grep_tool._which_rg", _which_ok)
    monkeypatch.setattr(GrepTool, "_run_rg", staticmethod(_fake_run))
    out = await GrepTool().execute(pattern="x", path=".")
    assert "<error>" in out and "无匹配" in out
    assert "<hint>" in out


async def test_grep_rg_self_error_via_exitcode2(monkeypatch: pytest.MonkeyPatch) -> None:
    """rg 退出码 2 = rg 自身错误(正则非法/路径不存在)→ 返回错误提示词。模拟。"""

    def _which_ok() -> str | None:
        return "/fake/rg"

    async def _fake_run(args, *, cwd):  # type: ignore[no-untyped-def]
        return 2, "", "regex error: unbalanced paren"

    monkeypatch.setattr("birdcode.tools.grep_tool._which_rg", _which_ok)
    monkeypatch.setattr(GrepTool, "_run_rg", staticmethod(_fake_run))
    out = await GrepTool().execute(pattern="(", path=".")
    assert "<error>" in out and "exit 2" in out
    assert "<hint>" in out and "glob" in out


async def test_grep_head_limit_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    """超 head_limit 截断并提示。模拟 rg 返回大量行。"""

    def _which_ok() -> str | None:
        return "/fake/rg"

    async def _fake_run(args, *, cwd):  # type: ignore[no-untyped-def]
        return 0, "\n".join(f"file{i}.py:match" for i in range(300)), ""

    monkeypatch.setattr("birdcode.tools.grep_tool._which_rg", _which_ok)
    monkeypatch.setattr(GrepTool, "_run_rg", staticmethod(_fake_run))
    out = await GrepTool().execute(pattern="match", path=".", head_limit=10)
    # 超阈 → ToolOutput(text=摘要, full=全量):摘要给 LLM,全量供 executor 落盘 sidecar
    assert isinstance(out, ToolOutput)
    assert "已截断" in out.text
    assert "共 300 行" in out.text
    assert "file299.py" in out.full  # full 含全部 300 行(尾部不丢)
    assert out.full.count("\n") == 299


def test_grep_description_and_name() -> None:
    d = GrepTool.description
    assert d  # 非空一句话
    assert "正则" in d or "搜索" in d  # 核心作用
    assert GrepTool.name == "grep"
