# tests/tools/test_fork_cwd.py
import asyncio

from birdcode.tools.base import Tool


def test_default_fork_ignores_cwd():
    """无状态工具默认 fork 返回 self,cwd 形参被忽略(不报错)。"""
    class _Stateless(Tool):
        name = "x"
        async def execute(self, **a): return ""
    t = _Stateless()
    assert t.fork_for_child(cwd="/tmp/wt") is t  # self,cwd 无害


def test_bash_fork_uses_cwd_when_provided():
    """BashTool.fork_for_child(cwd=...) 用传入 cwd 作种子(非父 _cwd 快照)。"""
    from birdcode.tools.bash_tool import BashTool
    parent = BashTool()
    parent._cwd = "/parent"  # type: ignore[SLF001]
    child = parent.fork_for_child(cwd="/wt/sub-abc")
    assert child is not parent
    assert child._cwd == "/wt/sub-abc"  # type: ignore[SLF001]  # 传入优先


def test_bash_fork_falls_back_to_parent_cwd():
    """cwd=None → 快照父 _cwd(旧行为,向后兼容)。"""
    from birdcode.tools.bash_tool import BashTool
    parent = BashTool()
    parent._cwd = "/parent"  # type: ignore[SLF001]
    child = parent.fork_for_child()
    assert child._cwd == "/parent"  # type: ignore[SLF001]


# --- Task 2: 6 文件工具 _cwd(相对路径按 per-agent cwd 解析)---


def test_read_fork_resolves_relative_to_cwd(tmp_path):
    """ReadTool fork(cwd=wt) → 相对路径相对 wt 解析(非进程 cwd)。"""
    from birdcode.tools.read_tool import ReadTool
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "f.txt").write_text("hello-wt", encoding="utf-8")
    tool = ReadTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(file_path="f.txt"))
    assert "hello-wt" in out


def test_write_fork_lands_in_cwd(tmp_path):
    """WriteTool fork(cwd=wt) → 相对路径写进 wt(非进程 cwd)。"""
    from birdcode.tools.write_tool import WriteTool
    wt = tmp_path / "wt"
    wt.mkdir()
    tool = WriteTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(file_path="new.txt", content="x"))
    assert "已写入" in out
    assert (wt / "new.txt").exists()
    assert (wt / "new.txt").read_text(encoding="utf-8") == "x"
    assert not (tmp_path / "new.txt").exists()  # 不落在 wt 的父目录


def test_edit_fork_resolves_relative_to_cwd(tmp_path):
    """EditTool fork(cwd=wt) → 相对路径相对 wt 解析(改 wt 内文件,读回验内容)。"""
    from birdcode.tools.edit_tool import EditTool
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "f.txt").write_text("old-text", encoding="utf-8")
    tool = EditTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(
        tool.execute(file_path="f.txt", old_string="old-text", new_string="new-text")
    )
    assert "已编辑" in out
    assert (wt / "f.txt").read_text(encoding="utf-8") == "new-text"


def test_delete_fork_resolves_relative_to_cwd(tmp_path):
    """DeleteTool fork(cwd=wt) → 相对路径相对 wt 解析(删 wt 内文件,非进程 cwd)。"""
    from birdcode.tools.delete_tool import DeleteTool
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "f.txt").write_text("x", encoding="utf-8")
    tool = DeleteTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(file_path="f.txt"))
    assert "已删除" in out
    assert not (wt / "f.txt").exists()


def test_glob_fork_uses_cwd_not_process(tmp_path):
    """GlobTool fork(cwd=wt) → pattern 相对 wt(非 Path.cwd())。"""
    from birdcode.tools.glob_tool import GlobTool
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "a.py").write_text("x", encoding="utf-8")
    tool = GlobTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(pattern="*.py"))
    assert "a.py" in out


def test_glob_fork_explicit_relative_path_resolves_to_cwd(tmp_path):
    """C2 回归:glob 显式相对 path(如 "sub")相对 _cwd(worktree)解析,而非进程 cwd。

    其它 5 个文件工具已按 _cwd 解析相对路径;glob 修复前显式相对 path 会留在相对形态,
    root.glob() 再相对进程 cwd 解析 → worktree 子 agent 串进主仓同名子目录。
    修复后:相对 root 先拼到 _cwd 下 → 命中 worktree/sub 内文件。
    """
    from birdcode.tools.glob_tool import GlobTool
    wt = tmp_path / "wt"
    sub = wt / "sub"
    sub.mkdir(parents=True)
    (sub / "in_wt.py").write_text("x", encoding="utf-8")
    # 进程 cwd 下的同名 sub(若 glob 误相对进程 cwd,会命中这里的串扰文件)
    proc_sub = tmp_path / "sub"
    proc_sub.mkdir()
    (proc_sub / "leak.py").write_text("x", encoding="utf-8")
    tool = GlobTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(pattern="*.py", path="sub"))
    assert "in_wt.py" in out
    assert "leak.py" not in out  # 没串进进程 cwd 的 sub


def test_grep_fork_passes_cwd_to_rg(tmp_path):
    """GrepTool fork(cwd=wt) → rg 子进程 cwd=wt(命中 wt 内文件)。"""
    from birdcode.tools.grep_tool import GrepTool
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "a.py").write_text("needle", encoding="utf-8")
    tool = GrepTool().fork_for_child(cwd=str(wt))
    out = asyncio.run(tool.execute(pattern="needle", output_mode="files_with_matches"))
    assert "a.py" in out


def test_filetools_no_arg_instantiation_backcompat():
    """无参实例化仍可用(cwd 默认 None → os.getcwd()),向后兼容(executor 注册路径)。"""
    import os

    from birdcode.tools.delete_tool import DeleteTool
    from birdcode.tools.edit_tool import EditTool
    from birdcode.tools.glob_tool import GlobTool
    from birdcode.tools.grep_tool import GrepTool
    from birdcode.tools.read_tool import ReadTool
    from birdcode.tools.write_tool import WriteTool
    for cls in (ReadTool, WriteTool, EditTool, DeleteTool, GlobTool, GrepTool):
        t = cls()
        assert t._cwd == os.getcwd()  # type: ignore[SLF001]
