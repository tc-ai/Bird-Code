# tests/permission/test_worktree_gate.py
"""Phase 2 §6.7:fork_worktree_async gate —— worktree 异步子 agent 的权限门。

L1-L4 复用父(skill/permission 锚主仓 D7/D8),L5 恒 approve(无 HITL),
sandbox_root=worktree(L2 锁 worktree,主仓路径 L2 硬拒)。对比 fork_async
(无 worktree 异步子 agent:L5 对写 reject 怕毁主仓),worktree 子 agent 写被
L2 限定在 worktree 内,故写工具可自动批——L2 兜底取代 fork_async 的拒写。
"""

from __future__ import annotations

from pathlib import Path

from birdcode.permission.gate import UiPermissionGate
from birdcode.tools.bash_tool import BashTool
from birdcode.tools.glob_tool import GlobTool
from birdcode.tools.write_tool import WriteTool


def _make_gate(main_repo: Path) -> UiPermissionGate:
    """父 gate:project_root=sandbox_root=main(主仓)。

    父 HITL(_perm)恒 reject——用于对比:worktree gate 应恒 approve,不调父回调。
    """

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    return UiPermissionGate(
        get_mode=_mode,  # type: ignore[arg-type]
        request_permission=_perm,  # type: ignore[arg-type]
        project_root=main_repo,
        sandbox_root=main_repo,
    )


async def test_worktree_gate_writes_inside_worktree_auto_approve(tmp_path: Path) -> None:
    """worktree gate:写 worktree 内文件 → L5 恒 approve(无 HITL,父 _perm 拒也不影响)。"""
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v = await g.check(WriteTool(), {"file_path": str(wt / "f.txt"), "content": "x"})
    assert v.action == "allow"  # L5 恒 approve(父 _perm 拒也不影响)
    assert v.layer == "L5"  # 走到 L5 always-approve,非 L4 bypass


async def test_worktree_gate_main_repo_path_rejected_at_l2(tmp_path: Path) -> None:
    """worktree gate:写主仓路径 → L2 拒(sandbox_root=worktree,主仓不在 roots)。"""
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v = await g.check(WriteTool(), {"file_path": str(main / "f.txt"), "content": "x"})
    assert v.action == "deny"
    assert v.layer == "L2"  # 主仓路径在 L2 被硬拦(不到 L5)


async def test_worktree_gate_bash_auto_approve_but_l1_still_holds(tmp_path: Path) -> None:
    """worktree gate:bash → L5 approve;但 L1 黑名单仍拦(rm -rf /)。"""
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v_ok = await g.check(BashTool(), {"command": "ls"})
    assert v_ok.action == "allow"
    assert v_ok.layer == "L5"  # bash 自动批(L1 黑名单未命中 → L5 approve)
    v_bad = await g.check(BashTool(), {"command": "rm -rf /"})
    assert v_bad.action == "deny" and v_bad.layer == "L1"  # L1 黑名单仍硬拦


# ---------------- glob pattern '..' 逃逸(F3 回归)----------------


async def test_worktree_gate_glob_parent_pattern_denied_at_l2(tmp_path: Path) -> None:
    """glob(pattern='../**', path=None)在 worktree 沙箱 → L2 拒(读隔离:不枚举主仓/上级)。

    F3 回归:_extract_subject 只取 file_path/path,不读 pattern → path=None 时 L2 整段跳过
    → worktree 子 agent 可 glob('../**') 枚举主仓。补审 pattern 的 .. 逃逸(仅 sandbox 约束时)。
    """
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v = await g.check(GlobTool(), {"pattern": "../**/*"})
    assert v.action == "deny"
    assert v.layer == "L2"  # pattern 含 .. → 解析到 worktree 外 → L2 硬拦


async def test_worktree_gate_glob_normal_pattern_not_denied(tmp_path: Path) -> None:
    """glob(pattern='**/*.py')无 .. → 不触发 pattern 补审 → 读默认 allow(不误伤正常 glob)。"""
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v = await g.check(GlobTool(), {"pattern": "**/*.py"})
    assert v.action == "allow"  # 无 .. → 不补审 → 读默认 allow


async def test_main_agent_glob_parent_pattern_not_denied(tmp_path: Path) -> None:
    """主 agent(sandbox_root=None):glob('../**') 不补审 → 读默认 allow(零回归)。

    补审仅对 worktree 沙箱(_sandbox_root 非空)生效;主 agent 读类 read-default 本就允许越界读。
    """
    main = tmp_path / "main"
    main.mkdir()

    async def _mode() -> str:
        return "default"

    async def _perm(_t: str, _s: str, _p: str | None) -> str:
        return "reject"

    g = UiPermissionGate(  # sandbox_root 缺省 None(主 agent,非 worktree)
        get_mode=_mode,  # type: ignore[arg-type]
        request_permission=_perm,  # type: ignore[arg-type]
        project_root=main,
    )
    v = await g.check(GlobTool(), {"pattern": "../**/*"})
    assert v.action == "allow"  # 主 agent 不补审 pattern → 读默认 allow


async def test_worktree_gate_glob_parent_pattern_with_explicit_path_denied(tmp_path: Path) -> None:
    """glob(path='.', pattern='../../../**')在 worktree 沙箱 → L2 拒(F3 完整版)。

    回归复审(04a546c):原补审 guard 是 `path_str is None`,但传显式 path('.')时 path_str
    非 None → 补审整段跳过;而上面 L2 只校验 path 自身(解析进 worktree 内即放行)→ worktree
    子 agent 用沙箱内 path + 足量 .. 的 pattern 仍可枚举主仓。修后去 path_str is None,对任意
    path 补审 pattern。path='sub' 同理(基=sandbox/sub,pattern 的 .. 仍逃逸)。
    """
    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    parent = _make_gate(main)
    g = parent.fork_worktree_async(sandbox_root=wt)
    v = await g.check(GlobTool(), {"pattern": "../../../**/*", "path": "."})
    assert v.action == "deny"
    assert v.layer == "L2"
    # 沙箱内子目录 + 上跳 pattern 也应拦(基=sandbox/sub,../../ 仍逃出 worktree)
    v2 = await g.check(GlobTool(), {"pattern": "../../**/*", "path": "sub"})
    assert v2.action == "deny"
    assert v2.layer == "L2"
