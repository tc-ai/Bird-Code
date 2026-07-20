# src/birdcode/utils/worktree.py
"""git worktree 支持:检测/解析/创建/清理原语。

Phase 1:多终端并行(--worktree CLI + auto-detect)。git 调用走 asyncio.subprocess
(与 grep_tool._run_rg / bash_tool 一致),_run_git 可 monkeypatch 做单测。
worktree 里 `.git` 是**文件**(内容 `gitdir: <main>/.git/worktrees/<name>`),
主仓库里 `.git` 是**目录**——这是 is_worktree / resolve_main_repo 的判据。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from birdcode.utils.logging import get_logger

log = get_logger("birdcode.utils.worktree")


def worktree_path(main_repo: Path, name: str) -> Path:
    """worktree 目录路径 <main_repo>/.birdcode/worktrees/<name>。

    单一来源:create_worktree(建)、cleanup_subagent_worktree(清)、create_worktree 快速恢复
    (探查)共用。路径方案一变(可配置根/迁址),只改此处,免得孤儿清理指向错路径 →
    remove_worktree 静默 no-op 留孤儿。
    """
    return main_repo / ".birdcode" / "worktrees" / name


def is_worktree(path: Path) -> bool:
    """path 是否是一个 git worktree 的根。

    worktree 的 `.git` 是文件(`gitdir: <main>/<gitdir>/worktrees/<name>`)。
    判 gitdir 路径含 `/worktrees/`(斜杠归一化后)——排除子模块(`/modules/`,无
    `/worktrees/`);兼容 separate-git-dir(`<gitdir>` 可能是 `.real.git` 等,非仅 `.git`)。
    主仓库的 `.git` 是目录(is_file() 即 False)。
    """
    git = path / ".git"
    if not git.is_file():
        return False
    try:
        text = git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    if not text.startswith("gitdir:"):
        return False
    gitdir = text[len("gitdir:") :].strip().replace("\\", "/")
    return "/worktrees/" in gitdir


def worktree_store_name(cwd: str | Path) -> str | None:
    """会话存储的 worktree 分段名(per-worktree 隔离)。

    cwd 是 worktree → 取目录名(`--worktree foo` 的 dir basename=foo,auto-detect 手动
    worktree 同);主仓/非 worktree → None(存主仓目录,不进 worktree/)。
    所有路径调用方(SessionStore/SubagentStore/runner/ui)统一用它从 cwd/ctx.cwd 推导,
    免穿参——会话 jsonl 与子 agent jsonl 落同一 worktree/<name>/ 目录。
    """
    p = Path(cwd)
    return p.name if is_worktree(p) else None


def resolve_main_repo(worktree_path: Path) -> Path | None:
    """从 worktree 根解析主仓库路径。

    读 `.git` 文件里的 `gitdir: <main>/<gitdir>/worktrees/<name>`,取 `<main>`。
    找 gitdir 路径上的 `worktrees` 段(恒为该名),其 parent=<gitdir>、再 parent=主仓——
    不再硬找 `.git` 段(separate-git-dir 的 <gitdir> 可能是 `.real.git` 等)。
    非 worktree/解析失败 → None。
    """
    git = worktree_path / ".git"
    if not git.is_file():
        return None
    try:
        text = git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:") :].strip())
    if not gitdir.is_absolute():
        gitdir = worktree_path / gitdir
    try:
        gitdir = gitdir.resolve()
    except OSError:
        pass
    # gitdir = <main>/<gitdir>/worktrees/[<sub>/]<name>;找 worktrees 段 → 主仓
    for parent in gitdir.parents:
        if parent.name == "worktrees":
            return parent.parent.parent
    return None


async def _run_git(
    args: list[str], *, cwd: Path | None = None, timeout: float = 60.0
) -> tuple[int, str, str]:
    """执行 git 子进程,返回 (returncode, stdout, stderr)。可 monkeypatch(单测)。

    stdin=DEVNULL(防 post-checkout hook / 凭据提示读 stdin → communicate() 死锁)
    + wait_for timeout(防 git 卡死等待 stdin)。超时 → kill 进程,返 (128,"","git 超时")。
    CancelledError(Ctrl+C/ESC 取消)→ kill 进程后透传(对齐 bash_tool:303,免孤儿)。
    凭据防挂三重防御:stdin=DEVNULL(读 stdin 拿 EOF)+ GIT_TERMINAL_PROMPT=0(不尝试终端 prompt)
    + GIT_ASKPASS=''(禁 askpass 助手,免弹 GUI/调外部程序挂起)。env 继承父进程 + 这两项。
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001 - kill 后 wait 失败也无妨,进程已停
            pass
        return (128, "", "git 超时")
    except asyncio.CancelledError:  # 取消 → 杀 git,免孤儿(对齐 bash_tool:303)
        proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        raise  # 透传(BaseException,不吞)
    code = proc.returncode if proc.returncode is not None else 0
    return (
        code,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


def _read_worktree_head_ref(wt: Path) -> str | None:
    """读 worktree 的 HEAD 符号引用分支名(快速恢复用,不调 git,仅读文件)。

    读 <wt>/.git 的 gitdir 指针 → <gitdir>/HEAD。HEAD='ref: refs/heads/<branch>' → <branch>;
    detached(SHA)/读失败/非 worktree → None。
    """
    git = wt / ".git"
    if not git.is_file():
        return None
    try:
        text = git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:") :].strip())
    if not gitdir.is_absolute():
        gitdir = wt / gitdir
    try:
        head = gitdir.joinpath("HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    prefix = "ref: refs/heads/"
    if head.startswith(prefix):
        return head[len(prefix) :].strip()
    return None


async def create_worktree(main_repo: Path, name: str, *, base_ref: str = "origin/HEAD") -> Path:
    """在 <main_repo>/.birdcode/worktrees/<name> 建 worktree,分支 worktree-<name>。

    快速恢复:dir 已存在 + 合法 worktree + 在 worktree-<name> 分支 → 直接复用(跳过 git,
    仅读几个文件)。被锁(stale lock)不影响复用——lock 只防 prune/remove,新会话 lock 幂等、
    退出 unlock。分支不匹配(手动切了/detached/非 worktree)→ 不复用,报状态不符(免进错分支)。
    **不删分支**——分支已存在 → checkout 复用(保留未合并 commit);不存在 → `-b` 新建。
    base_ref 默认 origin/HEAD;失败(无 remote/无 origin/HEAD)回退本地 HEAD。
    超时(rc=128,err="git 超时")【不】回退 HEAD(超时不是缺 ref,回退会 double 挂)。
    """
    if not (main_repo / ".git").exists():
        raise RuntimeError(f"--worktree 需在 git 仓库内使用: {main_repo} 无 .git")
    wt = worktree_path(main_repo, name)
    branch = f"worktree-{name}"
    # 快速恢复:dir 在 + HEAD==worktree-<name> → 复用(不调 git);dir 在但状态不符 → 报错
    if wt.exists():
        if _read_worktree_head_ref(wt) == branch:
            return wt
        raise FileExistsError(
            f"worktree 已存在但状态不符(期望分支 {branch}): {wt}——切回/删除后重试"
        )
    # 分支已存在 → checkout 复用(不 -b);不存在 → -b 新建
    rc_v, _, _ = await _run_git(["rev-parse", "--verify", f"refs/heads/{branch}"], cwd=main_repo)
    if rc_v == 0:
        rc, out, err = await _run_git(["worktree", "add", str(wt), branch], cwd=main_repo)
    else:
        rc, out, err = await _run_git(
            ["worktree", "add", "-b", branch, str(wt), base_ref], cwd=main_repo
        )
        if rc != 0 and base_ref != "HEAD" and "超时" not in err:  # 超时不回退
            log.warning("worktree 基线 %s 失败(%s),回退本地 HEAD", base_ref, err.strip())
            rc, out, err = await _run_git(
                ["worktree", "add", "-b", branch, str(wt), "HEAD"], cwd=main_repo
            )
    if rc != 0:
        raise RuntimeError(f"git worktree add 失败: {err.strip() or out.strip()}")
    return wt


async def list_worktrees(main_repo: Path) -> list[dict[str, str]]:
    """`git worktree list --porcelain` → [{path, branch, head}]。失败 → []。"""
    rc, out, _ = await _run_git(["worktree", "list", "--porcelain"], cwd=main_repo)
    if rc != 0:
        return []
    rows: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur:
                rows.append(cur)
            cur = {"path": line[len("worktree ") :]}
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch ") :]
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD ") :]
        elif line == "" and cur:
            rows.append(cur)
            cur = {}
    if cur:
        rows.append(cur)
    return rows


def _remove_junctions(root: Path) -> None:
    """删 root 下任意深度的 NTFS junction/symlink(按 link 条目删,不跟进目标)。

    Windows 上 git worktree remove 若遇 junction 会跟进删除目标目录内容
    (CC v2.1.205 前的坑);先清 link 条目规避。非 Windows / 无 link → no-op。
    """
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            p = Path(dirpath) / name
            try:
                if p.is_symlink() or p.is_junction():
                    p.unlink()
            except OSError:
                log.debug("删 link 失败: %s", p, exc_info=True)


async def remove_worktree(main_repo: Path, wt: Path, *, force: bool = False) -> None:
    """删 worktree。脏的需 force=True。Windows 先清 junction。

    单 `--force` 仍拒**已锁** worktree(需 `-ff`)→ force 路径首败重试 `-ff` 兜底。
    `-ff` 重试**仅限 force=True**(用户已确认丢弃);force=False(干净路径)首败不 -ff
    ——避免 TOCTOU 脏工作区被静默强删(干净检查与 remove 之间变脏时,force=False 失败
    应 prune+raise → 由 _cleanup_worktree 兜底保留,而非强删)。
    """
    _remove_junctions(wt)
    args = ["worktree", "remove", str(wt)]
    if force:
        args.append("--force")
    rc, _, err = await _run_git(args, cwd=main_repo)
    if rc != 0 and force:
        # force 路径仍可能 locked → -ff 兜底
        rc, _, err = await _run_git(
            ["worktree", "remove", str(wt), "--force", "--force"], cwd=main_repo
        )
    if rc != 0:
        # 元数据残留 → prune 清,再确认目录状态
        await _run_git(["worktree", "prune"], cwd=main_repo)
        if wt.exists():
            raise RuntimeError(f"git worktree remove 失败: {err.strip()}")


async def lock_worktree(main_repo: Path, wt: Path, reason: str) -> None:
    """锁 worktree 防误删(prune/自动清理)。失败(已锁)仅 debug。"""
    rc, _, err = await _run_git(["worktree", "lock", str(wt), "--reason", reason], cwd=main_repo)
    if rc != 0:
        log.debug("worktree lock 失败(可能已锁): %s", err.strip())


async def unlock_worktree(main_repo: Path, wt: Path) -> None:
    rc, _, _ = await _run_git(["worktree", "unlock", str(wt)], cwd=main_repo)
    # 解锁失败(未锁/已删)忽略


async def prune_worktrees(main_repo: Path) -> None:
    await _run_git(["worktree", "prune"], cwd=main_repo)


async def cleanup_subagent_worktree(main_repo: Path, wt: Path, *, succeeded: bool) -> None:
    """子 agent worktree 清理。绝不无脑 branch -D(防丢未合并 commit)。

    - failed → unlock + 留目录 + 留分支(回报主 agent 定夺,不毁活)。
    - succeeded → unlock + remove_worktree(force=False)(clean→删,dirty→RuntimeError→留目录)；
      目录删成(clean working tree)后再 git branch -d(已合并→删,未合并→rc≠0→留分支,commit 不丢)。
    """
    await unlock_worktree(main_repo, wt)  # 运行中 lock 防 prune;清理先解锁
    if not succeeded:
        log.debug("子 agent worktree 失败,保留目录+分支: %s", wt)
        return
    try:
        await remove_worktree(main_repo, wt, force=False)  # clean→删;dirty→RuntimeError
    except RuntimeError:
        log.debug("子 agent worktree 脏,保留目录+分支: %s", wt)
        return
    # 目录已删(clean working tree)。分支:git branch -d(safe,未合并则 rc≠0→留分支)
    branch = f"worktree-{wt.name}"
    rc, _, err = await _run_git(["branch", "-d", branch], cwd=main_repo)
    if rc != 0:
        log.debug("子 agent 分支未合并,保留(不丢 commit): %s (%s)", branch, err.strip())


async def current_head_sha(wt: Path) -> str | None:
    """wt 当前 HEAD 的 sha(git rev-parse HEAD,快速,不启 git log)。失败/空 → None。"""
    rc, out, _ = await _run_git(["rev-parse", "HEAD"], cwd=wt)
    sha = out.strip()
    return sha if rc == 0 and sha else None


def _parse_name_only(out: str) -> list[str]:
    """`git diff --name-only` 输出 → 路径列表(每行一个,无状态码)。"""
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _parse_porcelain(out: str) -> list[str]:
    """`git status --porcelain` 输出 → 路径列表。

    每行 `XY path`(前 2 状态码 + 1 空格);rename/copy `XY orig -> dest` 取 dest。
    带引号路径(含特殊字符)去引号。best-effort,不处理 C 转义。
    """
    result: list[str] = []
    for ln in out.splitlines():
        if len(ln) < 4:  # 至少 'XY ' + 1 字符路径
            continue
        path = ln[3:]
        if " -> " in path:  # rename/copy:取 dest
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path:
            result.append(path)
    return result


async def collect_worktree_changes(
    wt: Path, base_sha: str, *, max_files: int = 20
) -> tuple[list[str], bool]:
    """best-effort 收集 worktree 自 base_sha 起的净改变。

    返回 (artifacts 相对 wt 根的路径, committed)。
    - committed: `git rev-list --count base_sha..HEAD` > 0。
    - artifacts: (`git diff --name-only base_sha..HEAD`) ∪ (`git status --porcelain`) 去重。
    - wt 不存在 / base_sha 空 / rev-list 失败 → best-effort 降级,绝不抛。

    artifacts 超 max_files → 前 max_files 个 + 末尾截断标记行。
    """
    if not wt.exists() or not base_sha:
        return [], False
    # committed:HEAD 领先 base 的 commit 数
    rc, out, _ = await _run_git(["rev-list", "--count", f"{base_sha}..HEAD"], cwd=wt)
    if rc != 0:
        log.warning("collect: rev-list 失败(base=%s, wt=%s)", base_sha, wt)
        return [], False
    committed = out.strip().isdigit() and int(out.strip()) > 0
    files: set[str] = set()
    rc, out, _ = await _run_git(["diff", "--name-only", f"{base_sha}..HEAD"], cwd=wt)
    if rc == 0:
        files.update(_parse_name_only(out))
    rc, out, _ = await _run_git(["status", "--porcelain"], cwd=wt)
    if rc == 0:
        files.update(_parse_porcelain(out))
    artifacts = sorted(files)
    if len(artifacts) > max_files:
        artifacts = artifacts[:max_files] + [f"... 共 {len(artifacts)} 个文件(已截断)"]
    return artifacts, committed
