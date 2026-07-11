# src/birdcode/agent/system_prompt/env.py
"""环境采集:稳定环境(进缓存 block)+ 动态环境(进 <system-reminder>)。

- gather_stable_env():会话级稳定(platform/os/shell/home/cwd)。Phase 2 起按 resolved cwd
  缓存(_stable_env_cached @lru_cache)——主仓/worktree 各一条,各自字节一致、利于前缀缓存。
- gather_dynamic_env():每轮动态(日期 + git branch/dirty),git 失败静默降级。
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as _dt
import functools
import os
import platform as _platform
import sys
from pathlib import Path

_GIT_TIMEOUT = 2.0  # 秒;communicate 超时即降级,防 hung git 阻塞 agent 轮

# 模块级 ContextVar:仅由异步 worktree 子 agent 在 SubagentRunner.run() 内 set
# (自己 asyncio task,context 经 create_task 拷贝,set 不回泄父 task——故无 sync-inline
#  污染问题;sync 子 agent 不 set,继承父 None→Path.cwd())。主 agent 永不 set → Path.cwd()(兼容)。
_AGENT_CWD: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "birdcode_agent_cwd", default=None
)


def _resolve_cwd(cwd: str | None) -> str:
    """cwd 优先级:显式实参 > _AGENT_CWD contextvar > Path.cwd()。"""
    return cwd or _AGENT_CWD.get() or str(Path.cwd())


@functools.lru_cache(maxsize=32)
def _stable_env_cached(cwd: str) -> str:
    """按 cwd 键的稳定环境(Phase 2:主仓/worktree 各一条,不串)。

    已知张力(冻结的代价):工作目录整会话冻结。BashTool 维护自己可移动的
    _cwd(`bash_tool._maybe_update_cwd`),模型在 bash 里 `cd <dir>` 后,后续 bash 实际跑在
    新目录,而本行仍显示会话起始目录——可能误导相对路径判断。缓解:工具使用准则要求一律用
    绝对路径,且 bash 输出会反映真实 cwd。彻底修(把 cwd 移到动态块、或读取 bash._cwd)
    需跨层耦合且触及缓存敏感层,暂按"会话级冻结"接受,未来若加运行时 /cd 再连同 cache_clear 处理。
    """
    lines = [
        "环境",
        f"平台: {sys.platform}",
        f"操作系统: {_platform.platform()}",
        f"默认 shell: {_default_shell()}",
        f"主目录: {Path.home()}",
        f"工作目录: {cwd}",
    ]
    return "\n".join(lines)


def gather_stable_env(cwd: str | None = None) -> str:
    """会话级稳定环境。按 resolved cwd 缓存(主仓/worktree 各一条)→ 各自字节一致、利于前缀缓存。

    cwd 优先级:显式实参 > _AGENT_CWD contextvar(worktree 子 agent set)> Path.cwd()(主 agent)。
    """
    return _stable_env_cached(_resolve_cwd(cwd))


def _default_shell() -> str:
    if sys.platform == "win32":
        return "bash"  # 项目约定:Git Bash / MSYS
    return os.environ.get("SHELL", "sh")


async def gather_dynamic_env(
    *, registry: object | None = None, memory: str = "", cwd: str | None = None
) -> str:
    """每轮动态环境:日期 + git + (可选)延迟工具清单 + (可选)记忆索引。

    cwd 透传 _git_branch_dirty(同优先级:显式实参 > _AGENT_CWD > Path.cwd())。
    registry 给到时附上 deferred_reminder(每轮随发现状态变;位于缓存前缀之外,不破缓存)。
    memory 由调用方(base_llm._inject_reminder)每轮现读磁盘填入,故记忆更新及时可见。

    调用频率:经 base_llm._inject_reminder → build_system_reminder,每次 stream() 即每个
    agent 工具轮回跑一次(且在重试循环之外、LLM 调用之前同步执行),故 N 轮任务 ≈ N 次 git
    子进程(小仓库约 10-50ms,2s 超时兜底)。多数调用字节完全相同——branch/dirty 只在写类
    工具执行后才变。记忆扫描比 git 子进程还轻(几个小 .md),按"每轮现读"接受(保证准确性)。
    """
    date = _dt.date.today().isoformat()
    git = await _git_branch_dirty(cwd)
    parts = [f"日期: {date}", f"Git: {git}"]
    if registry is not None:
        reminder = getattr(registry, "deferred_reminder", lambda: "")()
        if reminder:
            parts.append(reminder)
    if memory:
        parts.append(memory)
    return "\n".join(parts)


async def _git_branch_dirty(cwd: str | None = None) -> str:
    resolved = _resolve_cwd(cwd)
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            resolved,
            "status",
            "--porcelain",
            "--branch",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return "非 git 仓库"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return "非 git 仓库"
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        return "非 git 仓库"
    out = stdout.decode("utf-8", errors="replace")
    lines = out.splitlines()
    branch = "unknown"
    if lines and lines[0].startswith("## "):
        head = lines[0][3:]
        if head.startswith("No commits yet on "):
            # "## No commits yet on main" → "main"(空仓库,尚无提交)
            branch = head.rsplit(" ", 1)[-1]
        elif head.startswith("HEAD (no branch)"):
            branch = "HEAD (detached)"
        else:
            # "## main...origin/main [ahead 1]" / "## main" → "main"
            branch = head.split("...")[0].split(" ")[0]
    dirty = any(ln.strip() and not ln.startswith("##") for ln in lines)
    return f"{branch}({'有未提交改动' if dirty else 'clean'})"
