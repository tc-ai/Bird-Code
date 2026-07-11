# src/birdcode/cli/app.py
"""BirdCode CLI entry (Typer)。加载配置 → 选 profile → 构造 provider → 启动 TUI。"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from birdcode.agent.factory import build_provider
from birdcode.agent.mock_provider import MockProvider
from birdcode.agent.provider import StreamingProvider
from birdcode.agent.system_prompt.instructions import load_project_instructions
from birdcode.config import loader as config_loader
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.memory.extractor import MemoryManager
from birdcode.session import paths as session_paths
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore
from birdcode.ui.app import BirdApp
from birdcode.utils.logging import get_logger
from birdcode.utils.paths import find_project_root

if TYPE_CHECKING:
    from birdcode.tools.registry import ToolRegistry

log = get_logger("birdcode.cli")

app = typer.Typer(add_completion=False, help="BirdCode — terminal AI coding agent.")


class ProfileError(Exception):
    """profile 选择失败（未知名称等）。"""


def _load(path: str | None, *, project_root: Path | None = None) -> AppConfig | None:
    return config_loader.load_config(
        Path(path) if path else None, project_root=project_root
    )


def _resolve_profile(cfg: AppConfig | None, name: str | None) -> ProviderProfile | None:
    """cfg 为 None → None（回退 MockProvider）。name 为 None → default（或首个）。"""
    if cfg is None:
        return None
    target = name or cfg.default
    if target is None:
        target = next(iter(cfg.providers))
    if target not in cfg.providers:
        raise ProfileError(f"未知 profile: {target!r}（可用: {', '.join(cfg.providers)}）")
    return cfg.providers[target]


def _make_provider(
    profile: ProviderProfile | None,
    cfg: AppConfig | None,
    delay: float,
    *,
    registry: ToolRegistry | None = None,
    mcp_instructions: Callable[[], dict[str, str]] | None = None,
) -> StreamingProvider:
    if profile is None:
        return MockProvider(delay=delay)
    assert cfg is not None  # profile 非 None 蕴含 cfg 非 None
    return build_provider(profile, cfg, registry=registry, mcp_instructions=mcp_instructions)


def _resolve_session_id(
    *,
    continue_last: bool,
    resume: str | None,
    project_root: Path,
    root: Path,
    worktree_name: str | None = None,
) -> tuple[str, bool]:
    """决定 sessionId 与是否需要重放历史。返回 (session_id, should_load)。

    纯函数(不启 TUI、不持久化),供直接单测。

    - 两者同给 → ProfileError(互斥,避免歧义)
    - --resume <id> → (id, True);文件不存在 → ProfileError(避免静默开新会话)
    - --continue   → 本项目最近 mtime 的 .jsonl 的 id;无历史 → (新 uuid, False)
    - 都不给      → (新 uuid, False)

    --resume <id> 先做格式校验(仅允许 [A-Za-z0-9-]{1,128}),再检查文件存在性。
    防止 ``../``、``/``、``\\``、空格等逃逸 encoded-cwd 目录的路径穿越(uuid4 是
    hex+dash,合法;新 uuid 也合法)。--continue 走 list_sessions,session_id 来自
    文件 stem,本身安全。
    """
    import re
    import uuid as _uuid

    if continue_last and resume is not None:
        raise ProfileError("--continue 与 --resume 互斥,请二选一。")
    if resume is not None:
        # 严格白名单格式校验,先于文件存在性。避免 ``../`` 等逃逸 encoded-cwd 目录。
        if not re.fullmatch(r"[A-Za-z0-9-]{1,128}", resume):
            raise ProfileError(
                f"--resume: 非法 sessionId {resume!r}(仅允许字母数字与连字符)。"
            )
        jf = session_paths.session_jsonl(root, resume, project_root, worktree_name=worktree_name)
        if not jf.exists():
            raise ProfileError(
                f"--resume: 无此会话 {resume!r}(用 /sessions 查看可用会话)。"
            )
        return resume, True
    if continue_last:
        summaries = SessionStore.list_sessions(root, project_root, worktree_name=worktree_name)
        if summaries:
            # list_sessions 已按 mtime 倒序,首条即最近
            return summaries[0].session_id, True
        return str(_uuid.uuid4()), False
    return str(_uuid.uuid4()), False


def _resolve_project_root(cwd: Path) -> Path:
    """worktree 里 → 主仓(resolve_main_repo);否则 find_project_root。"""
    from birdcode.utils.worktree import is_worktree, resolve_main_repo

    if is_worktree(cwd):
        main = resolve_main_repo(cwd)
        if main is not None:
            return main
    return find_project_root(cwd)


def _apply_worktree_env(cwd: Path) -> None:
    """在 worktree 里 → 设共享 venv env(UV_PROJECT_ENVIRONMENT=<main>/.venv + UV_NO_SYNC=1)。"""
    from birdcode.utils.worktree import is_worktree, resolve_main_repo

    if not is_worktree(cwd):
        return
    main = resolve_main_repo(cwd)
    if main is None:
        return
    os.environ.setdefault("UV_PROJECT_ENVIRONMENT", str(main / ".venv"))
    os.environ.setdefault("UV_NO_SYNC", "1")


async def _cleanup_worktree(main_repo: Path, wt: Path, *, interactive: bool) -> None:
    """owned worktree 退出清理(仅 --worktree 建的调用,见 run_tui owned_wt)。

    - 先 os.chdir(main_repo) 离开 wt —— Windows 下进程占 cwd 则 git worktree remove 失败。
    - unlock 先行:无论删否都解锁(lock 仅防 prune 误删,退出时该解)。
    - 干净 → remove_worktree(非 force);脏 + interactive + isatty → confirm 后 force 删;
      脏 + 非交互/非 TTY → 保留(绝不静默毁活)。
    - 正确修法:**不删分支**——删目录只删工作区,留 worktree-<name> 分支(commit 不丢,
      同名 re-add 由 create_worktree checkout 复用)。要彻底丢弃分支需用户手动 git branch -D。
    - remove 失败(如 locked)由 remove_worktree 内 -ff 重试兜底(仅 force 路径)。
    - 全程 try/except 不杀退出,但 typer.secho 出可见警告(不静默留孤儿)。
    - finally 复原 cwd(测试隔离);生产中 pre_cwd 可能=wt(已删)→留在 main_repo。
    """
    from birdcode.utils.worktree import remove_worktree, unlock_worktree

    pre_cwd = Path.cwd()
    try:
        try:
            os.chdir(main_repo)  # 离开 wt,释放 Windows 目录锁
        except OSError:
            pass
        await unlock_worktree(main_repo, wt)
        # 直接试 clean remove(git worktree remove 自带脏检查),省一次 git status 扫描。
        try:
            await remove_worktree(main_repo, wt)  # 干净→成;脏/locked→RuntimeError
            return
        except RuntimeError:
            pass  # 脏/locked → 落到提示
        if interactive and sys.stdin.isatty():  # isatty 守卫,非 TTY 不弹 confirm
            if typer.confirm(f"worktree {wt} 有改动,删除?(y/N)", default=False):
                await remove_worktree(main_repo, wt, force=True)
        # 脏/locked + 非交互/非 TTY/拒绝 → 保留目录与分支(不毁活)
    except Exception:
        log.warning("worktree 退出清理失败(不杀退出)", exc_info=True)
        typer.secho(
            f"⚠ worktree 清理失败,可能需手动清理: {wt}", fg=typer.colors.YELLOW
        )  # 可见信号,不静默留孤儿
    finally:
        try:  # 复原 cwd(测试隔离);wt 已删时 pre_cwd 不存在 → 留 main_repo
            if pre_cwd.exists():
                os.chdir(pre_cwd)
        except OSError:
            pass


def _build_session_ctx(
    session_id: str, project_root: Path, *, cwd: Path | None = None
) -> SessionContext:
    """构造 SessionContext(读 git 当前分支,失败 None;不抛异常)。

    subprocess 同步调用、超时 2s:启动时一次性读取,延迟可接受;失败降级 None。

    显式 encoding="utf-8", errors="replace"。git 输出 UTF-8(中文分支名等),
    默认 encoding 在 Windows 是 cp936 → 非 ASCII 字节解码乱码、strip 后偶发 None
    或 UnicodeDecodeError。强制 UTF-8 + replace 防御性降级。

    cwd 默认=project_root(向后兼容);worktree 场景传 worktree 目录,
    使 SessionContext.cwd 反映实际工作目录(而非主仓 project_root)。
    git rev-parse 也用 cwd(同参):worktree → 读 worktree 分支,非 main 分支。
    """
    git_branch: str | None = None
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=(cwd if cwd is not None else project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
        if out.returncode == 0:
            git_branch = out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - 任何失败(无 git/超时/非仓库)降级 None
        git_branch = None
    return SessionContext(
        session_id=session_id,
        cwd=str(cwd if cwd is not None else project_root),
        version="0.1.0",
        git_branch=git_branch,
    )


def run_tui(
    profile_name: str | None,
    theme: str,
    delay: float,
    config: str | None,
    continue_last: bool = False,
    resume: str | None = None,
    worktree_name: str | None = None,
) -> None:
    from birdcode.tools.executor import default_registry

    # 先 resolve project_root(worktree→主仓),供 config 加载 + --worktree 块复用。
    # 必须在 _load 前:项目层配置要锚主仓,否则 worktree 里启动漏载 <main>/.birdcode/config.yaml。
    cwd = Path.cwd()
    project_root = _resolve_project_root(cwd)
    cfg = _load(config, project_root=project_root)
    # --worktree:建/进 worktree(可选)。之后 auto-detect 统一处理 env(project_root 已钉主仓)。
    # owned_wt 标记「本进程 --worktree 建的 worktree」——仅它退出时清理(lock/remove);
    # 手动 worktree(无 flag 进的)不 owned,退出不动(防误删用户的工作区)。
    owned_wt: Path | None = None
    if worktree_name is not None:
        import asyncio

        from birdcode.utils.worktree import create_worktree

        # 复用已 resolve 的主仓 project_root(原 _resolve_project_root(Path.cwd()))。
        repo = project_root
        try:
            wt = asyncio.run(create_worktree(repo, worktree_name))
        except (RuntimeError, FileExistsError) as e:  # 目录已存在抛 FileExistsError
            typer.secho(str(e), fg=typer.colors.RED)
            return
        os.chdir(wt)
        owned_wt = wt
        cwd = Path.cwd()  # chdir 后更新(worktree 目录)
    # auto-detect + env(worktree→主仓);project_root 已在 _load 前 resolve(钉主仓,不变)
    _apply_worktree_env(cwd)
    # 项目指令(BirdCode.md):启动时一次性加载并拼装,prepend 进 system prompt(进缓存 block)。
    # 记忆索引不走这里——每轮从 <system-reminder> 现读注入(见 base_llm._inject_reminder),
    # 故提取落盘后后续轮次即见(代价:记忆块不进前缀缓存)。project_root 供 provider 现读记忆。
    proj_instructions = load_project_instructions(project_root)
    if cfg is not None:
        cfg.project_instructions = proj_instructions
        cfg.project_root = project_root
    try:
        if profile_name == "mock":
            profile: ProviderProfile | None = None
        else:
            profile = _resolve_profile(cfg, profile_name or None)
    except ProfileError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        return
    if cfg is None:
        log.info("未找到配置文件，回退 MockProvider（配置 ~/.birdcode/config.yaml）")
    # 注入链:同一 reg 实例贯穿 provider(发 tools=)与 executor(执行工具),
    # 保证「LLM 看到的工具集 = executor 能执行的工具集」绝不漂移。
    reg = default_registry()
    # mcp_instructions 是懒回调:provider 在 stream 时(此时 on_mount 已跑、tui 已建)
    # 才调用,读到已填充的 instructions。用 holder 容器持引用——provider 先于 tui 构造,
    # 直接在 lambda 里引用 tui 局部变量会触发 mypy possibly-undefined;holder 在 lambda
    # 定义前赋值,规避此问题。
    app_holder: list[BirdApp | None] = [None]

    def _mcp_instructions() -> dict[str, str]:
        app = app_holder[0]
        return app.mcp_instructions() if app is not None else {}

    provider = _make_provider(
        profile,
        cfg,
        delay,
        registry=reg,
        mcp_instructions=_mcp_instructions,
    )
    # 记忆提取管理器:仅真实 profile 启用(mock 无 LLM,提取无意义)。复用当前 provider
    # 端点/密钥,模型取 profile.hak_model or profile.model。/profile 运行时切换不即时反映。
    memory_mgr: MemoryManager | None = None
    if profile is not None:
        extract_model = profile.hak_model or profile.model
        memory_mgr = MemoryManager(
            provider, project_root=project_root, extract_model=extract_model
        )
    model_label = profile.name if profile is not None else "mock"

    # worktree 存储分段名(per-worktree 隔离)+ in_worktree 一次算,复用于
    # _resolve_session_id(定位会话目录)+ session_cwd + main_for_wt + enter-lock + exit-cleanup。
    from birdcode.utils.worktree import (
        is_worktree,
        lock_worktree,
        resolve_main_repo,
        worktree_store_name,
    )

    in_worktree = is_worktree(cwd)
    wt_store_name = worktree_store_name(cwd)
    # 会话:决定 sessionId + 是否重放历史。--continue/--resume 的 ProfileError
    # 在此转成红色提示退出(与 profile 错误一致的 UX)。
    try:
        session_id, should_load = _resolve_session_id(
            continue_last=continue_last,
            resume=resume,
            project_root=project_root,
            root=session_paths.default_root(),
            worktree_name=wt_store_name,
        )
    except ProfileError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        return
    # per-worktree 隔离:会话存 <encoded-主仓>/worktree/<name>/(--worktree <name> 或
    # auto-detect 手动 worktree,wt_store_name=目录名);主仓会话存 <encoded-主仓>/。
    # → birdcode -c(主仓)只找主仓会话,不挑 worktree 会话(堵错位);--worktree foo -c 只找 foo。
    # worktree → SessionContext.cwd = worktree 目录(jsonl 元数据 + resume 还原);
    # 非 worktree → cwd=None,_build_session_ctx 默认用 project_root(子目录启动也共享主仓)。
    main_for_wt = resolve_main_repo(cwd) if in_worktree else None
    session_cwd = cwd if in_worktree else None
    ctx = _build_session_ctx(session_id, project_root, cwd=session_cwd)
    store = SessionStore(ctx, project_root)  # worktree_name 从 ctx.cwd 自推导(=wt_store_name)

    # 屏蔽 SIGINT:Bash 子进程(cmd.exe)会破坏 Textual 的 console mode,使 ctrl+C 从
    # 「Textual 捕获为 key=copy」退化为 SIGINT → KeyboardInterrupt → 整个 TUI 退出
    # （触发 Bash 工具后 ctrl+C 直接退出、而非复制的 bug）。Textual 自身不设 SIGINT
    # handler，这里显式忽略：mode 正常时 ctrl+C 仍被 Textual 当 key=copy；mode 被破坏
    # 时 SIGINT 被忽略（不退出）。中断当前流用 Esc（action_interrupt）。
    # try 覆盖 lock_worktree + BirdApp 构造:任一抛异常,finally 仍跑 owned worktree 清理
    # (unlock+remove,留分支),否则 --worktree 新建的 worktree 会卡锁/孤立。
    # 仅 owned_wt(--worktree 建的)才 lock/清理;手动 worktree(in_worktree 但非 owned)不动。
    # lock/清理传 owned_wt(已 resolve 的 owned 路径,非 cwd)——使 owned_wt 被消费,非仅 flag。
    # prior_sigint 移到 try 前(安全读);SIGINT 未及修改时 finally 的 restore 是无害 no-op。
    prior_sigint = signal.getsignal(signal.SIGINT)
    try:
        if owned_wt is not None and main_for_wt is not None:
            import asyncio

            asyncio.run(lock_worktree(main_for_wt, owned_wt, "birdcode session active"))
        tui = BirdApp(
            provider,
            model=model_label,
            theme_name=theme,
            cfg=cfg,
            registry=reg,
            store=store,
            resume=should_load,
            project_instructions=proj_instructions,
            memory=memory_mgr,
        )
        app_holder[0] = tui
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        tui.run()
    finally:
        signal.signal(signal.SIGINT, prior_sigint)
        # 仅 owned worktree 清理(手动 worktree 不动);_cleanup_worktree 内部已 try/except。
        if owned_wt is not None and main_for_wt is not None:
            import asyncio

            asyncio.run(_cleanup_worktree(main_for_wt, owned_wt, interactive=True))
    # 退出回灌：把对话 print 到原生 scrollback
    try:
        transcript = tui.transcript()
        if transcript.strip():
            print(transcript)
    except Exception:
        log.debug("transcript dump failed", exc_info=True)


@app.command()
def main(
    profile_name: str | None = typer.Option(
        None, "--profile", "-p", help="profile 名（留空=用配置 default；mock=离线回退）。"
    ),
    theme: str = typer.Option("dark", "--theme", "-t", help="Theme: dark | light."),
    config: str | None = typer.Option(None, "--config", help="配置文件路径。"),
    delay: float = typer.Option(0.012, "--delay", help="Mock token 间隔(秒)。"),
    continue_last: bool = typer.Option(
        False, "--continue", "-c", help="接续本项目最近一次会话(按 mtime)。"
    ),
    resume: str | None = typer.Option(
        None, "--resume", "-r", help="接续指定 sessionId(用 /sessions 查可用 id)。"
    ),
    worktree: str | None = typer.Option(
        None, "--worktree", "-w", help="在隔离 worktree 里起会话(并行开发)。"
    ),
) -> None:
    """Start the BirdCode interactive terminal."""
    try:
        run_tui(profile_name, theme, delay, config, continue_last, resume, worktree_name=worktree)
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:  # never crash to a traceback
        log.exception("fatal error in TUI")
        typer.secho(f"BirdCode 遇到错误，详见 ~/.birdcode/debug.log: {exc}", fg=typer.colors.RED)


def run() -> None:
    """Console-script + `python -m` entry point."""
    app()


if __name__ == "__main__":
    run()
