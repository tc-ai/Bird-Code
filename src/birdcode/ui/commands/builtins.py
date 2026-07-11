"""10 个内置命令的注册。handler 只依赖 CommandContext,不绑 Textual。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from birdcode.ui.commands.command import Command, CommandContext, CommandType
from birdcode.ui.commands.registry import CommandRegistry


def _help_handler_factory(registry: CommandRegistry):
    async def _help(ctx: CommandContext, _args: str) -> None:
        cmds = registry.names()  # 排除 hidden、按名排序
        lines = ["可用命令:"]
        for name in cmds:
            c = registry.resolve(name)
            lines.append(f"  {c.usage:<22} {c.description}")
        lines.append("")
        lines.append("输入 /<命令> 执行;Tab 可补全命令名。")
        await ctx.show_message("\n".join(lines))

    return _help


async def _exit(ctx: CommandContext, _args: str) -> None:
    ctx.exit_app()


async def _server(ctx: CommandContext, _args: str) -> None:
    mgr = ctx.mcp_manager
    instructions = mgr.instructions if mgr is not None else {}
    if not instructions:
        await ctx.show_message("无已连接的 MCP server。")
        return
    lines = [f"MCP Server(已连接 {len(instructions)}):"]
    for srv, ins in instructions.items():
        lines.append(f"\n### {srv}")
        stripped = ins.strip() if isinstance(ins, str) else ""
        lines.append(stripped or "(server 未提供 instructions)")
    await ctx.show_message("\n".join(lines))


async def _tool(ctx: CommandContext, _args: str) -> None:
    reg = ctx.tools
    if reg is None:
        await ctx.show_message("工具注册表未就绪。")
        return
    tools = reg.to_anthropic_tools()
    if not tools:
        await ctx.show_message("无可用工具。")
        return
    lines = [f"当前可用工具({len(tools)}):"]
    for t in tools:
        name = str(t["name"])
        desc_raw = t.get("description")
        desc = ""
        if isinstance(desc_raw, str) and desc_raw.strip():
            desc = desc_raw.splitlines()[0].strip()
        lines.append(f"- {name} — {desc}" if desc else f"- {name}")
    await ctx.show_message("\n".join(lines))


async def _sessions(ctx: CommandContext, _args: str) -> None:
    from birdcode.session.store import SessionStore

    store = ctx.store
    if store is None:
        await ctx.show_message("(会话管理未启用)")
        return
    summaries = SessionStore.list_sessions(
        store._root,
        store.project_root,  # 主仓(worktree 会话按 worktree_name 分到 worktree/<name>/)
        worktree_name=store.worktree_name,
    )
    if not summaries:
        await ctx.show_message("(无历史会话)")
        return
    import datetime

    lines = [f"本项目历史会话({len(summaries)}):"]
    for s in summaries:
        when = datetime.datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M")
        summary = s.first_user_summary or "(无文本)"
        lines.append(f"{s.session_id}  {summary}  {when}  {s.turn_count} 轮")
    lines.append("用 birdcode --resume <id> 接续(运行中切换本轮不支持)。")
    await ctx.show_message("\n".join(lines))


async def _context(ctx: CommandContext, _args: str) -> None:
    from birdcode.agent.context import estimate_context_usage
    from birdcode.agent.system_prompt import build_system_text
    from birdcode.ui.pure import format_context_report
    from birdcode.utils.logging import get_logger

    try:
        cfg = ctx.cfg
        window = cfg.context_window if cfg is not None else 200_000
        threshold = cfg.autocompact_threshold if cfg is not None else 167_000
        mcp_instructions = {}
        mgr = ctx.mcp_manager
        if mgr is not None:
            mcp_instructions = mgr.instructions
        system_text = build_system_text(
            override=(cfg.system_prompt or None) if cfg is not None else None,
            mcp_instructions=mcp_instructions,
            project_instructions=cfg.project_instructions if cfg is not None else "",
        )
        registry = ctx.tools
        profile = cfg.providers.get(ctx.model) if cfg is not None else None
        protocol = profile.protocol if profile is not None else "anthropic"
        if registry is not None:
            tools = (
                registry.to_anthropic_tools()
                if protocol == "anthropic"
                else registry.to_openai_tools()
            )

            def _tool_name(t: dict[str, object]) -> str:
                name = t.get("name")
                if isinstance(name, str):
                    return name
                fn = t.get("function")
                if isinstance(fn, dict):
                    inner = fn.get("name")
                    if isinstance(inner, str):
                        return inner
                return ""

            mcp_tools = [t for t in tools if _tool_name(t).startswith("mcp__")]
        else:
            tools, mcp_tools = [], []
        history = ctx.controller.history if ctx.controller is not None else []
        last_measured = None
        for turn in reversed(history):
            u = turn.usage
            if u is not None and u.input_tokens:
                last_measured = u.input_tokens
                break
        usage = estimate_context_usage(
            window=window,
            autocompact_threshold=threshold,
            system_text=system_text,
            tools=tools,
            mcp_tools=mcp_tools,
            history=history,
            last_measured=last_measured,
        )
        model_name = profile.model if profile is not None else ctx.model
        await ctx.show_message(format_context_report(usage, model_name))
    except Exception:
        get_logger("birdcode.tui").exception("/context 失败")
        await ctx.show_message("/context 失败，详见 debug.log", kind="error")


async def _compact(ctx: CommandContext, _args: str) -> None:
    res, status = await ctx.compact_now()
    if status == "busy":
        await ctx.show_message("/compact:agent 正在运行,请等待或 Esc 中断后再压缩", kind="warn")
        return
    if status == "no_context":
        await ctx.show_message("/compact 需启用会话持久化(当前无 store/配置)", kind="warn")
        return
    if status == "empty" or res is None:
        await ctx.show_message("/compact:无可压缩内容(会话为空?)")
        return
    fallback = "(硬截断兜底)" if res.fell_back else ""
    msg = f"已压缩({res.trigger}):{res.pre_tokens}→{res.post_tokens} token".rstrip()
    if fallback:
        msg = f"{msg} {fallback}"
    await ctx.show_message(msg)


async def _clear(ctx: CommandContext, _args: str) -> None:
    new_sid = await ctx.clear_conversation()
    if new_sid is not None:
        await ctx.show_message(f"(已开新会话 {new_sid[:8]},旧会话保留)")


async def _profile(ctx: CommandContext, args: str) -> None:
    # 取首 token(恢复旧行为):/profile gpt-4 extra → 切 gpt-4,extra 不影响。
    parts = args.split()
    name = parts[0] if parts else ""
    if not name:
        await ctx.show_message("用法: /profile <name>", kind="warn")
        return
    ok = await ctx.switch_profile(name)
    if not ok:
        cfg = ctx.cfg
        avail = ", ".join(cfg.providers) if cfg is not None else "(no config)"
        # 不在文本里自带 ⚠ —— show_message 会对 warn/error 统一加前缀,否则渲染双图标。
        await ctx.show_message(f"未知 profile: {name}（可用: {avail}）", kind="warn")


DIFF_CAP = 100_000  # 字符;超出截断
_GIT_DIFF_TIMEOUT = 10.0  # 秒;git diff 应远快于此,超时 kill 防孤儿子进程
_REVIEW_PROMPT = """请审查以下未提交的代码改动（git diff HEAD），重点关注：
1) 正确性 bug 与边界情况；
2) 可简化 / 复用 / 提升效率之处；
3) 性能与安全隐患。
给出具体、可操作的修改建议，引用 文件:行号。

```diff
{diff}
```"""


async def run_git_diff_head(cwd: Path) -> str:
    """跑 `git diff HEAD`，返回 stdout（合并的未提交改动）。失败/超时抛异常。

    防护镜像 _git_branch_dirty:wait_for 超时 + CancelledError 都 kill+wait 子进程,
    避免孤儿 git 进程;超时抛 RuntimeError(由 /review 捕获提示,而非误报"无改动")。
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "HEAD",
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"git 不可用: {e}") from e
    try:
        stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=_GIT_DIFF_TIMEOUT)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("git diff HEAD 超时") from None
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"git diff HEAD 退出码 {proc.returncode}")
    return stdout.decode("utf-8", errors="replace")


async def _review(ctx: CommandContext, _args: str) -> None:
    from birdcode.utils.logging import get_logger
    from birdcode.utils.paths import find_project_root

    try:
        diff = await run_git_diff_head(find_project_root())
    except Exception:
        get_logger("birdcode.tui").exception("/review 取 diff 失败")
        await ctx.show_message("/review 获取 diff 失败，详见 debug.log", kind="error")
        return
    if not diff.strip():
        await ctx.show_message("无未提交改动（git diff HEAD 为空）", kind="warn")
        return
    if len(diff) > DIFF_CAP:
        diff = diff[:DIFF_CAP] + "\n…(diff 过长，已截断，完整改动请分批 review)"
    await ctx.send_user_message(_REVIEW_PROMPT.format(diff=diff))


def build_builtin_registry() -> CommandRegistry:
    r = CommandRegistry()
    # help 闭包持有 r(局部变量),运行时(已填满)才读 names()
    r.register(
        Command(
            name="/help",
            description="显示命令列表",
            usage="/help",
            type=CommandType.LOCAL,
            handler=_help_handler_factory(r),
            aliases=("/?",),
        )
    )
    r.register(
        Command(
            name="/exit",
            description="退出 BirdCode",
            usage="/exit",
            type=CommandType.LOCAL,
            handler=_exit,
            aliases=("/quit",),
        )
    )
    r.register(
        Command(
            name="/server",
            description="显示已连接 MCP server",
            usage="/server",
            type=CommandType.LOCAL,
            handler=_server,
        )
    )
    r.register(
        Command(
            name="/tool",
            description="列出当前可用工具",
            usage="/tool",
            type=CommandType.LOCAL,
            handler=_tool,
        )
    )
    r.register(
        Command(
            name="/sessions",
            description="列出本项目历史会话(只读)",
            usage="/sessions",
            type=CommandType.LOCAL,
            handler=_sessions,
            aliases=("/session",),
        )
    )
    r.register(
        Command(
            name="/context",
            description="上下文占用快照",
            usage="/context",
            type=CommandType.LOCAL,
            handler=_context,
        )
    )
    r.register(
        Command(
            name="/compact",
            description="手动压缩上下文",
            usage="/compact",
            type=CommandType.STATE,
            handler=_compact,
        )
    )
    r.register(
        Command(
            name="/clear",
            description="清空对话并开新会话",
            usage="/clear",
            type=CommandType.STATE,
            handler=_clear,
        )
    )
    r.register(
        Command(
            name="/profile",
            description="切换 provider profile",
            usage="/profile <name>",
            type=CommandType.STATE,
            handler=_profile,
        )
    )
    r.register(
        Command(
            name="/review",
            description="审查未提交改动(git diff HEAD)",
            usage="/review",
            type=CommandType.PROMPT,
            handler=_review,
        )
    )
    return r
