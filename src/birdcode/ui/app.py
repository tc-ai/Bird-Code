# src/birdcode/ui/app.py
"""BirdApp：全屏 Textual App，集成 provider/controller/widgets。

布局：Screen 下两个兄弟节点——上方 #scroll（banner + 各轮 Turn，可滚动），底部固定
composer（Separator/InputArea/Separator/StatusBar）。输入区不在滚动区里，屏位恒定，
终端光标(IME 锚点)不随滚动/流式漂移（Claude Code 式：上滚历史、底固定输入）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Markdown, Static, TextArea

from birdcode import __version__
from birdcode.agent.provider import (
    CompactionEnd,
    CompactionStart,
    Done,
    EditDiff,
    Error,
    Interrupted,
    ProviderEvent,
    StreamingProvider,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCallStart,
    ToolResult,
    TurnStart,
)
from birdcode.blocks import TextBlock
from birdcode.conversation import Message, TurnController
from birdcode.permission.gate import ModalResult, Mode
from birdcode.ui.icons import select_icons
from birdcode.ui.pure import parse_command
from birdcode.ui.theme import get_theme, supports_unicode
from birdcode.ui.widgets.input_area import InputArea
from birdcode.ui.widgets.separator import Separator
from birdcode.ui.widgets.status_bar import StatusBar
from birdcode.ui.widgets.thinking_block import ThinkingBlock
from birdcode.ui.widgets.tool_line import ToolLine
from birdcode.ui.widgets.transcript_scroll import TranscriptScroll
from birdcode.ui.widgets.turn import Turn
from birdcode.utils.logging import get_logger

if TYPE_CHECKING:
    from textual.widgets._markdown import MarkdownStream

    from birdcode.agent.context import CompactionResult
    from birdcode.agents.manager import SubagentManager
    from birdcode.agents.report import SubagentProgress
    from birdcode.agents.runner import ParentProvider
    from birdcode.agents.teammate import TeamManager
    from birdcode.config.schema import AppConfig
    from birdcode.conversation import Turn as ConversationTurn
    from birdcode.mcp.client import McpManager
    from birdcode.memory.extractor import MemoryManager
    from birdcode.memory.governance import GovernanceManager
    from birdcode.permission.gate import UiPermissionGate
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore
    from birdcode.tools.executor import ToolExecutor
    from birdcode.tools.registry import ToolRegistry
    from birdcode.ui.commands.command import Command
    from birdcode.ui.commands.registry import CommandRegistry
    from birdcode.ui.widgets.choice_prompt import ChoicePrompt
    from birdcode.ui.widgets.completion_menu import CompletionMenu
    from birdcode.ui.widgets.permission_prompt import PermissionPrompt
    from birdcode.ui.widgets.subagent_card import SubagentCard

log = get_logger("birdcode.tui")

# 顶端 banner 左侧的小鸟 ASCII 图(鸣鸟:冠羽 ▟▛ + 尾翼 ▛▘ + 右喙 >),契合项目名 BirdCode。
# 三行图,右侧对齐版本/模型/路径(仿 Claude Code)。_BIRD_ASCII 为无 Unicode 终端的回退。
_BIRD_UNICODE = ("  ▟▛██▙>", " ▟█████▛▘", "  ▘▘ ▝▝")
_BIRD_ASCII = ("  (o>", "  //\\", "  ^^")


def _read_sidechain_final_text(path: Path) -> str | None:
    """读侧链 jsonl 最后一条 assistant 文本块(子 agent 报告正文)。缺/坏 → None。"""
    from birdcode.session import codec

    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return None
    if not rows:
        return None
    # decode 成 Turn 后,从最后一个 turn 倒序找首条非空 assistant 文本。
    for turn in reversed(codec.decode_lines(rows)):
        for msg in reversed(turn.messages):
            if msg.role != "assistant":
                continue
            for b in msg.content:
                if isinstance(b, TextBlock) and b.text:
                    return b.text
    return None


def _reconstruct_subagent_notification(
    store: SessionStore, agent_id: str, status: str
) -> str | None:
    """crash 恢复:从侧链 sub-<id>.jsonl 重读最终报告 + meta 重构 <task-notification>。

    侧链在 enqueue 之前已写完关闭(runner.run 返回早于 manager._inject 写 enqueue),
    故 crash 卡在 enqueue↔user 行之间时侧链必然完整——是重构通知的权威来源,取代旧
    enqueue.content 冗余。侧链缺失 → 退化通知(仅 id+status,无正文),使该 enqueue 仍能
    被补投成 user 行进而被消费/dequeue,避免悬挂。
    """
    from birdcode.agents.definition import AgentDefinition
    from birdcode.agents.report import build_task_notification
    from birdcode.agents.runner import SubagentReport
    from birdcode.session import paths
    from birdcode.session.subagent_meta import read_subagent_meta

    root, project_root, session_id = store.root, store.project_root, store.ctx.session_id
    wt = store.worktree_name  # 侧链在 worktree/<name>/ 下
    meta = read_subagent_meta(
        paths.subagent_meta_path(root, session_id, project_root, agent_id, worktree_name=wt)
    )
    defn = AgentDefinition(
        name=meta.agent_type if meta else "subagent",
        description=meta.description if meta else "子 agent",
        system_prompt="",
    )
    report_text = _read_sidechain_final_text(
        paths.subagent_jsonl_path(root, session_id, project_root, agent_id, worktree_name=wt)
    )
    if report_text is None:
        report_text = "(子 agent 报告正文因侧链缺失未能恢复)"
    return build_task_notification(
        SubagentReport(
            is_completed=status == "completed",
            text=report_text,
            status=cast(Literal["completed", "error", "cancelled"], status),
            duration_ms=0,
            tokens=0,
            tool_use_count=0,
        ),
        defn,
        agent_id=agent_id,
        prompt="",
    )


class BirdApp(App[None]):
    CSS = """
    Screen { background: $background; color: $text; }
    #scroll { height: 1fr; scrollbar-gutter: stable; }
    #banner { color: #e6e6e6; margin: 0 0 1 0; }
    #composer { height: auto; }
    Screen.light { background: #f6f8fa; color: #1f2328; }
    .light #banner { color: #707088; }
    .light StatusBar { color: #707088; }
    .light ToolLine { color: #1f2328; }
    .light Separator { color: #d0d7de; }
    .light Turn .user { color: #2e5cd6; }
    .light ThinkingBlock { color: #9a6700; }
    """

    # 中断/退出改由 Esc / Ctrl+Q 负责；Ctrl+C、Ctrl+V 放行给 Textual 内建的
    # 复制/粘贴（TextArea.action_copy / screen.copy_text，走 OSC 52）。
    BINDINGS = [
        Binding("escape", "interrupt", priority=True),
        Binding("ctrl+q", "quit", show=False),
        # shift+tab 循环切换模式(plan→default→accept-edits→bypass);priority 抢占
        # 焦点逆向导航(Textual 默认 shift+tab=focus_previous)。
        Binding("shift+tab", "cycle_mode", show=False, priority=True),
        # shift+c 复制:非 priority → 输入框聚焦时 TextArea 先插入 'C'(action_copy 见输入框
        # 聚焦即 no-op),不抢键;焦点在 transcript 等处时才真正复制(走 OSC 52)。
        Binding("shift+c", "copy", show=False),
    ]

    model = reactive("mock")
    mode = reactive("default")
    cwd = reactive("birdcode")
    usage: reactive[TokenUsage | None] = reactive(None)
    queue_size = reactive(0)
    busy = reactive(False)

    def __init__(
        self,
        provider: StreamingProvider,
        *,
        model: str = "mock",
        theme_name: str = "dark",
        cfg: AppConfig | None = None,
        registry: ToolRegistry | None = None,
        store: SessionStore | None = None,
        resume: bool = False,
        project_instructions: str = "",
        memory: MemoryManager | None = None,
        governance: GovernanceManager | None = None,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._cfg = cfg
        # 会话持久化(None=未启用,如旧路径/无 store 测试);on_mount 注入 executor sink +
        # controller store + gate extra_roots;resume=True 时 on_mount 重放历史。
        self._store = store
        self._resume = resume
        # 用 _tool_registry 避免与 Textual App._registry(WeakSet[DOMNode])命名冲突。
        self._tool_registry = registry
        # McpManager 在 on_mount 中构造 + startup;先置 None 保证 provider getter 与
        # on_unmount 在任何路径（cfg=None / startup 抛异常）下都能安全读到此属性。
        self._mcp_manager: McpManager | None = None
        # 非阻塞启动持有的后台 task;on_unmount 先 cancel+await 它再 shutdown,
        # 避免 startup 与 shutdown 并发改 _sessions/_server_tasks。
        self._mcp_startup_task: asyncio.Task[None] | None = None
        # on_mount 把 executor/gate 存 self,/clear 后能重接 sink 与 extra_roots。
        # 否则 /clear 只更新 _store + controller._store,executor._output_sink 与
        # gate._extra_roots 仍指旧 session → 大输出落到旧 session、新 session 的
        # tool-results 在 gate extra_roots 外被 L2 拦(read_file 取不到落盘)。
        self._executor: ToolExecutor | None = None
        self._gate: UiPermissionGate | None = None
        # BirdCode.md 拼装结果(启动时一次性加载,会话内不变→利于 system 前缀缓存)。
        # /context 用此副本(无需经 cfg);真实 provider 走 cfg.project_instructions。
        self._project_instructions = project_instructions
        self._memory = memory
        self._governance = governance
        if command_registry is None:
            from birdcode.ui.commands.builtins import build_builtin_registry

            command_registry = build_builtin_registry()
        self._command_registry = command_registry
        self._completion_menu: CompletionMenu | None = None
        self._permission_prompt: PermissionPrompt | None = None
        self._choice_prompt: ChoicePrompt | None = None
        self.model = model
        self._theme = get_theme(theme_name)
        self._unicode = supports_unicode()
        self._icons = select_icons(unicode_supported=self._unicode)
        self.cwd = str(Path.cwd())  # 绝对路径(banner 显示完整路径,不再 shorten)
        self._md: Markdown | None = None
        self._md_stream: MarkdownStream | None = None
        self._tools: dict[str, ToolLine] = {}
        self._diff_pending: dict[str, ToolLine] = {}  # stage4:ToolResult→EditDiff 之间暂存 ToolLine
        # 自动压缩的橙色转圈行(CompactionStart 挂、CompactionEnd/Interrupted 收)。
        self._compaction_line: ToolLine | None = None
        # 同步子 agent 进度卡:agent_id→卡(首次 progress mount,后续 update;Done 统一清理)
        self._subagent_cards: dict[str, SubagentCard] = {}
        self._thinking: ThinkingBlock | None = None
        self._thinking_round = 0  # 多轮 agent_loop 的思考轮次计数(第 N 轮标注用)
        self._current_turn: Turn | None = None
        self._controller: TurnController | None = None
        # 异步子 agent 生命周期管理器(Phase2):on_mount 构造并注入 AgentTool;
        # None=未启用(无 cfg/store 路径)。Esc→cancel_all;/clear→rebind 到新会话。
        self._subagent_mgr: SubagentManager | None = None
        # agent teams(P5):on_mount 构造 + 注册 team 工具;Esc→cancel_all;/clear→reset。
        self._team_mgr: TeamManager | None = None
        self._follow_timer: Timer | None = None  # 生成期间的「钉底」定时器
        self._bg_tasks: set[asyncio.Task[None]] = set()  # 持有后台轮次任务引用 + 回收异常

    @property
    def controller(self) -> TurnController:
        assert self._controller is not None
        return self._controller

    @property
    def store(self) -> SessionStore | None:
        return self._store

    @property
    def tools(self) -> ToolRegistry | None:
        return self._tool_registry

    @property
    def mcp_manager(self) -> McpManager | None:
        return self._mcp_manager

    @property
    def cfg(self) -> AppConfig | None:
        return self._cfg

    @property
    def team_mgr(self) -> TeamManager | None:
        return self._team_mgr

    def list_teammates(self) -> list[dict[str, str]]:
        """列当前 team 的 teammate(name/agent_id/status/description),/agents 视图用。

        status 优先读 SubagentMeta(idle/running/completed/...);
        meta 缺则按 task.done() 推 done/unknown。
        """
        if self._team_mgr is None or self._store is None or self._cfg is None:
            return []
        from birdcode.session import paths
        from birdcode.session.subagent_meta import read_subagent_meta
        from birdcode.utils.worktree import worktree_store_name

        wt = worktree_store_name(self._store.ctx.cwd)
        project_root = self._cfg.project_root or Path(".")
        info: list[dict[str, str]] = []
        for name, h in self._team_mgr.teammates():
            mp = paths.subagent_meta_path(
                self._store.root, self._store.ctx.session_id, project_root, h.agent_id,
                worktree_name=wt,
            )
            m = read_subagent_meta(mp)
            info.append({
                "name": name,
                "agent_id": h.agent_id,
                "status": m.status if m else ("done" if h.done() else "unknown"),
                "description": m.description if m else "",
            })
        return info

    def read_teammate_transcript(self, agent_id: str) -> list[ConversationTurn]:
        """读某 teammate 的 sidechain transcript → list[Turn](/agents 看对话用)。"""
        if self._store is None or self._cfg is None:
            return []
        from birdcode.session import paths
        from birdcode.session.subagent_store import read_subagent_transcript
        from birdcode.utils.worktree import worktree_store_name

        wt = worktree_store_name(self._store.ctx.cwd)
        project_root = self._cfg.project_root or Path(".")
        p = paths.subagent_jsonl_path(
            self._store.root, self._store.ctx.session_id, project_root, agent_id,
            worktree_name=wt,
        )
        return read_subagent_transcript(p)

    def get_token_usage(self) -> TokenUsage | None:
        return self.usage

    def refresh_status(self) -> None:
        """CommandContext 协议入口:委托既有 _refresh_status(刷新状态行)。"""
        self._refresh_status()

    def exit_app(self) -> None:
        self.exit()

    def mcp_instructions(self) -> dict[str, str]:
        """MCP server instructions（供 provider 懒读取；on_mount 之后才有值）。

        作为 ``mcp_instructions`` 回调透传给 build_provider——provider 在 stream 时
        （此时 on_mount 已跑完）才调用，读到已填充的 instructions。无 MCP 时返回空 dict。
        """
        mgr = self._mcp_manager
        return mgr.instructions if mgr is not None else {}

    async def _start_mcp_background(self) -> None:
        """后台跑 MCP startup;绝不杀主循环。

        on_unmount 退出时会 cancel 本 task——CancelledError 是 BaseException、不被此处
        except Exception 接,正常向外传播,由 on_unmount 的 await 吞掉。各 server 的资源
        回滚由 _server_lifecycle 的 finally 保证(见 client.py)。
        """
        assert self._mcp_manager is not None
        try:
            await self._mcp_manager.startup()
        except Exception:  # noqa: BLE001 - MCP 启动失败绝不杀主循环
            log.exception("MCP 启动失败,继续运行(无 MCP)")

    def compose(self) -> ComposeResult:
        # 输入区(composer)固定在底部、不在滚动区里 → 其屏位恒定，终端光标(IME
        # 锚点)不再随滚动/流式漂移；#scroll 只放 banner 与各轮对话（Claude Code 式：
        # 上方滚动历史、底部固定输入）。
        with TranscriptScroll(id="scroll"):
            yield Static(self._build_banner(), id="banner", markup=False)
        with Container(id="composer"):
            yield Separator(char="─" if self._unicode else "-")
            yield InputArea(id="input")
            yield Separator(char="─" if self._unicode else "-")
            yield StatusBar(id="status")

    async def on_mount(self) -> None:
        from birdcode.permission.gate import UiPermissionGate
        from birdcode.tools.executor import ToolExecutor, default_registry

        # 兜底:__init__ 未传 registry(mock 启动路径)时,默认注册表也存起来——
        # 保证 /profile 切到真实 provider 时 self._tool_registry 非 None,可带 tools。
        if self._tool_registry is None:
            self._tool_registry = default_registry()
        # 一次性迁移:旧 sqlite 权限表(~/.birdcode/permissions.db)→ YAML。
        # 必须在构造 gate(读 YAML)之前跑,使迁移出的规则本轮即生效。无 db 则 no-op。
        from birdcode.permission.migrate import migrate_sqlite_to_yaml

        migrate_sqlite_to_yaml()
        # 注入权限 gate(L1-L5 决策流;yaml_paths/project_root 用 gate 默认值)。
        # mode 默认 default:逐次 HITL 确认(更安全的首启体验);可 /mode 或 shift+tab 切档。
        # 双轨外存:沙箱额外放行当前 session 的 tool-results 目录(在 ~/.birdcode
        # 项目外,L2 默认拒),让 LLM 能 read_file 取落盘的全量输出。
        extra_roots = list(self._cfg.extra_roots) if self._cfg is not None else []
        if self._store is not None:
            extra_roots.append(self._store.tool_results_dir)
        from birdcode.utils.worktree import is_worktree, resolve_main_repo

        _cwd = Path.cwd()  # CLI 已 chdir 到 worktree(若有);否则项目根
        _sandbox = _cwd if is_worktree(_cwd) else None
        gate = UiPermissionGate(
            get_mode=self._get_mode,
            request_permission=self._request_permission,
            extra_roots=extra_roots or None,
            project_root=(
                self._cfg.project_root if self._cfg is not None
                else (resolve_main_repo(_cwd) if is_worktree(_cwd) else None)
            ),
            sandbox_root=_sandbox,
        )
        # executor/gate 存 self,/clear 后能重接 sink 与 extra_roots(指向新 session)。
        # executor 注入 store 的 output_sink:超阈输出落盘 → 给 LLM 占位(三轨分离)。
        output_sink = self._store.as_output_sink() if self._store is not None else None
        self._executor = ToolExecutor(
            self._tool_registry,
            permission_gate=gate,
            output_sink=output_sink,
        )
        self._gate = gate
        # read_history 读当前会话 jsonl:注入 store 的 jsonl 路径(只读当前会话,不读其它会话)。
        # /clear 切到新 session 后在 _wire_read_history 里重接(同 update_output_sink 模式)。
        self._wire_read_history()
        # 上下文管理(Phase 1 安全网):有 store + cfg 才启用,否则 None(旧路径/无 store 测试)。
        # ContextManager 每轮 stream 前评估压缩(超阈→摘要)+ 413 应急硬截断;/compact 手动。
        from birdcode.agent.context import ContextManager

        context = (
            ContextManager(self._provider, self._store, self._cfg)
            if (self._store is not None and self._cfg is not None)
            else None
        )
        self._controller = TurnController(
            self._provider,
            on_event=self._on_event,
            on_status=self._on_status,
            executor=self._executor,
            app=self,
            store=self._store,
            context=context,
            memory=self._memory,
            governance=self._governance,
        )
        # MCP:构造 manager + 启动(隔离 per-server 失败)+ 注册 tool_search 常驻工具。
        # 外层 try 兜底任何意外异常(理论上 startup 内已隔离,但防御性双保险):
        # MCP 启动失败绝不杀 on_mount——继续运行(无 MCP),tool_search 仍注册(空发现)。
        from birdcode.mcp.client import McpManager
        from birdcode.mcp.tool_search import ToolSearchTool

        # 重挂载保护:若已有旧 manager(程序化重用/Textual 重新 mount),先收尾它——
        # cancel 旧 startup task + shutdown 旧 manager 的 lifecycle tasks(关传输/子进程),
        # 再建新 manager。否则旧 _server_tasks 成孤儿(传输/子进程泄漏)。
        old_mgr = self._mcp_manager
        old_task = self._mcp_startup_task
        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await old_task  # 检索异常(与 on_unmount 无条件 await 一致)
            except BaseException:  # noqa: BLE001
                pass
        if old_mgr is not None:
            try:
                await old_mgr.shutdown()
            except Exception:  # noqa: BLE001 - 重挂载清理失败不杀 on_mount
                log.debug("旧 MCP manager shutdown 失败", exc_info=True)
        self._mcp_manager = McpManager(
            (self._cfg.mcp_servers if self._cfg is not None else {}),
            registry=self._tool_registry,
        )
        # 非阻塞启动:后台连 MCP,on_mount 立即返回、UI 先就绪。代价:连完前的头一两轮
        # 缺 MCP 工具/instructions(连完后下轮自动出现,用户已接受)。失败仅 log,绝不杀主循环。
        self._mcp_startup_task = asyncio.create_task(self._start_mcp_background())
        # ask_user / tool_search 先于 capability 注册:让 register_skill_tools 的守卫看到
        # 它们——否则名为 ask_user/tool_search 的 skill 会绕过守卫、随后被核心工具静默覆盖,
        # 造成「斜杠命令指向 skill、模型工具却是核心工具」的不一致。二者不依赖 cfg/store。
        from birdcode.tools.ask_user import AskUserTool

        self._tool_registry.register(AskUserTool(app=self))
        self._tool_registry.register(ToolSearchTool(registry=self._tool_registry))
        # 子 agent(Agent 工具):需 provider/registry/gate/cfg/store/project_root 全就绪。
        # 注册在 stable 段(to_*_tools 缓存前缀),先于 MCP 异步注册,无竞态。
        # cfg/store 为 Optional(__init__ 测试/mock 路径可空):缺一则跳过注册——AgentTool
        # 强依赖 cfg.providers/store.ctx,无之无法派生;与 ContextManager 同守卫(上方 if)。
        if self._cfg is not None and self._store is not None:
            from birdcode.agents.builtins import build_capability_registry
            from birdcode.agents.manager import SubagentManager
            from birdcode.agents.registry import CapabilityConflictError
            from birdcode.tools.agent_tools import register_agent_tools, register_skill_tools
            from birdcode.ui.commands.registry import CommandConflictError
            from birdcode.ui.commands.skill_commands import register_skill_commands

            # 异步子 agent manager(Phase2):fork-skill 与 agent 共用。
            self._subagent_mgr = SubagentManager(store=self._store, controller=self.controller)
            # agent teams(P5):构造 team_mgr + 注册 team 工具到 lead registry(lead + teammate
            # 经 build_child_registry fork→self 共用;gate_exempt 免 async gate 拦)。
            from birdcode.agents.teammate import TeamManager as _TeamManager
            from birdcode.tools.send_message import SendMessageTool
            from birdcode.tools.task_tools import (
                TaskClaimTool,
                TaskCreateTool,
                TaskListTool,
                TaskUpdateTool,
            )

            self._team_mgr = _TeamManager()
            # 注册 lead 为 recipient:teammate→lead 通道(SendMessage(to="lead") 路由 +
            # teammate 完成通知 _on_done→_recipients["lead"])。lead 的 deliver =
            # controller.receive(格式化 peer 消息入队、空闲时后台起 drain)。全程同一
            # TurnController 实例(/clear 只重绑 _store),reset() 据此保留 lead 注册。
            self._team_mgr.register("lead", self.controller.receive)
            for _team_tool in (
                SendMessageTool(self._team_mgr),
                TaskCreateTool(self._team_mgr),
                TaskListTool(self._team_mgr),
                TaskUpdateTool(self._team_mgr),
                TaskClaimTool(self._team_mgr),
            ):
                self._tool_registry.register(_team_tool)
            # 每个 defn(agent 或 skill)→ 一个 tool;skill 另注册斜杠命令。
            common = dict(
                cfg=self._cfg,
                app=self,
                ctx=self._store.ctx,
                project_root=self._cfg.project_root or Path("."),
                parent_provider=cast("ParentProvider", self._provider),
                parent_registry=self._tool_registry,
                parent_gate=self._gate,
                spawn_depth=0,
                progress_cb=self._on_subagent_progress,
                subagent_mgr=self._subagent_mgr,
            )
            # skill/agent 名冲突(与核心工具 / 内置命令 / 彼此)→ 友好降级:记 warning + 跳过
            # 本次能力注册,App 照常启动(绝不抛 traceback 杀 on_mount)。冲突名已在异常
            # 信息里;fail-fast 语义保留——本次不加载能力,逼用户改名重启。
            try:
                caps = build_capability_registry(self._cfg.project_root or Path("."))
                register_agent_tools(self._tool_registry, caps, **common)
                skill_tools = register_skill_tools(self._tool_registry, caps, **common)
                register_skill_commands(self._command_registry, caps, skill_tools)
                # SpawnTeammate(P5):用 general-purpose defn;caps 无则跳过(友好降级)。
                gp_defn = next((d for d in caps.all() if d.name == "general-purpose"), None)
                if gp_defn is not None and self._team_mgr is not None:
                    from birdcode.tools.spawn_teammate_tool import SpawnTeammateTool

                    self._tool_registry.register(
                        SpawnTeammateTool(
                            team_mgr=self._team_mgr, defn=gp_defn,
                            cfg=self._cfg, app=self, ctx=self._store.ctx,
                            project_root=self._cfg.project_root or Path("."),
                            parent_provider=cast("ParentProvider", self._provider),
                            parent_registry=self._tool_registry,
                            parent_gate=self._gate, progress_cb=self._on_subagent_progress,
                        )
                    )
            except (CapabilityConflictError, CommandConflictError) as e:
                log.warning("能力(skill/agent)因名称冲突未注册:%s(请改名后重启)", e)
        # resume:加载主线填 controller.history + UI 静默重放(仅文本,工具行进待办)。
        # 失败不杀 on_mount——降级为空会话继续(用户可手动 /sessions 重新挑)。
        if self._store is not None and self._resume:
            try:
                turns = await self._controller.resume()
                if turns:
                    await self._replay_history(turns)
                # Phase2 resume 接线:补投 crash 在 enqueue↔user 行之间的通知 +
                # 重注入已投递未响应的通知(触发 wake)。best-effort,失败由外层 except 兜。
                await self._resume_pending_notifications()
            except Exception:
                log.debug("resume 重放失败,继续空会话", exc_info=True)
        self._apply_theme()
        self.query_one(InputArea).focus()
        self._refresh_status()

    async def _resume_pending_notifications(self) -> None:
        """resume:补投 crash 在 enqueue↔user 行之间的通知;重注入已投递未响应的(触发 wake)。

        两轴(与 codec 辅助对应,正交):
        - 投递补全(find_pending_notifications):enqueue 无匹配 user 行(且无 dequeue/remove)
          → 从子 agent 侧链 sub-<id>.jsonl 重读最终报告 + meta 重构 <task-notification>
          user 行写回(侧链在 enqueue 前已落盘,是权威来源;enqueue 行不再冗余存 content)。
        - 重注入未响应(find_unresponded_notifications):已投递 user 行但无 dequeue → 唤醒
          主 agent 经 _process_wake 消费(实装)。补投后重读 rows——新写回的 user 行同样
          无 dequeue,会落入未响应轴被唤醒,与统一运行时 wake 路径一致。

        best-effort:无 store/controller 时直接 return;单条补投异常只 log 不杀 resume
        (外层 on_mount try 已兜整体异常,此为细粒度隔离)。
        """
        if self._store is None or self._controller is None:
            return
        from birdcode.session.codec import (
            find_pending_notifications,
            find_unresponded_notifications,
        )

        rows = self._store.mainline_rows()
        # 1) 投递补全:enqueue 无 user 行 → 从侧链重构 <task-notification> user 行写回。
        for obj in find_pending_notifications(rows):
            aid = obj.get("agentId") or ""
            status = obj.get("status") or "completed"
            if not aid:
                continue
            notification = _reconstruct_subagent_notification(self._store, aid, status)
            if notification is None:
                continue
            try:
                await self._store.append(
                    Message(role="user", content=[TextBlock(text=notification)]),
                    is_task_notification=True,
                    agent_id=aid,
                )
            except Exception:
                log.debug("resume 补投通知失败 aid=%s", aid, exc_info=True)
        # 2) 重注入未响应 → 唤醒(重读 rows:补投的 user 行也已纳入)。
        if find_unresponded_notifications(self._store.mainline_rows()):
            self._controller.notify_wake()

    async def _get_mode(self) -> Mode:
        assert self.mode in ("plan", "default", "accept-edits", "bypass")
        return self.mode  # type: ignore[return-value]

    async def _request_permission(
        self, tool_name: str, summary: str, path: str | None
    ) -> ModalResult:
        # 委托 controller(push_screen + await Future 在 controller 里;Esc=Reject 传播)
        assert self._controller is not None
        return await self._controller.request_permission(tool_name, summary, path)

    async def mount_permission_prompt(
        self,
        tool_name: str,
        summary: str,
        on_result: Callable[[str], None],
    ) -> None:
        """挂底部行内权限菜单(Claude Code 风格,取代旧 ModalScreen)。

        controller.request_permission 调此;on_result = controller 的 future 回调。
        choose(数字键/enter/shift+tab/Esc) → _wrap → 卸载 + on_result → controller
        await 的 future set。挂载失败 → reject(不卡 gate)。
        """
        from birdcode.ui.widgets.permission_prompt import PermissionPrompt

        def _wrap(result: object) -> None:
            self._dismiss_permission_prompt()  # 先卸载(UI 即时消失)再 set future
            on_result(str(result))

        prompt = PermissionPrompt(tool_name, summary, on_result=_wrap)
        self._permission_prompt = prompt
        try:
            await self.query_one("#composer").mount(prompt)
            prompt.focus()
        except Exception:
            log.debug("mount permission prompt failed", exc_info=True)
            self._permission_prompt = None
            on_result("reject")

    def _dismiss_permission_prompt(self) -> None:
        """卸载权限菜单(同步,幂等)。无菜单时 no-op。"""
        p = self._permission_prompt
        if p is None:
            return
        self._permission_prompt = None
        try:
            p.remove()
        except Exception:
            log.debug("permission prompt remove failed", exc_info=True)

    async def ask_user(self, question: str, options: list[dict]) -> str:
        """AskUserTool.execute 委托:在对话区(#scroll)挂 ChoicePrompt,等用户选预设/自定义/Esc。

        ChoicePrompt 是 inline 容器:选 Type something 在菜单内原地变 InputArea 输入
        (Submitted 在容器内 @on+event.stop 捕获,不冒泡 app 主输入)。ask_user 期间隐藏底部
        主输入(#input),强制用户只在 ChoicePrompt 的 #choice_input 输入,免误进 type-ahead;
        结束(finally)恢复 #input + focus。
        """
        from birdcode.ui.widgets.choice_prompt import ChoicePrompt

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def _on_result(value: str) -> None:
            if not future.done():
                future.set_result(value)

        prompt = ChoicePrompt(question, options, on_result=_on_result)
        self._choice_prompt = prompt
        # 隐藏底部主输入:ask_user 期间用户只在 #choice_input 输入,免误在 #input 打字进 type-ahead。
        main_input = None
        try:
            main_input = self.query_one("#input", InputArea)
            main_input.display = False
        except Exception:  # noqa: BLE001 - 无 #input(测试/非 BirdApp)→ 跳过,不影响 ask_user
            main_input = None
        try:
            await self.query_one("#scroll").mount(prompt)  # 对话位置,跟随对话
            prompt.focus()
        except Exception:
            log.debug("mount choice prompt failed", exc_info=True)
            self._choice_prompt = None
            if main_input is not None:
                main_input.display = True  # 挂载失败也要恢复
            return "用户中断选择"
        try:
            return await future
        except asyncio.CancelledError:
            return "用户中断选择"
        finally:
            self._dismiss_choice_prompt()
            if main_input is not None:
                main_input.display = True
                main_input.focus()

    def _dismiss_choice_prompt(self) -> None:
        """卸载方案选择菜单(同步,幂等)。无菜单时 no-op。"""
        p = self._choice_prompt
        if p is None:
            return
        self._choice_prompt = None
        try:
            p.remove()
        except Exception:
            log.debug("choice prompt remove failed", exc_info=True)

    def _wire_read_history(self) -> None:
        """把当前 store 的 jsonl 路径注入 read_history 工具(只读当前会话,不读其它会话)。

        on_mount + /clear(切到新 session)时调,与 update_output_sink / gate extra_root
        同模式。registry 不含该工具(自定义 registry / 测试)→ 静默跳过。best-effort:
        失败只 log,不杀主循环(read_history 降级为「未启用」提示,不影响其它工具)。
        """
        from birdcode.tools.read_history import ReadHistoryTool

        reg = self._tool_registry
        tool = reg.get("read_history") if reg is not None else None
        if not isinstance(tool, ReadHistoryTool):
            return
        try:
            tool.set_jsonl(self._store.jsonl_path if self._store is not None else None)
        except Exception:
            log.warning("注入 read_history jsonl 路径失败", exc_info=True)

    def _rebind_agent_tool(
        self, *, ctx: SessionContext | None = None, provider: ParentProvider | None = None
    ) -> None:
        """/clear(重绑 ctx)/profile(重绑 provider)后,把新值传导给所有已注册的 agent tool。

        每个 agent tool(_AgentTool)在 on_mount 捕获 ctx/parent_provider;App 层状态变更
        (/clear 重开 store、/profile 重建 provider)若不重绑,子 agent 会用旧 session 目录 /
        旧 profile。与 executor.update_output_sink / gate.replace_extra_root /
        _wire_read_history 同模式。遍历所有 is_agent_tool(builtin + 自定义,数量不定)。
        best-effort:registry 无 agent tool(自定义/测试 registry)→ 静默跳过。
        """
        reg = self._tool_registry
        if reg is None:
            return
        for name in reg.names():
            tool = reg.get(name)
            if tool is None or not getattr(tool, "is_agent_tool", False):
                continue
            if ctx is not None:
                tool.rebind(ctx=ctx)
            if provider is not None:
                tool.rebind(provider=provider)

    def _apply_theme(self) -> None:
        d = self._theme.design
        try:
            screen = self.screen
            screen.styles.background = d["background"]
            screen.styles.color = d["text"]
            if self._theme.name == "light":
                screen.add_class("light")
            else:
                screen.remove_class("light")
        except Exception:
            log.debug("apply_theme failed", exc_info=True)

    def watch_busy(self, busy: bool) -> None:
        self._refresh_status()
        if busy:
            # anchor(on_mount 已开)负责「新内容增长时贴底、用户上滑即自动取消跟随」,
            # 覆盖了旧实现里事件驱动 _stick 抓不到的异步 Markdown 高度增长。生成期间仍跑
            # 低频定时器,但回调改为 _tick_follow:仅在贴底时续上 anchor(用户滚回底部→
            # 恢复跟随),上滑时不动→视图不被拉回(修:同步子 agent 期间可自由上滑读历史)。
            if self._follow_timer is None:
                self._follow_timer = self.set_interval(0.08, self._tick_follow)
        elif self._follow_timer is not None:
            self._follow_timer.stop()
            self._follow_timer = None
            self.call_after_refresh(self._finish_turn_scroll)  # 收尾:贴底则钉一次,上滑不打扰

    def watch_queue_size(self, _v: int) -> None:
        self._refresh_status()

    def watch_usage(self, _v: TokenUsage | None) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status", StatusBar).refresh_from(
                mode=self.mode,
                unicode_supported=self._unicode,
                queue_size=self.queue_size,
            )
        except Exception:
            pass

    def _display_model(self) -> str:
        """banner 显示的模型:取 profile.model(真实 id,如 deepseek-v4-pro);
        profile 缺失(无 cfg / mock 启动)退化为 profile 名(self.model)。"""
        if self._cfg is not None:
            prof = self._cfg.providers.get(self.model)
            if prof is not None:
                return prof.model
        return self.model

    def _build_banner(self) -> Text:
        """顶端 banner:小鸟图(着色)+ BirdCode vX.X.X / 当前模型 / 当前路径。

        图行用主题 accent 色(金/黄,呼应「鸟」);三行文字继承 #banner 的 $secondary。
        /profile 切模型、cwd 变更后由 watch_model/watch_cwd → _refresh_banner 重绘。
        """
        art = _BIRD_UNICODE if self._unicode else _BIRD_ASCII
        right = [f"BirdCode v{__version__}", self._display_model(), self.cwd]
        width = max(len(a) for a in art)
        art_style = self._theme.design.get("accent") or "yellow"
        out = Text()
        for i, (a, t) in enumerate(zip(art, right, strict=True)):
            out.append(Text(a.ljust(width), style=art_style))
            out.append("  ")
            out.append(t)
            if i < len(art) - 1:
                out.append("\n")
        return out

    def _refresh_banner(self) -> None:
        try:
            self.query_one("#banner", Static).update(self._build_banner())
        except Exception:
            pass

    def watch_model(self, _model: str) -> None:
        # /profile 切模型 → 重绘 banner(状态行不再显示模型)。
        self._refresh_banner()

    def watch_cwd(self, _cwd: str) -> None:
        self._refresh_banner()

    async def show_message(self, text: str, *, kind: str = "info") -> None:
        """命令输出统一入口:挂一条 Static 到 #scroll。kind=warn/error 前缀 ⚠。"""
        prefix = "⚠ " if kind in ("warn", "error") else ""
        scroll = self.query_one("#scroll")
        await scroll.mount(Static(f"{prefix}{text}", classes="turn"))

    async def send_user_message(self, text: str) -> None:
        """把预设 prompt 当用户消息送进对话(/review 用)。

        后台跑 controller.submit(与普通用户输入同路径、入 _bg_tasks),不阻塞消息泵——
        否则 /review 的整轮 AI 对话会内联占住 _on_submitted,Esc 无法中断、状态不刷新。
        """
        task = asyncio.create_task(self.controller.submit(text))
        self._bg_tasks.add(task)
        task.add_done_callback(self._reap_bg_task)

    async def _on_status(self) -> None:
        assert self._controller is not None
        self.busy = self._controller.busy
        self.queue_size = self._controller.queue_size

    async def _on_subagent_progress(self, progress: SubagentProgress) -> None:
        """子 agent 进度:首次 mount SubagentCard 到滚动区,后续 update 文本。

        同步子 agent 运行期间,卡片挂在 #scroll;每轮 Done 经 progress_cb 推一次进度
        (无独立计时器)。主轮次 Done 时由 _on_event 统一清理卡片(子 agent 报告已成
        Agent 工具的 ToolResult)。
        """
        if not self.is_running:
            return
        from birdcode.ui.widgets.subagent_card import SubagentCard

        card = self._subagent_cards.get(progress.agent_id)
        if card is None:
            card = SubagentCard()
            self._subagent_cards[progress.agent_id] = card
            scroll = self.query_one("#scroll")
            was_at_bottom = self._at_bottom()
            await scroll.mount(card)
            if was_at_bottom:
                self._stick()
        card.update_progress(progress)

    def _clear_subagent_cards(self) -> None:
        """轮次结束(任意终态:Done/Error/Interrupted)清理子 agent 进度卡。

        同步子 agent 运行期间挂的卡片必须随【主轮次】结束一并清掉;仅 Done 分支清理会
        在 Esc 中断(Interrupted)或护栏 Error 结束时残留屏幕。三个终态分支统一调用。
        """
        for card in self._subagent_cards.values():
            card.remove()
        self._subagent_cards.clear()

    def _reap_bg_task(self, task: asyncio.Task[None]) -> None:
        """后台轮次任务的完成回调：移除引用 + 取走异常。

        set.discard 只从集合移除、不读 .exception()，故失败的后台任务会留下「未检索
        异常」，回收时 asyncio 经事件循环报 'Task exception was never retrieved'。这里
        读出异常并落日志，既回收引用又消除告警。cancelled 任务读 .exception() 会抛
        CancelledError，先放过。
        """
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("background turn task failed", exc_info=exc)

    async def _dispatch(self, cmd: Command, args: str) -> None:
        try:
            await cmd.handler(self, args)  # self 即 CommandContext
        except Exception:
            log.exception("命令 %s 执行失败", cmd.name)
            await self.show_message(f"{cmd.name} 失败，详见 debug.log", kind="error")
        self._stick()
        self._refresh_status()

    @on(InputArea.Submitted)
    async def _on_submitted(self, event: InputArea.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return
        if text.startswith("/"):
            # 斜杠输入一律走命令路径:裸 "/" 或 "/ " → parse_command 返回 None,
            # 静默忽略(旧行为),绝不把 "/" 当用户消息发给 LLM。
            parsed = parse_command(text)
            if parsed is None:
                return
            if self._command_registry is None:
                return
            cmd = self._command_registry.resolve(parsed.name)
            if cmd is None:
                await self.show_message(
                    f"未知命令 {parsed.name}，输入 /help 查看可用命令", kind="warn"
                )
                self._stick()  # 未知命令分支不走 _dispatch,需手动钉底(否则反馈可能在视口外)
            else:
                await self._dispatch(cmd, parsed.args)
            return
        task = asyncio.create_task(self.controller.submit(event.text))
        self._bg_tasks.add(task)
        task.add_done_callback(self._reap_bg_task)

    @on(InputArea.CompletionRequested)
    async def _on_completion_requested(self, event: InputArea.CompletionRequested) -> None:
        """补全浮层:挂 #composer 末尾(StatusBar 下方)→ composer 长高、输入框上移让位。

        已开则就地 update 候选(输入实时过滤时复用同一浮层,避免每键重挂闪烁)。
        """
        menu = self._completion_menu
        if menu is not None:
            menu.update(event.candidates)
            return
        from birdcode.ui.widgets.completion_menu import CompletionMenu

        menu = CompletionMenu(event.candidates)
        self._completion_menu = menu
        try:
            await self.query_one("#composer").mount(menu)
        except Exception:
            log.debug("mount completion menu failed", exc_info=True)
            self._completion_menu = None

    def _dismiss_completion(self) -> None:
        """卸载补全浮层(同步)。无浮层时 no-op。"""
        m = self._completion_menu
        if m is None:
            return
        self._completion_menu = None
        try:
            m.remove()
        except Exception:
            log.debug("completion menu remove failed", exc_info=True)

    async def compact_now(self) -> tuple[CompactionResult | None, str]:
        """/compact 主体:busy 校验 + compact_now。返回 (result, status)。

        status ∈ {'no_context','busy','ok','empty'}。
        """
        ctrl = self._controller
        if ctrl is None or ctrl._context is None:  # noqa: SLF001
            return None, "no_context"
        if ctrl.busy:
            return None, "busy"
        res = await ctrl._context.compact_now(history=ctrl.history, trigger="manual")  # noqa: SLF001
        return (res, "ok") if res is not None else (None, "empty")

    async def clear_conversation(self) -> str | None:
        """/clear 主体:abort+清UI+清history+开新session+rehook。返回新 session id(或 None)。

        异常安全:controller._store 必须先于 sink/gate rehook 赋值——reopen 已关闭旧
        store,后续 _on_message 全靠 controller 指向新(开)store。read_history 先接(它有内部
        try,自身不会抛),保证「只读当前会话」不变式必成立,即使后面 sink/gate rehook 抛异常。
        """
        self.controller.abort()
        await self._close_markdown()
        scroll = self.query_one("#scroll")
        for w in list(scroll.children):
            if w.id != "banner":
                w.remove()
        self._tools.clear()
        self._diff_pending.clear()
        self._current_turn = None
        self._thinking_round = 0  # 重置思考轮次(/clear 开新会话)
        self.controller.history.clear()
        self.usage = None
        if self._store is None:
            return None
        import uuid as _uuid

        new_sid = str(_uuid.uuid4())
        old_tool_results = self._store.tool_results_dir
        self._store = self._store.reopen(new_sid)
        # 先切 controller(见异常安全注释):rehook 抛异常时 controller 已指向新 store,
        # 不会每条 append 写已关句柄丢消息。
        self.controller._store = self._store  # noqa: SLF001
        # rehook 是 best-effort:失败只降级(大输出走 inline / 落盘 read_file 被拦),
        # 绝不让异常逃出 clear_conversation(否则 controller 状态未切完 + UI 半残)。
        try:
            self._wire_read_history()
            if self._executor is not None:
                self._executor.update_output_sink(self._store.as_output_sink())
            if self._gate is not None:
                # replace(非 append)session 级 tool-results 根,防累积。
                self._gate.replace_extra_root(old_tool_results, self._store.tool_results_dir)
            # AgentTool 的 ctx 仍指向旧 session → 重绑到新 session(否则子 agent 侧链写旧目录)。
            self._rebind_agent_tool(ctx=self._store.ctx)
            # 异步子 agent manager 也重绑到新会话句柄(否则完成通知注入旧 session)。
            if self._subagent_mgr is not None:
                self._subagent_mgr.rebind(store=self._store, controller=self.controller)
            # agent teams:/clear 切新 session → 旧 team 终止 + 清(team_mgr 引用保留,
            # team 工具不重注册;reset 内 cancel_all + 清 registry/board)。
            if self._team_mgr is not None:
                self._team_mgr.reset()
        except Exception:
            log.warning("/clear 重接 sink/gate/read_history 失败", exc_info=True)
        return new_sid

    async def switch_profile(self, name: str) -> bool:
        """/profile 主体:重建 provider + 切 controller/context/memory。未知 profile 返回 False。"""
        if self._cfg is None or name not in self._cfg.providers:
            return False
        from birdcode.agent.factory import build_provider

        prof = self._cfg.providers[name]
        # 透传 registry:切换到真实 provider 也带 tools(与启动注入链一致)。
        # 透传 mcp_instructions:on_mount 已建好 manager,bound method 懒读最新值。
        new_provider = build_provider(
            prof,
            self._cfg,
            registry=self._tool_registry,
            mcp_instructions=self.mcp_instructions,
        )
        self.controller.set_provider(new_provider)
        # ContextManager 也要换 provider,否则摘要 complete() 仍走旧 profile。
        ctx = self.controller._context  # noqa: SLF001 - 同包内省,跨模块读 controller 的 context 句柄
        if ctx is not None:
            ctx.set_provider(new_provider)
        self.model = name
        # 记忆提取也跟 provider:否则切 profile 后提取仍走旧端点/密钥/模型,旧 key 失效
        # 即静默失败。mock→real 时 _memory 还是 None,这里补构造并挂到 controller。
        extract_model = prof.hak_model or prof.model
        if self._memory is None:
            from birdcode.memory.extractor import MemoryManager
            from birdcode.utils.paths import find_project_root

            # 用钉好的 project_root(主仓),不用裸 find_project_root()——后者在 worktree
            # 里(CLI 已 chdir)返回 worktree,记忆会写进 worktree、退出清理时被毁。
            # _cfg.project_root 启动时已设(=主仓);or 兜底仅满足类型,实际不触发。
            self._memory = MemoryManager(
                new_provider,
                project_root=self._cfg.project_root or find_project_root(),
                extract_model=extract_model,
            )
            self.controller._memory = self._memory  # noqa: SLF001
        else:
            self._memory.set_provider(new_provider, extract_model=extract_model)
        # 记忆治理同 memory:mock→real 时 _governance 还是 None,补构造并挂 controller;
        # 已有则跟 provider 切换。与 _memory 同步构造(cli 启动处),故此处 find_project_root
        # 已由上方 memory 分支 import 到作用域。
        if self._governance is None:
            from birdcode.memory.governance import GovernanceManager

            self._governance = GovernanceManager(
                new_provider,
                project_root=self._cfg.project_root or find_project_root(),
                govern_model=extract_model,
                session_store=self._store,
            )
            self.controller._governance = self._governance  # noqa: SLF001
        else:
            self._governance.set_provider(new_provider, govern_model=extract_model)
        # self._provider 是 AgentTool 等捕获方的真相源:切换后必须同步,否则子 agent 继承
        # 【初始】profile(永不更新)。同时重绑已注册 AgentTool 的 parent_provider。
        self._provider = new_provider
        self._rebind_agent_tool(provider=cast("ParentProvider", new_provider))
        return True

    async def _replay_history(self, turns: list[ConversationTurn]) -> None:
        """resume 后静默重放历史 Turn 到 #scroll(不调 provider,纯渲染)。

        简化版:仅重放 user TextBlock(开新轮次)+ assistant TextBlock(写 Markdown);
        跳过 tool_use / tool_result / thinking block(完整还原 ToolLine/thinking 状态
        复杂度高,进待办)。重放后钉底,让用户立即看到完整对话文字。
        """
        for turn in turns:
            for msg in turn.messages:
                if msg.synthetic:
                    continue  # 合成消息不重放(避免每次崩溃 resume 弹固定气泡)
                if msg.is_task_notification:
                    continue  # 子 agent 完成通知不当用户气泡重放(系统注入,非用户提问)
                if msg.role == "user":
                    text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
                    if text:
                        await self.begin_assistant_turn(text)
                elif msg.role == "assistant":
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text:
                            await self._ensure_markdown()
                            assert self._md_stream is not None
                            await self._md_stream.write(b.text)
            await self._close_markdown()
        self._pin_to_bottom()

    async def _on_event(self, event: ProviderEvent) -> None:
        # 退出/拆卸期间控制器可能仍在吐事件（被取消 → Interrupted）；此时 UI 已不在，
        # 直接丢弃，绝不因 query_one 找不到节点而抛异常。
        if not self.is_running:
            return
        # 关键：在本次事件改动 DOM「之前」读是否贴底。mount/write 之后 virtual_size
        # 已增长，max_y 会跟着变大，再用「当前 scroll_y >= max_y-1」判断会永远为假，
        # 于是输入框被新内容顶到屏外却不回滚（提交后输入框消失的 bug）。
        was_at_bottom = self._at_bottom()
        scroll = self.query_one("#scroll")
        if isinstance(event, TurnStart):
            # controller 在每轮开头把 user_text 随流带回 → 据此开轮次（不再用
            # _current_user_text 共享态延迟读取，修 type-ahead 竞态）。
            await self.begin_assistant_turn(event.user_text)
        elif isinstance(event, ThinkingDelta):
            if self._thinking is None:
                self._thinking_round += 1
                block = ThinkingBlock(self._icons, round_no=self._thinking_round)
                if self._md is not None:
                    await scroll.mount(block, before=self._md)
                else:
                    await scroll.mount(block)
                self._thinking = block
                block.start()
            self._thinking.write(event.text)
        elif isinstance(event, TextDelta):
            self._finish_thinking()
            await self._ensure_markdown()
            assert self._md_stream is not None  # _ensure_markdown 保证已建
            await self._md_stream.write(event.text)
        elif isinstance(event, ToolCallStart):
            # 工具行挂在当前助手轮次下（与回复同组）；无当前轮次则单开一个
            turn = self._current_turn
            if turn is None:
                turn = Turn()
                await scroll.mount(turn)
                self._current_turn = turn
            tool_line = await turn.add_tool(self._icons)
            self._tools[event.id] = tool_line
            tool_line.start(event.name, event.args)
        elif isinstance(event, ToolResult):
            pending = self._tools.pop(event.id, None)
            if pending is not None:
                pending.finish(event.ok, event.summary, truncated=event.truncated)
                if event.output:  # 有完整正文 → 挂折叠区(默认收起)
                    pending.set_output(event.output, truncated=event.truncated)
                # 暂存:agent_loop 紧随 ToolResult emit EditDiff(成功时),用它挂 diff 子区。
                self._diff_pending[event.id] = pending
        elif isinstance(event, EditDiff):
            # Edit/Write 成功后的 (old,new) → 挂 diff 高亮子区。
            line = self._diff_pending.pop(event.id, None)
            if line is not None:
                line.set_diff(event.old, event.new)
        elif isinstance(event, CompactionStart):
            # 自动压缩:挂一条橙色转圈行(仿工具调用,复用 ToolLine 的 spinner/收尾)。
            # 挂到当前轮次下(TurnStart 已先于 agent_loop 发出 → _current_turn 通常存在)。
            turn = self._current_turn
            if turn is None:
                turn = Turn()
                await scroll.mount(turn)
                self._current_turn = turn
            line = await turn.add_tool(self._icons, color="dark_orange")
            line.start(label="上下文空间不足，正在进行压缩")
            self._compaction_line = line
        elif isinstance(event, CompactionEnd):
            # 压缩结束:停转圈、显示结果摘要(橙色)。fell_back 标硬截断兜底。
            line = self._compaction_line
            if line is not None:
                summary = event.summary or "压缩完成"
                if event.fell_back:
                    summary = f"{summary}（硬截断兜底）"
                line.finish(ok=True, summary=summary)
                self._compaction_line = None
        elif isinstance(event, Done):
            if event.usage is not None:
                self.usage = event.usage
            await self._close_markdown()
            # 清理只读工具残留的 _diff_pending(它们无 EditDiff 跟随 → 不 pop);轮次结束统一清。
            self._diff_pending.clear()
            # 子 agent 进度卡随主轮次结束清理(同步子 agent 已跑完,报告已成 Agent ToolResult)。
            self._clear_subagent_cards()
        elif isinstance(event, Error):
            await self._close_markdown()
            # 轮次以护栏 Error 结束(非 Done):子 agent 进度卡同样需清理,否则残留。
            self._clear_subagent_cards()
            await scroll.mount(Static(f"⚠ {event.message}", classes="turn"))
        elif isinstance(event, Interrupted):
            await self._close_markdown()
            # 正在执行的工具(转 spinner)没机会收到 ToolResult → 主动停 spinner 并标中断。
            # 否则中断后工具行一直转,盖在 [interrupted] 上方,用户看不出「已中断」。
            for line in self._tools.values():
                line.finish(ok=False, summary="⚡ 中断")
            self._tools.clear()
            # 自动压缩转圈行同样没机会收到 CompactionEnd → 主动收尾,免得中断后仍转。
            if self._compaction_line is not None:
                self._compaction_line.finish(ok=False, summary="⚡ 中断")
                self._compaction_line = None
            # 同步子 agent 运行中被 Esc 中断:卡片随轮次结束清理(与 Done/Error 同)。
            self._clear_subagent_cards()
            await scroll.mount(Static("[interrupted]", classes="turn"))
        # 贴底时跟随新内容（walk-down）；用户主动上滑读历史时不被拉回。
        if was_at_bottom:
            self._stick()

    async def begin_assistant_turn(self, user_text: str) -> None:
        # 每条用户消息开新计数:否则 _thinking_round 跨轮次累加 → 第二个问题的首段
        # 思考会被标成「第 4 轮」(延续上一问的多轮),标注与轮次错位。
        self._thinking_round = 0
        scroll = self.query_one("#scroll")
        turn = Turn()
        await scroll.mount(turn)
        await turn.add_user(user_text)
        self._md = await turn.add_assistant_markdown()
        self._current_turn = turn
        self._md_stream = Markdown.get_stream(self._md)

    def _finish_thinking(self) -> None:
        """收尾思考块：停 spinner 定时器并清引用。幂等（已 None 则 no-op）。"""
        if self._thinking is not None:
            self._thinking.finish()
            self._thinking = None

    async def _ensure_markdown(self) -> None:
        """确保有可写的 _md_stream。

        多轮 agent_loop 中,每轮 Done 都会 _close_markdown 停掉当前 md 流。下一轮模型若
        再输出 TextDelta(如工具调用后的总结),到达时 _md_stream 已是 None —— 若不重建,
        正文会被静默丢弃(用户只看到工具结果、看不到总结)。重建一个 md 挂到滚动区末尾
        的新 Turn(第 2 轮起 thinking 已挂 scroll 末尾,新 Turn 紧随其后,顺序正确)。
        """
        if self._md_stream is not None:
            return
        scroll = self.query_one("#scroll")
        turn = Turn()
        await scroll.mount(turn)
        self._current_turn = turn
        self._md = await turn.add_assistant_markdown()
        self._md_stream = Markdown.get_stream(self._md)

    async def _close_markdown(self) -> None:
        self._finish_thinking()
        if self._md_stream is not None:
            try:
                await self._md_stream.stop()
            except Exception:
                log.debug("stop markdown stream failed", exc_info=True)
        self._md_stream = None
        self._md = None
        self._current_turn = None

    def action_interrupt(self) -> None:
        # ask_user 选择/输入态:Esc = 退出整个 ask_user(菜单态/输入态统一),
        # tool_result="用户中断选择",控制权回用户重新输入。
        if self._choice_prompt is not None:
            self._choice_prompt.choose("用户中断选择")
            return
        # 权限菜单打开时:Esc = No(reject),不中断 agent。
        if self._permission_prompt is not None:
            self._permission_prompt.choose("reject")
            return
        # 补全浮层打开时:Esc 先关浮层,不中断 agent。
        if self._completion_menu is not None:
            self._dismiss_completion()
            return
        assert self._controller is not None
        # Esc：中断当前流/工具；空闲时 controller.interrupt() 是 no-op。退出改由 Ctrl+Q。
        self._controller.interrupt()
        # Esc 同时终止在跑的异步子 agent(不分忙闲——空闲时也可能有后台子 agent 在跑)。
        if self._subagent_mgr is not None:
            self._subagent_mgr.cancel_all()
        if self._team_mgr is not None:
            self._team_mgr.cancel_all()

    async def action_quit(self) -> None:
        self.exit()

    async def on_unmount(self) -> None:
        """App 关闭时清理 MCP 连接(per-server lifecycle task:transport+session 全链)。

        Textual 在 _shutdown 中 await _dispatch_message(Unmount()),故 on_unmount 会
        被同步 await——覆盖所有退出路径(Ctrl+Q / /exit / 异常 / 关窗)。非阻塞启动后,
        退出时 startup task 可能仍在跑:先 cancel+await 它(各 _server_lifecycle 的 finally 回滚
        自己的 transport),再 shutdown——避免与 shutdown 并发改 _sessions/_server_tasks。
        """
        # 0) controller 先进入退出态(#1):抑制后续 notify_wake/receive 起新后台 drain +
        #    取消已起的 orphan drain。必须先于 cancel/join——退出收尾链(cancel_all→join_all→
        #    _supervise→_inject→notify_wake)会在退出期触发 notify_wake,不抑制度就起一个 on_unmount
        #    既不 await 也不 cancel 的 orphan drain,在后续 MCP/store await 期跑 run_agent_loop。
        if self._controller is not None:
            self._controller.begin_shutdown()
        try:
            # 1) 取消在跑的 teammate / 异步子 agent:退出时别让它们悬挂(meta 卡 running/idle、
            #    侧链写一半)。cancel 经 runner CancelledError 路径落 meta=cancelled + close store。
            if self._team_mgr is not None:
                self._team_mgr.cancel_all()
            if self._subagent_mgr is not None:
                self._subagent_mgr.cancel_all()
            # F5:await 两者 task 终止 → runner.run() finally 内的 shielded worktree cleanup
            # (git unlock/remove/branch-d)在 live loop 上跑完。不 await 则关 loop 时砍断 cleanup
            # → 每次裸退出留一批 git-locked worktree(teammate/async-subagent 的 agent_id 随机,
            # 跨会话不复用,create_worktree 的快速恢复救不了)。#9:两者并发 gather(各自内部 5s
            # 超时),避免串行 5s+5s≈10s 退出卡顿;return_exceptions 兜底,join_all 自身不抛。
            joins: list = []
            if self._team_mgr is not None:
                joins.append(self._team_mgr.join_all())
            if self._subagent_mgr is not None:
                joins.append(self._subagent_mgr.join_all())
            if joins:
                await asyncio.gather(*joins, return_exceptions=True)
            # 2) 收尾后台 startup task。无条件 await(即使已完成):检索其异常,防 asyncio
            # "Task exception was never retrieved" 警告 + 静默丢弃(startup 尾巴的 BaseException
            # 如 SystemExit 逃过 _start_mcp_background 的 except Exception 时)。cancel 仅未完成者。
            task = self._mcp_startup_task
            if task is not None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 - CancelledError/startup 异常都吞,继续 shutdown
                    pass
            # 3) 关闭已托管的 sessions/transports
            mgr = self._mcp_manager
            if mgr is not None:
                try:
                    await mgr.shutdown()
                except Exception:
                    log.debug("MCP shutdown failed", exc_info=True)
        finally:
            # 4) 关闭会话 store(刷 jsonl 文件句柄;无 store 时 no-op)。#2:放 finally——退出
            #    路径(被 cancel / 上游异常)也保证关句柄,不因 join/mcp 期被打断而跳过。
            #    (数据本就每行 append 即 flush,这里只关句柄;不关只留句柄泄漏,不丢数据。)
            if self._store is not None:
                try:
                    self._store.close()
                except Exception:
                    log.debug("store close failed", exc_info=True)

    def action_cycle_mode(self) -> None:
        """shift+tab 循环切档:plan→default→accept-edits→bypass→plan。

        反应式 self.mode 变更会触发 StatusBar 刷新(watch_* 未挂 mode,故手动 _refresh_status)。
        """
        # 权限菜单(行内 PermissionPrompt)期间:shift+tab 选 session(本会话按类别宽放),
        # 不切档。shift+tab 是 app 级 priority 绑定,prompt 自带 binding 抢不过,在此分流;
        # 若放行切档会冒泡把会话误切进 bypass(此后写操作自动 L4 放行、跳过 HITL)。
        # ask_user 选择/输入期:shift+tab 不切档(防误进 bypass)。
        if self._choice_prompt is not None:
            return
        if self._permission_prompt is not None:
            self._permission_prompt.choose("session")
            return
        # 循环顺序:default→plan→accept-edits→bypass→default(plan 最安全,bypass 最危险)。
        order = ["default", "plan", "accept-edits", "bypass"]
        try:
            i = order.index(self.mode)
        except ValueError:
            i = 1  # 兜底:异常值落到 default 之后(下一档 plan)
        self.mode = order[(i + 1) % len(order)]
        self._refresh_status()

    def action_copy(self) -> None:
        """shift+c 复制:输入框聚焦时 no-op(让 TextArea 正常插入大写 C,不抢键);
        焦点在 transcript 等处时委托 screen.copy_text(走 OSC 52)复制选区。
        """
        # 非优先级绑定 → 输入框聚焦时 TextArea 已先插入 'C';这里再见聚焦态 no-op,双保险。
        if isinstance(self.focused, InputArea):
            return
        copy = getattr(self.screen, "copy_text", None)
        if callable(copy):
            try:
                copy()
            except Exception:
                log.debug("shift+c 复制失败", exc_info=True)

    def _at_bottom(self) -> bool:
        """当前视图是否贴在底部（用于决定新内容是否跟随滚动）。"""
        try:
            s = self.query_one("#scroll")
            max_y = max(0, s.virtual_size.height - s.size.height)
            return s.scroll_y >= max_y - 1
        except Exception:
            return True

    def _stick(self) -> None:
        """本次改动刷新后，把视图钉到底部。"""
        self.call_after_refresh(self._pin_to_bottom)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # 用户在输入框打字/换行 → 输入框伸缩后贴底，历史向上滚（不被挤出可视区）
        self.call_after_refresh(self._pin_to_bottom)

    def _pin_to_bottom(self) -> None:
        try:
            self.query_one("#scroll").scroll_end(animate=False)
        except Exception:
            pass

    def _tick_follow(self) -> None:
        """生成期间跟随定时器回调。

        仅当 TranscriptScroll.user_following=True(用户在跟随,没上滑读历史)时 scroll_end
        钉底——覆盖事件驱动 _stick 抓不到的异步 Markdown 高度增长(walk-down)。用户上滑后
        user_following 已被 watch_scroll_y 置 False,此处跳过→视图不被拉回(修:同步子 agent
        运行期间可自由上滑读历史)。用户滚回底部后 watch_scroll_y 自动恢复 user_following。
        """
        try:
            s = self.query_one("#scroll", TranscriptScroll)
        except Exception:
            return
        if s.user_following:
            s.scroll_end(animate=False)

    def _finish_turn_scroll(self) -> None:
        """轮次收尾:跟随态则钉一次保证整轮可见;用户上滑读历史则不打扰。"""
        try:
            s = self.query_one("#scroll", TranscriptScroll)
        except Exception:
            return
        if s.user_following:
            s.scroll_end(animate=False)

    def transcript(self) -> str:
        assert self._controller is not None
        lines: list[str] = []
        for turn in self._controller.history:
            for msg in turn.messages:
                text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
                if msg.role == "user":
                    lines.append(f"> {text}")
                elif msg.role == "assistant":
                    lines.append(text)
            if turn.interrupted:
                lines.append("[interrupted]")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
