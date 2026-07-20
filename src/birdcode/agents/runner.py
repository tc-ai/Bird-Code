# src/birdcode/agents/runner.py
"""子 agent 运行时:派生 → 跑 → 终止 → 报告销毁(同步,Phase1)。"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid as _uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

from birdcode.agent.agent_loop import run_agent_loop
from birdcode.agent.context import ContextManager
from birdcode.agent.factory import build_provider
from birdcode.agent.provider import Done, Error, ProviderEvent, TokenUsage, ToolCallStart
from birdcode.agent.system_prompt.env import _AGENT_CWD
from birdcode.agents.definition import AgentDefinition
from birdcode.blocks import TextBlock, ThinkingBlock
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.conversation import Message, Turn
from birdcode.permission.gate import PermissionGate
from birdcode.session import paths
from birdcode.session.models import SessionContext, SubagentMeta
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.session.subagent_store import SubagentStore
from birdcode.session.timeutil import utc_iso
from birdcode.tools.executor import ToolExecutor
from birdcode.tools.registry import ToolRegistry
from birdcode.utils.logging import get_logger
from birdcode.utils.worktree import (
    cleanup_subagent_worktree,
    collect_worktree_changes,
    create_worktree,
    current_head_sha,
    lock_worktree,
    worktree_store_name,
)

log = get_logger("birdcode.agents.runner")

MAX_SPAWN_DEPTH = 1  # 递归硬禁止(子工具表本就不含 Agent;此为防御性 belt-and-suspenders)


class ParentProvider(Protocol):
    """resolve_profile / AgentTool 等跨模块共用的父 provider 视图:一个 .profile 属性。

    不直接用 StreamingProvider:StreamingProvider 协议【未】声明 .profile
    (_BaseLLMProvider 具体基类已暴露 .profile,MockProvider/测试桩均无)。
    调用方只用 .profile,故以最小协议表达真实契约(结构性,ISP)。
    App 层传 AnthropicProvider/OpenAIProvider(均继承 _BaseLLMProvider.profile)即可满足。
    """

    @property
    def profile(self) -> ProviderProfile: ...


def build_child_registry(
    parent: ToolRegistry | None,
    disallowed: tuple[str, ...],
    *,
    is_async: bool = False,
    cwd: str | None = None,
) -> ToolRegistry:
    """子工具表 = 父 registry 减三类:agent tool、disallowed、会话级有状态工具。

    parent=None → 空 registry(测试/无工具场景)。
    agent tool(builtin + 自定义,均 _AgentTool 实例,is_agent_tool=True)一律排除:子 agent 不能
    再派生(递归禁止)。用标记属性而非名字/isinstance:自定义 agent 名任意(不能枚举),且
    isinstance 需 import tools.agent_tools → 与本模块(被其 import)循环。
    会话级有状态工具(tool.session_scoped=True,如 read_history/tool_search)排除:它们持有
    父会话状态(主线 jsonl / 父 registry),原样共享会让子 agent 读/写父 agent 状态。
    每实例可变工具(如 BashTool._cwd)经 fork_for_child 派生全新实例,隔离子 agent 对状态的
    改动不污染父 agent。无状态工具(读类 / MCP 代理)默认 fork 返回 self,行为不变。
    is_async=True(异步子 agent):ask_user(is_ask_user=True)无法后台与用户交互 → 换成
    AskUserAsyncStub(execute 返回拒文案),不 fork 原 tool——模型仍见该能力、调用即拒。
    cwd(Phase 2):worktree 子 agent 派生时传 worktree dir → fork_for_child(cwd=cwd)
    注入到 BashTool/6 文件工具的 _cwd。None(非 worktree 子 agent)→ fork_for_child(cwd=None),
    无状态工具返回 self,BashTool 等快照父 _cwd(旧行为,向后兼容)。
    """
    child = ToolRegistry()
    if parent is None:
        return child
    disallowed_set = set(disallowed)
    for name in parent.names():
        tool = parent.get(name)
        if tool is None:
            continue
        if getattr(tool, "is_agent_tool", False) or name in disallowed_set:
            continue
        if tool.session_scoped:
            continue
        # 异步子 agent:ask_user 无法交互 → 换 stub(execute 返回拒文案),不 fork 原 tool。
        if is_async and getattr(tool, "is_ask_user", False):
            from birdcode.tools.ask_user import AskUserAsyncStub

            child.register(AskUserAsyncStub())
            continue
        child.register(tool.fork_for_child(cwd=cwd))  # cwd 注入:worktree dir 或 None
    return child


def resolve_profile(
    defn_model: str, override: str, cfg: AppConfig, parent_provider: ParentProvider
) -> ProviderProfile:
    """model 三级优先:override(工具参数) > defn_model(AgentDefinition) > 继承父。

    '' → 继承父 profile(但仍要用 child_registry 重建 provider)。
    未知 profile 名 → ValueError(AgentTool 捕获转错误回传)。
    """
    chosen = override or defn_model
    if not chosen:
        return parent_provider.profile
    if chosen not in cfg.providers:
        raise ValueError(f"未知 profile '{chosen}'。可用:{list(cfg.providers)}")
    return cfg.providers[chosen]


def build_child_gate(
    parent_gate: PermissionGate,
    *,
    allow_hitl: bool,
    worktree_dir: Path | None = None,
) -> PermissionGate:
    """同步子 agent(allow_hitl=True):复用父 gate(L1-L5 全照常,含 HITL)。
    异步非 worktree(allow_hitl=False, worktree_dir=None):UiPermissionGate.fork_async()——
      L1-L4 同父,L5 对 bash approve、其余写 reject(怕毁主仓)。
    异步 worktree(worktree_dir 非空, Phase 2):fork_worktree_async(sandbox_root=worktree_dir)——
      L1-L4 同父但 sandbox 锁 worktree(主仓 L2 拒),L5 恒 approve(无 HITL,写自动批)。
      worktree 恒异步(worktree_dir 非空隐含 allow_hitl=False)。

    getattr 鸭子类型探测 fork_worktree_async/fork_async:PermissionGate 协议未声明二者
    (仅 UiPermissionGate 有)。非 UiPermissionGate 父(测试桩/MCP 代理)在异步路径回退复用父。
    """
    if worktree_dir is not None:
        # worktree 恒异步(隐含 allow_hitl=False);优先判,免 sync+worktree 误组合
        # (allow_hitl 在前)回退父 gate——那样 sandbox=主仓而工具 _cwd=worktree,口径分歧。
        # 生产中 execute 对 isolation=worktree force-async 使该组合不出现;此处 belt-and-suspenders。
        fork = getattr(parent_gate, "fork_worktree_async", None)
        if callable(fork):
            return cast(PermissionGate, fork(sandbox_root=worktree_dir))
        return parent_gate  # 回退(非 UiPermissionGate,不常见)
    if allow_hitl:
        return parent_gate
    fork = getattr(parent_gate, "fork_async", None)
    if callable(fork):
        return cast(PermissionGate, fork())  # fork_async→UiPermissionGate(满足 Protocol)
    return parent_gate  # 回退:非 UiPermissionGate,复用父(不常见)


_WORKTREE_ADDENDUM = (
    "\n\n【worktree 模式】你的工作目录是一个独立 git worktree(路径见环境块「工作目录」)。"
    "完成后,在最终报告里说明:产物落在哪个 worktree、改/建了哪些文件"
    "——主 agent 据此定位你的成果。"
)


def _system_override(defn: AgentDefinition, isolation: str | None) -> str:
    """子 agent system_prompt;worktree 模式追加产物报告提示(让最终 assistant 可靠报产物路径)。"""
    sp = defn.system_prompt or ""
    if isolation == "worktree":
        return sp + _WORKTREE_ADDENDUM
    return sp


@dataclass
class SubagentReport:
    is_completed: bool
    text: str
    status: Literal["completed", "error", "cancelled"]
    duration_ms: int
    tokens: int
    tool_use_count: int
    artifacts: list[str] = field(default_factory=list)  # worktree 净改变文件清单
    committed: bool = False  # 是否已 commit 到 worktree 分支


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _update_meta(meta_path: Path, **changes: object) -> None:
    """read-modify-write SubagentMeta(状态/计时随生命周期更新)。"""
    existing = read_subagent_meta(meta_path)
    if existing is None:
        return  # launched 写失败则无 meta 可更新;不杀运行
    if existing.status == "lost":
        # lost = 用户 2/No 永久丢弃,终态冻结。runner 的生命周期写(completed/error/cancelled)
        # 不得覆盖 lost——否则被丢弃的 agent 又变可续,跨会话再弹,违背"永久丢弃"。防窄竞态:
        # async 子 agent 的 cancel 落盘晚于用户 2/No 时,except CancelledError 的
        # _update_meta(status=cancelled) 不应改写已落的 lost。
        return
    data = existing.model_dump(by_alias=True)
    data.update(changes)
    try:
        write_subagent_meta(meta_path, SubagentMeta.model_validate(data))
    except Exception:  # noqa: BLE001 - meta 更新失败不杀运行,仅记录(可观测)
        log.debug("write_subagent_meta 失败,忽略(不杀运行)", exc_info=True)


async def _fill_artifacts(report: SubagentReport, wt: Path, base_sha: str) -> None:
    """best-effort 收集 worktree 产物填入 report(worktree cleanup 之前调)。

    base_sha 空 / collect 抛异常 → 保持默认([], False),绝不传播(不杀报告交付/cleanup)。
    """
    if not base_sha:
        return
    try:
        artifacts, committed = await collect_worktree_changes(wt, base_sha)
        report.artifacts = artifacts
        report.committed = committed
    except Exception:  # noqa: BLE001 - 收集失败不影响报告交付
        log.debug("收集子 agent worktree 产物失败,忽略", exc_info=True)


class SubagentRunner:
    """派生并同步运行一个子 agent(独立上下文)。跑到终态 → 报告 → 销毁。

    同步:run() 在主 agent 的 tool 调用 task 里直接 await run_agent_loop;Esc 经
    CancelledError 自然传入,finally 收尾后 re-raise。
    """

    def __init__(
        self,
        *,
        defn: AgentDefinition,
        prompt: str,
        description: str,
        tool_use_id: str,
        model_override: str,
        spawn_depth: int,
        is_async: bool,
        isolation: Literal[None, "worktree"] = None,
        parent_provider: ParentProvider,  # 跨模块 Protocol(只用 .profile)
        parent_registry: ToolRegistry | None,
        parent_gate: PermissionGate | None,
        cfg: AppConfig,
        app: object | None,
        ctx: SessionContext,
        project_root: Path,
        root: Path | None = None,
        progress_cb: Callable[..., Awaitable[None]] | None = None,
        agent_id: str | None = None,
        resume_from: Path | None = None,
    ) -> None:
        self.defn = defn
        self.prompt = prompt
        self.description = description
        self.tool_use_id = tool_use_id
        self.model_override = model_override
        self.spawn_depth = spawn_depth
        self.is_async = is_async
        self.is_sync = not is_async
        self.isolation = isolation  # Phase2:worktree 隔离标记(在 run() 用)
        self.parent_provider = parent_provider
        self.parent_registry = parent_registry
        self.parent_gate = parent_gate
        self.cfg = cfg
        self.app = app
        self.ctx = ctx
        self.project_root = project_root
        self.root = root
        self.progress_cb = progress_cb
        # 续跑(T4):注入旧 agent_id 复用(指向既有侧链/meta);None→新生成 sub-{uuid12}。
        self.agent_id = agent_id if agent_id is not None else f"sub-{_uuid.uuid4().hex[:12]}"
        self.resume_from = resume_from  # 非空 → resume 模式:加载侧链为 history、跳过播种
        self._worktree_name = worktree_store_name(ctx.cwd)  # 侧链与主会话同 worktree/<name>/
        self.sidechain_path = paths.subagent_jsonl_path(
            self.root or paths.default_root(),
            ctx.session_id,
            project_root,
            self.agent_id,
            worktree_name=self._worktree_name,
        )

    async def run(self) -> SubagentReport:
        started = time.monotonic()
        meta_path = paths.subagent_meta_path(
            self.root or paths.default_root(),
            self.ctx.session_id,
            self.project_root,
            self.agent_id,
            worktree_name=self._worktree_name,
        )
        # #3:续跑保留旧 meta(原任务描述/派生时间/worktree 起点),仅靠下方 _update_meta 刷 status;
        # 若此处仍无条件 write(launched),prompt 会被覆写成 direction(=self.prompt,如"继续"——
        # reminder/可续跑清单就显示"继续"而非原任务)、spawnedAt 重置(计费丢首次派生时间)、
        # baseSha=None(worktree 续跑把首跑已 commit 的推进当起点 → 产物清单残缺)。旧 meta 缺失
        # (损坏/手工删)→ 回退写 fresh,使生命周期跟踪仍可用。
        resume_existing = read_subagent_meta(meta_path) if self.resume_from is not None else None
        if resume_existing is None:
            write_subagent_meta(
                meta_path,
                SubagentMeta(
                    agentId=self.agent_id,
                    agentType=self.defn.name,
                    description=self.description,
                    toolUseId=self.tool_use_id,
                    spawnDepth=self.spawn_depth,
                    isAsync=self.is_async,
                    model=self.defn.model,
                    status="launched",
                    spawnedAt=utc_iso(),
                    resolvedModel=None,
                    completedAt=None,
                    totalDurationMs=None,
                    totalTokens=None,
                    prompt=self.prompt,
                    isolation=self.isolation,
                    baseSha=None,
                ),
            )
        store = SubagentStore(self.ctx, self.project_root, self.agent_id, root=self.root)
        state = {"tokens": 0, "input_tokens": 0, "tool_use_count": 0}
        error_msg: str | None = None  # 护栏/Provider Error 的消息(_emit 写,run() 读)

        async def _emit(ev: ProviderEvent) -> None:
            nonlocal error_msg
            # 不向主 UI 流式展示文本;仅累计 tokens/工具数 + 节流进度。
            if isinstance(ev, ToolCallStart):
                state["tool_use_count"] += 1
            elif isinstance(ev, Error):
                # 护栏(max_tool_rounds / max_same_failures / 连续上下文超限)或 provider 自身
                # 出错时,run_agent_loop 是 emit(Error) 后【正常 return】(不抛异常)。若不在此
                # 捕获,run() 的正常返回路径会把失败子 agent 报成 status='completed'(空正文),
                # 主 agent 误以为成功。记录消息,run() 返回后据此判定 status='error'。
                error_msg = ev.message
            elif isinstance(ev, Done):
                # 本轮正常 Done:清除此前任何非终态 Error 的残留。护栏 Error(max_tool_rounds /
                # max_same_failures / 连续超限)后 run_agent_loop 立即 return、不会再有 Done,
                # 故 Done 必代表「成功完成一轮」——若不在此清,流中途 Error(已 yield token,
                # base_llm 不重试→Error+return)后下一轮恢复完成时 error_msg 仍残留 → 成功被
                # 误报 status='error'(反向 false-negative)。
                error_msg = None
                if ev.usage is not None:
                    # tokens:总开销(input+output)→ SubagentReport.totalTokens(完整统计)。
                    # input_tokens:减去缓存的输入(Anthropic input_tokens 已扣 cache_read/
                    # cache_creation)→ 进度卡 ↓ 显示(用户要"扣缓存的输入 token")。
                    state["tokens"] += ev.usage.input_tokens + ev.usage.output_tokens
                    state["input_tokens"] += ev.usage.input_tokens
                if self.progress_cb is not None:
                    from birdcode.agents.report import SubagentProgress

                    try:
                        await self.progress_cb(
                            SubagentProgress(
                                agent_id=self.agent_id,
                                description=self.description,
                                elapsed_ms=_elapsed_ms(started),
                                input_tokens=state["input_tokens"],
                                tool_use_count=state["tool_use_count"],
                                phase="running",
                            )
                        )
                    except Exception:  # noqa: BLE001 - progress 是 best-effort,绝不杀子 agent
                        log.debug("subagent progress_cb 失败,忽略(不杀子 agent)", exc_info=True)

        async def _on_message(
            msg: Message,
            *,
            usage: TokenUsage | None = None,
            stop_reason: str | None = None,
        ) -> None:
            await store.append(
                msg,
                is_assistant=(msg.role == "assistant"),
                usage=usage,
                stop_reason=stop_reason,
            )

        async def _progress_tick() -> None:
            """每秒刷一次进度(仅 elapsed_ms;tokens/tool_use_count 复用当前累计 state,
            仍只在每轮 Done 变化)。让长跑子 agent 的时间持续走,而非轮次粒度停滞。"""
            from birdcode.agents.report import SubagentProgress

            while True:
                await asyncio.sleep(1)
                if self.progress_cb is None:
                    continue
                try:
                    await self.progress_cb(
                        SubagentProgress(
                            agent_id=self.agent_id,
                            description=self.description,
                            elapsed_ms=_elapsed_ms(started),
                            input_tokens=state["input_tokens"],
                            tool_use_count=state["tool_use_count"],
                            phase="running",
                        )
                    )
                except Exception:  # noqa: BLE001 - tick best-effort,绝不杀子 agent
                    log.debug("subagent progress tick 失败,忽略", exc_info=True)

        tick_task: asyncio.Task[None] | None = None
        if self.progress_cb is not None:
            tick_task = asyncio.create_task(_progress_tick())
        # Phase 2 worktree 生命周期局部(init before try → finally 可引用)
        worktree_dir: Path | None = None
        # #3:续跑沿用旧 meta 的 worktree 起点 → collect 的净改变含首跑推进(产物清单完整);
        # 首跑 / 旧 meta 缺 base → ""(下文 worktree 块派生)。
        base_sha = (resume_existing.base_sha or "") if resume_existing is not None else ""
        _cwd_token: contextvars.Token[str | None] | None = None
        succeeded = False
        report: SubagentReport | None = None  # 终态报告;cancelled 分支留 None→raise
        try:
            # Phase 2:worktree 隔离 — create/lock/_AGENT_CWD 先于 build_provider
            # (系统提示 env 块用 worktree cwd)。失败 → except Exception 捕获;worktree_dir 未设
            # 则 finally 跳过 cleanup,已设则 cleanup(succeeded=False)→ 留目录。
            if self.isolation == "worktree":
                worktree_dir = await create_worktree(
                    self.project_root, self.agent_id, base_ref="origin/HEAD"
                )
                await lock_worktree(self.project_root, worktree_dir, "subagent active")
                _cwd_token = _AGENT_CWD.set(str(worktree_dir))
                # 记 worktree 起点 sha(本次开始前 HEAD;复用路径下=本次开始前状态)。
                # finally 头据此收集本次净改变。失败 → ""(collect 降级 [])。
                # #3:续跑已沿用旧 meta 的 base_sha(上方),不重派生——否则把首跑推进当起点。
                if not base_sha:
                    base_sha = await current_head_sha(worktree_dir) or ""
                    _update_meta(meta_path, baseSha=base_sha)  # 供续跑收产物清单
            profile = resolve_profile(
                self.defn.model, self.model_override, self.cfg, self.parent_provider
            )
            child_registry = build_child_registry(
                self.parent_registry,
                self.defn.disallowed_tools,
                is_async=self.is_async,
                cwd=str(worktree_dir) if worktree_dir else None,
            )
            child_gate = (
                build_child_gate(
                    self.parent_gate,
                    allow_hitl=self.is_sync,
                    worktree_dir=worktree_dir,
                )
                if self.parent_gate is not None
                else None
            )
            child_executor = ToolExecutor(child_registry, permission_gate=child_gate)
            # mcp_instructions 懒回调透传(Phase1 漏):与主 provider 一致传【绑定方法本身】
            # (build_provider 形参类型 Callable[[], dict[str, str]] | None,base_llm 在 stream 时
            # 调用——传 dict 会在 _system_text 调用时 TypeError)。app 无该方法或 app=None → None。
            mcp_instructions: Callable[[], dict[str, str]] | None = None
            if self.app is not None:
                getter = getattr(self.app, "mcp_instructions", None)
                if callable(getter):
                    mcp_instructions = getter
            child_provider = build_provider(
                profile,
                self.cfg,
                registry=child_registry,
                system_override=_system_override(self.defn, self.isolation),
                mcp_instructions=mcp_instructions,
            )
            # 一次性子 agent:不接 ContextManager(history 不跨轮累积,无需 maybe_compact)。
            child_context: ContextManager | None = None
            history: list[Turn]
            if self.resume_from is not None:
                # resume 模式(T4):侧链当 history(只重放旧轮次,含已执行工具——不重执,
                # 见 Global Constraints);direction(self.prompt)作新 user turn 落侧链
                # (规避 Claude Code #11712:resume 必须把用户新方向作为真实 user turn,
                # 而非靠推断旧回复)。跳过 fresh 播种(history 已有旧首 turn)。
                _update_meta(meta_path, status="running")  # 标记已重新在跑(供 T19 幂等守卫)
                # #2:经 store 加载+末尾修复+【持久化合成消息】(镜像 SessionStore.load_mainline)。
                # 仅内存修复会让磁盘侧链残留 orphan tool_use(中断期未落盘的 tool_result),
                # direction 追加后变成中段 orphan;二次中断 → decode 切多 Turn、repair 只看
                # turns[-1] → 下轮 API 400 或 tool_use 被静默剥离(子 agent 不知自己发过 bash,
                # 可能重做副作用)。传 self.resume_from 保留「从 resume_from 读」的契约(生产中
                # 与 store._jsonl 同路径)。
                history = await store.load_and_repair_sidechain(self.resume_from)
                turn = Turn(messages=[Message(role="user", content=[TextBlock(text=self.prompt)])])
                await store.append(turn.messages[0])
            else:
                _update_meta(meta_path, status="running")
                turn = Turn(messages=[Message(role="user", content=[TextBlock(text=self.prompt)])])
                # 侧链首行:user prompt。run_agent_loop 只对【自己追加】的消息调 on_message
                # (assistant / tool_result),不会回写初始 user turn 消息——与
                # TurnController._process 一致,此处手动落盘,使侧链含完整对话
                # (user + assistant ≥ 2 行)。
                await store.append(turn.messages[0])
                history = []
            # 一次性子 agent:跑一轮 run_agent_loop 到自然结束(无 tool_use)即止。
            # run_agent_loop 以 history 为只读先验上下文(context=child_context)。
            await run_agent_loop(
                provider=child_provider,
                turn=turn,
                history=history,
                executor=child_executor,
                emit=_emit,
                on_message=_on_message,
                context=child_context,
            )
            if error_msg is not None:
                # 护栏 / provider Error 终止:run_agent_loop 已正常 return,但子 agent 并未真正
                # 完成。报 error + 把护栏消息回传主 agent(否则空正文被当成功)。
                _update_meta(
                    meta_path,
                    status="error",
                    completedAt=utc_iso(),
                    totalDurationMs=_elapsed_ms(started),
                    totalTokens=state["tokens"],
                )
                report = SubagentReport(
                    is_completed=False,
                    text=f"子 agent 出错: {error_msg}",
                    status="error",
                    duration_ms=_elapsed_ms(started),
                    tokens=state["tokens"],
                    tool_use_count=state["tool_use_count"],
                )
            else:
                report_text = _final_assistant_text(turn)
                is_completed = is_terminal_report_text(report_text)
                duration = _elapsed_ms(started)
                report = SubagentReport(
                    is_completed=is_completed,
                    text=report_text,
                    status="completed",
                    duration_ms=duration,
                    tokens=state["tokens"],
                    tool_use_count=state["tool_use_count"],
                )
                succeeded = True  # Phase 2:worktree cleanup 用(finally 不引用 report.status)
                _update_meta(
                    meta_path,
                    status="completed",
                    completedAt=utc_iso(),
                    resolvedModel=getattr(profile, "model", None),
                    totalDurationMs=report.duration_ms,
                    totalTokens=report.tokens,
                )
        except asyncio.CancelledError:
            _update_meta(
                meta_path,
                status="cancelled",
                completedAt=utc_iso(),
                totalDurationMs=_elapsed_ms(started),
                totalTokens=state["tokens"],
            )
            store.close()
            raise  # 同步:让取消继续上传到主 agent_loop
        except Exception as exc:  # noqa: BLE001
            _update_meta(
                meta_path,
                status="error",
                completedAt=utc_iso(),
                totalDurationMs=_elapsed_ms(started),
                totalTokens=state["tokens"],
            )
            report = SubagentReport(
                is_completed=False,
                text=f"子 agent 异常: {exc}",
                status="error",
                duration_ms=_elapsed_ms(started),
                tokens=state["tokens"],
                tool_use_count=state["tool_use_count"],
            )
        finally:
            # 收集 worktree 产物(best-effort,在 cleanup 删 worktree 之前;report 非 None 才填)。
            if worktree_dir is not None and report is not None:
                await _fill_artifacts(report, worktree_dir, base_sha)
            if tick_task is not None:
                tick_task.cancel()
            store.close()
            # Phase 2 worktree 清理:_AGENT_CWD 复位 + cleanup_subagent_worktree
            if _cwd_token is not None:
                _AGENT_CWD.reset(_cwd_token)
            if worktree_dir is not None:
                # shield:Esc(cancel_all→manager.cancel_all)落在清理窗口时,让 git 清理跑完
                # (免 worktree 留 git-locked)。已有终态报告(completed/error,report 非 None)→
                # 交付结果,不因取消被 _supervise 改写成 'cancelled';run_agent_loop 被取消
                # (report None)→ 维持取消语义上传。
                cleanup_task = asyncio.create_task(
                    cleanup_subagent_worktree(self.project_root, worktree_dir, succeeded=succeeded)
                )
                try:
                    await asyncio.shield(cleanup_task)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    # 清理被取消(Esc)或抛非取消异常(_run_git OSError/EMFILE/git 不可用):
                    # best-effort 等 shield 内的清理跑完(免 git-locked/孤儿),再按是否有终态
                    # 报告决定。两路径对称:report 非 None(completed/error)→ 交付,绝不因清理
                    # 失败/取消掩盖已完成的报告;report None(run_agent_loop 被取消)→ 维持取消上传。
                    try:
                        await cleanup_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        log.debug("子 agent worktree 清理失败,可能留孤儿目录/分支", exc_info=True)
                    if report is not None:
                        return report  # 已终态:清理失败/取消不该丢报告
                    raise  # 无报告(本就被取消):维持取消语义上传
        assert report is not None  # cancelled 分支已 raise;completed/error 已赋值
        return report


# 子 agent「进行中」报告的魔法前缀(runner 写的中途快照 "## 待办清单 ...")。
# runner 完成判定与 resume 交叉校验(_sidechain_looks_complete)共用此常量,避免两处分叉
# (曾致 review #1:resume 漏过滤 synthetic 占位,把进行中的「## 待办清单」翻转成「已完成」)。
_TERMINAL_REPORT_PREFIX = "## 待办清单"


def is_terminal_report_text(text: str) -> bool:
    """子 agent 报告正文是否表示【已完成】(非进行中的待办清单)。

    runner 中途会把进行状态写成形如「## 待办清单\\n- step」的快照——以此为前缀 → 返 False
    (未完成);其余(含空文本)→ True(完成)。空文本语义因调用方而异:runner 视空报告为
    完成(返 True);resume 的 _sidechain_looks_complete 自行先判空返 False,再调本函数。
    """
    return not text.lstrip().startswith(_TERMINAL_REPORT_PREFIX)


def _final_assistant_text(turn: Turn) -> str:
    """最后一条 assistant 的报告正文。

    优先 text block;若无则回退【同一条】assistant 的 thinking block——reasoning 模型
    (如 deepseek-v4)可能整轮把最终答复写进 thinking、stop_reason=end_turn 却无 text,
    此时只有取 thinking 才能拿到报告。只看最后一条 assistant,不回退更早轮次:更早轮的
    text 是中间话(如「我再确认一下…」),曾致完成通知正文变成中间问句、真正报告丢失。
    """
    for msg in reversed(turn.messages):
        if msg.role != "assistant":
            continue
        for b in msg.content:
            if isinstance(b, TextBlock) and b.text:
                return b.text
        for b in msg.content:
            if isinstance(b, ThinkingBlock) and b.text:
                return b.text
        return ""
    return ""
