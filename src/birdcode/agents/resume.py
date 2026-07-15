# src/birdcode/agents/resume.py
"""续跑核心(触发无关):复用 agent_id + 加载侧链 + 按 is_async 派发。所有续跑经此入口。

触发方(resume UI / "继续" 路由 / 将来的 ESC-continue)调本模块;授权闸门在触发方,
本模块假定已授权。幂等:manager.has_live(agent_id) 为真(已有活 task)→ 直接返回 in_progress,
不重复 launch。

回连用 agent_id 查 meta(绝不用 tool_use_id——sync/async 的 tool_use_id 都是占位符)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from birdcode.agents.manager import SubagentManager
from birdcode.agents.runner import SubagentRunner, is_terminal_report_text
from birdcode.blocks import TextBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.session.codec import load_sidechain_turns
from birdcode.session.models import SubagentMeta
from birdcode.session.paths import subagent_jsonl_path, subagent_meta_path
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta
from birdcode.session.timeutil import utc_iso
from birdcode.utils.logging import get_logger
from birdcode.utils.worktree import create_worktree, remove_worktree, worktree_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from birdcode.agents.definition import AgentDefinition
    from birdcode.agents.registry import AgentRegistry
    from birdcode.agents.runner import ParentProvider, SubagentReport
    from birdcode.config.schema import AppConfig
    from birdcode.permission.gate import PermissionGate
    from birdcode.session.models import SessionContext
    from birdcode.tools.registry import ToolRegistry

log = get_logger("birdcode.agents.resume")


@dataclass
class ResumeDeps:
    """resume_subagent 所需依赖(由触发方/App 注入)。"""

    manager: SubagentManager
    root: Path
    session_id: str
    project_root: Path
    worktree_name: str | None
    # runner 构造依赖(复用 _AgentTool 那套)
    agent_registry: AgentRegistry            # .resolve(agent_type) 查 AgentDefinition
    parent_provider: ParentProvider
    parent_registry: ToolRegistry | None
    parent_gate: PermissionGate | None
    cfg: AppConfig
    app: object | None                       # 对齐 runner.__init__ 的 app: object | None
    ctx: SessionContext
    spawn_depth: int = 1
    progress_cb: Callable[..., Awaitable[None]] | None = None


@dataclass
class ResumeResult:
    # sync_done:inline 结果在 .text / async_launched:完成走通知 / in_progress:已在跑
    outcome: str
    text: str = ""      # sync 模式的 inline 报告(async 模式为 ack)


async def resume_subagent(*, agent_id: str, direction: str, deps: ResumeDeps) -> ResumeResult:
    meta_path = subagent_meta_path(
        deps.root, deps.session_id, deps.project_root, agent_id,
        worktree_name=deps.worktree_name,
    )
    meta = read_subagent_meta(meta_path)
    if meta is None:
        return ResumeResult(outcome="sync_done", text=f"无法续跑:{agent_id} 的 meta 缺失或损坏")

    # 显式 completed 拒绝续跑:discover 不会列出 completed,但主 agent 可能被陈旧 reminder
    # (一次性注入、不随 meta 刷新)误导而对【已完成】agent 调 resume_agent。此前无此守卫 → T17
    # 只管「侧链像终态」的启发式,显式 completed 会落到构造 runner + run() → 把已完成 agent 从头
    # 重跑(配合旧 meta 被覆写成 direction,连原任务都丢)。completed 已无未完成工作可续,拒绝
    # 严格正确(legitimate 续跑不会指向 completed——reminder 只列非终态)。error(可重试)/
    # cancelled(用户显式中断)仍可续。
    if meta.status == "completed":
        return ResumeResult(
            outcome="sync_done",
            text=f"{agent_id} 已完成,无需续跑",
        )

    if deps.manager.has_live(agent_id):  # 幂等:已在跑,不重复 launch
        return ResumeResult(outcome="in_progress", text=f"{agent_id} 已在运行中")

    # AgentRegistry 真实方法名是 .resolve(非 .get);返回 None → 该 agent 类型已不存在
    defn = deps.agent_registry.resolve(meta.agent_type)
    if defn is None:
        return ResumeResult(
            outcome="sync_done", text=f"无法续跑:agent 类型 {meta.agent_type} 已不存在"
        )

    sidechain = subagent_jsonl_path(
        deps.root, deps.session_id, deps.project_root, agent_id,
        worktree_name=deps.worktree_name,
    )
    # worktree 子 agent 续跑:派发前同步探查 create_worktree(create_worktree 自愈 stale lock +
    # 复用 agent_id)。状态不符(分支被改/detached)→ FileExistsError → 降级注入 + 清孤儿。
    #
    # 据实调整(brief 原想 try/except 包住 launch_async/run()):(1) 异步路径异常在后台 _supervise
    # task 内被 manager 兜成 error 通知,不冒泡到本函数;(2) 同步路径 runner.run() 的 except Exception
    # 会吞 FileExistsError。两路都接不到 → 改在派发前【同步探查】,对 sync/async 统一生效。
    # runner.run() 二次调 create_worktree 命中快速恢复路径(dir 在 + HEAD 匹配 → 直接复用,不调 git),
    # 幂等无副作用。非 worktree(isolation=None)完全不受影响。
    #
    # 顺序:worktree 探查在 T17 交叉校验之前——状态不符的 worktree 无论侧链是否像终态都需要
    # force 清理(_degrade_to_inject 含 remove_worktree);若先走 T17 会留孤儿 worktree。
    if meta.isolation == "worktree":
        try:
            await create_worktree(deps.project_root, agent_id)
        except (FileExistsError, RuntimeError) as e:
            # FileExistsError=状态不符(分支被改/detached);RuntimeError=无 .git / git worktree
            # add 失败(锁竞争/磁盘满/索引损坏)。两者都降级注入 + 清孤儿,不冒泡成工具异常
            # (write 工具契约:返回友好文本而非抛)。
            log.warning("worktree %s 状态不符/创建失败,降级注入:%s", agent_id, e)
            return await _degrade_to_inject(meta, deps, agent_id, sidechain)

    # 空侧链检查(T18):侧链无任何真实 assistant 产出(子 agent 死太早:只有种子 user、缺文件、
    # 全坏行)→ 无可恢复内容 → 不续跑,返回「interrupted,无输出」友好文本。
    #
    # 顺序:必在 T17 交叉校验之前——仅种子 user 的侧链经 repair_trailing_edge 会补一条 synthetic
    # 占位 assistant(文案不以「## 待办清单」开头)→ T17 会误判「像终态」当完成注入,但其实无任何
    # 真实产出;T18 提前拦截(_has_assistant_output 跳过 synthetic 占位)返「无输出」。在 T14 worktree
    # 探查之后:T14 的降级清理(状态不符 worktree)仍优先返回,本块只处理 worktree 已自愈/非 worktree。
    turns = _load_sidechain_turns_safe(sidechain)
    if not _has_assistant_output(turns):
        log.info("空侧链:agent %s 无 assistant 产出 → 标 interrupted 无输出,不续跑", agent_id)
        return ResumeResult(
            outcome="sync_done",
            text=f"{agent_id} 中断过早,无可恢复产出(interrupted,无输出)",
        )

    # 交叉校验(T17):meta 非终态(running/idle 等,可能未及时更新)但侧链末尾 assistant 像终态
    # 报告(子 agent 实际完成了,只是 meta 没刷)→ 当"完成未注入"处理:落点 A 注入,不续跑
    # (避免把已完成的当未完成重跑)。判不准(无文本 / "## 待办清单" 头)→ 走续跑(明确像终态
    # 才注入)。terminal meta 不属本路径职责。worktree 已自愈/非 worktree 才到此。
    if (
        meta.status not in ("completed", "error", "cancelled")
        and _sidechain_looks_complete(sidechain)
    ):
        log.info("交叉校验:agent %s meta=%s 但侧链像终态 → 注入不续跑", agent_id, meta.status)
        return await _inject_as_completed(meta, deps, agent_id, sidechain)

    runner = SubagentRunner(
        defn=defn, prompt=direction, description=meta.description or defn.description,
        tool_use_id=meta.tool_use_id, model_override="", spawn_depth=deps.spawn_depth,
        is_async=meta.is_async,
        # meta.isolation 是 str|None,但数据值仅 None/"worktree"(本系统只写这两值);
        # cast 到 runner 期望的 Literal[None,"worktree"],勿让任意 str 静默通过。
        isolation=cast(Literal[None, "worktree"], meta.isolation),
        parent_provider=deps.parent_provider, parent_registry=deps.parent_registry,
        parent_gate=deps.parent_gate, cfg=deps.cfg, app=deps.app, ctx=deps.ctx,
        project_root=deps.project_root, root=deps.root, progress_cb=deps.progress_cb,
        agent_id=agent_id, resume_from=sidechain,         # 复用原 id + resume 模式
    )

    if meta.is_async:
        ack = await deps.manager.launch_async(runner)     # 后台;完成走 _inject→notification
        return ResumeResult(outcome="async_launched", text=ack.ack_text)
    report = await runner.run()                           # sync:阻塞 inline
    return ResumeResult(outcome="sync_done", text=report.text)


def _load_sidechain_turns_safe(sidechain_path: Path) -> list[Turn]:
    """读侧链 turns(容错):文件缺/全坏行/读异常 → [](等价无产出,供 _has_assistant_output 判 False)。

    load_sidechain_turns 自身文档承诺文件缺/全坏行 → []不抛;此处额外 try/except 兜底
    (任何读失败 = 无可恢复产出,按空处理,不杀续跑入口)。镜像 _read_sidechain_final_text 的防御。
    """
    if not sidechain_path.exists():
        return []
    try:
        return load_sidechain_turns(sidechain_path)
    except Exception:  # noqa: BLE001 - 读失败按「无产出」处理,不杀续跑入口
        return []


def _has_assistant_output(turns: list[Turn]) -> bool:
    """侧链 turns 里是否有真实(非 synthetic)assistant 产出。

    判据:存在任意 role==assistant 且 synthetic=False 的消息 → True;否则 False。空侧链
    (只有种子 user)/ 缺文件 → load_sidechain_turns 返回 [] 或仅含 user 的 Turn;此时
    repair_trailing_edge 会为「末尾 user」补一条 synthetic=True 的占位 assistant(收尾用,
    非 LLM 产出,文案「(上一轮因会话中断未完成,已自动补全收尾)」)—— synthetic 占位不算真实
    产出,故必须用 not msg.synthetic 排除,否则仅种子 user 的侧链会被误判有产出。
    """
    for turn in turns:
        for msg in turn.messages:
            if msg.role == "assistant" and not msg.synthetic:
                return True
    return False


def _read_sidechain_final_text(sidechain_path: Path) -> str | None:
    """读侧链 jsonl 最后一条 assistant 文本块(子 agent 报告正文)。缺/坏 → None。

    镜像 ui.app._read_sidechain_final_text,但不跨层 import ui(agents ← ui 是反向依赖);
    复用 session.codec.load_sidechain_turns(文件缺/全坏行 → [],不抛)。
    """
    if not sidechain_path.exists():
        return None
    try:
        turns = load_sidechain_turns(sidechain_path)
    except Exception:  # noqa: BLE001 - 侧链读失败不杀降级路径
        return None
    for turn in reversed(turns):
        for msg in reversed(turn.messages):
            # 跳过 synthetic 占位/收尾(repair 补的「(上一轮因会话中断…已自动补全收尾)」):
            # 否则进行中侧链 [seed, assistant("## 待办清单…", tool_use_X)] 经 repair 末尾补一条
            # synthetic assistant 占位 → 本函数命中占位 → _sidechain_looks_complete 误判「像终态」
            # → 把进行中真实产出当完成注入、不续跑。与 _has_assistant_output 同用 not synthetic。
            if msg.role != "assistant" or msg.synthetic:
                continue
            for b in msg.content:
                if isinstance(b, TextBlock) and b.text:
                    return b.text
    return None


def _sidechain_looks_complete(sidechain_path: Path) -> bool:
    """侧链是否像【自然完成】的报告(交叉校验:meta 可能未及时刷成 completed)。

    首选【权威信号】:末轮 ``stop_reason``(`decode_lines`` 从末条真实 assistant 行回填到 Turn)。
    - ``end_turn``:模型自然结束本轮(纯文本收尾、无待执工具)→ 视为完成、注入不续跑。
    - ``tool_use`` / ``max_tokens`` / ``stop_sequence`` 等:停在调工具/截断/停序列 → 未完成 → 续跑。
    这正是 API 的完成语义,远比"看文本是否以 ## 待办清单 开头"可靠(后者无法区分中间过渡文本与
    终态报告——实测 explore 子 agent 死在 grep 中途,末条 assistant 是 [过渡文本 + tool_use]、
    stop_reason=tool_use,旧文本启发式误判完成、注入占位吞掉产出)。

    回退(无 stop_reason:其它 provider 不发 / 旧会话缺字段):末条真实 assistant 含 tool_use → 中途;
    否则用文本启发式(非「## 待办清单」头 → 像完成)。即 stop_reason 缺席时退回 tool_use 代理 +
    文本判据。

    取向:判不准→【续跑】。把真完成当未完成重跑只是浪费;把真未完成当完成(注入占位)会丢产出——
    更糟。文本判据与 runner 共用 ``is_terminal_report_text``(单一来源)。
    """
    turns = _load_sidechain_turns_safe(sidechain_path)
    last_real_asst: Message | None = None
    for turn in reversed(turns):
        for msg in reversed(turn.messages):
            if msg.role == "assistant" and not msg.synthetic:
                last_real_asst = msg
                break
        if last_real_asst is not None:
            break
    if last_real_asst is None:
        return False  # 无真实 assistant(_has_assistant_output 已更早拦截,此处兜底)
    # 权威:末轮 stop_reason = 末条真实 assistant 的 stop_reason(end_turn=自然完成)。
    stop_reason = turns[-1].stop_reason if turns else None
    if stop_reason is not None:
        return stop_reason == "end_turn"
    # 回退:无 stop_reason —— tool_use 代理(末条 assistant 含 tool_use → 中途 → 续跑)。
    if any(isinstance(b, ToolUseBlock) for b in last_real_asst.content):
        return False
    text = ""
    for b in last_real_asst.content:
        if isinstance(b, TextBlock) and b.text:
            text = b.text
            break
    if not text.strip():
        return False
    return is_terminal_report_text(text)


async def _inject_task_notification(
    meta: SubagentMeta, deps: ResumeDeps, agent_id: str,
    report: SubagentReport, defn: AgentDefinition,
) -> tuple[str, bool]:
    """落点 A 三步注入(镜像 manager._inject):enqueue + is_task_notification user 行 + notify_wake。

    供 _degrade_to_inject(worktree 状态不符降级,T14)与 _inject_as_completed(交叉校验已完成,
    T17)共用,避免注入逻辑重复。返回 (notification_text, injected):store/controller 缺失
    (测试桩/无 manager 内部)或注入异常 → injected=False,notification 文本仍返回(由调用方
    兜底回传,不丢产出)。queue_operation 的 status 跟随 report.status(completed/error 各自语义)。
    """
    from birdcode.agents.report import build_task_notification

    notification = build_task_notification(
        report, defn, agent_id=agent_id, prompt=meta.prompt or "",
        is_worktree=(meta.isolation == "worktree"),
    )
    store = getattr(deps.manager, "_store", None)
    controller = getattr(deps.manager, "_controller", None)
    if store is None or controller is None:
        return notification, False
    try:
        await store.append_queue_operation(
            operation="enqueue", agent_id=agent_id,
            tool_use_id=meta.tool_use_id, status=report.status,
        )
        await store.append(
            Message(role="user", content=[TextBlock(text=notification)]),
            is_task_notification=True, agent_id=agent_id,
        )
        controller.notify_wake()
    except Exception:  # noqa: BLE001 - 注入失败不杀调用方路径
        log.debug("通知注入失败 agent_id=%s", agent_id, exc_info=True)
        return notification, False
    return notification, True


def _mark_meta_terminal(
    meta: SubagentMeta, deps: ResumeDeps, agent_id: str,
    *, status: Literal["completed", "error", "cancelled"],
) -> None:
    """注入完成通知后把 meta 刷成终态(completed/error),同步 completed_at。

    修复 final review Finding:此前 _inject_as_completed / _degrade_to_inject 只注入通知、
    不更 meta.status → meta 停在非终态(running/idle)。后果:(a) 再次显式 resume 同 agent
    → T17 交叉校验重复触发(侧链仍像终态、meta 仍非终态)→ 重复 enqueue+notification;
    (b) find_resumable_subagents 永远扫到该非终态 meta,每会话 reminder 都列一个实际已完成的
    agent,直到 /clear。meta_path 计算镜像 resume_subagent 顶部。write_subagent_meta 自身
    IO 失败只 log,不杀注入路径(通知已落,meta 漏刷是降级,不是数据丢失)。
    """
    meta_path = subagent_meta_path(
        deps.root, deps.session_id, deps.project_root, agent_id,
        worktree_name=deps.worktree_name,
    )
    updated = meta.model_copy(update={"status": status, "completed_at": utc_iso()})
    write_subagent_meta(meta_path, updated)


async def _inject_as_completed(
    meta: SubagentMeta, deps: ResumeDeps, agent_id: str, sidechain: Path,
) -> ResumeResult:
    """交叉校验:meta 非终态但侧链末尾 assistant 像终态报告 → 当"完成未注入"处理(落点 A 注入)。

    复用 T14 _degrade_to_inject 的三步注入(经 _inject_task_notification 公共 helper)。report
    标 is_completed=True / status="completed"(子 agent 实际完成了,只是 meta 没刷)。不续跑、
    不清 worktree(非状态不符;worktree 若存在由 runner finally 正常清理或后续 GC,不在本路径职责)。
    """
    from birdcode.agents.definition import AgentDefinition
    from birdcode.agents.runner import SubagentReport

    final_text = _read_sidechain_final_text(sidechain)
    if not final_text:
        final_text = "(子 agent 侧链显示已完成,但无可用报告正文)"
    defn = AgentDefinition(
        name=meta.agent_type or "subagent",
        description=meta.description or "子 agent",
        system_prompt="",
    )
    report = SubagentReport(
        is_completed=True, text=final_text, status="completed",
        duration_ms=0, tokens=0, tool_use_count=0,
    )
    notification, injected = await _inject_task_notification(
        meta, deps, agent_id, report, defn,
    )
    # 标终态:meta 停在非终态(running/idle)会导致(a)再次 resume 重复触发交叉校验 → 重复
    # 注入+notification;(b)find_resumable_subagents 永久列为可续跑。注入完成即刷 status=completed。
    _mark_meta_terminal(meta, deps, agent_id, status="completed")
    if injected:
        text = f"子 agent {agent_id} 侧链显示已完成,已补发完成通知(不续跑)"
    else:
        text = notification  # 无 store/controller:notification 兜底回传,不丢产出
    return ResumeResult(outcome="sync_done", text=text)


async def _degrade_to_inject(
    meta: SubagentMeta, deps: ResumeDeps, agent_id: str, sidechain: Path,
) -> ResumeResult:
    """worktree 状态不符无法续跑 → 降级:侧链最后产出包成落点 A notification 注入 + force 清孤儿。

    不抛、不续跑。注入走 _inject_task_notification 公共 helper(T17 抽出,镜像 manager._inject
    三步:enqueue + is_task_notification user 行 + notify_wake),使主 agent 下一轮看到「部分产出」
    完成通知。store/controller 缺失(测试桩/无 manager 内部)→ injected=False,notification 进
    ResumeResult.text 兜底回传(不丢产出)。清孤儿用 remove_worktree(force=True)(状态不符可能
    脏/锁;cleanup_subagent_worktree 无 force 参,且 succeeded=False 留目录、succeeded=True 用
    force=False 对脏目录会失败 → 直接 remove_worktree(force=True) 最贴合「清孤儿」语义)。
    """
    from birdcode.agents.definition import AgentDefinition
    from birdcode.agents.runner import SubagentReport

    final_text = _read_sidechain_final_text(sidechain)
    if not final_text:
        final_text = "(子 agent 因 worktree 状态不符无法续跑,侧链无可用产出)"
    defn = AgentDefinition(
        name=meta.agent_type or "subagent",
        description=meta.description or "子 agent",
        system_prompt="",
    )
    report = SubagentReport(
        is_completed=False, text=final_text, status="error",
        duration_ms=0, tokens=0, tool_use_count=0,
    )
    notification, injected = await _inject_task_notification(
        meta, deps, agent_id, report, defn,
    )
    # 标终态:worktree 状态不符属异常,与上方 report.status / queue-operation status 一致标 error。
    # 不选 cancelled(cancelled 是可续跑状态,会继续被 discover 列为可续跑;且语义是用户中断,不符)。
    _mark_meta_terminal(meta, deps, agent_id, status="error")
    # force 清孤儿 worktree(状态不符,可能脏/锁)。路径经 worktree_path 单一来源(与 create 一致)。
    wt_path = worktree_path(deps.project_root, agent_id)
    try:
        await remove_worktree(deps.project_root, wt_path, force=True)
    except Exception:  # noqa: BLE001 - 清理失败不杀降级路径
        log.debug("清孤儿 worktree 失败 %s", wt_path, exc_info=True)
    if injected:
        text = f"子 agent {agent_id} 的 worktree 状态不符,已降级注入其部分产出"
    else:
        text = notification  # 无 store/controller:notification 兜底回传,不丢产出
    return ResumeResult(outcome="sync_done", text=text)
