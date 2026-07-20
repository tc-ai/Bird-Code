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
from birdcode.agents.runner import SubagentRunner, _update_meta, is_terminal_report_text
from birdcode.blocks import TextBlock, ToolUseBlock
from birdcode.conversation import Message, Turn
from birdcode.session.codec import load_sidechain_turns
from birdcode.session.paths import subagent_jsonl_path, subagent_meta_path
from birdcode.session.subagent_meta import read_subagent_meta
from birdcode.utils.logging import get_logger
from birdcode.utils.worktree import create_worktree, remove_worktree, worktree_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from birdcode.agents.registry import AgentRegistry
    from birdcode.agents.runner import ParentProvider
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
    agent_registry: AgentRegistry  # .resolve(agent_type) 查 AgentDefinition
    parent_provider: ParentProvider
    parent_registry: ToolRegistry | None
    parent_gate: PermissionGate | None
    cfg: AppConfig
    app: object | None  # 对齐 runner.__init__ 的 app: object | None
    ctx: SessionContext
    spawn_depth: int = 1
    progress_cb: Callable[..., Awaitable[None]] | None = None


@dataclass
class ResumeResult:
    # sync_done:inline 结果在 .text / async_launched:完成走通知 / in_progress:已在跑
    outcome: str
    text: str = ""  # sync 模式的 inline 报告(async 模式为 ack)


async def _finalize_resume(
    deps: ResumeDeps, meta_path: Path, agent_id: str, status: str
) -> None:
    """resume 短路返回(完成/空侧链/worktree-mismatch)时收敛:meta 非终态 → 写 completed;
    落 queue-operation dequeue 标记。使 _resume_pending_notifications 下次把该 agentId 视为
    已处理(不再 re-hint,修窗口 X/Y 死循环)。best-effort(meta 写失败不杀 resume)。"""
    if status not in ("completed", "error"):
        try:
            _update_meta(meta_path, status="completed")
        except Exception:  # noqa: BLE001
            log.debug("_finalize_resume 写 meta 失败 aid=%s(best-effort)", agent_id, exc_info=True)
    await deps.manager.mark_resumed(agent_id, "completed")


async def _cleanup_orphan_worktree(project_root: Path, agent_id: str) -> None:
    """worktree 状态不符时强制清理孤儿(remove_worktree force=True),下次 create_worktree
    重建干净。best-effort(失败仅 log)。"""
    wt = worktree_path(project_root, agent_id)
    try:
        await remove_worktree(project_root, wt, force=True)
    except Exception:  # noqa: BLE001
        log.debug("清理孤儿 worktree 失败 wt=%s(best-effort)", wt, exc_info=True)


async def resume_subagent(*, agent_id: str, direction: str, deps: ResumeDeps) -> ResumeResult:
    meta_path = subagent_meta_path(
        deps.root,
        deps.session_id,
        deps.project_root,
        agent_id,
        worktree_name=deps.worktree_name,
    )
    meta = read_subagent_meta(meta_path)
    if meta is None:
        return ResumeResult(outcome="sync_done", text=f"无法续跑:{agent_id} 的 meta 缺失或损坏")

    # lost = 用户按 2/No 永久丢弃,不可续。discover 不返回 lost、reminder 不列 lost,
    # 正常不会触发;此守卫纯 belt-and-suspenders(防陈旧 reminder 误导主 agent 调 resume_agent)。
    if meta.status == "lost":
        return ResumeResult(
            outcome="sync_done",
            text=f"{agent_id} 已被用户丢弃，不再续跑",
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
        deps.root,
        deps.session_id,
        deps.project_root,
        agent_id,
        worktree_name=deps.worktree_name,
    )
    # 完成判定 + 空侧链:前置于 worktree probe(completed/空侧链不重建 worktree,#4)。
    # 短路返回时 _finalize_resume 写 meta + 落 dequeue 收敛标记(修窗口 X/Y 死循环,#3)。
    turns = _load_sidechain_turns_safe(sidechain)
    if meta.status == "completed" or _sidechain_looks_complete(sidechain):
        final = _read_sidechain_final_text(sidechain)
        log.info("resume:agent %s 完成(meta=%s) → 返回侧链最后内容", agent_id, meta.status)
        await _finalize_resume(deps, meta_path, agent_id, meta.status)
        return ResumeResult(
            outcome="sync_done",
            text=final or f"{agent_id} 完成但侧链无可用报告正文",
        )
    # 空侧链(无真实 assistant 产出)→ 无可恢复内容 → 返回「interrupted,无输出」+ 收敛标记。
    # _has_assistant_output 跳过 repair 补的 synthetic 占位(仅种子 user 的侧链会被 repair 补一条
    # synthetic assistant,文案不像终态 → completed 不命中;此处拦截返「无输出」)。
    if not _has_assistant_output(turns):
        log.info("空侧链:agent %s 无 assistant 产出 → 标 interrupted 无输出,不续跑", agent_id)
        await _finalize_resume(deps, meta_path, agent_id, "completed")
        return ResumeResult(
            outcome="sync_done",
            text=f"{agent_id} 中断过早,无可恢复产出(interrupted,无输出)",
        )

    # worktree probe(仅未完成才探查:completed/空已上面短路,不重建孤儿 worktree)。
    # 状态不符(分支被改/detached)→ FileExistsError → force-clean 孤儿(#5)+ 返回侧链 + 收敛标记。
    # 同步探查因 runner.run() 的 except 会吞 FileExistsError,故派发前探查对 sync/async 统一生效。
    if meta.isolation == "worktree":
        try:
            await create_worktree(deps.project_root, agent_id)
        except (FileExistsError, RuntimeError) as e:
            log.warning("worktree %s 状态不符/创建失败,清理孤儿 + 返回侧链最后:%s", agent_id, e)
            await _cleanup_orphan_worktree(deps.project_root, agent_id)  # force-clean 自愈(#5)
            final = _read_sidechain_final_text(sidechain)
            await _finalize_resume(deps, meta_path, agent_id, meta.status)
            return ResumeResult(
                outcome="sync_done",
                text=final or f"{agent_id} worktree 状态不符,无法续跑(侧链无可用报告)",
            )

    runner = SubagentRunner(
        defn=defn,
        prompt=direction,
        description=meta.description or defn.description,
        tool_use_id=meta.tool_use_id,
        model_override="",
        spawn_depth=deps.spawn_depth,
        is_async=meta.is_async,
        # meta.isolation 是 str|None,但数据值仅 None/"worktree"(本系统只写这两值);
        # cast 到 runner 期望的 Literal[None,"worktree"],勿让任意 str 静默通过。
        isolation=cast(Literal[None, "worktree"], meta.isolation),
        parent_provider=deps.parent_provider,
        parent_registry=deps.parent_registry,
        parent_gate=deps.parent_gate,
        cfg=deps.cfg,
        app=deps.app,
        ctx=deps.ctx,
        project_root=deps.project_root,
        root=deps.root,
        progress_cb=deps.progress_cb,
        agent_id=agent_id,
        resume_from=sidechain,  # 复用原 id + resume 模式
    )

    if meta.is_async:
        ack = await deps.manager.launch_async(runner)  # 后台;完成走 _inject→notification
        return ResumeResult(outcome="async_launched", text=ack.ack_text)
    report = await runner.run()  # sync:阻塞 inline
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

    本函数是侧链最终报告正文的唯一读取点(早期 ui 层有同名镜像,已统一到此处);
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
