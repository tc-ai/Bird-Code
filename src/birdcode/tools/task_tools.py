# src/birdcode/tools/task_tools.py
"""agent teams:task 工具——共享看板 CRUD(TaskCreate/List/Update)。

team-scoped(持 TeamManager),同 SendMessage 模式:注册进 lead 主 registry,teammate 经
build_child_registry fork 返 self 继承。created_by 取 _AGENT_NAME(lead→"lead" / teammate→其名)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from birdcode.agents.mailbox import _AGENT_NAME
from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.agents.teammate import TeamManager


class TaskCreateInput(BaseModel):
    title: str = Field(..., description="任务标题")
    assignee: str | None = Field(None, description="指派给某 teammate(可选;push 模式由 lead 设)")
    blocked_by: list[str] = Field(
        default_factory=list,
        description="依赖的 task id 列表;这些全部完成后本任务才从 blocked 解锁为 pending",
    )


class TaskCreateTool(Tool):
    name = "TaskCreate"
    description = (
        "在共享看板上建一个任务。可选指派给 teammate(push)、声明依赖 blocked_by"
        "(依赖完成后才解锁可抢)。"
    )
    parameters = TaskCreateInput
    kind = "write"
    parallel_safe = False
    gate_exempt = True  # 纯内存(board dict),无 FS/命令副作用
    team_scoped = True  # 仅 teammate(mailbox)继承;一次性 subagent 排除

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(
        self, *, title: str, assignee: str | None = None,
        blocked_by: list[str] | None = None,
    ) -> str:
        t = self.team.task_board.create(
            title=title, assignee=assignee, created_by=_AGENT_NAME.get(),
            blocked_by=blocked_by or [],
        )
        suffix = f"(→{assignee})" if assignee else ""
        dep = f"(依赖:{','.join(t.blocked_by)})" if t.blocked_by else ""
        return f"已建任务 #{t.id}: {t.title}{suffix}{dep}"


class TaskListInput(BaseModel):
    status: Literal["pending", "blocked", "in_progress", "completed"] | None = Field(
        None,
        description="状态过滤(pending/blocked/in_progress/completed);None=全部",
    )


class TaskListTool(Tool):
    name = "TaskList"
    description = (
        "列出看板上任务(id/status/title/assignee)。可选按 status 过滤"
        "(pull 时常用 pending 看可抢的)。"
    )
    parameters = TaskListInput
    kind = "read"
    parallel_safe = True
    team_scoped = True  # 仅 teammate(mailbox)继承;一次性 subagent 排除

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(self, *, status: str | None = None) -> str:
        tasks = self.team.task_board.list()
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if not tasks:
            return "(无任务)"
        lines = []
        for t in tasks:
            a = f" →{t.assignee}" if t.assignee else ""
            dep = f" 依赖:{','.join(t.blocked_by)}" if t.blocked_by else ""
            lines.append(f"#{t.id} [{t.status}] {t.title}{a}{dep}")
        return "\n".join(lines)


class TaskUpdateInput(BaseModel):
    id: str = Field(..., description="任务 id")
    status: Literal["pending", "in_progress", "completed"] | None = Field(
        None,
        description=(
            "新状态:pending|in_progress|completed。领取任务请用 TaskClaim(原子 CAS),"
            "勿直接设 in_progress——后者非原子,双人抢同一任务会 last-writer-wins。"
            "blocked 不可手动设(只能由依赖产生/解锁)"
        ),
    )
    assignee: str | None = Field(None, description="改指派(可选;空串/纯空白等同未指派)")


class TaskUpdateTool(Tool):
    name = "TaskUpdate"
    description = "更新任务状态或指派。"
    parameters = TaskUpdateInput
    kind = "write"
    parallel_safe = False
    gate_exempt = True  # 纯内存(board dict),无 FS/命令副作用
    team_scoped = True  # 仅 teammate(mailbox)继承;一次性 subagent 排除

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(
        self, *, id: str, status: str | None = None, assignee: str | None = None,
    ) -> str:
        t = self.team.task_board.update(id, status=status, assignee=assignee)
        if t is None:
            return f"未知任务 #{id}"
        # 状态机守卫拒绝(F4/F5/F9):合法转移 update 已应用 → t.status==请求值;t.status != 请求值
        # 必为守卫拒绝(update 既未改 status、也未动 assignee——F3 保证整笔原子)。旧实现把这种
        # 「未变的 status」报成「已更新 #N: status=<旧值>」,LLM 收不到拒绝信号会继续推进未完成
        # 的任务。显式回拒 + 原因,让 LLM 自决(等依赖 / 换任务 / 改用 TaskClaim)。
        if status is not None and t.status != status:
            if status == "blocked":
                reason = "blocked 不可手动设置(只能由 create 带依赖产生、或前置完成时自动解锁)"
            elif t.status == "blocked":
                reason = "任务处于 blocked 且依赖未全部完成,不能转出(待依赖完成自动解锁后再推进)"
            else:
                reason = f"状态 {t.status}→{status} 不被状态机允许"
            return (
                f"拒绝更新 #{id}: {reason}(status 保持 {t.status},assignee 未改)。"
                "可 TaskList 查看当前依赖状态。"
            )
        # 只回显【实际传入】的字段:旧实现无条件附 status=,仅改 assignee 时会误报未变的 status。
        parts = []
        if status is not None:
            parts.append(f"status={t.status}")
        if assignee is not None:
            parts.append(f"assignee={t.assignee}")
        if not parts:
            return f"任务 #{id} 未变更(未指定 status/assignee)"
        return f"已更新 #{id}: " + ", ".join(parts)


class TaskClaimInput(BaseModel):
    id: str = Field(..., description="要领取的任务 id(须为 pending)")


class TaskClaimTool(Tool):
    """领取(claim)一个 pending 任务——pull 模式核心,teammate 自抢。

    原子地 pending→in_progress + 落自己名下(经 TaskBoard.claim 同步 CAS)。替代裸
    TaskUpdate(status=in_progress) 的非原子路径(后者双人抢同一任务 last-writer-wins)。
    assignee 取 _AGENT_NAME(teammate 名);同 SendMessage/TaskCreate 模式。
    """

    name = "TaskClaim"
    description = (
        "领取(claim)一个 pending 任务:原子地标为 in_progress 并落到自己名下。"
        "pull 模式——自抢未指派(或指派给自己)的 pending 任务。已被抢/已完成/指派他人则失败。"
    )
    parameters = TaskClaimInput
    kind = "write"
    parallel_safe = False
    gate_exempt = True  # 纯内存(board dict CAS),无 FS/命令副作用
    team_scoped = True  # 仅 teammate(mailbox)继承;一次性 subagent 排除

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(self, *, id: str) -> str:
        agent = _AGENT_NAME.get()
        t = self.team.task_board.claim(id, agent)
        if t is not None:
            return f"已领取 #{id}: {t.title}(status=in_progress, assignee={agent})"
        # claim 失败:回原因让 LLM 自决(换任务 / 等待 / 找指派者)
        existing = self.team.task_board.get(id)
        if existing is None:
            return f"领取失败 #{id}: 任务不存在"
        if existing.status != "pending":
            return f"领取失败 #{id}: 当前 status={existing.status}(非 pending,无法领取)"
        return f"领取失败 #{id}: 已指派给 {existing.assignee}(非你,不可抢)"
