# src/birdcode/tools/task_tools.py
"""agent teams:task 工具——共享看板 CRUD(TaskCreate/List/Update)。

team-scoped(持 TeamManager),同 SendMessage 模式:注册进 lead 主 registry,teammate 经
build_child_registry fork 返 self 继承。created_by 取 _AGENT_NAME(lead→"lead" / teammate→其名)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from birdcode.agents.mailbox import _AGENT_NAME
from birdcode.tools.base import Tool

if TYPE_CHECKING:
    from birdcode.agents.teammate import TeamManager


class TaskCreateInput(BaseModel):
    title: str = Field(..., description="任务标题")
    assignee: str | None = Field(None, description="指派给某 teammate(可选;push 模式由 lead 设)")


class TaskCreateTool(Tool):
    name = "TaskCreate"
    description = "在共享看板上建一个任务(flat,无依赖)。可选指派给 teammate。"
    parameters = TaskCreateInput
    kind = "write"
    parallel_safe = False

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(self, *, title: str, assignee: str | None = None) -> str:
        t = self.team.task_board.create(
            title=title, assignee=assignee, created_by=_AGENT_NAME.get()
        )
        suffix = f"(→{assignee})" if assignee else ""
        return f"已建任务 #{t.id}: {t.title}{suffix}"


class TaskListTool(Tool):
    name = "TaskList"
    description = "列出看板上所有任务(id/status/title/assignee)。"
    parameters = BaseModel  # 无参
    kind = "read"
    parallel_safe = True

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(self) -> str:  # noqa: ARG002 - executor 传 **{},签名无 kw
        tasks = self.team.task_board.list()
        if not tasks:
            return "(无任务)"
        lines = []
        for t in tasks:
            a = f" →{t.assignee}" if t.assignee else ""
            lines.append(f"#{t.id} [{t.status}] {t.title}{a}")
        return "\n".join(lines)


class TaskUpdateInput(BaseModel):
    id: str = Field(..., description="任务 id")
    status: str | None = Field(None, description="新状态:pending|in_progress|completed")
    assignee: str | None = Field(None, description="改指派(可选)")


class TaskUpdateTool(Tool):
    name = "TaskUpdate"
    description = "更新任务状态或指派。"
    parameters = TaskUpdateInput
    kind = "write"
    parallel_safe = False

    def __init__(self, team: TeamManager) -> None:
        super().__init__()
        self.team = team

    async def execute(
        self, *, id: str, status: str | None = None, assignee: str | None = None,
    ) -> str:
        t = self.team.task_board.update(id, status=status, assignee=assignee)
        if t is None:
            return f"未知任务 #{id}"
        # 只回显【实际传入】的字段:旧实现无条件附 status=,仅改 assignee 时会误报未变的 status。
        parts = []
        if status is not None:
            parts.append(f"status={t.status}")
        if assignee is not None:
            parts.append(f"assignee={t.assignee}")
        if not parts:
            return f"任务 #{id} 未变更(未指定 status/assignee)"
        return f"已更新 #{id}: " + ", ".join(parts)
