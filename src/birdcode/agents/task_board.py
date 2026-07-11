# src/birdcode/agents/task_board.py
"""agent teams:flat 共享任务看板(内存)。

无 asyncio.Lock:单进程 asyncio,所有方法同步(无 await)→ 原子、无竞争。
若未来引入跨 await 的 RMW(如 get→await→update),那时再加 Lock。依赖图(blocks/
blocked_by)留后续;pull/claim 留后续。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    """看板上一个任务(flat,无依赖)。"""

    id: str
    title: str
    status: str = "pending"  # pending|in_progress|completed
    assignee: str | None = None
    created_by: str = ""


class TaskBoard:
    """team 共享的 flat 任务看板。所有方法同步(无 await)→ asyncio 下原子。"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._next_id = 0

    def create(self, *, title: str, assignee: str | None = None, created_by: str = "") -> Task:
        self._next_id += 1
        tid = str(self._next_id)
        t = Task(id=tid, title=title, status="pending", assignee=assignee, created_by=created_by)
        self._tasks[tid] = t
        return t

    def get(self, tid: str) -> Task | None:
        return self._tasks.get(tid)

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def update(
        self, tid: str, *, status: str | None = None, assignee: str | None = None,
    ) -> Task | None:
        t = self._tasks.get(tid)
        if t is None:
            return None
        if status is not None:
            t.status = status
        if assignee is not None:
            t.assignee = assignee
        return t
