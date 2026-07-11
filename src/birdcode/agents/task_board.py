# src/birdcode/agents/task_board.py
"""agent teams:flat 共享任务看板(内存)。

无 asyncio.Lock:单进程 asyncio,所有方法同步(无 await)→ 原子、无竞争。
若未来引入跨 await 的 RMW(如 get→await→update),那时再加 Lock。依赖图(blocks/
blocked_by)留后续(P4)。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    """看板上一个任务(可带依赖图)。"""

    id: str
    title: str
    status: str = "pending"  # pending|blocked|in_progress|completed
    assignee: str | None = None
    created_by: str = ""
    blocked_by: list[str] = field(default_factory=list)  # 依赖的 task id(正向)
    blocks: list[str] = field(default_factory=list)  # 反向:被我阻塞的 task id


class TaskBoard:
    """team 共享的 flat 任务看板。所有方法同步(无 await)→ asyncio 下原子。"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._next_id = 0

    def create(
        self, *, title: str, assignee: str | None = None, created_by: str = "",
        blocked_by: list[str] | None = None,
    ) -> Task:
        self._next_id += 1
        tid = str(self._next_id)
        deps = list(blocked_by or [])
        # 校验:依赖必须已存在(无悬空;且只能依赖更早 task → 结构保证 DAG,无需环检测)。
        missing = [d for d in deps if d not in self._tasks]
        if missing:
            raise ValueError(f"blocked_by 指向不存在的任务 id: {missing}")
        # 判初始 status:有未完成依赖 → blocked;否则 pending(无依赖或依赖全 completed)
        status = "pending"
        if deps and not all(self._tasks[d].status == "completed" for d in deps):
            status = "blocked"
        t = Task(
            id=tid, title=title, status=status, assignee=assignee,
            created_by=created_by, blocked_by=deps,
        )
        self._tasks[tid] = t
        # 建反向边:每个前置的 blocks 追加新 id(自动解锁 O(1) 查后继用)。校验过 → 必存在。
        for dep_id in deps:
            self._tasks[dep_id].blocks.append(tid)
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
        # 完成自动解锁:遍历后继(t.blocks),blocked_by 全 completed 的 blocked→pending。
        # 单进程同步段原子,无需 Lock;只认 completed 触发(前置 in_progress/blocked 不解锁)。
        if status == "completed":
            for succ_id in t.blocks:
                succ = self._tasks.get(succ_id)
                if succ is None or succ.status != "blocked":
                    continue
                if all(self._tasks[d].status == "completed" for d in succ.blocked_by):
                    succ.status = "pending"
        return t

    def claim(self, tid: str, agent_name: str) -> Task | None:
        """原子 claim(CAS;同步无 await → 整段原子,无需 Lock)。pull 模式核心。

        pending + (assignee 是 None 或 agent_name) → status=in_progress + assignee=agent_name,
        返回 Task。否则(非 pending / 指派给别人 / 不存在)返回 None。

        语义:
        - pull:未指派 pending → 任意 teammate 抢 → in_progress + 落自己名下。
        - push 指派给自己(assignee=自己, pending)→ 自己 claim 推进到 in_progress("我接手了")。
        - push 指派给别人 → 不可抢(尊重 push 派活)。

        原子性:同步段无 await,asyncio 不会在中间穿插 → 两 teammate 先后 claim 同一 pending,
        第二个必见 in_progress → None(根除裸 TaskUpdate 的 last-writer-wins double-claim)。
        """
        t = self._tasks.get(tid)
        if t is None:
            return None
        if t.status != "pending":
            return None
        if t.assignee not in (None, agent_name):
            return None
        t.status = "in_progress"
        t.assignee = agent_name
        return t
