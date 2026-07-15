# tests/agents/test_resume_crosscheck_completed.py
"""Task 17:交叉校验 —— meta 非终态 vs 侧链实际已完成 → 注入不续跑。

覆盖:
- meta status=running/idle(非终态),但侧链末尾 assistant 是终态报告(有实质正文且
  非"## 待办清单"开头)→ 当"完成未注入"处理:落点 A 注入(enqueue + is_task_notification
  user 行 + notify_wake),不构造 runner、不 launch/run(不续跑)。
- 保守取向回归:meta 非终态 + 侧链末尾 assistant 以"## 待办清单"开头(runner.py:486 的
  进行中标志)→ 视为未完成 → 走续跑(构造 runner + launch)。
- 保守取向回归:meta 非终态 + 侧链缺/无 assistant 正文 → 走续跑。
- terminal meta(completed/error/cancelled)+ 侧链像终态 → 不触发交叉校验(本路径只管
  meta 非终态的不一致;terminal meta 不属 T17 职责)。

完成判据镜像 runner.py:486(`is_completed = not text.lstrip().startswith("## 待办清单")`)。
保守取向(task-17 brief Step 2 末句):判不准→注入更安全 —— 本函数落在"有实质文本 + 非待办
清单头"时才判完成;无文本/有待办清单头仍走续跑(明确像终态才注入,避免把真未完成当完成丢产出)。

mock 范式抄 tests/agents/test_resume_worktree.py(_FakeRegistry/_FakeManager/_FakeStore/
_FakeController/替身 Runner),只 mock runner/manager/registry/store,meta + sidechain 走真实
文件读写。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.discover import find_resumable_subagents
from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.agents.runner import SubagentReport
from birdcode.session.models import SubagentMeta
from birdcode.session.paths import subagent_jsonl_path, subagent_meta_path, subagents_dir
from birdcode.session.subagent_meta import read_subagent_meta, write_subagent_meta


@dataclass
class _LaunchAck:
    agent_id: str
    ack_text: str


class _FakeRegistry:
    """AgentRegistry 替身:.resolve 返回注入 defn 或 None。"""

    def __init__(self, defn: AgentDefinition | None) -> None:
        self._defn = defn

    def resolve(self, name: str) -> AgentDefinition | None:  # noqa: ARG002
        return self._defn


@dataclass
class _StoreCall:
    method: str
    kwargs: dict[str, object]


class _FakeStore:
    """记录 append / append_queue_operation(验落点 A 三步注入)。"""

    def __init__(self) -> None:
        self.calls: list[_StoreCall] = []

    async def append(self, msg: object, **kwargs: object) -> str:
        self.calls.append(_StoreCall("append", {"msg": msg, **kwargs}))
        return "uuid"

    async def append_queue_operation(self, **kwargs: object) -> None:
        self.calls.append(_StoreCall("append_queue_operation", kwargs))


class _FakeController:
    def __init__(self) -> None:
        self.woken = 0

    def notify_wake(self) -> None:
        self.woken += 1


class _FakeManager:
    """SubagentManager 替身:has_live + launch_async + 暴露 _store/_controller(注入用)。"""

    def __init__(
        self, *, store: _FakeStore | None = None, controller: _FakeController | None = None,
        live_agent_ids: set[str] | None = None,
    ) -> None:
        self._store = store
        self._controller = controller
        self._live: dict[str, object] = {aid: object() for aid in (live_agent_ids or set())}
        self.launched: list[object] = []

    def has_live(self, agent_id: str) -> bool:
        return agent_id in self._live

    async def launch_async(self, runner: object) -> _LaunchAck:
        self.launched.append(runner)
        return _LaunchAck(agent_id=runner.agent_id, ack_text=f"已派发 {runner.agent_id}")  # type: ignore[attr-defined]


class _FakeRunner:
    """SubagentRunner 替身:捕获构造(验"未续跑"= 未构造)+ 固定 run() 返回。"""

    constructed: list[_FakeRunner] = []

    def __init__(self, **kw: object) -> None:
        self.captured_kw = kw
        self.agent_id = kw.get("agent_id")
        type(self).constructed.append(self)

    async def run(self) -> SubagentReport:
        return SubagentReport(True, "续跑结果", "completed", 10, 5, 1)


def _defn() -> AgentDefinition:
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


def _write_meta(
    tmp_path: Path, agent_id: str, *, status: str = "running", is_async: bool = True,
    isolation: str | None = None,
) -> None:
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id, agentType="general-purpose", description="旧任务",
            toolUseId="(async-agent)", isAsync=is_async, status=status, isolation=isolation,
        ),
    )


def _write_sidechain(tmp_path: Path, agent_id: str, *, text: str) -> Path:
    """写一条配对 user+assistant 的侧链(供 _read_sidechain_final_text 读最后 assistant 文本)。"""
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    user_line = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "原任务"}]},
    })
    asst_line = json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    })
    p.write_text("\n".join([user_line, asst_line]) + "\n", encoding="utf-8")
    return p


def _deps(tmp_path: Path, *, manager: _FakeManager) -> ResumeDeps:
    return ResumeDeps(
        manager=manager, root=tmp_path, session_id="s1", project_root=tmp_path,
        worktree_name=None, agent_registry=_FakeRegistry(_defn()),
        parent_provider=None, parent_registry=None, parent_gate=None, cfg=None, app=None,
        ctx=None, spawn_depth=1, progress_cb=None,
    )


# ---- 主测试:meta 非终态 + 侧链像终态 → 注入不续跑 ----


@pytest.mark.asyncio
async def test_nonterminal_meta_but_sidechain_terminal_injects_not_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """meta status=running,但侧链末尾 assistant 是终态报告(实质正文 + 非待办清单头)
    → 落点 A 注入,不构造 runner、不 launch/run。

    断言:未构造 runner / 未 launch;落点 A 三步(enqueue + is_task_notification user 行 +
    notify_wake);notification 含侧链最后 assistant 文本 + 标 Completed: yes。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x1", status="running", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-x1", text="任务已完成:共修了 3 个测试,全部通过。")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x1", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 不续跑:未构造 runner、未 launch
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    # 落点 A 三步注入
    methods = [c.method for c in store.calls]
    assert "append_queue_operation" in methods
    assert "append" in methods
    append_call = next(c for c in store.calls if c.method == "append")
    assert append_call.kwargs.get("is_task_notification") is True
    assert append_call.kwargs.get("agent_id") == "sub-x1"
    notification_text = append_call.kwargs["msg"].content[0].text  # type: ignore[attr-defined]
    assert "任务已完成" in notification_text  # 侧链最后 assistant 文本进 notification
    assert "Completed: yes" in notification_text  # 标为完成(非 error)
    assert controller.woken == 1
    # result
    assert result.outcome == "sync_done"


# ---- 保守取向回归:侧链以"## 待办清单"开头(进行中标志)→ 走续跑 ----


@pytest.mark.asyncio
async def test_sidechain_with_todo_list_header_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """meta running + 侧链末尾 assistant 以"## 待办清单"开头(runner.py:486 进行中标志)
    → 视为未完成 → 走续跑(构造 runner + launch_async)。

    保守红线:明确进行中标志不误判为完成。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x2", status="running", is_async=True, isolation=None)
    _write_sidechain(
        tmp_path, "sub-x2",
        text="## 待办清单\n- [ ] 还没做完的项\n- [ ] 另一个待办",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x2", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 续跑:构造了 runner + launch
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    # 未走注入
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "async_launched"


# ---- 边界:terminal meta(completed)+ 侧链像终态 → 不触发交叉校验,走正常续跑派发 ----


@pytest.mark.asyncio
async def test_terminal_meta_does_not_trigger_crosscheck_inject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """meta status=completed(终态)+ 侧链像终态 → 交叉校验不触发(本路径只管 meta 非终态的不一致)。

    即便 sidechain 看起来已完成,terminal meta 不属 T17 职责 → 走正常派发(launch_async)。
    回归红线:交叉校验门控必须含"meta 非终态"条件,不在 terminal meta 上误燃。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x4", status="completed", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-x4", text="任务已完成:全部通过。")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x4", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 走正常派发(未触发交叉校验注入)
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    assert store.calls == []
    assert result.outcome == "async_launched"


# ---- Final review fix:注入完成后 meta 必须刷终态,避免重复注入 + discover 永久列出 ----


@pytest.mark.asyncio
async def test_inject_as_completed_marks_meta_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """交叉校验注入完成后,meta.status 应刷成 completed(并带 completed_at)。

    回归红线:此前注入只 append notification、不更 meta → meta 停 running。修复后必须落终态。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-t1", status="running", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-t1", text="任务已完成:全部通过。")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    await resume_subagent(
        agent_id="sub-t1", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, "sub-t1")
    meta = read_subagent_meta(meta_path)
    assert meta is not None
    assert meta.status == "completed"  # 终态,不再停 running
    assert meta.completed_at  # 非空 ISO 时间戳
    assert meta.completed_at.endswith("Z")


@pytest.mark.asyncio
async def test_inject_as_completed_no_duplicate_on_re_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """注入完成(刷终态)后再次 resume 同 agent → 不重复注入 notification。

    修复前:meta 停 running → 再次 resume 时 T17 交叉校验门控(meta 非终态 & 侧链像终态)仍为真
    → 重复 enqueue+notification。修复后 meta=completed → 门控失效,不重复注入(落到正常派发路径)。
    断言:第二次 resume 用全新 store,store.calls 不含 append_queue_operation / append notification。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-t2", status="running", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-t2", text="任务已完成:修了 2 个 bug。")

    # 第一次 resume:触发交叉校验注入 + 刷终态
    store1 = _FakeStore()
    mgr1 = _FakeManager(store=store1, controller=_FakeController())
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)
    await resume_subagent(
        agent_id="sub-t2", direction="继续", deps=_deps(tmp_path, manager=mgr1),
    )
    first_inject_count = sum(1 for c in store1.calls if c.method == "append_queue_operation")
    assert first_inject_count == 1  # 首次注入了一次

    # 第二次 resume:全新 store,断言不重复注入(meta 已 completed,T17 门控失效)
    _FakeRunner.constructed.clear()
    store2 = _FakeStore()
    mgr2 = _FakeManager(store=store2, controller=_FakeController())
    await resume_subagent(
        agent_id="sub-t2", direction="继续", deps=_deps(tmp_path, manager=mgr2),
    )
    inject_methods = [c.method for c in store2.calls]
    assert "append_queue_operation" not in inject_methods  # 不重复 enqueue
    # 不重复 notification user 行(append 若有也是续跑派发,不带 is_task_notification)
    for c in store2.calls:
        if c.method == "append":
            assert c.kwargs.get("is_task_notification") is not True


@pytest.mark.asyncio
async def test_inject_as_completed_removes_from_discover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """注入完成后,find_resumable_subagents 不再列出该 agent(completed 非可续跑)。

    修复前:meta 停 running → discover 永远扫到,每会话 reminder 都列实际已完成的 agent。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-t3", status="running", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-t3", text="任务已完成:全部通过。")

    sub_dir = subagents_dir(tmp_path, "s1", tmp_path)

    # 注入前:discover 列出该 agent(running 可续跑)
    before = [m.agent_id for m in find_resumable_subagents(sub_dir)]
    assert "sub-t3" in before

    store = _FakeStore()
    mgr = _FakeManager(store=store, controller=_FakeController())
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)
    await resume_subagent(
        agent_id="sub-t3", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 注入后:discover 不再列出(completed 非可续跑)
    after = [m.agent_id for m in find_resumable_subagents(sub_dir)]
    assert "sub-t3" not in after
