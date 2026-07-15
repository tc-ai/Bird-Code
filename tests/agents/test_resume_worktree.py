# tests/agents/test_resume_worktree.py
"""Task 14:worktree 子 agent 续跑 —— create_worktree 自愈 + 状态不符降级注入。

覆盖:
- worktree meta(isolation="worktree")续跑:resume_subagent 探查 create_worktree(复用
  agent_id)自愈;成功 → 透传 isolation 构造 runner + launch_async 派发(异步)。
- create_worktree 抛 FileExistsError(分支不符,无法自愈)→ 降级:侧链最后产出包成
  落点 A notification 注入(enqueue + is_task_notification user 行 + notify_wake)
  + force 清孤儿 worktree。

设计说明(brief 据实调整):brief 原想 try/except 包住 launch_async/run()——但(1)异步路径
异常在后台 _supervise task 不冒泡到 resume_subagent;(2)同步路径 runner.run() 内 except Exception
会吞 FileExistsError。故改为 resume_subagent 在派发前【同步探查】create_worktree,状态不符即降级,
对 sync/async 两路统一生效。runner.run() 二次调 create_worktree 命中快速恢复路径(dir 在 +
HEAD 匹配 → 直接复用,不调 git),幂等。

mock 范式抄 tests/agents/test_resume_subagent.py(_FakeRegistry/_FakeManager/替身 Runner)。
create_worktree / remove_worktree 走模块级 monkeypatch(免真实 git)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.agents.runner import SubagentReport
from birdcode.session.models import SubagentMeta
from birdcode.session.paths import subagent_jsonl_path, subagent_meta_path
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
    """记录 append / append_queue_operation(验降级注入走落点 A 三步)。"""

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
    """SubagentManager 替身:has_live + launch_async + 暴露 _store/_controller(降级注入用)。"""

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
    """SubagentRunner 替身:捕获构造 kw(验 isolation 透传 + agent_id 复用)。"""

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
    tmp_path: Path, *, agent_id: str, is_async: bool = True,
    isolation: str | None = "worktree",
) -> None:
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id, agentType="general-purpose", description="旧 worktree 任务",
            toolUseId="(async-agent)", isAsync=is_async, status="running", isolation=isolation,
        ),
    )


def _write_sidechain(tmp_path: Path, agent_id: str, *, text: str = "旧的部分产出") -> Path:
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


def _deps(
    tmp_path: Path, *, manager: _FakeManager,
    defn: AgentDefinition | None = None,
) -> ResumeDeps:
    return ResumeDeps(
        manager=manager, root=tmp_path, session_id="s1", project_root=tmp_path,
        worktree_name=None, agent_registry=_FakeRegistry(defn if defn is not None else _defn()),
        parent_provider=None, parent_registry=None, parent_gate=None, cfg=None, app=None,
        ctx=None, spawn_depth=1, progress_cb=None,
    )


# ---- Test 1:自愈成功 → 透传 isolation + create_worktree 复用 agent_id + launch_async ----


@pytest.mark.asyncio
async def test_resume_worktree_reuses_isolation_and_self_heals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """worktree 续跑:create_worktree(复用 agent_id)自愈成功 → 透传 isolation + launch_async 派发。

    断言:create_worktree 用复用的 agent_id("sub-wt-old")探查;runner 透传 isolation="worktree";
    launch_async 被调一次。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-wt-old", is_async=True, isolation="worktree")
    # 有真实产出 → T18 空侧链检查放行;用「## 待办清单」头使 T17 判未完成、不误注入,抵达 launch
    _write_sidechain(tmp_path, "sub-wt-old", text="## 待办清单\n- [ ] 进行中")
    mgr = _FakeManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    create_calls: list[tuple[Path, str]] = []

    async def _spy_create(main_repo: Path, name: str, *, base_ref: str = "origin/HEAD") -> Path:  # noqa: ARG001
        create_calls.append((main_repo, name))
        return tmp_path / ".birdcode" / "worktrees" / name  # 模拟自愈:返回已存在 dir

    monkeypatch.setattr(resume_mod, "create_worktree", _spy_create)

    result = await resume_subagent(
        agent_id="sub-wt-old", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    assert result.outcome == "async_launched"
    # create_worktree 用复用的 agent_id 探查(自愈)
    assert len(create_calls) == 1
    assert create_calls[0][1] == "sub-wt-old"
    # runner 透传 isolation="worktree" + 复用 agent_id
    assert len(_FakeRunner.constructed) == 1
    runner_kw = _FakeRunner.constructed[0].captured_kw
    assert runner_kw["isolation"] == "worktree"
    assert runner_kw["agent_id"] == "sub-wt-old"
    # launch_async 被调一次
    assert len(mgr.launched) == 1


# ---- Test 2:状态不符 → 降级注入(落点 A)+ force 清孤儿 worktree ----


@pytest.mark.asyncio
async def test_resume_worktree_state_mismatch_degrades_to_inject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """create_worktree 抛 FileExistsError(分支不符)→ 降级:注入侧链最后产出 + force 清孤儿。

    断言:未构造 runner / 未 launch;落点 A 三步注入(enqueue + is_task_notification user 行 +
    notify_wake)且 notification 含侧链最后 assistant 文本;remove_worktree(force=True)清孤儿。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-wt-dead", is_async=True, isolation="worktree")
    _write_sidechain(tmp_path, "sub-wt-dead", text="旧的部分产出 ABC")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)

    async def _boom_create(main_repo: Path, name: str, *, base_ref: str = "origin/HEAD") -> Path:  # noqa: ARG001
        raise FileExistsError(f"worktree 已存在但状态不符: {name}")

    monkeypatch.setattr(resume_mod, "create_worktree", _boom_create)

    remove_calls: list[tuple[Path, Path, bool]] = []

    async def _spy_remove(main_repo: Path, wt: Path, *, force: bool = False) -> None:
        remove_calls.append((main_repo, wt, force))

    monkeypatch.setattr(resume_mod, "remove_worktree", _spy_remove)

    result = await resume_subagent(
        agent_id="sub-wt-dead", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 降级:未构造 runner、未 launch
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    # 落点 A 注入:enqueue + task-notification user 行 + wake
    methods = [c.method for c in store.calls]
    assert "append_queue_operation" in methods
    assert "append" in methods
    append_call = next(c for c in store.calls if c.method == "append")
    assert append_call.kwargs.get("is_task_notification") is True
    assert append_call.kwargs.get("agent_id") == "sub-wt-dead"
    notification_text = append_call.kwargs["msg"].content[0].text  # type: ignore[attr-defined]
    assert "旧的部分产出 ABC" in notification_text  # 侧链最后 assistant 文本进 notification
    assert controller.woken == 1
    # 清孤儿 worktree(force=True)
    assert len(remove_calls) == 1
    _, wt_removed, force = remove_calls[0]
    assert force is True
    assert wt_removed.name == "sub-wt-dead"
    # result
    assert result.outcome == "sync_done"


# ---- Final review fix:降级注入后 meta 必须刷终态(error),避免 discover 永久列出 ----


@pytest.mark.asyncio
async def test_degrade_to_inject_marks_meta_terminal_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """worktree 状态不符降级注入后,meta.status 应刷成 error(并带 completed_at)。

    修复红线:降级注入的是侧链最后产出但 worktree 状态不符属异常 → 与 report.status /
    queue-operation status 一致标 error(非 cancelled——cancelled 可续跑,会被 discover 反复列出)。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-wt-term", is_async=True, isolation="worktree")
    _write_sidechain(tmp_path, "sub-wt-term", text="旧的部分产出 XYZ")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)

    async def _boom_create(main_repo: Path, name: str, *, base_ref: str = "origin/HEAD") -> Path:  # noqa: ARG001
        raise FileExistsError(f"worktree 已存在但状态不符: {name}")

    monkeypatch.setattr(resume_mod, "create_worktree", _boom_create)

    async def _noop_remove(main_repo: Path, wt: Path, *, force: bool = False) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(resume_mod, "remove_worktree", _noop_remove)

    await resume_subagent(
        agent_id="sub-wt-term", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, "sub-wt-term")
    meta = read_subagent_meta(meta_path)
    assert meta is not None
    assert meta.status == "error"  # 终态,与 report.status / queue-operation status 一致
    assert meta.completed_at  # 非空 ISO 时间戳
    assert meta.completed_at.endswith("Z")


# ---- 回归:非 worktree(isolation=None)不走 create_worktree 探查,不影响原 sync 路径 ----


@pytest.mark.asyncio
async def test_resume_non_worktree_does_not_probe_create_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """isolation=None(非 worktree 子 agent)不调 create_worktree,走原 sync run() 路径。

    回归红线:worktree 探查/降级只对 isolation="worktree" 生效,不污染普通子 agent 续跑。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-plain", is_async=False, isolation=None)
    # 有真实产出 → T18 放行;「## 待办清单」头使 T17 不误注入,抵达 sync run() 路径
    _write_sidechain(tmp_path, "sub-plain", text="## 待办清单\n- [ ] 进行中")
    mgr = _FakeManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    async def _fail_create(*args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("非 worktree 续跑不应调 create_worktree")

    monkeypatch.setattr(resume_mod, "create_worktree", _fail_create)

    result = await resume_subagent(
        agent_id="sub-plain", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    assert result.outcome == "sync_done"
    assert result.text == "续跑结果"  # _FakeRunner.run() 的报告正文
    assert len(_FakeRunner.constructed) == 1
    assert _FakeRunner.constructed[0].captured_kw["isolation"] is None
