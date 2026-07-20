# tests/agents/test_resume_worktree.py
"""Task 14:worktree 子 agent 续跑 —— create_worktree 自愈 + 状态不符降级注入。

覆盖:
- worktree meta(isolation="worktree")续跑:resume_subagent 探查 create_worktree(复用
  agent_id)自愈;成功 → 透传 isolation 构造 runner + launch_async 派发(异步)。
- create_worktree 抛 FileExistsError(分支不符,无法自愈)→ B 方案:返回侧链最后产出(模型驱动,
  不系统注入)、**不 force-clean 孤儿**(mismatch 时 worktree 可能正被活跃使用,force 会毁未提交
  工作;孤儿由 agent 正常完成时的 cleanup_subagent_worktree 清理或安全驻留)+ 落收敛 dequeue。

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
from birdcode.session.subagent_meta import write_subagent_meta


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
        self,
        *,
        store: _FakeStore | None = None,
        controller: _FakeController | None = None,
        live_agent_ids: set[str] | None = None,
    ) -> None:
        self._store = store
        self._controller = controller
        self._live: dict[str, object] = {aid: object() for aid in (live_agent_ids or set())}
        self.launched: list[object] = []
        self.resumed: list[str] = []

    def has_live(self, agent_id: str) -> bool:
        return agent_id in self._live

    async def mark_resumed(self, agent_id: str, status: str = "completed") -> None:  # noqa: ARG002
        self.resumed.append(agent_id)

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
    tmp_path: Path,
    *,
    agent_id: str,
    is_async: bool = True,
    isolation: str | None = "worktree",
) -> None:
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id,
            agentType="general-purpose",
            description="旧 worktree 任务",
            toolUseId="(async-agent)",
            isAsync=is_async,
            status="running",
            isolation=isolation,
        ),
    )


def _write_sidechain(tmp_path: Path, agent_id: str, *, text: str = "旧的部分产出") -> Path:
    """写一条配对 user+assistant 的侧链(供 _read_sidechain_final_text 读最后 assistant 文本)。"""
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    user_line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "原任务"}]},
        }
    )
    asst_line = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        }
    )
    p.write_text("\n".join([user_line, asst_line]) + "\n", encoding="utf-8")
    return p


def _deps(
    tmp_path: Path,
    *,
    manager: _FakeManager,
    defn: AgentDefinition | None = None,
) -> ResumeDeps:
    return ResumeDeps(
        manager=manager,
        root=tmp_path,
        session_id="s1",
        project_root=tmp_path,
        worktree_name=None,
        agent_registry=_FakeRegistry(defn if defn is not None else _defn()),
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        cfg=None,
        app=None,
        ctx=None,
        spawn_depth=1,
        progress_cb=None,
    )


# ---- Test 1:自愈成功 → 透传 isolation + create_worktree 复用 agent_id + launch_async ----


@pytest.mark.asyncio
async def test_resume_worktree_reuses_isolation_and_self_heals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        agent_id="sub-wt-old",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """create_worktree 抛 FileExistsError(分支不符)→ B 方案:返回侧链最后产出,不续跑、不注入、
    **不 force-clean 孤儿**(mismatch 恰发生在 worktree 被活跃使用时,force 会删未提交工作)。

    侧链用「## 待办清单」头使 _sidechain_looks_complete 判未完成 → 越过完成短路、抵达 worktree
    probe(旧用「旧的部分产出 ABC」被 is_terminal_report_text 判完成短路,probe 零覆盖)。
    断言:probe 被调(create_worktree 命中)/ 未构造 runner / 未 launch / 不注入 / 不清孤儿
    (remove_worktree 未调)/ 落收敛 dequeue(mark_resumed)/ 返回侧链最后 assistant 文本。
    """
    import birdcode.utils.worktree as wt_mod

    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-wt-dead", is_async=True, isolation="worktree")
    _write_sidechain(tmp_path, "sub-wt-dead", text="## 待办清单\n- [ ] 进行中")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)

    create_calls: list[tuple[Path, str]] = []

    async def _boom_create(main_repo: Path, name: str, *, base_ref: str = "origin/HEAD") -> Path:  # noqa: ARG001
        create_calls.append((main_repo, name))
        raise FileExistsError(f"worktree 已存在但状态不符: {name}")

    remove_calls: list[tuple[tuple, dict]] = []

    async def _spy_remove(*args: object, **kwargs: object) -> None:
        remove_calls.append((args, kwargs))

    monkeypatch.setattr(resume_mod, "create_worktree", _boom_create)
    monkeypatch.setattr(wt_mod, "remove_worktree", _spy_remove)

    result = await resume_subagent(
        agent_id="sub-wt-dead",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # probe 被调(证明走的是 mismatch 分支,非完成短路)
    assert len(create_calls) == 1
    assert create_calls[0][1] == "sub-wt-dead"
    # B 方案:不续跑、不系统注入、不 force-clean 孤儿(worktree 被活跃用时 force 会毁未提交工作)
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    assert store.calls == []  # 不 inject(无 enqueue/notification)
    assert remove_calls == []  # 不清孤儿(F1:去掉破坏性 force-clean)
    assert controller.woken == 0
    # 收敛:短路返回落 dequeue 标记(_resume_pending / discover 下次不再列)
    assert mgr.resumed == ["sub-wt-dead"]
    assert result.outcome == "sync_done"
    assert "## 待办清单" in result.text  # 返回侧链最后 assistant 文本(模型驱动)


# ---- Final review fix:降级注入后 meta 必须刷终态(error),避免 discover 永久列出 ----


@pytest.mark.asyncio
async def test_resume_non_worktree_does_not_probe_create_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        agent_id="sub-plain",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    assert result.outcome == "sync_done"
    assert result.text == "续跑结果"  # _FakeRunner.run() 的报告正文
    assert len(_FakeRunner.constructed) == 1
    assert _FakeRunner.constructed[0].captured_kw["isolation"] is None
