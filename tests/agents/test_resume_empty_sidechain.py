# tests/agents/test_resume_empty_sidechain.py
"""Task 18:空/仅种子侧链(无任何真实 assistant 产出)→ 标 interrupted 无输出,不续跑。

覆盖:
- 侧链只有种子 user、无 assistant 产出(死太早)→ 不构造 runner、不 launch/run,
  返回 ResumeResult(sync_done, text 含「无输出」)。
- 侧链文件缺/全坏行 → load_sidechain_turns 返回 [](等价无产出)→ 同样「无输出」不续跑。
- 回归:侧链有真实 assistant 产出 → T18 不拦截,走正常派发(launch_async)。

判据关键(_has_assistant_output):load_sidechain_turns 末尾恒经 repair_trailing_edge,
对「末尾 user」会补一条 synthetic=True 的占位 assistant(文案「(上一轮因会话中断未完成,
已自动补全收尾)」)。故判据必须是「非 synthetic 的 assistant 消息」——synthetic 占位不算
真实产出,仅种子 user 的侧链(经 repair 补 synthetic assistant)仍判 False。

顺序(T18 检查插在 T14 worktree 探查之后、T17 交叉校验之前):空侧链在 T17 之前拦截——
否则 T17 会因「无 assistant 文本」判未完成→续跑,而 T18 要拦截返「无输出」。非空侧链不受
影响(T18 通过 → 落到 T17/正常派发)。

mock 范式抄 tests/agents/test_resume_crosscheck_completed.py(_FakeRegistry/_FakeManager/
_FakeStore/_FakeController/替身 Runner),meta + sidechain 走真实文件读写。
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
    """记录 append / append_queue_operation(验未走落点 A 注入)。"""

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
    """SubagentManager 替身:has_live + launch_async + 暴露 _store/_controller。"""

    def __init__(
        self, *, store: _FakeStore | None = None, controller: _FakeController | None = None,
    ) -> None:
        self._store = store
        self._controller = controller
        self.launched: list[object] = []

    def has_live(self, agent_id: str) -> bool:  # noqa: ARG002
        return False

    async def launch_async(self, runner: object) -> _LaunchAck:
        self.launched.append(runner)
        return _LaunchAck(agent_id=runner.agent_id, ack_text=f"已派发 {runner.agent_id}")  # type: ignore[attr-defined]


class _FakeRunner:
    """SubagentRunner 替身:捕获构造(验「未续跑」= 未构造)+ 固定 run() 返回。"""

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


def _write_seed_only_sidechain(tmp_path: Path, agent_id: str) -> Path:
    """写一条仅种子 user、无 assistant 的侧链(死太早)。

    repair_trailing_edge 会为此补一条 synthetic=True 的占位 assistant;_has_assistant_output
    必须跳过 synthetic 才能正确判 False。
    """
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    user_line = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "种子任务"}]},
    })
    p.write_text(user_line + "\n", encoding="utf-8")
    return p


def _write_real_assistant_sidechain(tmp_path: Path, agent_id: str, text: str) -> Path:
    """写一条含真实 assistant 产出的侧链(T18 应放行)。"""
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


# ---- 主测试:侧链只有种子 user、无 assistant 产出 → 「无输出」不续跑 ----


@pytest.mark.asyncio
async def test_seed_only_sidechain_returns_no_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """侧链只有种子 user、无任何真实 assistant 产出(子 agent 死太早)→ 不续跑,
    返回 ResumeResult(sync_done, text 含「无输出」)。

    断言:未构造 runner / 未 launch / 未走落点 A 注入(store 无调用、controller 未 wake);
    result.text 含「无输出」与 agent_id。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-e1", status="running", is_async=True, isolation=None)
    _write_seed_only_sidechain(tmp_path, "sub-e1")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-e1", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # 不续跑:未构造 runner、未 launch
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    # 未走落点 A 注入(T18 直接返 ResumeResult,不入队/不 wake)
    assert store.calls == []
    assert controller.woken == 0
    # result:text 含「无输出」+ agent_id,outcome=sync_done
    assert result.outcome == "sync_done"
    assert "无输出" in result.text
    assert "sub-e1" in result.text


# ---- 边界:侧链文件缺 → 等价无产出 → 「无输出」不续跑 ----


@pytest.mark.asyncio
async def test_missing_sidechain_file_returns_no_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """侧链文件不存在(load_sidechain_turns 返回 [])→ 等价无 assistant 产出 → 「无输出」不续跑。

    回归红线:缺侧链文件不崩、不续跑、不误注入完成通知。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-e2", status="idle", is_async=True, isolation=None)
    # 故意不写侧链文件

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-e2", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "sync_done"
    assert "无输出" in result.text


# ---- 回归:侧链有真实 assistant 产出 → T18 放行,走正常派发 ----


@pytest.mark.asyncio
async def test_real_assistant_output_passes_t18_and_dispatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """侧链含真实(非 synthetic)assistant 产出 → _has_assistant_output True → T18 不拦截,
    落到正常派发(构造 runner + launch_async)。

    回归红线:T18 不误伤有产出的侧链。用「## 待办清单」头(进行中标志,T17 判未完成→续跑)
    确保不被 T17 交叉校验拦截,从而验证 T18 放行后能正常走到派发。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-e3", status="running", is_async=True, isolation=None)
    _write_real_assistant_sidechain(
        tmp_path, "sub-e3", text="## 待办清单\n- [ ] 还没做完",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-e3", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # T18 放行 → 构造 runner + launch
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    assert result.outcome == "async_launched"


# ---- 边界:sync 子 agent + 空侧链 → 同样「无输出」(不调 run()) ----


@pytest.mark.asyncio
async def test_sync_empty_sidechain_returns_no_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """sync(is_async=False)子 agent + 空侧链 → 同样不续跑(不调 runner.run()),
    返回「无输出」。T18 对 sync/async 统一生效。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-e4", status="running", is_async=False, isolation=None)
    _write_seed_only_sidechain(tmp_path, "sub-e4")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-e4", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )

    # sync 也未构造 runner、未 run(T18 在派发前拦截)
    assert _FakeRunner.constructed == []
    assert result.outcome == "sync_done"
    assert "无输出" in result.text
