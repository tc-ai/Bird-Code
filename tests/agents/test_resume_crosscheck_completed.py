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

    async def mark_resumed(self, agent_id: str, status: str = "completed") -> None:
        self.resumed.append(agent_id)

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
    tmp_path: Path,
    agent_id: str,
    *,
    status: str = "running",
    is_async: bool = True,
    isolation: str | None = None,
) -> None:
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id,
            agentType="general-purpose",
            description="旧任务",
            toolUseId="(async-agent)",
            isAsync=is_async,
            status=status,
            isolation=isolation,
        ),
    )


def _write_sidechain(
    tmp_path: Path,
    agent_id: str,
    *,
    text: str,
    stop_reason: str | None = None,
) -> Path:
    """写一条配对 user+assistant 的侧链(供 _read_sidechain_final_text 读最后 assistant 文本)。

    stop_reason 非空时写入 assistant 行的 message.stop_reason(模拟真实落盘),供交叉校验主路径。
    """
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    user_line = json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "原任务"}]},
        }
    )
    asst_msg: dict[str, object] = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }
    if stop_reason is not None:
        asst_msg["stop_reason"] = stop_reason
    asst_line = json.dumps({"type": "assistant", "message": asst_msg})
    p.write_text("\n".join([user_line, asst_line]) + "\n", encoding="utf-8")
    return p


def _deps(tmp_path: Path, *, manager: _FakeManager) -> ResumeDeps:
    return ResumeDeps(
        manager=manager,
        root=tmp_path,
        session_id="s1",
        project_root=tmp_path,
        worktree_name=None,
        agent_registry=_FakeRegistry(_defn()),
        parent_provider=None,
        parent_registry=None,
        parent_gate=None,
        cfg=None,
        app=None,
        ctx=None,
        spawn_depth=1,
        progress_cb=None,
    )


# ---- 主测试:meta 非终态 + 侧链像终态 → 注入不续跑 ----


@pytest.mark.asyncio
async def test_nonterminal_meta_but_sidechain_terminal_injects_not_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """meta status=running,但侧链末尾 assistant 是终态报告(实质正文 + 非待办清单头)
    → B 方案:返回侧链最后内容(resume_agent 内部读侧链),不构造 runner、不 launch/run、不注入。

    断言:未构造 runner / 未 launch;不系统注入(无 enqueue/notification);result.text 含侧链最后文本。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x1", status="running", is_async=True, isolation=None)
    _write_sidechain(
        tmp_path,
        "sub-x1",
        text="任务已完成:共修了 3 个测试,全部通过。",
        stop_reason="end_turn",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x1",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # B 方案:不续跑、不系统注入;resume_agent 返回侧链最后内容(模型驱动)
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    assert store.calls == []  # 不 inject(无 enqueue/notification)
    assert controller.woken == 0
    assert result.outcome == "sync_done"
    assert "任务已完成" in result.text  # 返回侧链最后 assistant 文本


# ---- 保守取向回归:侧链以"## 待办清单"开头(进行中标志)→ 走续跑 ----


@pytest.mark.asyncio
async def test_sidechain_with_todo_list_header_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """meta running + 侧链末尾 assistant 以"## 待办清单"开头(runner.py:486 进行中标志)
    → 视为未完成 → 走续跑(构造 runner + launch_async)。

    保守红线:明确进行中标志不误判为完成。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x2", status="running", is_async=True, isolation=None)
    _write_sidechain(
        tmp_path,
        "sub-x2",
        text="## 待办清单\n- [ ] 还没做完的项\n- [ ] 另一个待办",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x2",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # 续跑:构造了 runner + launch
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    # 未走注入
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "async_launched"


# ---- 边界:completed meta + 显式 resume → 拒绝续跑(不重跑、不注入)----


@pytest.mark.asyncio
async def test_completed_meta_refused_not_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """meta status=completed + 显式 resume_agent → 拒绝(不重跑、不注入)。

    completed agent 已无未完成工作;唯一会指向它的路径是陈旧 reminder(注入后不随 meta 刷新)
    误导主 agent 再调 resume_agent。此前 completed 会落到构造 runner + run() 把已完成 agent
    从头重跑(配合旧 meta 被覆写成 direction,连原任务都丢)。completed 守卫在交叉校验之前
    拦截:不构造 runner、不 launch、不注入,返回 sync_done。交叉校验门控仍含「meta 非终态」
    条件(此处更前已挡);error/cancelled 仍可续(此处不拦)。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x4", status="completed", is_async=True, isolation=None)
    _write_sidechain(tmp_path, "sub-x4", text="任务已完成:全部通过。")

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x4",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # 拒绝续跑:不构造 runner、不 launch、不注入、不唤醒
    assert _FakeRunner.constructed == []
    assert mgr.launched == []
    # #3 收敛:completed 短路返回时落 dequeue 标记(_resume_pending 下次不再 re-hint)
    assert mgr.resumed == ["sub-x4"]
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "sync_done"
    assert "完成" in result.text


# ---- Final review fix:注入完成后 meta 必须刷终态,避免重复注入 + discover 永久列出 ----


@pytest.mark.asyncio
async def test_midtool_todo_list_not_misjudged_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """review #1:侧链以 [user, assistant("## 待办清单…", tool_use_X)] 中断(典型 mid-tool)。

    repair_trailing_edge 先补 synthetic tool_result_X(user),末尾变 user 又补 synthetic
    assistant 占位「(上一轮因会话中断…已自动补全收尾)」。修 #1 前:_read_sidechain_final_text
    命中末尾 synthetic 占位(非「## 待办清单」头)→ _sidechain_looks_complete 误判完成 →
    把占位当完成报告注入、不续跑(吞掉进行中的真实产出)。修后:跳过 synthetic,读到真实
    「## 待办清单」→ 未完成 → 走续跑(构造 runner + launch)。讽刺点:若读到的是真实文本,
    它以「## 待办清单」开头本就会正确判 False 续跑——synthetic 漏过滤把正确决策翻转成错误。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x5", status="running", is_async=True, isolation=None)
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, "sub-x5")
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "做任务"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "## 待办清单\n- [ ] 还没做完"},
                                {
                                    "type": "tool_use",
                                    "id": "toolu_mid",
                                    "name": "bash",
                                    "input": {"cmd": "ls"},
                                },
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x5",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # 进行中(## 待办清单)→ 走续跑,不当完成注入
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "async_launched"


@pytest.mark.asyncio
async def test_midtool_intermediate_text_not_misjudged_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """硬中断 mid-tool:末条 assistant 是【过渡文本(非 ## 待办清单)+ tool_use】→ 必须续跑。

    实测场景:explore 子 agent 死在 grep 中途,侧链末条 = [thinking + "再查一下谁调用了…"
    + grep tool_use×2](紧随的 tool_result 写一半被杀、成半截行)。旧 _sidechain_looks_complete
    只看文本:"再查一下…" 不以 ## 待办清单 开头 → is_terminal_report_text=True → 误判完成 →
    注入占位、标 completed,吞掉进行中的探索产出。修后:末条真实 assistant 含 tool_use → 续跑。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x6", status="running", is_async=True, isolation=None)
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, "sub-x6")
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "探索 memory 模块"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "stop_reason": "tool_use",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "再查一下外部谁调用了 memory 模块的核心类。",
                                },
                                {
                                    "type": "tool_use",
                                    "id": "call_g1",
                                    "name": "grep",
                                    "input": {"pattern": "MemoryManager"},
                                },
                                {
                                    "type": "tool_use",
                                    "id": "call_g2",
                                    "name": "grep",
                                    "input": {"pattern": "GovernanceManager"},
                                },
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x6",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    # mid-tool(末条 assistant 含 tool_use)→ 续跑,不注入
    assert len(_FakeRunner.constructed) == 1
    assert len(mgr.launched) == 1
    assert store.calls == []
    assert controller.woken == 0
    assert result.outcome == "async_launched"


@pytest.mark.asyncio
async def test_non_end_turn_stop_reasons_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """stop_reason 非 end_turn(max_tokens 截断)→ 续跑(仅 end_turn 算自然完成)。

    守护:完成判据是 ``stop_reason == "end_turn"``,而非"非 tool_use"——故 max_tokens /
    stop_sequence 等其它非自然终止也走续跑,不被文本启发式误判完成。
    """
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-x7", status="running", is_async=True, isolation=None)
    _write_sidechain(
        tmp_path,
        "sub-x7",
        text="生成到一半被 max_tokens 截断的报告……",
        stop_reason="max_tokens",
    )

    store = _FakeStore()
    controller = _FakeController()
    mgr = _FakeManager(store=store, controller=controller)
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-x7",
        direction="继续",
        deps=_deps(tmp_path, manager=mgr),
    )

    assert len(mgr.launched) == 1
    assert store.calls == []
    assert result.outcome == "async_launched"


# ---- review #12:完成判据单一来源 helper ----


def test_is_terminal_report_text_helper():
    """review #12:is_terminal_report_text 单一来源(runner 完成判定 + resume 交叉校验共用)。"""
    from birdcode.agents.runner import is_terminal_report_text

    assert is_terminal_report_text("任务完成:全部通过") is True
    assert is_terminal_report_text("## 待办清单\n- step") is False  # 进行中标志
    assert is_terminal_report_text("  ## 待办清单") is False  # lstrip 容忍前导空白
    assert is_terminal_report_text("") is True  # runner 视空报告为完成(resume 自行先判空)
