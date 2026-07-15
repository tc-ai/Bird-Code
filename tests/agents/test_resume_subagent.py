# tests/agents/test_resume_subagent.py
"""Task 5:resume_subagent 续跑核心 —— 复用 agent_id + 按 is_async 派发 + 幂等。

覆盖:
- async meta → launch_async 后台派发,且复用原 agent_id(传给 SubagentRunner)
- sync meta → runner.run() 阻塞 inline,返回 report.text
- 幂等:_live 已有该 agent_id → in_progress,不构造 runner、不 launch
- meta 缺失 → 友好 ResumeResult(sync_done + 说明),不抛
- agent_type 在 registry 中不存在 → 友好 ResumeResult,不抛

mock 范式抄 tests/agents/test_manager.py(_FakeStore/_FakeController/替身 Runner 暴露 agent_id)
+ tests/agents/test_runner_resume_mode.py(MonkeyPatch 模块级符号:那里 patch build_provider,
  这里 patch resume.SubagentRunner)。meta 走真实 write_subagent_meta/read_subagent_meta,
  paths 走真实函数 —— 只 mock runner/manager/registry 三件。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.agents.runner import SubagentReport
from birdcode.session.models import SubagentMeta
from birdcode.session.paths import subagent_jsonl_path, subagent_meta_path
from birdcode.session.subagent_meta import write_subagent_meta


@dataclass
class _LaunchAck:
    """对镜 manager.LaunchAck:launch_async 回执(只用到 ack_text)。"""

    agent_id: str
    ack_text: str


class _FakeRegistry:
    """AgentRegistry 替身:暴露真实方法名 .resolve(非 .get),返回注入的 defn 或 None。"""

    def __init__(self, defn: AgentDefinition | None) -> None:
        self._defn = defn

    def resolve(self, name: str) -> AgentDefinition | None:  # noqa: ARG002
        return self._defn


class _FakeManager:
    """SubagentManager 替身:暴露 _live dict + has_live + launch_async(记录入参,不后台)。

    _live 初始填入 live_agent_ids,使幂等分支(has_live=True)可测。has_live 镜像真
    SubagentManager 公共方法(_live dict 只读访问器,供 resume.py 幂等守卫用)。
    """

    def __init__(self, *, live_agent_ids: set[str] | None = None) -> None:
        self._live: dict[str, object] = {aid: object() for aid in (live_agent_ids or set())}
        self.launched: list[object] = []

    def has_live(self, agent_id: str) -> bool:
        return agent_id in self._live

    async def launch_async(self, runner: object) -> _LaunchAck:
        self.launched.append(runner)
        return _LaunchAck(agent_id=runner.agent_id, ack_text=f"已派发 {runner.agent_id}")  # type: ignore[attr-defined]


class _FakeRunner:
    """SubagentRunner 替身:捕获构造 kw(验证 agent_id 复用 + resume_from)+ 固定 run() 返回。"""

    constructed: list[_FakeRunner] = []  # 类级计数:幂等测试断言"未构造"

    def __init__(self, **kw: object) -> None:
        self.captured_kw = kw
        self.agent_id = kw.get("agent_id")
        self.resume_from = kw.get("resume_from")
        type(self).constructed.append(self)

    async def run(self) -> SubagentReport:
        return SubagentReport(True, "续跑结果文本", "completed", 10, 5, 1)


def _defn() -> AgentDefinition:
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


def _write_meta(tmp_path: Path, *, agent_id: str, is_async: bool) -> None:
    """写真实 meta.json,使 resume_subagent 内 read_subagent_meta 走真实路径。"""
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id, agentType="general-purpose", description="旧任务",
            toolUseId="(async-agent)", isAsync=is_async, status="running",
        ),
    )


def _write_sidechain_with_output(tmp_path: Path, agent_id: str) -> None:
    """写一条含真实(非 synthetic)assistant 产出的侧链,让 T18 空侧链检查放行。

    assistant 文本用「## 待办清单」头(runner.py:486 进行中标志)→ T17 交叉校验判未完成 →
    不误注入完成、走续跑派发,使本文件到达它要验证的 launch_async / run() 路径。
    """
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    user_line = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "原任务"}]},
    })
    asst_line = json.dumps({
        "type": "assistant",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "## 待办清单\n- [ ] 进行中"}]},
    })
    p.write_text("\n".join([user_line, asst_line]) + "\n", encoding="utf-8")


def _deps(
    tmp_path: Path, *, manager: _FakeManager, defn: AgentDefinition | None,
) -> ResumeDeps:
    return ResumeDeps(
        manager=manager, root=tmp_path, session_id="s1", project_root=tmp_path,
        worktree_name=None, agent_registry=_FakeRegistry(defn),
        parent_provider=None, parent_registry=None, parent_gate=None, cfg=None, app=None,
        ctx=None, spawn_depth=1, progress_cb=None,
    )


# ---- async:launch_async + 复用 agent_id + resume_from 指向侧链 ----


async def test_resume_async_launches_and_reuses_agent_id(tmp_path: Path, monkeypatch):
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-old", is_async=True)
    _write_sidechain_with_output(tmp_path, "sub-old")  # 有产出 → T18 放行,抵达 launch_async
    mgr = _FakeManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-old", direction="继续,改成 X",
        deps=_deps(tmp_path, manager=mgr, defn=_defn()),
    )

    assert result.outcome == "async_launched"
    assert "sub-old" in result.text  # ack 文本含复用的 agent_id
    # 复用原 agent_id(非新生成 sub-{uuid})+ resume_from 指向该 agent 的侧链
    assert len(_FakeRunner.constructed) == 1
    runner = _FakeRunner.constructed[0]
    assert runner.captured_kw["agent_id"] == "sub-old"
    expected_sidechain = subagent_jsonl_path(tmp_path, "s1", tmp_path, "sub-old")
    assert runner.captured_kw["resume_from"] == expected_sidechain
    assert len(mgr.launched) == 1  # launch_async 被调一次


# ---- sync:runner.run() 阻塞 inline,返回 report.text ----


async def test_resume_sync_runs_inline_and_returns_report_text(tmp_path: Path, monkeypatch):
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-sync", is_async=False)
    _write_sidechain_with_output(tmp_path, "sub-sync")  # 有产出 → T18 放行,抵达 run()
    mgr = _FakeManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-sync", direction="接着做",
        deps=_deps(tmp_path, manager=mgr, defn=_defn()),
    )

    assert result.outcome == "sync_done"
    assert result.text == "续跑结果文本"  # 来自 FakeRunner.run() 的 SubagentReport.text
    assert mgr.launched == []  # sync 不走 launch_async


# ---- 幂等:_live 已有该 agent_id → in_progress,不构造 runner、不 launch ----


async def test_resume_idempotent_when_live_has_agent(tmp_path: Path, monkeypatch):
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-running", is_async=True)
    mgr = _FakeManager(live_agent_ids={"sub-running"})
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    result = await resume_subagent(
        agent_id="sub-running", direction="继续",
        deps=_deps(tmp_path, manager=mgr, defn=_defn()),
    )

    assert result.outcome == "in_progress"
    assert "sub-running" in result.text
    assert "运行中" in result.text
    assert _FakeRunner.constructed == []  # 幂等:未构造 runner
    assert mgr.launched == []  # 未 launch


# ---- meta 缺失 → 友好 ResumeResult,不抛 ----


async def test_resume_meta_missing_returns_friendly(tmp_path: Path):
    # 不写 meta → read_subagent_meta 返回 None
    result = await resume_subagent(
        agent_id="sub-gone", direction="继续",
        deps=_deps(tmp_path, manager=_FakeManager(), defn=_defn()),
    )

    assert result.outcome == "sync_done"
    assert "缺失" in result.text or "损坏" in result.text


# ---- agent_type 在 registry 中不存在 → 友好 ResumeResult,不抛 ----


async def test_resume_agent_type_not_found_returns_friendly(tmp_path: Path):
    _write_meta(tmp_path, agent_id="sub-ghost", is_async=False)
    result = await resume_subagent(
        agent_id="sub-ghost", direction="继续",
        deps=_deps(tmp_path, manager=_FakeManager(), defn=None),  # registry.resolve → None
    )

    assert result.outcome == "sync_done"
    assert "general-purpose" in result.text  # meta.agent_type 写进了文案
