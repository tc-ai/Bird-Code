# tests/agents/test_resume_idempotency.py
"""Task 19:幂等(二次 resume 不重复 launch)+ 递归中断自洽。

覆盖:
- 二次 resume 同一活 agent:第一次 launch_async(_live 自然加入该 id,镜像真实 manager),
  第二次 has_live 命中守卫 → outcome=in_progress,launch 不重复(launch_count==1)。
- 递归中断:续跑后又被中断(_live 清空、meta 留非终态 running、侧链更长)→
  find_resumable_subagents(T2)再发现(running ∈ RESUMABLE_STATUSES)→ 再 resume 同机制。

与 test_resume_subagent.py::test_resume_idempotent_when_live_has_agent 的区别:后者预填
_live 测守卫单点;本文件 _LiveTrackingManager.launch_async 写实 _live(加入 agent_id),
使「第一次 launch → 第二次命中守卫」自然走完整序列,验证幂等在真实 launch 路径上成立。

mock 范式抄 tests/agents/test_resume_subagent.py(_FakeRegistry/_FakeRunner + 真实 meta/
paths,只 monkeypatch resume.SubagentRunner),meta 走真实 write/read_subagent_meta。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from birdcode.agents import resume as resume_mod
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.discover import find_resumable_subagents
from birdcode.agents.resume import ResumeDeps, resume_subagent
from birdcode.agents.runner import SubagentReport
from birdcode.session.models import SubagentMeta
from birdcode.session.paths import (
    subagent_jsonl_path,
    subagent_meta_path,
    subagents_dir,
)
from birdcode.session.subagent_meta import write_subagent_meta


@dataclass
class _LaunchAck:
    """对镜 manager.LaunchAck:launch_async 回执(只用到 agent_id + ack_text)。"""

    agent_id: str
    ack_text: str


class _FakeRegistry:
    """AgentRegistry 替身:暴露真实方法名 .resolve,返回注入的 defn 或 None。"""

    def __init__(self, defn: AgentDefinition | None) -> None:
        self._defn = defn

    def resolve(self, name: str) -> AgentDefinition | None:  # noqa: ARG002
        return self._defn


class _LiveTrackingManager:
    """SubagentManager 替身:launch_async 写实 _live(加入 agent_id,镜像真实 manager)。

    与 test_resume_subagent._FakeManager 区别:后者 launch_async 不改 _live(靠预填
    live_agent_ids 测幂等)。本类让第一次 launch 自然把 agent_id 加进 _live → 第二次
    resume 自然命中 has_live 守卫,验证「二次 resume 不重复 launch」的真实序列。
    has_live 镜像真实 SubagentManager 公共方法(_live dict 只读访问器)。
    """

    def __init__(self) -> None:
        self._live: dict[str, object] = {}
        self.launch_count: int = 0
        self.launched: list[object] = []

    def has_live(self, agent_id: str) -> bool:
        return agent_id in self._live

    async def launch_async(self, runner: object) -> _LaunchAck:
        aid = runner.agent_id  # type: ignore[attr-defined]
        self._live[aid] = object()  # 镜像真实 launch_async:_live[id] = task
        self.launched.append(runner)
        self.launch_count += 1
        return _LaunchAck(agent_id=aid, ack_text=f"已派发 {aid}")

    def simulate_interrupt(self, agent_id: str) -> None:
        """模拟「在跑的 task 又被中断」:_live 弹出(task 终止),launch_count 不重置。

        真实 _supervise 在 finally 内 _live.pop(agent_id);此处直接 pop,模拟 task 结束
        但 meta 未刷成终态(T4 resume 分支已写 status=running,中断未及更新)。
        """
        self._live.pop(agent_id, None)


class _FakeRunner:
    """SubagentRunner 替身:捕获构造 kw(验证 agent_id 复用 + resume_from)+ 固定 run()。

    不写 meta(真实 T4 runner resume 分支会写 status=running;本文件测的是 resume 入口
    幂等/再发现机制,meta 写入由测试用例显式 _write_meta 模拟,逻辑更聚焦可读)。
    """

    constructed: list[_FakeRunner] = []

    def __init__(self, **kw: object) -> None:
        self.captured_kw = kw
        self.agent_id = kw.get("agent_id")
        self.resume_from = kw.get("resume_from")
        type(self).constructed.append(self)

    async def run(self) -> SubagentReport:
        return SubagentReport(True, "续跑结果文本", "completed", 10, 5, 1)


def _defn() -> AgentDefinition:
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


def _write_meta(tmp_path: Path, *, agent_id: str, status: str = "running") -> None:
    """写真实 meta.json,使 resume_subagent 内 read_subagent_meta 走真实路径。"""
    meta_path = subagent_meta_path(tmp_path, "s1", tmp_path, agent_id)
    write_subagent_meta(
        meta_path,
        SubagentMeta(
            agentId=agent_id, agentType="general-purpose", description="旧任务",
            toolUseId="(async-agent)", isAsync=True, status=status,
        ),
    )


def _write_sidechain(tmp_path: Path, agent_id: str, *, extra_turn: bool = False) -> None:
    """写含真实(非 synthetic)assistant 产出的侧链;T18 空侧链检查放行、T17 判未完成。

    assistant 文本用「## 待办清单」头(runner.py 完成判据的反向标志)→ T17 交叉校验
    判未完成 → 走续跑派发路径(抵达 launch_async)。extra_turn=True 模拟续跑后又跑了
    一轮(侧链更长,递归中断场景的可观测差异)。
    """
    p = subagent_jsonl_path(tmp_path, "s1", tmp_path, agent_id)
    lines = [
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "原任务"}]},
        }),
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "## 待办清单\n- [ ] 进行中"}]},
        }),
    ]
    if extra_turn:
        lines.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "继续做"}]},
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "## 待办清单\n- [ ] 仍在进行"}]},
        }))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deps(tmp_path: Path, *, manager: _LiveTrackingManager) -> ResumeDeps:
    return ResumeDeps(
        manager=manager, root=tmp_path, session_id="s1", project_root=tmp_path,
        worktree_name=None, agent_registry=_FakeRegistry(_defn()),
        parent_provider=None, parent_registry=None, parent_gate=None, cfg=None, app=None,
        ctx=None, spawn_depth=1, progress_cb=None,
    )


# ---- 二次 resume 同一活 agent → in_progress,不重复 launch ----


async def test_second_resume_while_live_returns_in_progress(
    tmp_path: Path, monkeypatch,
):
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-live")
    _write_sidechain(tmp_path, "sub-live")
    mgr = _LiveTrackingManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    first = await resume_subagent(
        agent_id="sub-live", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )
    # 第一次 launch 后 _live 自然含 sub-live(镜像真实 manager)→ 第二次命中 has_live 守卫
    second = await resume_subagent(
        agent_id="sub-live", direction="再继续", deps=_deps(tmp_path, manager=mgr),
    )

    assert first.outcome == "async_launched"
    assert second.outcome == "in_progress"
    assert "sub-live" in second.text
    assert "运行中" in second.text
    # 核心:第二次命中守卫未重复 launch(launch 仍只发生一次)
    assert mgr.launch_count == 1
    assert mgr.has_live("sub-live") is True  # 第一次 launch 的 task 仍在 _live 里


# ---- 递归中断:续跑后又被中断 → find_resumable 再发现 → 再 resume 同机制 ----


async def test_recursive_interruption_self_similar(
    tmp_path: Path, monkeypatch,
):
    _FakeRunner.constructed.clear()
    _write_meta(tmp_path, agent_id="sub-recur")
    _write_sidechain(tmp_path, "sub-recur")
    mgr = _LiveTrackingManager()
    monkeypatch.setattr(resume_mod, "SubagentRunner", _FakeRunner)

    # 第一次 resume(async):launch → _live 加入 sub-recur
    first = await resume_subagent(
        agent_id="sub-recur", direction="继续", deps=_deps(tmp_path, manager=mgr),
    )
    assert first.outcome == "async_launched"
    assert mgr.launch_count == 1

    # 模拟递归中断:续跑的 task 又被中断。
    # - _live 清空(真实 _supervise finally 内 pop,task 终止)
    # - meta 仍非终态 running(T4 resume 分支已写 running,中断未及更新成终态)
    # - 侧链更长(续跑后又跑了一轮才被中断)
    mgr.simulate_interrupt("sub-recur")
    _write_meta(tmp_path, agent_id="sub-recur", status="running")
    _write_sidechain(tmp_path, "sub-recur", extra_turn=True)

    # find_resumable_subagents(T2)再发现:running ∈ RESUMABLE_STATUSES
    sub_dir = subagents_dir(tmp_path, "s1", tmp_path)
    resumable = find_resumable_subagents(sub_dir)
    assert "sub-recur" in {m.agent_id for m in resumable}

    # 再 resume:同机制(_live 已清空 → 重新 launch,launch_count 递增到 2)
    second = await resume_subagent(
        agent_id="sub-recur", direction="再次继续", deps=_deps(tmp_path, manager=mgr),
    )

    second_resume_works = second.outcome == "async_launched" and mgr.launch_count == 2
    assert second_resume_works
    # 再 launch 后 _live 又含该 id(同机制:可被下一次 resume 幂等守卫)
    assert mgr.has_live("sub-recur") is True
