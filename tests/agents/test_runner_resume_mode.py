# tests/agents/test_runner_resume_mode.py
"""Task 4:SubagentRunner resume 模式 —— 注入 agent_id + resume_from 加载侧链作 history。

覆盖:
- __init__ 接受 agent_id 注入(复用旧 id,非 sub-{uuid} 新生成)+ resume_from 存档。
- 不注入时保持原行为(sub-{uuid12})—— 回归红线。
- run() resume 分支:resume_from 非空 → load_sidechain_turns 当 history(只重放不重执、
  跳过播种)、direction(self.prompt)作新 user turn 落侧链(规避 Claude Code #11712)。

stub 构造抄自 tests/agents/test_runner_run.py(_FakeProvider/_cfg/_ctx/_defn +MonkeyPatch
build_provider 范式),不从零发明。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from birdcode.agent.provider import Done, TextDelta, TokenUsage
from birdcode.agents.definition import AgentDefinition
from birdcode.agents.runner import SubagentRunner
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.session.models import SessionContext


class _FakeProvider:
    """实现 StreamingProvider 协议:yield 一段文本 + Done,无工具。"""

    def __init__(self, profile):
        self.profile = profile

    def stream(self, messages, *, history):  # noqa: ARG002
        async def _gen():
            yield TextDelta(text="续跑完成")
            yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn")

        return _gen()


class _HistoryCapturingProvider:
    """捕获 stream() 收到的 history(验证 resume 加载了侧链);yield 文本 + Done。"""

    def __init__(self, profile, captured: dict[str, object]):
        self.profile = profile
        self._captured = captured

    def stream(self, messages, *, history):  # noqa: ARG002
        self._captured["history"] = history

        async def _gen():
            yield TextDelta(text="续跑完成")
            yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="end_turn")

        return _gen()


def _cfg() -> AppConfig:
    return AppConfig(
        providers={"p": ProviderProfile(name="p", protocol="anthropic", model="m",
                                        base_url="http://x", api_key="k")},
        default="p",
    )


def _ctx() -> SessionContext:
    return SessionContext(session_id="s1", cwd=".", version="", git_branch=None)


def _defn() -> AgentDefinition:
    return AgentDefinition(name="general-purpose", description="d", system_prompt="SP")


def _build_test_runner(
    *, agent_id: str | None, resume_from: Path | None, prompt: str, tmp_path: Path,
) -> SubagentRunner:
    """最小依赖构造 SubagentRunner(抄 tests/agents/test_runner_run.py 范式)。

    run() 路径需要 build_provider:caller 自行用 MonkeyPatch 替换为 _FakeProvider/
    _HistoryCapturingProvider(见各 test)。本 helper 仅组装 runner。
    """
    return SubagentRunner(
        defn=_defn(), prompt=prompt, description="d", tool_use_id="call_resume",
        model_override="", spawn_depth=1, is_async=False,
        parent_provider=_FakeProvider(profile=_cfg().providers["p"]), parent_registry=None,
        parent_gate=None, cfg=_cfg(), app=None, ctx=_ctx(),
        project_root=tmp_path, root=tmp_path,
        agent_id=agent_id, resume_from=resume_from,
    )


def _write_paired_sidechain(path: Path, user_text: str = "原任务") -> None:
    """写一条已配对的完整 Turn(user + assistant-text),不触发 repair。"""
    path.write_text(
        "\n".join([
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": user_text}]},
            }),
            json.dumps({
                "type": "assistant",
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": "旧回复"}]},
            }),
        ]) + "\n",
        encoding="utf-8",
    )


# ---- Step 1:__init__ 注入 ----


def test_runner_accepts_injected_agent_id_and_resume_from(tmp_path: Path):
    """注入旧 agent_id + resume_from → 复用(非新生成)+ 存档指向侧链。"""
    fake_sidechain = tmp_path / "agent-sub-old123.jsonl"
    _write_paired_sidechain(fake_sidechain, user_text="原任务")
    runner = _build_test_runner(
        agent_id="sub-old123", resume_from=fake_sidechain, prompt="继续,改成 X",
        tmp_path=tmp_path,
    )
    assert runner.agent_id == "sub-old123"  # 复用,非新生成
    assert runner.resume_from == fake_sidechain


def test_runner_defaults_agent_id_when_not_injected(tmp_path: Path):
    """不注入 agent_id → 保持原行为:sub-{uuid12}。回归红线(不破坏既有构造)。"""
    runner = _build_test_runner(
        agent_id=None, resume_from=None, prompt="新任务", tmp_path=tmp_path,
    )
    assert runner.agent_id.startswith("sub-")
    assert len(runner.agent_id) == len("sub-") + 12
    assert runner.resume_from is None


# ---- run() resume 分支 ----


async def test_run_resume_loads_sidechain_as_history_and_appends_direction(tmp_path: Path):
    """resume 模式:run() 加载侧链当 history 喂 LLM(只重放)、direction 作新 user turn 落侧链。

    规避 #11712:用户新方向必须是真实 user turn 落侧链,而非靠推断旧回复。
    断言:provider.stream 收到非空 history(含旧 user "原任务")+ 当前 sidechain 新增
    direction 的 user 行(由 store.append 在 resume 分支写入)。
    """
    from birdcode.agents import runner

    fake_sidechain = tmp_path / "agent-sub-resume1.jsonl"
    _write_paired_sidechain(fake_sidechain, user_text="原任务")

    captured: dict[str, object] = {}

    def _spy(profile, app, *, registry=None, system_override=None, mcp_instructions=None):  # noqa: ARG001
        return _HistoryCapturingProvider(profile=profile, captured=captured)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "build_provider", _spy)
        runner_obj = _build_test_runner(
            agent_id="sub-resume1", resume_from=fake_sidechain, prompt="继续,改成 X",
            tmp_path=tmp_path,
        )
        sidechain_path = runner_obj.sidechain_path  # store.append 写入此路径
        report = await runner_obj.run()

    # history 加载了侧链旧 user(非空,包含"原任务")—— 证明侧链当 history 重放
    hist = captured.get("history")
    assert isinstance(hist, list)
    assert len(hist) >= 1
    old_user_texts: list[str] = []
    for t in hist:
        for m in t.messages:
            if m.role == "user":
                for b in m.content:
                    if hasattr(b, "text"):
                        old_user_texts.append(b.text)
    assert any("原任务" in t for t in old_user_texts)
    # 用户新方向已落侧链(规避 #11712):sidechain 含 direction 的 user 行
    assert sidechain_path.exists()
    assert "继续,改成 X" in sidechain_path.read_text(encoding="utf-8")
    # report 正常完成(resume 路径不崩)
    assert report.status == "completed"
