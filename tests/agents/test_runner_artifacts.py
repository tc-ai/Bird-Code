# tests/agents/test_runner_artifacts.py
from __future__ import annotations

from pathlib import Path

import pytest

from birdcode.agents import runner as runner_mod
from birdcode.agents.runner import SubagentReport, _fill_artifacts


@pytest.mark.asyncio
async def test_fill_artifacts_populates_report(tmp_path: Path, monkeypatch):
    """collect 成功 → report.artifacts/committed 被填。"""

    async def _fake_collect(wt: Path, base_sha: str, *, max_files: int = 20):
        return (["sort.py", "foo.txt"], True)

    monkeypatch.setattr(runner_mod, "collect_worktree_changes", _fake_collect)
    report = SubagentReport(True, "done", "completed", 10, 5, 1)
    await _fill_artifacts(report, tmp_path, "base123")
    assert report.artifacts == ["sort.py", "foo.txt"]
    assert report.committed is True


@pytest.mark.asyncio
async def test_fill_artifacts_empty_base_noop(tmp_path: Path, monkeypatch):
    """base_sha 空 → 不调 collect,report 保持默认。"""
    called = {"n": 0}

    async def _fake_collect(wt, base_sha, *, max_files=20):
        called["n"] += 1
        return (["x"], True)

    monkeypatch.setattr(runner_mod, "collect_worktree_changes", _fake_collect)
    report = SubagentReport(True, "done", "completed", 10, 5, 1)
    await _fill_artifacts(report, tmp_path, "")
    assert called["n"] == 0
    assert report.artifacts == []
    assert report.committed is False


@pytest.mark.asyncio
async def test_fill_artifacts_collect_raises_keeps_default(tmp_path: Path, monkeypatch):
    """collect 抛异常 → 吞掉(report 保持默认),不传播。"""

    async def _fake_collect(wt, base_sha, *, max_files=20):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(runner_mod, "collect_worktree_changes", _fake_collect)
    report = SubagentReport(True, "done", "completed", 10, 5, 1)
    await _fill_artifacts(report, tmp_path, "base123")  # 不抛
    assert report.artifacts == []
    assert report.committed is False
