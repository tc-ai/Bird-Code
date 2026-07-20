from pathlib import Path

import pytest

from birdcode.agents.builtins import build_capability_registry
from birdcode.agents.registry import CapabilityConflictError


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_merge_agents_and_skills(tmp_path):
    agent_dir = tmp_path / "agents"
    skill_dir = tmp_path / "skill"
    agent_user = tmp_path / "user_agents"
    agent_user.mkdir()
    _write(agent_dir / "x.md", "---\nname: x\ndescription: ax\n---\napersona")  # agent(非透明)
    _write(skill_dir / "y.md", "---\nname: y\ndescription: sy\n---\nsbody")  # skill(透明)
    reg = build_capability_registry(
        tmp_path,
        project_skill_dir=skill_dir,
        _project_agent_dir=agent_dir,
        _user_agent_dir=agent_user,
    )
    x = reg.resolve("x")
    y = reg.resolve("y")
    assert x is not None and x.is_transparent is False
    assert y is not None and y.is_transparent is True


def test_cross_dir_same_name_conflicts(tmp_path):
    agent_dir = tmp_path / "agents"
    skill_dir = tmp_path / "skill"
    agent_user = tmp_path / "user_agents"
    agent_user.mkdir()
    _write(agent_dir / "dup.md", "---\nname: dup\ndescription: a\n---\nap")
    _write(skill_dir / "dup.md", "---\nname: dup\ndescription: s\n---\nsb")
    with pytest.raises(CapabilityConflictError):
        build_capability_registry(
            tmp_path,
            project_skill_dir=skill_dir,
            _project_agent_dir=agent_dir,
            _user_agent_dir=agent_user,
        )


def test_skill_colliding_with_builtin_agent_raises(tmp_path):
    """skill named 'explore' collides with always-loaded builtin agent → raise."""
    skill_dir = tmp_path / "skill"
    agent_user = tmp_path / "user_agents"
    agent_user.mkdir()
    _write(skill_dir / "explore.md", "---\nname: explore\ndescription: fake explore\n---\nbody")
    with pytest.raises(CapabilityConflictError, match=r"skill 'explore'"):
        build_capability_registry(tmp_path, project_skill_dir=skill_dir, _user_agent_dir=agent_user)
