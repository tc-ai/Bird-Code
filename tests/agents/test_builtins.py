from pathlib import Path

from birdcode.agents.builtins import BUILTIN_AGENTS, build_agent_registry, build_capability_registry
from birdcode.agents.definition import AgentSource


def test_builtins_present():
    names = {d.name for d in BUILTIN_AGENTS}
    assert {"explore", "plan", "general-purpose"} <= names
    for d in BUILTIN_AGENTS:
        assert d.source is AgentSource.BUILTIN


def test_builtin_kind_parallel_safe():
    """D3/D4:explore/plan=read+可并行;gp=write+串行。"""
    by_name = {d.name: d for d in BUILTIN_AGENTS}
    assert by_name["explore"].kind == "read"
    assert by_name["explore"].parallel_safe is True
    assert by_name["plan"].kind == "read"
    assert by_name["plan"].parallel_safe is True
    assert by_name["general-purpose"].kind == "write"
    assert by_name["general-purpose"].parallel_safe is False


def test_explore_is_readonly():
    from birdcode.agents.runner import build_child_registry
    from birdcode.tools.executor import default_registry

    explore = next(d for d in BUILTIN_AGENTS if d.name == "explore")
    plan = next(d for d in BUILTIN_AGENTS if d.name == "plan")
    # 用 registry 真实工具名(非类名 Write/Edit/Delete —— build_child_registry 精确匹配,
    # 类名永不命中 → 黑名单静默失效)。bash 不在禁用之列:explore/plan 靠 system_prompt
    # 约束「只用于只读操作」(ls/git log/git diff/find/cat),故 bash 允许(代价见 builtins 注释)。
    for d in (explore, plan):
        assert {"write_file", "edit_file", "delete_file"} <= set(d.disallowed_tools)
    # 回归保护:黑名单名字必须真实存在于 registry,且被 build_child_registry 真正排除
    # (旧 bug:disallowed 用 'Write' 等类名,精确匹配永不命中 → 此断言会失败)。
    parent = default_registry()
    assert set(explore.disallowed_tools) <= set(parent.names())  # 名字真实
    child = build_child_registry(parent, explore.disallowed_tools)
    child_names = set(child.names())
    assert not ({"write_file", "edit_file", "delete_file"} & child_names)  # 真正被排除
    assert {"read_file", "grep", "glob"} <= child_names  # 只读工具保留
    assert "bash" in child_names  # bash 不再硬块(explore/plan 可做只读 bash)


def test_build_registry_layers(tmp_path: Path):
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "proj"
    user_dir.mkdir()
    proj_dir.mkdir()
    # user 覆盖 builtin 的 explore
    (user_dir / "explore.md").write_text(
        "---\nname: explore\ndescription: user-explore\n---\nUSER\n", encoding="utf-8"
    )
    (proj_dir / "custom.md").write_text(
        "---\nname: custom\ndescription: proj-only\n---\nPROJ\n", encoding="utf-8"
    )
    r = build_agent_registry(tmp_path, user_dir=user_dir, project_dir=proj_dir)
    assert r.resolve("explore").description == "user-explore"  # user 覆盖 builtin
    assert r.resolve("general-purpose") is not None  # builtin 仍在
    assert r.resolve("custom") is not None  # project 新增


def test_build_capability_registry_test_override_isolates_home(tmp_path, monkeypatch):
    """#2:只传 _project_agent_dir(不传 _user_agent_dir)时,不得回落加载真实 ~/.birdcode/agents。

    回归:旧实现把未传的 _user_agent_dir=None 透传给 build_agent_registry,后者将 None
    解释为「用 Path.home()/​.birdcode/​agents」,于是测试加载开发机本机 agent(若本机有
    commit.md 等还会与测试 skill 同名冲突 → 非确定性失败)。此处把 home 指向含 spy.md 的
    临时目录,断言 spy **不**进 registry。
    """
    home = tmp_path / "fakehome"
    (home / ".birdcode" / "agents").mkdir(parents=True)
    (home / ".birdcode" / "agents" / "spy.md").write_text(
        "---\nname: spy\ndescription: should-not-load\n---\nSP\n", encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    proj_agents = tmp_path / "proj_agents"  # 显式传入(空)
    caps = build_capability_registry(
        tmp_path,
        _project_agent_dir=proj_agents,
        project_skill_dir=tmp_path / "skill",
    )
    names = caps.names()
    assert "spy" not in names  # 真实 home agent 被隔离,不泄漏进测试
    assert {"explore", "plan", "general-purpose"} <= set(names)  # builtin 仍就位
