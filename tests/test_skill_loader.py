from pathlib import Path

from birdcode.agents.definition import AgentSource
from birdcode.agents.skill_loader import build_skill_registry, load_md_skills


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_file_skill(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    _write(d / "commit.md",
           "---\nname: commit\ndescription: make commit\nmode: inline\n---\n# steps\n$ARGUMENTS")
    defs = load_md_skills(d, AgentSource.PROJECT)
    assert len(defs) == 1
    s = defs[0]
    assert s.name == "commit"
    assert s.is_transparent is True
    assert s.mode == "inline"
    assert s.body.startswith("# steps")
    assert "$ARGUMENTS" in s.body
    assert s.system_prompt == ""


def test_dir_skill_reads_skill_md(tmp_path):
    d = tmp_path / "skill"
    _write(d / "brainstorming" / "SKILL.md",
           "---\nname: brainstorming\ndescription: brainstorm\n---\nbody")
    _write(d / "brainstorming" / "extra.md", "ignored")  # 非 SKILL.md 不加载
    defs = load_md_skills(d, AgentSource.PROJECT)
    assert [x.name for x in defs] == ["brainstorming"]
    assert defs[0].body == "body"


def test_mode_defaults_inline(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    _write(d / "s.md", "---\nname: s\ndescription: d\n---\nb")
    assert load_md_skills(d, AgentSource.PROJECT)[0].mode == "inline"


def test_mode_fork(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    _write(d / "s.md", "---\nname: s\ndescription: d\nmode: fork\n---\nb")
    assert load_md_skills(d, AgentSource.PROJECT)[0].mode == "fork"


def test_missing_name_or_description_skipped(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    _write(d / "bad.md", "---\ndescription: no name\n---\nbody")
    assert load_md_skills(d, AgentSource.PROJECT) == []


def test_bad_frontmatter_skipped(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    _write(d / "broken.md", "no frontmatter at all")
    assert load_md_skills(d, AgentSource.PROJECT) == []


def test_build_skill_registry_layers(tmp_path):
    user_d = tmp_path / "user"
    proj_d = tmp_path / "proj"
    _write(user_d / "a.md", "---\nname: a\ndescription: ua\n---\nubody")
    _write(proj_d / "a.md", "---\nname: a\ndescription: pa\n---\npbody")  # project 覆盖 user
    _write(proj_d / "b.md", "---\nname: b\ndescription: pb\n---\nbody")
    reg = build_skill_registry(tmp_path, user_dir=user_d, project_dir=proj_d)
    a = reg.resolve("a")
    assert a is not None and a.body == "pbody"  # project 胜
    assert reg.resolve("b") is not None
    assert all(x.is_transparent for x in reg.all())
