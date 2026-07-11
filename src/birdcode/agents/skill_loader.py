"""从 .birdcode/skill/ 加载 skill 定义(透明变体,与 agent 合并)。

镜像 agents/loader.py 的 _parse_one/load_md_agents,差异:
- body(frontmatter 后正文)→ defn.body(任务模板,含 $ARGUMENTS),非 system_prompt。
- mode 可选(默认 inline);is_transparent 恒 True(目录决定)。
- 同时支持文件型(foo.md)与目录型(foo/SKILL.md);目录内其它文件不加载(模型按需 Read)。
- name+description 必填(缺→跳过+告警);坏 frontmatter 跳过,不杀启动。
- kind/parallel_safe 不解析(skill fork 保守 write/False)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import yaml

from birdcode.agents.definition import AgentDefinition, AgentSource, split_frontmatter
from birdcode.agents.registry import AgentRegistry
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.agents.skill_loader")


def _parse_skill_one(path: Path, source: AgentSource) -> AgentDefinition | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        log.warning("读 skill md 失败,跳过: %s", path, exc_info=True)
        return None
    split = split_frontmatter(raw)
    if split is None:
        return None
    fm_text, body_text = split
    try:
        meta = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        log.warning("skill md frontmatter 坏,跳过: %s", path, exc_info=True)
        return None
    if not isinstance(meta, dict):
        return None
    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        log.warning("skill md 缺 name/description,跳过: %s", path)
        return None
    body = body_text.strip()
    mode_raw = meta.get("mode")
    mode = cast(Literal["inline", "fork"], mode_raw if mode_raw in ("inline", "fork") else "inline")
    disallowed = meta.get("disallowed_tools") or []
    if isinstance(disallowed, str):
        disallowed = [disallowed]
    elif not isinstance(disallowed, (list, tuple)):
        log.warning("skill md disallowed_tools 非列表,降级为空: %s", path)
        disallowed = []
    rb_raw = meta.get("run_in_background")
    run_in_background = rb_raw if isinstance(rb_raw, bool) else False
    return AgentDefinition(
        name=str(name),
        description=str(description),
        system_prompt="",  # skill 无 persona,body 才是任务
        disallowed_tools=tuple(str(t) for t in disallowed),
        model=str(meta.get("model") or ""),
        source=source,
        path=path,
        run_in_background=run_in_background,
        mode=mode,
        is_transparent=True,  # 目录决定
        body=body,
    )


def load_md_skills(directory: Path, source: AgentSource) -> list[AgentDefinition]:
    """扫 directory 下文件型(*.md)+ 目录型(*/SKILL.md)→ skill 定义列表。

    目录不存在 → 空。解析失败的单文件跳过,不阻断。
    """
    if not directory.is_dir():
        return []
    paths = sorted(set(list(directory.glob("*.md")) + list(directory.glob("*/SKILL.md"))))
    out: list[AgentDefinition] = []
    for path in paths:
        defn = _parse_skill_one(path, source)
        if defn is not None:
            out.append(defn)
    return out


def build_skill_registry(
    project_root: Path,
    *,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> AgentRegistry:
    """组装三层 skill registry:builtin(空,暂无内置 skill)→ user(~/.birdcode/skill)
    → project(<root>/.birdcode/skill)。后注册覆盖先(project > user)。user_dir/project_dir
    仅测试覆盖用。所有 defn is_transparent=True。
    """
    r = AgentRegistry()
    user = user_dir if user_dir is not None else Path.home() / ".birdcode" / "skill"
    for defn in load_md_skills(user, AgentSource.USER):
        r.register(defn)
    proj = project_dir if project_dir is not None else project_root / ".birdcode" / "skill"
    for defn in load_md_skills(proj, AgentSource.PROJECT):
        r.register(defn)
    return r
