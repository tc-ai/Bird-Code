"""从 .birdcode/agents/*.md 加载用户/项目子 agent 定义。

frontmatter(`---` 之间,YAML)→ 字段;正文 → system_prompt。未知字段忽略(前向兼容)。
**name + description 必填**(作 tool 必须有名+描述;缺任一跳过+告警)。可选:disallowed_tools
/ model / kind / parallel_safe / run_in_background(坏值降级)。
坏/无 frontmatter 跳过(不抛,不杀注册)。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from birdcode.agents.definition import AgentDefinition, AgentSource, split_frontmatter
from birdcode.utils.logging import get_logger

log = get_logger("birdcode.agents.loader")


def _parse_one(path: Path, source: AgentSource) -> AgentDefinition | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        log.warning("读 agent md 失败,跳过: %s", path, exc_info=True)
        return None
    split = split_frontmatter(raw)
    if split is None:
        return None
    fm_text, body_text = split
    try:
        meta = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        log.warning("agent md frontmatter 坏,跳过: %s", path, exc_info=True)
        return None
    if not isinstance(meta, dict):
        return None
    system_prompt = body_text.strip()
    # name + description 必填(作 tool 必须有名+描述)。缺任一 → 跳过 + 告警。
    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        log.warning("agent md 缺 name/description,跳过: %s", path)
        return None
    disallowed = meta.get("disallowed_tools") or []
    if isinstance(disallowed, str):
        disallowed = [disallowed]
    elif not isinstance(disallowed, (list, tuple)):
        # 非可迭代标量(YAML `disallowed_tools: yes`→True、`: 3`→int):坏 frontmatter。
        # 契约「坏值跳过、不抛」:此处 try 之外,标量会让 tuple(str(t) for t in disallowed)
        # 抛 TypeError、经 load_md_agents 崩 on_mount 启动。降级空黑名单,不抛。
        log.warning("agent md disallowed_tools 非列表,降级为空: %s", path)
        disallowed = []
    # kind/parallel_safe 可选(自定义 .md 可覆盖默认)。坏值降级 write/False。
    kind_raw = meta.get("kind")
    kind = kind_raw if kind_raw in ("read", "write") else "write"
    ps_raw = meta.get("parallel_safe")
    parallel_safe = ps_raw if isinstance(ps_raw, bool) else False
    rb_raw = meta.get("run_in_background")
    run_in_background = rb_raw if isinstance(rb_raw, bool) else False
    return AgentDefinition(
        name=str(name),
        description=str(description),
        system_prompt=system_prompt,
        disallowed_tools=tuple(str(t) for t in disallowed),
        model=str(meta.get("model") or ""),
        source=source,
        path=path,
        kind=kind,
        parallel_safe=parallel_safe,
        run_in_background=run_in_background,
    )


def load_md_agents(directory: Path, source: AgentSource) -> list[AgentDefinition]:
    """扫 directory/*.md → AgentDefinition 列表(跳过坏文件)。目录不存在 → 空。"""
    if not directory.is_dir():
        return []
    out: list[AgentDefinition] = []
    for path in sorted(directory.glob("*.md")):
        defn = _parse_one(path, source)
        if defn is not None:
            out.append(defn)
    return out
