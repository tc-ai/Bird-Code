"""YAML 配置加载：解析、env 覆盖密钥、校验、路径解析。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import cast

import yaml

from birdcode.config.schema import AppConfig
from birdcode.utils.logging import get_logger
from birdcode.utils.paths import find_project_root

log = get_logger("birdcode.config")


class ConfigError(Exception):
    """配置无效（缺密钥、非法值、结构错误等）。"""


_VAR_RE = re.compile(r"\$\{(\w+)\}")
_LEFTOVER_RE = re.compile(r"\$\{[^}]*\}")  # 兜底:捕获 \w+ 匹配不上的 ${...}(如 ${MY-VAR})


def expand_env_vars(value: object) -> object:
    """递归对 dict/list/str 里的 ${VAR} 做展开。

    缺失变量 → 替换为空串 + warning(不杀启动)。非 str 值原样返回。
    对合并后的 raw dict 调用,在 model_validate 之前。
    """
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            val = os.environ.get(name)
            if val is None:
                log.warning("配置引用了未设置的环境变量 ${%s},替换为空串", name)
                return ""
            return val

        out = _VAR_RE.sub(_sub, value)
        # ${MY-VAR}/${host.name}/${VAR-default} 等非 \w+ 名不会被 _VAR_RE 消费,原样残留→
        # 静默变成错误的认证串。显式告警(与「未设置变量」一致提示),避免晦涩失败。
        for ref in _LEFTOVER_RE.findall(out):
            log.warning("配置引用 %s 无法展开(变量名须为 \\w+;如需字面量请避免 ${...} 语法)", ref)
        return out
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def deep_merge(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    """递归合并:b 盖 a。dict→递归;其余→ b 整体覆盖。"""
    out: dict[str, object] = dict(a)
    for k, bv in b.items():
        av = out.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            out[k] = deep_merge(av, bv)
        else:
            out[k] = bv
    return out


def _default_config_paths(project_root: Path | None = None) -> list[Path]:
    """两层:[user, project] —— user 在前、project 在后,project 盖 user。

    project_root 显式传入(已 resolve worktree→主仓)优先;否则 find_project_root()。
    worktree 里启动时 find_project_root() 返回当前 worktree(.git 文件作标记),
    而 config.yaml 不进 git(含密钥,本地)→ worktree 里无此文件 → 项目层漏载。
    传入主仓 project_root 使项目层锚到 <main>/.birdcode/config.yaml。
    """
    pr = project_root if project_root is not None else find_project_root()
    return [
        Path.home() / ".birdcode" / "config.yaml",
        pr / ".birdcode" / "config.yaml",
    ]


def load_config(
    path: Path | None = None,
    *,
    paths: list[Path] | None = None,
    project_root: Path | None = None,
) -> AppConfig | None:
    """加载配置。

    - 显式 path(单文件,测试用)→ 单文件模式,行为同旧版。
    - paths(多文件,测试用)→ 按序 deep-merge,后者盖前者。
    - 都不给 → 两层默认 [user, project];project_root 锚定项目层(worktree→主仓)。
    任一文件不存在→跳过;全不存在→None。
    """
    if paths is not None:
        file_paths = paths
    elif path is not None:
        file_paths = [path]
    else:
        file_paths = _default_config_paths(project_root)

    merged: dict[str, object] = {}
    loaded: list[Path] = []
    for p in file_paths:
        if not p.exists():
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 解析失败 {p}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"配置顶层应为映射,{p} 实得 {type(raw).__name__}")
        merged = deep_merge(merged, raw)
        loaded.append(p)
    if not loaded:
        return None

    merged = cast(dict[str, object], expand_env_vars(merged))  # 合并后、校验前展开 ${VAR}

    try:
        cfg = AppConfig.model_validate(merged)
    except Exception as e:
        raise ConfigError(f"配置 schema 校验失败: {e}") from e
    _inject_names(cfg)
    _apply_env_overrides(cfg)
    _validate(cfg)
    # 对【所有实际加载的文件】查权限,不只是最后一个:两层合并时用户级(~/.birdcode,
    # 含 api_key)也要查——旧实现只查 last_path(=项目层),用户级密钥文件漏查。
    for p in loaded:
        _check_perms(p)
    return cfg


def _inject_names(cfg: AppConfig) -> None:
    for name, profile in cfg.providers.items():
        profile.name = name


def _apply_env_overrides(cfg: AppConfig) -> None:
    """ANTHROPIC_API_KEY / OPENAI_API_KEY 覆盖对应 protocol profile 的 api_key。

    注意：同一 protocol 的所有 profile 共享同一个 env-var key（例如两个 openai
    profile 都用 OPENAI_API_KEY），因此无法通过 env 给它们设置不同的 key——
    需要不同 key 时请直接在 YAML 里填写。
    """
    mapping = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    for profile in cfg.providers.values():
        env_val = os.environ.get(mapping.get(profile.protocol, ""))
        if env_val:
            profile.api_key = env_val


def _validate(cfg: AppConfig) -> None:
    if not cfg.providers:
        raise ConfigError("未配置任何 provider（将回退 MockProvider，但配置文件已存在故视为错误）")
    if cfg.default is not None and cfg.default not in cfg.providers:
        raise ConfigError(f"default 指向不存在的 profile: {cfg.default!r}")
    for profile in cfg.providers.values():
        if not profile.api_key:
            env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[profile.protocol]
            raise ConfigError(
                f"profile {profile.name!r} 缺 api_key（在配置里填写或设置 env {env}）"
            )
        if profile.thinking is not None and profile.thinking.budget_tokens >= cfg.max_tokens:
            raise ConfigError(
                f"profile {profile.name!r}: thinking.budget_tokens 须 < max_tokens"
                f"（{profile.thinking.budget_tokens} ≥ {cfg.max_tokens}）"
            )


def _check_perms(p: Path) -> None:
    try:
        mode = p.stat().st_mode & 0o777
        if mode & 0o077:
            log.warning("配置文件 %s 权限过宽（%o），建议 0600（含 api_key）", p, mode)
    except OSError:
        pass
