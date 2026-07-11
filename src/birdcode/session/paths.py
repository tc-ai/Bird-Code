# src/birdcode/session/paths.py
"""会话存储路径解析:encoded-cwd 编码 + sessionId→文件/目录路径。

对标 Claude Code 的 ~/.claude/projects/<encoded-cwd>/ 布局。
encoded-cwd 把工作目录路径里的分隔符(/ \\ :)和盘符冒号折成 '-'。
"""
from __future__ import annotations

from pathlib import Path

# 路径分隔符 + Windows 盘符冒号 → '-'
_SEPARATORS = str.maketrans({"/": "-", "\\": "-", ":": "-"})


def encode_cwd(path: Path) -> str:
    """C:\\Python-project\\BirdCode → C--Python-project-BirdCode;/home/u/repo → -home-u-repo。

    不调 resolve():让 POSIX 风格路径在 Windows 上也按字面字符串编码,
    保证测试跨平台一致;生产中 cwd 总是绝对路径,无需规范化。
    """
    return str(path).translate(_SEPARATORS)


def default_root() -> Path:
    """全局会话根:~/.birdcode/projects/(与 debug.log/permissions 同处 ~/.birdcode)。"""
    return Path.home() / ".birdcode" / "projects"


def sessions_dir(
    root: Path, project_root: Path, *, worktree_name: str | None = None,
) -> Path:
    """本项目所有会话的父目录:<root>/<encoded-cwd>/[worktree/<name>/](per-worktree 隔离)。"""
    d = root / encode_cwd(project_root)
    return d / "worktree" / worktree_name if worktree_name else d


def session_jsonl(
    root: Path, session_id: str, project_root: Path, *, worktree_name: str | None = None,
) -> Path:
    """主会话日志:<root>/<encoded>/[worktree/<name>/]<sessionId>.jsonl。"""
    return sessions_dir(root, project_root, worktree_name=worktree_name) / f"{session_id}.jsonl"


def session_dir(
    root: Path, session_id: str, project_root: Path, *, worktree_name: str | None = None,
) -> Path:
    """同名资源目录:<root>/<encoded>/[worktree/<name>/]<sessionId>/。"""
    return sessions_dir(root, project_root, worktree_name=worktree_name) / session_id


def tool_results_dir(
    root: Path, session_id: str, project_root: Path, *, worktree_name: str | None = None,
) -> Path:
    """工具输出外存:<...>/<sessionId>/tool-results/。"""
    return session_dir(root, session_id, project_root, worktree_name=worktree_name) / "tool-results"


def tool_result_path(
    root: Path, session_id: str, project_root: Path, tool_use_id: str,
    *, worktree_name: str | None = None,
) -> Path:
    """单个工具输出落盘:<...>/tool-results/<tool_use_id>.txt。"""
    return tool_results_dir(
        root, session_id, project_root, worktree_name=worktree_name
    ) / f"{tool_use_id}.txt"


def subagents_dir(
    root: Path, session_id: str, project_root: Path, *, worktree_name: str | None = None,
) -> Path:
    """子 agent 外存目录:<...>/<sessionId>/subagents/(与 tool-results 同构)。"""
    return session_dir(root, session_id, project_root, worktree_name=worktree_name) / "subagents"


def subagent_jsonl_path(
    root: Path, session_id: str, project_root: Path, agent_id: str,
    *, worktree_name: str | None = None,
) -> Path:
    """单个子 agent 会话日志:<...>/subagents/agent-<agent_id>.jsonl。"""
    return subagents_dir(
        root, session_id, project_root, worktree_name=worktree_name
    ) / f"agent-{agent_id}.jsonl"


def subagent_meta_path(
    root: Path, session_id: str, project_root: Path, agent_id: str,
    *, worktree_name: str | None = None,
) -> Path:
    """单个子 agent 元数据:<...>/subagents/agent-<agent_id>.meta.json(唯一可变)。"""
    return subagents_dir(
        root, session_id, project_root, worktree_name=worktree_name
    ) / f"agent-{agent_id}.meta.json"
