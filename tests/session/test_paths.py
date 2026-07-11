# tests/session/test_paths.py
from pathlib import Path

from birdcode.session import paths


def test_encode_cwd_windows_drive():
    """Windows 盘符与分隔符全折成 '-'(对标 CC)。"""
    assert paths.encode_cwd(Path(r"C:\Python-project\BirdCode")) == "C--Python-project-BirdCode"


def test_encode_cwd_posix():
    assert paths.encode_cwd(Path("/home/u/repo")) == "-home-u-repo"


def test_sessions_dir_under_root():
    root = Path("/tmp/sess")
    sd = paths.sessions_dir(root, Path(r"C:\proj"))
    assert sd == Path("/tmp/sess/C--proj")


def test_session_jsonl_and_dir_paths():
    root = Path("/tmp/sess")
    sid = "abc123"
    proj = Path(r"C:\proj")
    assert paths.session_jsonl(root, sid, proj) == Path("/tmp/sess/C--proj/abc123.jsonl")
    assert paths.session_dir(root, sid, proj) == Path("/tmp/sess/C--proj/abc123")


def test_tool_results_subpath():
    root = Path("/tmp/sess")
    p = paths.tool_result_path(root, "sid1", Path(r"C:\proj"), "call_xyz")
    assert p == Path("/tmp/sess/C--proj/sid1/tool-results/call_xyz.txt")


def test_default_root_under_home():
    assert paths.default_root() == Path.home() / ".birdcode" / "projects"


def test_subagents_paths():
    root = Path("/r")
    proj = Path("C:/proj")
    sd = paths.subagents_dir(root, "s1", proj)
    assert sd == Path("/r") / paths.encode_cwd(proj) / "s1" / "subagents"
    assert (
        paths.subagent_jsonl_path(root, "s1", proj, "a0685194cbfe9829e")
        == sd / "agent-a0685194cbfe9829e.jsonl"
    )
    assert (
        paths.subagent_meta_path(root, "s1", proj, "a0685194cbfe9829e")
        == sd / "agent-a0685194cbfe9829e.meta.json"
    )


def test_worktree_name_isolation():
    """F10:worktree_name → 多套 worktree/<name>/;None(默认)→ 主仓目录。"""
    root = Path("/tmp/sess")
    proj = Path(r"C:\proj")
    assert paths.sessions_dir(root, proj, worktree_name="foo") == Path(
        "/tmp/sess/C--proj/worktree/foo"
    )
    assert paths.sessions_dir(root, proj) == Path("/tmp/sess/C--proj")  # None=主仓
    assert paths.session_jsonl(root, "s1", proj, worktree_name="foo") == Path(
        "/tmp/sess/C--proj/worktree/foo/s1.jsonl"
    )
    assert paths.subagent_jsonl_path(root, "s1", proj, "a1", worktree_name="foo") == Path(
        "/tmp/sess/C--proj/worktree/foo/s1/subagents/agent-a1.jsonl"
    )
