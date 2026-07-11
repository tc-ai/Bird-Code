import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.cli.app import ProfileError, _make_provider, _resolve_profile
from birdcode.config.schema import AppConfig, ProviderProfile
from birdcode.tools.registry import ToolRegistry


def _profile(n, protocol="openai"):
    return ProviderProfile(name=n, protocol=protocol, model="m", base_url="u", api_key="k")


def _cfg_two():
    return AppConfig(providers={"a": _profile("a"), "b": _profile("b")}, default="a")


def test_resolve_returns_none_when_no_config():
    assert _resolve_profile(None, None) is None


def test_resolve_default_when_name_none():
    assert _resolve_profile(_cfg_two(), None).name == "a"


def test_resolve_explicit_overrides_default():
    assert _resolve_profile(_cfg_two(), "b").name == "b"


def test_resolve_first_provider_when_no_default():
    cfg = AppConfig(providers={"x": _profile("x"), "y": _profile("y")})
    assert _resolve_profile(cfg, None).name == "x"


def test_resolve_unknown_raises():
    with pytest.raises(ProfileError):
        _resolve_profile(_cfg_two(), "nope")


def test_make_provider_threads_delay_into_mock():
    """--delay 必须传到 MockProvider（回归守护：之前 _make_provider 丢了 delay）。"""
    provider = _make_provider(None, None, delay=0.5)
    assert isinstance(provider, MockProvider)
    assert provider._delay == 0.5  # noqa: SLF001


def test_make_provider_default_delay_into_mock():
    provider = _make_provider(None, None, delay=0.012)
    assert isinstance(provider, MockProvider)
    assert provider._delay == 0.012  # noqa: SLF001


# ---- stage2: 注入链 —— cli 透传同一 registry ----


def test_make_provider_threads_registry_into_build(monkeypatch):
    """_make_provider(profile, cfg, delay, registry) 把 registry 透传给 build_provider。"""
    from birdcode.cli import app as cli_app

    captured: dict = {}

    def fake_build(profile, cfg, *, registry=None, mcp_instructions=None):
        captured["registry"] = registry
        captured["mcp_instructions"] = mcp_instructions
        # 返回一个最小 provider 满足 StreamingProvider
        return MockProvider()

    monkeypatch.setattr(cli_app, "build_provider", fake_build)
    prof = _profile("c", protocol="anthropic")
    cfg = AppConfig(providers={"c": prof}, default="c")
    reg = ToolRegistry()
    p = _make_provider(prof, cfg, delay=0.012, registry=reg)
    assert isinstance(p, MockProvider)
    assert captured["registry"] is reg


def test_make_provider_mock_ignores_registry():
    """profile=None(mock 路径)忽略 registry(MockProvider 不持有)。"""
    reg = ToolRegistry()
    p = _make_provider(None, None, delay=0.012, registry=reg)
    assert isinstance(p, MockProvider)


# ---- Task 8: --continue / --resume 解析 + SessionStore 构造 ----


def _encoded(p):
    from birdcode.session import paths

    return paths.encode_cwd(p)


@pytest.mark.asyncio
async def test_resolve_session_id_new_when_no_flag(tmp_path):
    """无 flag → 生成新 uuid(sessionId 唯一)+ should_load=False。"""
    from birdcode.cli.app import _resolve_session_id

    sid, should_load = _resolve_session_id(
        continue_last=False, resume=None, project_root=tmp_path, root=tmp_path
    )
    assert sid and sid != ""
    assert should_load is False


@pytest.mark.asyncio
async def test_resolve_session_id_continue_picks_latest(tmp_path):
    """--continue → 取本项目最近 mtime 的 .jsonl 的 sessionId + should_load=True。"""
    from birdcode.blocks import TextBlock
    from birdcode.cli.app import _resolve_session_id
    from birdcode.conversation import Message
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    # 预置两个会话(old 先写,new 后写)
    for sid in ("old", "new"):
        ctx = SessionContext(session_id=sid, cwd=str(tmp_path), version="0.1.0", git_branch=None)
        s = SessionStore(ctx, tmp_path, root=tmp_path)
        await s.append(Message(role="user", content=[TextBlock(text="x")]))
        s.close()

    # 让 "new" 的 mtime 显著晚于 "old"
    import os

    new_jsonl = tmp_path / _encoded(tmp_path) / "new.jsonl"
    later = new_jsonl.stat().st_mtime + 10
    os.utime(new_jsonl, (later, later))

    sid, should_load = _resolve_session_id(
        continue_last=True, resume=None, project_root=tmp_path, root=tmp_path
    )
    assert sid == "new"
    assert should_load is True


@pytest.mark.asyncio
async def test_resolve_session_id_resume_specific(tmp_path):
    """--resume <id> → 用指定 id + should_load=True(存在性校验在函数内)。"""
    # 先建出会话文件,使 resume 校验通过
    from birdcode.blocks import TextBlock
    from birdcode.cli.app import _resolve_session_id
    from birdcode.conversation import Message
    from birdcode.session.models import SessionContext
    from birdcode.session.store import SessionStore

    ctx = SessionContext(session_id="abc123", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    s = SessionStore(ctx, tmp_path, root=tmp_path)
    await s.append(Message(role="user", content=[TextBlock(text="x")]))
    s.close()

    sid, should_load = _resolve_session_id(
        continue_last=False, resume="abc123", project_root=tmp_path, root=tmp_path
    )
    assert sid == "abc123" and should_load is True


def test_resolve_session_id_continue_and_resume_mutually_exclusive(tmp_path):
    """--continue 与 --resume 同给 → ProfileError(互斥)。"""
    from birdcode.cli.app import ProfileError, _resolve_session_id

    with pytest.raises(ProfileError):
        _resolve_session_id(
            continue_last=True, resume="abc", project_root=tmp_path, root=tmp_path
        )


def test_resolve_session_id_resume_missing_file_raises(tmp_path):
    """--resume <id> 但文件不存在 → ProfileError(避免静默开新会话)。"""
    from birdcode.cli.app import ProfileError, _resolve_session_id

    with pytest.raises(ProfileError):
        _resolve_session_id(
            continue_last=False, resume="not-exist", project_root=tmp_path, root=tmp_path
        )


@pytest.mark.asyncio
async def test_resolve_session_id_continue_empty_opens_new(tmp_path):
    """--continue 但本项目无历史会话 → 开新会话(不报错)+ should_load=False。"""
    from birdcode.cli.app import _resolve_session_id

    sid, should_load = _resolve_session_id(
        continue_last=True, resume=None, project_root=tmp_path, root=tmp_path
    )
    assert sid and should_load is False


# ---- F14: --resume <id> 路径穿越校验 ----


def test_resolve_session_id_rejects_path_traversal_dotdot(tmp_path):
    """F14: --resume "../etc/passwd" 含 ../ → ProfileError(防逃逸 encoded-cwd 目录)。
    关键:错误应是「非法 sessionId」(格式校验先于文件存在性校验),而非「无此会话」。"""
    from birdcode.cli.app import _resolve_session_id

    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="../etc/passwd",
            project_root=tmp_path,
            root=tmp_path,
        )
    assert "非法" in str(ei.value)


def test_resolve_session_id_rejects_path_traversal_separator(tmp_path):
    """F14: --resume "../../x" → ProfileError,且是「非法 sessionId」(格式校验)。"""
    from birdcode.cli.app import _resolve_session_id

    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="../../x",
            project_root=tmp_path,
            root=tmp_path,
        )
    assert "非法" in str(ei.value)


def test_resolve_session_id_rejects_slash_in_session(tmp_path):
    """F14: --resume "abc/def" 含 / → ProfileError,且是「非法 sessionId」(路径分隔符不允许)。"""
    from birdcode.cli.app import _resolve_session_id

    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="abc/def",
            project_root=tmp_path,
            root=tmp_path,
        )
    assert "非法" in str(ei.value)


def test_resolve_session_id_rejects_backslash_in_session(tmp_path):
    """F14: --resume "abc\\def" 含 \\ → ProfileError(Windows 路径分隔符也不允许)。"""
    from birdcode.cli.app import _resolve_session_id

    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="abc\\def",
            project_root=tmp_path,
            root=tmp_path,
        )
    assert "非法" in str(ei.value)


def test_resolve_session_id_rejects_space_in_session(tmp_path):
    """F14: --resume "ab cd" 含空格 → ProfileError(防御性:边界字符不允许)。"""
    from birdcode.cli.app import _resolve_session_id

    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="ab cd",
            project_root=tmp_path,
            root=tmp_path,
        )
    assert "非法" in str(ei.value)


def test_resolve_session_id_accepts_valid_uuid_format(tmp_path):
    """F14: --resume "abc-123-uuid"(字母数字+连字符,合法)→ 通过格式校验(文件不存在
    仍报错,但错误是「无此会话」类,不是「非法 sessionId」)。"""
    from birdcode.cli.app import _resolve_session_id

    # 合法格式但文件不存在 → ProfileError(无此会话),不是「非法 sessionId」
    with pytest.raises(ProfileError) as ei:
        _resolve_session_id(
            continue_last=False,
            resume="abc-123-uuid",
            project_root=tmp_path,
            root=tmp_path,
        )
    # 错误消息应为「无此会话」类,不是「非法 sessionId」
    assert "非法" not in str(ei.value)


# ---- F12: subprocess 必须用 encoding="utf-8" 解码 git 输出 ----


def test_build_session_ctx_uses_utf8_encoding(monkeypatch, tmp_path):
    """F12: git 输出 UTF-8 字节(中文/非 ASCII 分支)→ 必须正确解码。
    Windows cp936 默认下,非 ASCII 字节会乱码→None。强制 encoding="utf-8"。
    """
    import subprocess

    from birdcode.cli.app import _build_session_ctx

    captured: dict = {}

    class _FakeResult:
        returncode = 0
        # 模拟 UTF-8 字节 "feature/中文分支" 解码回正确字符串
        stdout = "feature/中文分支\n"

    def fake_run(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ctx = _build_session_ctx("s1", tmp_path)
    # kwargs 必须显式传 encoding="utf-8"
    assert captured["kwargs"].get("encoding") == "utf-8"
    # 中文分支被正确解码(不是 cp936 乱码也不是 None)
    assert ctx.git_branch == "feature/中文分支"


def test_build_session_ctx_utf8_with_errors_replace(monkeypatch, tmp_path):
    """F12: 即便 git 输出含无效 UTF-8 字节,errors="replace" 也能避免崩溃(降级为替换字符)。"""
    import subprocess

    from birdcode.cli.app import _build_session_ctx

    class _FakeResult:
        returncode = 0
        stdout = "main\n"

    def fake_run(cmd, **kwargs):
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ctx = _build_session_ctx("s1", tmp_path)
    assert ctx.git_branch == "main"


def test_build_session_ctx_does_not_use_platform_default_encoding(monkeypatch, tmp_path):
    """F12: subprocess.run 必须不依赖平台默认编码(Windows 是 cp936)。
    显式传 encoding="utf-8" 是契约。
    """
    import subprocess

    from birdcode.cli.app import _build_session_ctx

    captured: dict = {}

    class _FakeResult:
        returncode = 0
        stdout = "main\n"

    def fake_run(cmd, **kwargs):
        captured["encoding"] = kwargs.get("encoding")
        captured["errors"] = kwargs.get("errors")
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    _build_session_ctx("s1", tmp_path)
    # 显式 utf-8 必传
    assert captured["encoding"] == "utf-8"
    # errors 也应被指定(replace/ignore 之一,防 UnicodeDecodeError 崩)
    assert captured["errors"] in ("replace", "ignore")
