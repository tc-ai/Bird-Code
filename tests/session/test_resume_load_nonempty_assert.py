# tests/session/test_resume_load_nonempty_assert.py
"""Task 21:防御 Claude Code #15837(jsonl 数据完整但 resume 未加载进上下文 → 失忆)。

加载后断言:jsonl 非空但 decode 后 turns 为空(矛盾状态)→ log warning(只记录,不抛)。
覆盖主会话(TurnController.resume)+ 侧链(load_sidechain_turns)两处。
合法空会话(无 jsonl / 空文件 / 全可解 turns)不 warn。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from birdcode.session import paths
from birdcode.session.codec import load_sidechain_turns
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


async def _noop_event(_ev: object) -> None:
    pass


async def _noop_status() -> None:
    pass


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _broken_user_row() -> dict[str, object]:
    """user 行但 content 全是不可识别 block → message_from_dict 返回 None → decode 丢整条。

    _read_mainline_rows / _read_jsonl_rows 仍保留此行(合法 dict、非 sidechain),
    构成「jsonl 非空但 turns 空」的矛盾状态(#15837 场景)。
    """
    return {
        "type": "user",
        "uuid": "u1",
        "parentUuid": None,
        "message": {"role": "user", "content": [{"type": "totally_unknown_block"}]},
    }


# ---- 主会话:TurnController.resume ----


@pytest.mark.asyncio
async def test_resume_warns_when_history_empty_but_jsonl_nonempty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """主线 jsonl 有内容但 decode 后 turns 空 → log warning(含「截断」/「空」),不抛。"""
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.conversation import TurnController

    ctx = SessionContext(
        session_id="badload", cwd=str(tmp_path), version="0.1.0", git_branch=None
    )
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    jf = tmp_path / paths.encode_cwd(tmp_path) / "badload.jsonl"
    _write_jsonl(jf, [_broken_user_row()])

    ctrl = TurnController(
        MockProvider(delay=0.0),
        on_event=_noop_event,
        on_status=_noop_status,
        store=store,
    )
    # get_logger 置 propagate=False(只写文件,见 utils/logging.py);caplog 依赖向 root
    # 传播,这里临时放开以捕获记录(同 test_agent_loop.py 的既有手法)。
    conv_log = logging.getLogger("birdcode.conversation")
    conv_log.propagate = True
    caplog.set_level(logging.WARNING, logger="birdcode.conversation")
    try:
        turns = await ctrl.resume()
    finally:
        conv_log.propagate = False

    assert turns == []  # 矛盾状态:确实加载出空
    assert any(
        ("截断" in r.message or "空" in r.message) and r.levelno == logging.WARNING
        for r in caplog.records
    ), [r.message for r in caplog.records]
    store.close()


@pytest.mark.asyncio
async def test_resume_no_warn_when_jsonl_genuinely_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """合法全新会话(jsonl 不存在)→ resume 返回空,但绝不 warn(不是矛盾状态)。"""
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.conversation import TurnController

    ctx = SessionContext(
        session_id="fresh", cwd=str(tmp_path), version="0.1.0", git_branch=None
    )
    store = SessionStore(ctx, tmp_path, root=tmp_path)
    ctrl = TurnController(
        MockProvider(delay=0.0),
        on_event=_noop_event,
        on_status=_noop_status,
        store=store,
    )
    conv_log = logging.getLogger("birdcode.conversation")
    conv_log.propagate = True
    caplog.set_level(logging.WARNING, logger="birdcode.conversation")
    try:
        turns = await ctrl.resume()
    finally:
        conv_log.propagate = False

    assert turns == []
    assert not any("截断" in r.message for r in caplog.records)
    store.close()


# ---- 侧链:load_sidechain_turns ----


def test_load_sidechain_warns_when_rows_nonempty_but_turns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """侧链 jsonl 有内容但 decode 后 turns 空 → log warning,不抛。"""
    p = tmp_path / "sc.jsonl"
    _write_jsonl(p, [_broken_user_row()])

    codec_log = logging.getLogger("birdcode.session.codec")
    codec_log.propagate = True
    caplog.set_level(logging.WARNING, logger="birdcode.session.codec")
    try:
        turns = load_sidechain_turns(p)
    finally:
        codec_log.propagate = False

    assert turns == []
    assert any(
        ("截断" in r.message or "空" in r.message) and r.levelno == logging.WARNING
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_load_sidechain_no_warn_when_file_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """侧链文件不存在 → [](合法无侧链),不 warn。"""
    codec_log = logging.getLogger("birdcode.session.codec")
    codec_log.propagate = True
    caplog.set_level(logging.WARNING, logger="birdcode.session.codec")
    try:
        turns = load_sidechain_turns(tmp_path / "nope.jsonl")
    finally:
        codec_log.propagate = False

    assert turns == []
    assert not any("截断" in r.message for r in caplog.records)


def test_load_sidechain_no_warn_when_turns_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """侧链正常解出 turns → 不 warn(只在矛盾状态触发)。"""
    p = tmp_path / "sc.jsonl"
    _write_jsonl(
        p,
        [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi back"}],
                },
            },
        ],
    )
    codec_log = logging.getLogger("birdcode.session.codec")
    codec_log.propagate = True
    caplog.set_level(logging.WARNING, logger="birdcode.session.codec")
    try:
        turns = load_sidechain_turns(p)
    finally:
        codec_log.propagate = False

    assert len(turns) >= 1
    assert not any("截断" in r.message for r in caplog.records)
