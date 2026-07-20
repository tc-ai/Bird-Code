# tests/test_permission_prompt.py
"""PermissionPrompt 行内菜单:3 编号项 + 类别标签 + 选择回调(取代旧 ModalScreen 测试)。"""

from birdcode.ui.widgets.permission_prompt import PermissionPrompt


def _capture() -> tuple[list[str], object]:
    box: list[str] = []

    def cb(result: object) -> None:
        box.append(str(result))

    return box, cb


def test_render_edits_category():
    _, cb = _capture()
    p = PermissionPrompt("write_file", "/a/b.py (3 行)", on_result=cb)
    out = p.render()
    assert "Write /a/b.py (3 行)" in out  # write_file→Write
    assert "1. Yes" in out
    assert "allow all edits during this session" in out
    assert "3. No" in out


def test_render_commands_category():
    _, cb = _capture()
    out = PermissionPrompt("bash", "git status", on_result=cb).render()
    assert "Bash git status" in out
    assert "allow all commands during this session" in out


def test_render_other_category_uses_tool_display():
    _, cb = _capture()
    out = PermissionPrompt("mcp__srv__tool", "do x", on_result=cb).render()
    # mcp 原样显示完整命名空间;session 文案用 all {tool}
    assert "mcp__srv__tool do x" in out
    assert "allow all mcp__srv__tool during this session" in out


def test_default_selection_is_yes():
    _, cb = _capture()
    out = PermissionPrompt("write_file", "f", on_result=cb).render()
    lines = out.splitlines()
    assert lines[1].startswith("▶ ")  # 默认选中第 1 项 Yes
    assert "Yes" in lines[1]


def test_move_changes_selection():
    _, cb = _capture()
    p = PermissionPrompt("bash", "c", on_result=cb)
    p.action_move_down()  # sel 0→1(session)
    lines = p.render().splitlines()
    assert lines[2].startswith("▶ ")
    assert "session" in lines[2]


def test_choose_approve_calls_callback():
    box, cb = _capture()
    PermissionPrompt("write_file", "f", on_result=cb).choose("approve")
    assert box == ["approve"]


def test_choose_is_idempotent():
    box, cb = _capture()
    p = PermissionPrompt("bash", "c", on_result=cb)
    p.choose("session")
    p.choose("reject")  # 已 done,第二次忽略
    assert box == ["session"]


def test_action_bindings_route_to_choose():
    box, cb = _capture()
    PermissionPrompt("write_file", "f", on_result=cb).action_approve()
    PermissionPrompt("write_file", "f", on_result=cb).action_session()
    PermissionPrompt("write_file", "f", on_result=cb).action_reject()
    assert box == ["approve", "session", "reject"]


def test_confirm_uses_current_selection():
    box, cb = _capture()
    p = PermissionPrompt("write_file", "f", on_result=cb)
    p.action_move_down()  # 选中 session
    p.action_confirm()
    assert box == ["session"]
