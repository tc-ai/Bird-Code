# src/birdcode/ui/widgets/permission_prompt.py
"""权限确认行内菜单(Claude Code 风格):挂 #composer 底部,3 编号项。

取代 PermissionModalScreen 弹窗。Yes(本次放行)/ allow-all-session(本会话按类别宽放)/
No。session 按类别——edits→Write+Edit+Delete、commands→Bash(规则串由 gate
._add_session_by_category 生成,本 widget 只管文案)。

键路由:挂载后 focus() 消费自带 BINDINGS(y/1/2/n/3/↑↓/enter)。shift+tab、Esc 是
app 级 priority 绑定(cycle_mode/interrupt),本 widget 抢不过 → 由 app 检查活跃
prompt 后分流调 choose("session")/choose("reject")。幂等:首次 choose 触发回调。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from textual.binding import Binding
from textual.widgets import Static

# 与 gate.ModalResult(去 always 后)一致;解耦不复用 gate 类型,避免 ui→permission 反向依赖。
Result = Literal["approve", "reject", "session"]

_EDITS = {"write_file", "edit_file", "delete_file"}


def _tool_display(tool_name: str) -> str:
    """write_file→Write、bash→Bash、mcp__srv__tool→原样(命名空间不可拆,显示完整名)。"""
    if tool_name.startswith("mcp__"):
        return tool_name
    return tool_name.split("_")[0].capitalize()


def _session_label(tool_name: str) -> str:
    """session 选项「本会话放行范围」文案,对应 gate._add_session_by_category 的类别。"""
    if tool_name in _EDITS:
        return "allow all edits during this session"
    if tool_name == "bash":
        return "allow all commands during this session"
    return f"allow all {_tool_display(tool_name)} during this session"


class PermissionPrompt(Static):
    """底部行内 3 项权限菜单(Yes / allow-all-session / No)。

    挂载后须 focus() 才消费自带 BINDINGS。app 层持引用,选择回调后由 app 卸载。
    """

    can_focus = True  # Static 默认不可聚焦;开启后 focus() 生效、自带 BINDINGS 才消费按键

    DEFAULT_CSS = """
    PermissionPrompt {
        border: tall $warning;
        background: $surface;
        color: $text;
        padding: 0 1;
        width: auto;
        height: auto;
    }
    """

    BINDINGS = [
        Binding("y", "approve", show=False),
        Binding("1", "approve", show=False),
        Binding("2", "session", show=False),
        Binding("n", "reject", show=False),
        Binding("3", "reject", show=False),
        Binding("up", "move_up", show=False),
        Binding("down", "move_down", show=False),
        Binding("enter", "confirm", show=False),
    ]

    def __init__(
        self, tool_name: str, summary: str, *, on_result: Callable[[Result], None],
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._summary = summary
        self._on_result = on_result
        self._options: list[tuple[Result, str]] = [
            ("approve", "Yes"),
            ("session", f"Yes, {_session_label(tool_name)} (shift+tab)"),
            ("reject", "No"),
        ]
        self._sel = 0  # 默认 Yes(最保守)
        self._done = False

    def render(self) -> str:  # type: ignore[override]
        head = f"⚠ {_tool_display(self._tool_name)} {self._summary}".rstrip()
        lines = [head]
        for i, (_r, label) in enumerate(self._options):
            marker = "▶ " if i == self._sel else "  "
            lines.append(f"{marker}{i + 1}. {label}")
        return "\n".join(lines)

    def _move(self, delta: int) -> None:
        if self._done:
            return
        self._sel = (self._sel + delta) % len(self._options)
        self.refresh()

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def action_confirm(self) -> None:
        """enter = 提交当前高亮项。"""
        if not self._done:
            self.choose(self._options[self._sel][0])

    def action_approve(self) -> None:
        self.choose("approve")

    def action_session(self) -> None:
        self.choose("session")

    def action_reject(self) -> None:
        self.choose("reject")

    def choose(self, result: Result) -> None:
        """提交结果。app 分流 shift+tab(→session)/Esc(→reject) 与本 widget binding 都走这。

        幂等:首次触发回调,后续(如 Esc 跟在 enter 后)忽略。
        """
        if self._done:
            return
        self._done = True
        self._on_result(result)
