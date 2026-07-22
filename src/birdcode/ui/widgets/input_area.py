# src/birdcode/ui/widgets/input_area.py
"""带框输入区：TextArea 子类。

- 输入以 "/" 开头（且无空格）时自动弹命令补全浮层，随输入实时过滤；键入空格或删掉
  "/" 则收起、输入框沉底。
- enter：浮层开 → 执行选中命令；否则行尾 '\\' 续行换行、否则提交。
- tab：浮层开 → 填选中命令名 + 空格（继续输参数）；浮层关且 "/" 开头 → 单匹配补全。
  shift+tab 已被 app 的 cycle_mode 占用，不冲突。
- 浮层打开时：↑↓ 移动高亮；Esc 由 app.action_interrupt 收口（关浮层或中断 agent）。
- ctrl+c / ctrl+v：交给 TextArea 内建 copy/paste（OSC 52）。
- 高度随行数伸缩（1..10）。

注意：必须覆盖 _on_key（非 on_key）——TextArea 的字符插入与 enter->\\n 都在
内部的 async _on_key 里；on_key 在其 MRO 中不存在。
"""

from __future__ import annotations

from pathlib import Path

from textual import events
from textual.message import Message
from textual.widgets import TextArea

from birdcode.ui.pure import (
    AtToken,
    FileCandidate,
    enter_action,
    list_file_candidates,
    parse_at_token,
)
from birdcode.ui.widgets.file_completion_menu import FileCompletionMenu


def command_prefix(line: str) -> str:
    """取行从开头到首个空格的命令前缀(含 "/")。非 "/" 开头 → ""。"""
    if not line.startswith("/"):
        return ""
    return line.split(maxsplit=1)[0]


class InputArea(TextArea):
    DEFAULT_CSS = (
        "InputArea { border: none; padding: 0 1; height: 1; } InputArea:focus { border: none; }"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._active_at: AtToken | None = None

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    class CompletionRequested(Message):
        """多匹配补全:请求 app 挂 CompletionMenu 浮层。"""

        def __init__(self, candidates: list) -> None:
            self.candidates = candidates
            super().__init__()

    class FileCompletionRequested(Message):
        """@ 文件补全:请求 app 挂 FileCompletionMenu 浮层。"""

        def __init__(self, candidates: list[FileCandidate], truncated: bool, at: AtToken) -> None:
            self.candidates = candidates
            self.truncated = truncated
            self.at = at
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        # 补全浮层打开时优先处理导航/确认;Esc 由 app.action_interrupt 收口。
        menu = getattr(self.app, "_completion_menu", None)
        if isinstance(menu, FileCompletionMenu):
            if await self._on_at_menu_key(event, menu):
                return
            # 未拦截的键(可见字符/退格)放行 → 文本变化触发 _route_completion 重新过滤
        elif menu is not None:
            if event.key == "up":
                menu.move(-1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "down":
                menu.move(1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                # 执行选中命令:直接提交其名(不留输入框)。
                name = menu.selected_name()
                self.app._dismiss_completion()
                event.stop()
                event.prevent_default()
                self._submit_text(name)
                return
            if event.key == "tab":
                # 填选中命令名 + 空格,继续输参数(不提交)。
                name = menu.selected_name()
                self.text = name + " "
                self._cursor_to_end()
                self.app._dismiss_completion()
                event.stop()
                event.prevent_default()
                return
            # 其它键(字母/退格等):不关浮层,放行 → 文本变化触发 _maybe_auto_complete 重新过滤。

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._submit_or_continue()
            return
        if event.key == "tab" and self.text.startswith("/"):
            event.stop()
            event.prevent_default()
            await self._try_complete()
            return
        # ctrl+c / ctrl+v 不再拦截 → 交给 TextArea 内建的 copy/paste（OSC 52）。
        # 中断/退出改为应用级绑定：Esc 中断、Ctrl+Q 退出。
        await super()._on_key(event)

    async def _on_at_menu_key(self, event: events.Key, menu: FileCompletionMenu) -> bool:
        """@ 浮层打开时的键位处理。返回 True 表示已拦截。"""
        if event.key in ("up", "down"):
            menu.move(-1 if event.key == "up" else 1)
            event.stop()
            event.prevent_default()
            return True
        if event.key in ("tab", "enter"):
            cand = menu.selected_candidate()
            if cand is not None:
                # 目录(Tab):补 full_path(带尾 /)→ changed 自动展开下一级;
                # 文件或 Enter:补 full_path + 空格(闭合 token)。
                trail = "" if (event.key == "tab" and cand.is_dir) else " "
                self._replace_at_token(cand.full_path + trail)
                self.app._dismiss_completion()
                self._active_at = None
            event.stop()
            event.prevent_default()
            return True
        return False

    def _replace_at_token(self, replacement: str) -> None:
        """把当前 @token([start_col, cursor_col) 区间)替换为 replacement,光标到替换末尾。"""
        at = self._active_at
        if at is None:
            return
        row, col = self.cursor_location
        start = at.start_col
        if col < start:
            return
        self.delete((row, start), (row, col))
        self.insert(replacement, (row, start))
        self.move_cursor((row, start + len(replacement)))
        self.refresh()

    async def _try_complete(self) -> None:
        """Tab 补全:单匹配直接补全名+空格;多匹配 post CompletionRequested。"""
        from birdcode.ui.commands.builtins import build_builtin_registry

        app = self.app
        # 优先用 app 已注入的注册表;未注入时回退到内置注册表(测试路径)
        reg = getattr(app, "_command_registry", None) or build_builtin_registry()
        prefix = command_prefix(self.text)
        if not prefix:
            return
        # 取小写匹配(parse_command 也转小写)→ Tab 补全与 Enter 执行大小写行为一致。
        candidates = reg.complete(prefix.lower())
        if len(candidates) == 1:
            self._replace_command_prefix(candidates[0].name)
        elif len(candidates) >= 2:
            self.post_message(InputArea.CompletionRequested(candidates))
        # 0 候选 → 不动作

    def _replace_command_prefix(self, new_name: str) -> None:
        """替换命令名前缀、保留其后已输内容(参数/换行);光标移到末尾。

        new_name 为补全后的命令名(如 "/help",不含尾空格)。无参数时补一个空格
        待输;已有参数则原样保留,避免整体替换静默丢弃用户已输的参数/多行内容。
        """
        text = self.text
        prefix = command_prefix(text)
        rest = text[len(prefix) :] if prefix and text.startswith(prefix) else ""
        if rest.strip():
            self.text = new_name + rest  # 保留已输参数/多行
        else:
            self.text = new_name + " "  # 无参数 → 补空格待输
        self._cursor_to_end()
        self.refresh()

    def _cursor_to_end(self) -> None:
        """光标移到文本末尾(文档最后一行行尾)。

        TextArea.move_cursor 接受 (row, col) 位置,无 end_of_line 关键字;
        用 self.text 直接算末行行列,避免依赖 document 的 line_count API。
        """
        text = self.text
        row = text.count("\n")
        last_line = text.rsplit("\n", 1)[-1]
        self.move_cursor((row, len(last_line)))

    def _submit_or_continue(self) -> None:
        row, _col = self.cursor_location
        line = self.document.get_line(row)
        if enter_action(line) == "continue":
            self.delete((row, len(line) - 1), (row, len(line)))  # 删尾反斜杠
            self.insert("\n", (row, len(line) - 1))  # 换行
            return
        text = self.text
        if not text.strip():
            return
        self._submit_text(text)

    def _submit_text(self, text: str) -> None:
        """清空输入框并提交 text。清空后立即重绘把终端光标拉回(0,0),避免中文 IME 锚到旧位置。"""
        self.text = ""
        self.refresh()
        self.post_message(InputArea.Submitted(text))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # 随【软换行后的视觉行数】伸缩（1..10）；soft_wrap=True 时长行自动折行
        n = max(1, min(self.wrapped_document.height, 10))
        self.styles.height = n
        self._route_completion()

    def _maybe_auto_complete(self) -> None:
        """输入以 "/" 开头且尚在命令名段(无空格) → 弹/更新补全浮层;否则关闭。

        浮层随输入实时过滤;键入空格(开始输参数)或删掉 "/" → 关闭、输入框沉底。
        app 未提供补全机制(单元测试 harness)时为 no-op。
        """
        app = self.app
        dismiss = getattr(app, "_dismiss_completion", None)
        if dismiss is None:
            return  # app 未提供补全机制(非 BirdApp 测试 harness)→ 不自动补全
        text = self.text
        if not (text.startswith("/") and " " not in text):
            dismiss()
            return
        reg = getattr(app, "_command_registry", None)
        if reg is None:
            dismiss()
            return
        cands = reg.complete(command_prefix(text).lower())
        if not cands:
            dismiss()
            return
        self.post_message(InputArea.CompletionRequested(cands))

    def _route_completion(self) -> None:
        """命令名段(/开头无空格)→ 命令补全;否则 → @ 文件补全。二者互斥。

        非命令态下,仅在没有活跃 @ 会话(_active_at is None)时才调用
        _maybe_auto_complete——它在非命令态会 dismiss,可顺带关掉命令态残留
        的命令浮层;若有 @ 会话则交给 _maybe_complete_at 专管 @ 浮层,避免
        _maybe_auto_complete 误关 @ 浮层。
        """
        if self.text.startswith("/") and " " not in self.text:
            self._maybe_auto_complete()
            return
        if self._active_at is None:
            self._maybe_auto_complete()
        self._maybe_complete_at()

    def _maybe_complete_at(self) -> None:
        """光标处若有活跃 @ token → 弹/更新 @ 浮层;否则(且我们正管 @ 浮层时)关闭。

        app 无补全机制(非 BirdApp 测试 harness)时为 no-op。
        """
        app = self.app
        dismiss = getattr(app, "_dismiss_completion", None)
        if dismiss is None:
            return
        row, col = self.cursor_location
        line = self.document.get_line(row)
        at = parse_at_token(line, col)
        if at is None:
            if self._active_at is not None:
                self._active_at = None
                dismiss()
            return
        if "/" in at.prefix:
            base_rel, _, name_prefix = at.prefix.rpartition("/")
        else:
            base_rel, name_prefix = "", at.prefix
        base_dir = Path.cwd() / base_rel if base_rel else Path.cwd()
        cands, truncated = list_file_candidates(base_dir, name_prefix, base_rel)
        if not cands:
            if self._active_at is not None:
                self._active_at = None
                dismiss()
            return
        self._active_at = at
        self.post_message(InputArea.FileCompletionRequested(cands, truncated, at))
