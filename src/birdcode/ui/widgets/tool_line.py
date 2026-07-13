# src/birdcode/ui/widgets/tool_line.py
"""工具状态行：● spinner → ⎿ ✓/✗，带可折叠的结果区(set_output)。

start/finish 契约不变(向后兼容)。set_output 挂一个默认收起的折叠区,
点击或调用 toggle 展开 —— 长输出(rg 多行命中 / Read 大文件)默认折叠。
仿 ThinkingBlock 的 toggle/on_click 折叠范式。
"""

from __future__ import annotations

import difflib
import json

from pygments.lexers import get_lexer_for_filename  # type: ignore[import-untyped]
from pygments.util import ClassNotFound  # type: ignore[import-untyped]
from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.timer import Timer
from textual.widgets import Static

from birdcode.ui.icons import Icons

_DIFF_LINE_LIMIT = 200  # diff 高亮行数上限,避免大文件 Write 全量 diff 卡顿
# 工具输出默认收起;点击展开时正文最多显示这么多行(免得撑满整个终端、显杂乱)
_MAX_EXPAND_LINES = 15


def _lexer_for_path(path: str) -> str | None:
    """按文件名扩展名推断 pygments lexer 别名(如 'python'/'javascript');无法识别返回 None。"""
    try:
        return str(get_lexer_for_filename(path).aliases[0])
    except ClassNotFound:
        return None
    except Exception:  # 奇怪扩展名/路径可能抛其他异常,兜底为不高亮
        return None


_TOOL_ARG_PREVIEW_MAX = 60  # 运行行参数预览上限(字符);超长截断,避免多行 JSON 撑爆行


def _preview_args(args: str, *, max_len: int = _TOOL_ARG_PREVIEW_MAX) -> str:
    """工具调用参数的单行预览:折叠空白(含字面 \\n/\\t 转义)+ 截断。

    原 args 是原始 JSON 串(explore/prompt 等含大段换行文本)→ 直接拼进运行行会撑成多行、
    极难看。此处压成单行并截断,保留简短预览(如 read_file 的路径)。
    """
    if not args:
        return ""
    flat = args.replace("\\n", " ").replace("\\t", " ")
    flat = " ".join(flat.split())
    if len(flat) <= max_len:
        return flat
    return flat[:max_len] + "…"


def format_tool_running(name: str, args: str, frame: str) -> str:
    preview = _preview_args(args)
    return f"{frame} {name}({preview})" if preview else f"{frame} {name}"


class ToolLine(Static):
    """单条工具调用状态 + 可折叠结果区。"""

    # $text 而非 $success:正文(文件内容)与工具状态行同属这个 Static,共用一色。
    # 若用 $success(绿),Read 回来的整篇文件内容会全绿、难读。状态行靠 ⎿ ✓ 图标区分。
    DEFAULT_CSS = "ToolLine { color: $text; }"

    def __init__(self, icons: Icons, *, color: str | None = None) -> None:
        # markup=False:工具输出(文件内容)含 [TextBlock(...)] 这类方括号。Rich 的 escape
        # 只转义 [小写tag] 形式(漏掉大写/含括号的如 [TextBlock(..)]),仍会 MarkupError 崩溃;
        # 在 widget 级直接关闭 markup 解析最稳妥(update 渲染纯文本)。
        super().__init__(markup=False)
        self._icons = icons
        # color 非空时,运行转圈行 + 收尾 head 均用此色(自动压缩橙色提示复用本 widget)。
        self._color: str | None = color
        # label 非空时,运行行显示「{spinner} {label}」(替代默认「{spinner} name(args)」)。
        self._label: str | None = None
        self._name: str = ""
        self._args: str = ""
        self._frame: int = 0
        self._timer: Timer | None = None
        self._truncated: bool = False  # 接收 executor 截断标记
        # 折叠结果区状态
        self._output: str = ""
        self._has_output: bool = False
        self._expanded: bool = False
        # finish 状态(set_output 可能先于/后于 finish 到,需兜底默认值)
        self._last_ok: bool = True
        self._last_summary: str = ""
        # 正文语言(Read 按 file_path 扩展名推断)→ 展开时用 Syntax 语法高亮
        self._language: str | None = None
        # Write/Edit 成功后的 diff 渲染对象(并入 _render_display 的 Group,不 mount 子区)
        self._diff_syntax: Syntax | None = None
        # 缓存正文渲染对象(Syntax/Text):set_output 建一次,toggle 复用,避免每次展开
        # 都重跑 pygments lexer 解析大文件(展开卡顿/点击无反应的根因)
        self._body_render: RenderableType | None = None

    # ---- 现有契约(不变)----

    def start(self, name: str = "", args: str = "", *, label: str | None = None) -> None:
        self._name, self._args = name, args
        self._label = label
        self._frame = 0
        self._render_running()
        self._timer = self.set_interval(0.12, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._icons.spinner)
        self._render_running()

    def _running_renderable(self) -> RenderableType:
        frame = self._icons.spinner[self._frame]
        body = (
            f"{frame} {self._label}"
            if self._label is not None
            else format_tool_running(self._name, self._args, frame)
        )
        return Text(body, style=self._color) if self._color else body

    def _render_running(self) -> None:
        self.update(self._running_renderable())

    def finish(self, ok: bool, summary: str, *, truncated: bool = False) -> None:
        self._truncated = truncated
        self._last_ok = ok
        self._last_summary = summary
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        # finish 先渲染结果行;若已有 output(理论上 set_output 在 finish 后调,
        # 这里防御先到),_refresh 会带上折叠提示
        self._refresh()

    # ---- 新增:可折叠结果区(stage3)----

    def set_output(self, text: str, truncated: bool = False) -> None:
        """挂可折叠结果区。默认收起;点击展开看正文(最多 _MAX_EXPAND_LINES 行)。

        正文直接渲染在本 Static 内(与状态行同 widget),靠 DEFAULT_CSS 的 $text
        正常色显示——不另起子区(Static 非容器,mount 子区在大块正文下渲染不稳)。
        """
        self._output = text
        self._truncated = truncated
        self._has_output = True
        # 默认收起:免得工具输出撑满终端;点击展开看正文(上限 _MAX_EXPAND_LINES 行)
        self._expanded = False
        self._language = self._guess_language()
        self._body_render = self._build_body_render()
        self._refresh()

    def _build_body_render(self) -> RenderableType:
        """构造正文渲染对象并缓存(Syntax 高亮 / 纯 Text)。展开最多 _MAX_EXPAND_LINES 行,
        set_output 建一次、toggle/_refresh 复用——避免每次展开重跑 pygments lexer(卡顿源)。"""
        lines = self._output.splitlines()
        body_text = "\n".join(lines[:_MAX_EXPAND_LINES])
        if len(lines) > _MAX_EXPAND_LINES:
            body_text += f"\n... [仅前 {_MAX_EXPAND_LINES} 行,共 {len(lines)} 行]"
        if self._language:
            return Syntax(body_text, self._language, theme="ansi_dark", word_wrap=True)
        return Text(body_text)

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._refresh()

    def on_click(self, event: events.Click) -> None:
        """点击 head 行(第 0 行)展开/收起;展开后点正文(y≥1)不收起——便于选中拷贝。"""
        if not self._has_output:
            return
        if self._expanded and event.y >= 1:
            return  # 点正文(文件内容):不 toggle,留给选中
        event.stop()
        self.toggle()

    def _render_head(self) -> str:
        """工具状态行(含工具名):⎿ <name> ✓ <summary>。"""
        mark = self._icons.success if self._last_ok else self._icons.fail
        name_part = f"{self._name} " if self._name else ""
        return f"  {self._icons.result} {name_part}{mark} {self._last_summary}"

    def _render_text(self) -> str:
        """纯文本视图(head + 展开时的正文,不高亮)。

        供测试断言;也作无语言/折叠时的渲染源(update(str) 在裸测试无 app 时也安全)。
        """
        head = self._render_head()
        if not self._has_output:
            return head
        if self._expanded:
            body = self._output.rstrip()
            tag = " [已截断]" if self._truncated else ""
            return f"{head}\n{body}{tag}"
        hint = "点击展开" + ("(已截断)" if self._truncated else "")
        return f"{head} · {hint}"

    def _guess_language(self) -> str | None:
        """从工具名+参数推断正文语言(目前仅 Read:按 file_path 扩展名)。

        语言信息已在 ToolCallStart 时经 start(name, args) 存入 _name/_args —— 无需改
        executor/provider 事件链。其他工具返回 None(纯文本显示)。
        """
        if self._name != "read_file" or not self._args:
            return None
        try:
            parsed = json.loads(self._args)
        except json.JSONDecodeError:
            return None
        file_path = parsed.get("file_path") if isinstance(parsed, dict) else None
        return _lexer_for_path(str(file_path)) if file_path else None

    def _render_display(self) -> RenderableType:
        """实际渲染:Group(head 绿 + 正文)。

        head 用 green 体现「工具调用」;正文优先级:diff(Write/Edit)> 按语言 Syntax
        高亮(Read)> 纯 Text(Grep/Bash…)。Group 在裸测试下 update 不抛(单个 Text 才
        抛 NoActiveApp),故 set_output/set_diff→_refresh 无 app 也安全。关键是 diff 也
        并入此 Group,不再 self.mount 子区——Static 非容器,mount 子区曾致 Write 工具行
        「一闪而过」。
        """
        # 成功绿 / 失败红:失败时绿底 ✗ 会让人误判成成功;color 覆盖(压缩橙色提示)优先。
        head_style = self._color or ("red" if not self._last_ok else "green")
        head = Text(self._render_head(), style=head_style)
        if not (self._has_output or self._diff_syntax is not None):
            return Group(head)
        if not self._expanded:
            hint = "点击展开" + ("(已截断)" if self._truncated else "")
            return Group(head, Text(f" · {hint}", style="dim"))
        if self._diff_syntax is not None:
            body: RenderableType = self._diff_syntax
        elif self._body_render is not None:
            body = self._body_render  # 缓存:set_output 建一次,toggle 复用不重跑 lexer
        else:
            body = Text(self._output.rstrip())
        parts: list[RenderableType] = [head, body]
        if self._truncated:
            parts.append(Text(" [已截断]", style="dim"))
        return Group(*parts)

    def _refresh(self) -> None:
        # widget 级 markup=False(见 __init__)。str 内容不解析 markup;renderable(Syntax/Group)
        # 自带样式,高亮不受 markup=False 影响。
        self.update(self._render_display())

    # ---- stage4: Edit diff 高亮子区(独立于 set_output 折叠子区)----

    @staticmethod
    def _build_diff_syntax(old: str, new: str, label: str) -> Syntax:
        """生成 Rich Syntax(diff lexer):+ 绿、- 红、@@ 蓝。"""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"{label} (旧)",
                tofile=f"{label} (新)",
                lineterm="",
            )
        )
        # lineterm="" 让 unified_diff 用各 splitlines 自带的换行;header 行无尾换行需补
        if not diff.endswith("\n"):
            diff += "\n"
        # 限制 diff 行数:大文件(如全量 Write)的整篇 diff 会让 Rich Syntax 高亮卡顿
        diff_lines = diff.splitlines(keepends=True)
        if len(diff_lines) > _DIFF_LINE_LIMIT:
            diff = (
                "".join(diff_lines[:_DIFF_LINE_LIMIT])
                + f"... [diff 已截断,共 {len(diff_lines)} 行,仅显示前 {_DIFF_LINE_LIMIT}] ...\n"
            )
        return Syntax(diff, "diff", theme="ansi_dark", word_wrap=True)

    def set_diff(self, old: str, new: str) -> None:
        """生成 diff 并并入本 Static 的渲染 Group(不再 mount 子区)。

        Static 非布局容器,往它 self.mount 子区在大内容下渲染不可靠——曾导致 Write/Edit
        成功后工具行「一闪而过」。改为持有 _diff_syntax,由 _render_display 统一进 Group。
        """
        label = self._name or "file"
        self._diff_syntax = self._build_diff_syntax(old, new, label)
        self._refresh()
