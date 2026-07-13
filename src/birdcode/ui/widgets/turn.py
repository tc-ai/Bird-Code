# src/birdcode/ui/widgets/turn.py
"""单轮容器：用户问题 + 助手 Markdown + 工具行。

便于 /clear 批量移除（query('.turn').remove()）与未来按轮折叠/编辑。
用 classes（非 id）标识子元素，避免多轮时 id 冲突。
"""

from __future__ import annotations

from textual.containers import Container
from textual.widgets import Markdown, Static

from birdcode.ui.icons import Icons
from birdcode.ui.widgets.tool_line import ToolLine


class Turn(Container):
    DEFAULT_CSS = """
    Turn { padding: 0; height: auto; }
    /* 用户消息用紫色(Tokyo Night #bb9af7):原 $primary=#7aa2f7 蓝在黑底上与背景重合看不清。 */
    /* light 主题在 app.py 用 .light Turn .user 覆盖回蓝(#2e5cd6,白底上高对比)。 */
    Turn .user { color: #bb9af7; }
    """

    def __init__(self) -> None:
        super().__init__(classes="turn")

    async def add_user(self, text: str) -> None:
        # 空文本(wake 轮 TurnStart(user_text=''))不挂空 .user Static——否则渲染出空用户
        # 气泡,与「UI 不应把它当用户气泡」的意图相悖。
        if not text:
            return
        await self.mount(Static(text, classes="user"))

    async def add_assistant_markdown(self) -> Markdown:
        md = Markdown("")
        await self.mount(md)
        return md

    async def add_tool(self, icons: Icons, *, color: str | None = None) -> ToolLine:
        tool = ToolLine(icons, color=color)
        await self.mount(tool)
        return tool
