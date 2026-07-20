# tests/test_replay_order.py
"""_replay_history 渲染顺序:同一 ConversationTurn 内「先工具调用 → 工具返回 → 模型再写正文」
时,正文须落在工具行【下方】,对齐 live 路径(每轮 Done → _close_markdown,下轮正文经
_ensure_markdown 重建到滚动区末尾的新 Turn)。

regression:_replay_history 曾在一个 ConversationTurn 内不关 md,所有 assistant TextBlock
都写进 Turn 起始(挂在工具行之前)的 md_A → 工具行渲染到回复正文之后(用户报「工具调用成功
渲染在回复后面」)。Turn 边界见 codec.decode_lines:含 TextBlock 的 user 行开新轮,tool_result
回填的 user 行归入当前轮,故 user→asst(tool_use)→user(result)→asst(text) 同属一轮。
"""

from __future__ import annotations

import pytest
from textual.widgets import Markdown

from birdcode.agent.mock_provider import MockProvider
from birdcode.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from birdcode.conversation import Message
from birdcode.conversation import Turn as ConversationTurn
from birdcode.ui.app import BirdApp
from birdcode.ui.widgets.tool_line import ToolLine


@pytest.mark.asyncio
async def test_replay_assistant_text_after_tool_lands_below_tool_line():
    """user→asst(tool_use)→user(result,成功)→asst(text):回复正文必须在工具行下方。"""
    turns = [
        ConversationTurn(
            messages=[
                Message(role="user", content=[TextBlock(text="继续")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="t1",
                            name="resume_agent",
                            input={"agent_id": "sub-x", "direction": "go"},
                            agent_id="sub-x",
                        )
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="t1",
                            content="所有信息已收集完毕。\n\n## 总结\n正文详情…",
                            is_error=False,
                        )
                    ],
                ),
                Message(role="assistant", content=[TextBlock(text="好的，探索完毕，如下。")]),
            ]
        )
    ]
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app._replay_history(turns)
        await pilot.pause()
        scroll = app.query_one("#scroll")
        # 按 DOM 顺序收集 Markdown 与 ToolLine(query 迭代为文档序)。
        nodes = list(scroll.query("Markdown, ToolLine"))
        tool_idx = next(i for i, w in enumerate(nodes) if isinstance(w, ToolLine))
        md_idxs = [i for i, w in enumerate(nodes) if isinstance(w, Markdown)]
        assert md_idxs, "应至少有一个 assistant 正文 md"
        # 末尾 md = 回复正文(fix 后:工具行轮的 md_A 空 + 关闭 → 回复另建 md_B 挂末尾)。
        # 工具行必须在最后一处正文 md 之前,否则即「工具行渲染到回复之后」order bug。
        assert tool_idx < md_idxs[-1], (
            f"工具行(idx={tool_idx})应在回复正文 md(idx={md_idxs[-1]})之前"
        )


@pytest.mark.asyncio
async def test_replay_text_then_tool_keeps_text_before_tool():
    """单 assistant 消息内 [TextBlock, ToolUseBlock]:正文在前、工具行在后(顺序不变)。"""
    turns = [
        ConversationTurn(
            messages=[
                Message(role="user", content=[TextBlock(text="查一下")]),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="我来搜一下。"),
                        ToolUseBlock(id="t2", name="grep", input={"pattern": "foo"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="t2", content="3 处命中", is_error=False)],
                ),
            ]
        )
    ]
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app._replay_history(turns)
        await pilot.pause()
        scroll = app.query_one("#scroll")
        nodes = list(scroll.query("Markdown, ToolLine"))
        tool_idx = next(i for i, w in enumerate(nodes) if isinstance(w, ToolLine))
        md_idx = next(i for i, w in enumerate(nodes) if isinstance(w, Markdown))
        assert md_idx < tool_idx, "同消息内正文应在前、工具行在后"
