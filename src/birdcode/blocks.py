# src/birdcode/blocks.py
"""Content block types: the unified content model for Message.

一条 Message 的 content 是块列表,可装 text / thinking(含 signature)/ tool_use /
tool_result。这是抹平 Anthropic blocks 与 OpenAI 字段风格的统一中间表示。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    """Anthropic extended thinking。signature 专为多轮连续性——工具调用时必须
    原样(含 signature、置于 tool_use 之前)回传,否则 API 报 400。"""

    text: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    """assistant 发起的工具调用。input 是已解析的参数 dict。"""

    id: str
    name: str
    input: dict[str, object]


@dataclass
class ToolResultBlock:
    """回传给模型的工具结果。作为 user 消息里的一个块(Anthropic 模型),
    而非独立 role。is_error=True 时模型据其自我纠正。"""

    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock
