# src/birdcode/tools/echo.py
"""Stub 工具:原样回显输入文本。用于阶段 1 端到端验证工具循环。"""

from __future__ import annotations

from pydantic import BaseModel

from birdcode.tools.base import Tool


class _EchoInput(BaseModel):
    text: str


class EchoTool(Tool):
    name = "echo"
    hidden = True  # stage1 stub:不暴露给 LLM,仅 mock_provider 端到端/测试可执行
    description = (
        "回显输入文本(阶段 1 stub)。何时用:验证工具调用链路。"
        "参数:text(要回显的文本)。输出:原样返回该文本。"
    )
    parameters = _EchoInput
    kind = "read"
    parallel_safe = True

    async def execute(self, *, text: str) -> str:  # type: ignore[override]
        return text
