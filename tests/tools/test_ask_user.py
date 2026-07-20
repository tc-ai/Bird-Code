# tests/tools/test_ask_user.py
"""ask_user 工具:execute 委托 app.ask_user、异步 stub 返回拒文案、Input 校验。"""

import pytest

from birdcode.tools.ask_user import AskOption, AskUserAsyncStub, AskUserInput, AskUserTool


def test_ask_user_input_validates():
    inp = AskUserInput(
        question="选哪个?",
        options=[AskOption(label="A", description="a"), AskOption(label="B", description="b")],
    )
    assert inp.question == "选哪个?"
    assert [o.label for o in inp.options] == ["A", "B"]


@pytest.mark.asyncio
async def test_ask_user_tool_delegates_to_app():
    """execute 委托 app.ask_user,原样透传 question/options(model_dump 后 list[dict])。"""
    received: dict = {}

    class _App:
        async def ask_user(self, question: str, options: list) -> str:
            received["q"] = question
            received["opts"] = options
            return "A"

    tool = AskUserTool(app=_App())
    out = await tool.execute(question="选哪个?", options=[{"label": "A", "description": "a"}])
    assert out == "A"
    assert received == {"q": "选哪个?", "opts": [{"label": "A", "description": "a"}]}


@pytest.mark.asyncio
async def test_ask_user_async_stub_returns_reject():
    """异步 stub execute 恒返回拒文案(异步子 agent 用)。"""
    out = await AskUserAsyncStub().execute(
        question="q", options=[{"label": "A", "description": "a"}]
    )
    assert "异步" in out


def test_ask_user_tool_attrs():
    tool = AskUserTool(app=None)
    assert tool.name == "ask_user"
    assert tool.kind == "read"
    assert tool.parallel_safe is False
    assert tool.is_ask_user is True


def test_ask_user_async_stub_no_marker():
    """stub 不带 is_ask_user 标记(避免 build_child_registry 递归替换)。"""
    stub = AskUserAsyncStub()
    assert stub.name == "ask_user"  # 同名(替换原 tool)
    assert getattr(stub, "is_ask_user", False) is False
