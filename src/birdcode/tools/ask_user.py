# src/birdcode/tools/ask_user.py
"""ask_user 工具:让模型向用户提问、给方案让其选择/自定义输入。

模型遇到需用户拍板的分叉时调用:question + options(每个 label+description) →
ChoicePrompt 弹底部行内菜单,用户选预设 / 输入自定义 / Esc 取消 → 结果回 tool_result。
异步子 agent 无法交互:build_child_registry 把本工具换成 AskUserAsyncStub(execute 恒拒)。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from birdcode.tools.base import Tool


class AskOption(BaseModel):
    label: str = Field(..., description="选项简称,用户选中后作为结果回传给模型")
    description: str = Field(..., description="选项的详细说明,展示给用户辅助决策")


class AskUserInput(BaseModel):
    question: str = Field(..., description="给用户的问题或提示语")
    options: list[AskOption] = Field(..., description="供用户选择的方案列表(建议 2 到若干个)")


class AskUserTool(Tool):
    """主 agent / 同步子 agent 用:execute 委托 app.ask_user 弹 ChoicePrompt 等用户响应。

    options 经 executor model_dump 后为 list[dict](label/description),原样透传 app。
    """

    name = "ask_user"
    description = (
        "向用户提问并给出若干方案让其选择,或让用户输入自定义意见。"
        "当需要用户在多个方案间拍板、或需用户补充信息时调用。"
    )
    parameters = AskUserInput
    kind = "read"  # 只问不改,免 gate HITL(问问题本身不该再被权限门拦)
    parallel_safe = False  # 阻塞等用户 + 独占 UI,必须串行
    # build_child_registry 据此在异步子 agent 把本工具换成 AskUserAsyncStub(标记属性,
    # 非 isinstance:与 is_agent_tool 同理,规避循环 import)。
    is_ask_user = True

    def __init__(self, app: object) -> None:
        self._app = app

    async def execute(self, *, question: str, options: list) -> str:  # type: ignore[override]
        # executor model_dump 后 options 是 list[dict](label/description);透传给 app。
        return await self._app.ask_user(question, options)


class AskUserAsyncStub(Tool):
    """异步子 agent 的 ask_user 替身:execute 恒返回拒文案。

    异步子 agent 后台跑、无法与用户交互,故 ask_user 在其工具表里被换成本 stub
    (build_child_registry is_async 分支)。模型仍看到 ask_user 能力、调用即得拒,
    据此自行决策(不阻塞、不卡死)。
    """

    name = "ask_user"
    description = AskUserTool.description
    parameters = AskUserInput
    kind = "read"
    parallel_safe = False

    async def execute(self, *, question: str, options: list) -> str:  # type: ignore[override]
        return "异步环境无法提问,请自行决策。"
