# tests/session/test_e2e_resume.py
"""端到端 resume 集成测试:串起 session→executor→agent_loop→controller 全链。

会话1 跑一轮(MockProvider echo 工具往返)→ 退出 → 会话2 resume → MockProvider
接着聊(看到旧 history)→ 验证 jsonl 链连续性(parentUuid 不断裂)。
"""

import json

import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.session import paths
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore
from birdcode.tools.executor import ToolExecutor, default_registry


async def _noop_event(_ev) -> None:
    pass


async def _noop_status() -> None:
    pass


@pytest.mark.asyncio
async def test_resume_then_continue_with_mock_provider(tmp_path):
    """e2e:会话1 跑一轮 → 退出 → 会话2 resume → MockProvider 接着聊 + 链连续。"""
    # 局部 import 规避 ui↔conversation 循环依赖链(同其他测试文件风格)
    from birdcode.conversation import TurnController

    ctx = SessionContext(session_id="e2e", cwd=str(tmp_path), version="0.1.0", git_branch=None)

    # —— 会话1:跑一轮 "hello world"(MockProvider 会 echo 工具往返)——
    store1 = SessionStore(ctx, tmp_path, root=tmp_path)
    ctrl1 = TurnController(
        MockProvider(delay=0.0),
        on_event=_noop_event,
        on_status=_noop_status,
        executor=ToolExecutor(default_registry()),
        store=store1,
    )
    await ctrl1.submit("hello world")
    assert len(ctrl1.history) == 1
    # user + assistant(tool_use) + user(tool_result) + assistant(text)
    assert len(ctrl1.history[0].messages) == 4
    store1.close()

    # —— 会话2:resume ——
    store2 = SessionStore(ctx, tmp_path, root=tmp_path)
    ctrl2 = TurnController(
        MockProvider(delay=0.0),
        on_event=_noop_event,
        on_status=_noop_status,
        executor=ToolExecutor(default_registry()),
        store=store2,
    )
    turns = await ctrl2.resume()
    assert len(turns) == 1
    # resume 后 history 还原
    assert ctrl2.history[0].messages[0].content[0].text == "hello world"

    # 接着聊一轮
    await ctrl2.submit("again")
    assert len(ctrl2.history) == 2  # 旧的 1 + 新的 1
    store2.close()

    # —— 验证 jsonl:会话1(4 行)+ 会话2 新轮(4 行)= 8 行,链连续 ——
    jf = tmp_path / paths.encode_cwd(tmp_path) / "e2e.jsonl"
    raw_lines = jf.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines if line.strip()]
    assert len(rows) == 8
    # 链连续:每行 parentUuid 指向上一行 uuid(行0 parent 为 null)
    assert rows[0]["parentUuid"] is None
    for i in range(1, len(rows)):
        assert rows[i]["parentUuid"] == rows[i - 1]["uuid"], f"行 {i} 链断裂"


@pytest.mark.asyncio
async def test_persisted_tool_output_survives_resume(tmp_path):
    """e2e 双轨外存:超阈工具输出落盘 → resume 后给 LLM 的仍是占位(完整在落盘文件)。

    会话1 用大输出工具触发落盘 → 退出 → 会话2 resume → 该 tool_result 行 content
    仍是 <persisted-output> 占位(不是完整原文),完整在 tool-results/<id>.txt。
    """
    from pydantic import BaseModel

    from birdcode.blocks import TextBlock, ToolResultBlock
    from birdcode.conversation import Message
    from birdcode.tools import Tool, ToolRegistry

    class _BigInput(BaseModel):
        text: str

    class _BigTool(Tool):
        name = "big"
        description = "返回超长文本,触发双轨外存落盘"
        parameters = _BigInput
        kind = "read"
        parallel_safe = True
        max_result_chars = 1000  # 低阈值,使 32K+ 输出触发落盘(默认 100K 下不会)

        async def execute(self, **args):  # type: ignore[override]
            return "x" * (32 * 1024 + 100)  # 大输出,超 _BigTool.max_result_chars(1000)

    reg = ToolRegistry()
    reg.register(_BigTool())

    ctx = SessionContext(session_id="persist", cwd=str(tmp_path), version="0.1.0", git_branch=None)
    store1 = SessionStore(ctx, tmp_path, root=tmp_path)

    # 直接用 executor 跑一次大输出工具,把 tool_result Message 手动 append(绕过 provider)
    ex = ToolExecutor(reg, output_sink=store1.as_output_sink())
    from birdcode.blocks import ToolUseBlock

    results = await ex.execute_batch([ToolUseBlock(id="big1", name="big", input={"text": "y"})])
    r = results[0]
    assert r.persisted_path is not None  # 落盘了
    tool_msg = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="big1", content=r.llm_content, is_error=False)],
    )
    await store1.append(Message(role="user", content=[TextBlock(text="跑大工具")]))
    await store1.append(tool_msg)
    store1.close()

    # 落盘文件存在
    persisted = paths.tool_result_path(tmp_path, "persist", tmp_path, "big1")
    assert persisted.exists() and len(persisted.read_text(encoding="utf-8")) > 32 * 1024

    # resume:load 主线,验证 tool_result content 仍是占位
    store2 = SessionStore(ctx, tmp_path, root=tmp_path)
    turns = await store2.load_mainline()
    store2.close()
    tool_turn_msg = turns[0].messages[1]  # user(tool_result)
    tr = next(b for b in tool_turn_msg.content if isinstance(b, ToolResultBlock))
    assert "<persisted-output>" in tr.content  # 占位往返无损
    # 完整原文(32KB+)不在 jsonl 行里——占位只含前 2048 字符 preview + 模板,远小于完整原文
    assert len(tr.content) < 32 * 1024
