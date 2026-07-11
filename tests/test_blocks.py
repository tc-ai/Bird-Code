from birdcode.blocks import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_textblock_holds_text():
    b = TextBlock(text="hi")
    assert b.text == "hi"


def test_thinkingblock_has_optional_signature():
    b = ThinkingBlock(text="hm", signature="sig")
    assert b.signature == "sig"
    assert ThinkingBlock(text="x").signature == ""


def test_tooluseblock_holds_input_dict():
    b = ToolUseBlock(id="t1", name="echo", input={"text": "abc"})
    assert b.input == {"text": "abc"}


def test_toolresultblock_defaults():
    b = ToolResultBlock(tool_use_id="t1", content="ok")
    assert b.is_error is False
    assert b.tool_use_id == "t1"


def test_all_blocks_are_content_blocks():
    # ContentBlock 是覆盖全部 4 种块类型的联合;确保无遗漏、可直接作 list 元素类型。
    blocks: list[ContentBlock] = [
        TextBlock(text="hi"),
        ThinkingBlock(text="hm", signature="sig"),
        ToolUseBlock(id="t1", name="echo", input={}),
        ToolResultBlock(tool_use_id="t1", content="ok"),
    ]
    assert len(blocks) == 4
