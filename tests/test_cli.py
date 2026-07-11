# tests/test_cli.py
import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.ui.app import BirdApp
from birdcode.ui.widgets.input_area import InputArea


@pytest.mark.asyncio
async def test_transcript_printable_after_turn():
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        await app._on_submitted(InputArea.Submitted("hello world"))
        await pilot.pause(delay=2.0)
    tx = app.transcript()
    assert "hello world" in tx
    assert "─" not in tx and "⌛" not in tx  # 不含框线/状态行
