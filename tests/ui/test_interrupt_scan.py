from types import SimpleNamespace

from birdcode.ui.app import BirdApp


async def test_interrupt_scan_calls_refresh() -> None:
    fake = SimpleNamespace()
    called: list[bool] = []

    async def _refresh(*, force_show: bool = False) -> None:
        called.append(force_show)

    # 直接挂到 fake 实例上(_on_interrupt_scan 经 self._refresh_resume_prompt 查找):
    # fake 是 SimpleNamespace,monkeypatch BirdApp 类属性不会被 SimpleNamespace 查到。
    # (同 test_refresh_resume_prompt.py 的 fake._mount_resume_prompt 挂载模式。)
    fake._refresh_resume_prompt = _refresh
    await BirdApp._on_interrupt_scan(fake)  # type: ignore[arg-type]
    assert called == [False]  # ESC 后非 force,走新增差集
