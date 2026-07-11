import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.ui.app import BirdApp
from birdcode.ui.commands.command import Command, CommandType
from birdcode.ui.widgets.completion_menu import CompletionMenu
from birdcode.ui.widgets.input_area import InputArea, command_prefix


async def _noop(_ctx, _args):
    return None


def _make_cands(n: int) -> list:
    """n 个测试用 Command:/c00../c{n-1}(零填充 2 位 → 字符串序即数值序)。"""
    return [
        Command(
            name=f"/c{i:02d}",
            description=f"d{i}",
            usage=f"/c{i:02d}",
            type=CommandType.LOCAL,
            handler=_noop,
        )
        for i in range(n)
    ]


def test_command_prefix_basic():
    assert command_prefix("/he") == "/he"
    assert command_prefix("/profile gpt") == "/profile"  # 空格后是参数


def test_command_prefix_non_slash_empty():
    assert command_prefix("hello") == ""
    assert command_prefix("") == ""


def test_completion_menu_move_and_select():
    cands = [
        Command(
            name="/server", description="d1", usage="/server", type=CommandType.LOCAL, handler=_noop
        ),
        Command(
            name="/sessions",
            description="d2",
            usage="/sessions",
            type=CommandType.LOCAL,
            handler=_noop,
        ),
    ]
    m = CompletionMenu(cands)
    assert m.selected_name() == "/server"  # 初始 sel=0
    m.move(1)
    assert m.selected_name() == "/sessions"
    m.move(1)  # 回绕到头
    assert m.selected_name() == "/server"
    m.move(-1)  # 反向回绕
    assert m.selected_name() == "/sessions"


@pytest.mark.asyncio
async def test_completion_menu_mounts_and_esc_dismisses():
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.ui.app import BirdApp
    from birdcode.ui.widgets.input_area import InputArea

    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        reg = app._command_registry
        cands = reg.complete("/s")  # /server, /sessions (>=2)
        assert len(cands) >= 2
        await app._on_completion_requested(InputArea.CompletionRequested(cands))
        await pilot.pause()
        assert app._completion_menu is not None
        # 导航
        app._completion_menu.move(1)
        assert app._completion_menu.selected_name() in {"/server", "/sessions"}
        # action_interrupt(Esc) 浮层开时应关浮层、不中断
        app.action_interrupt()
        assert app._completion_menu is None


@pytest.mark.asyncio
async def test_completion_dismiss_via_method():
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.ui.app import BirdApp
    from birdcode.ui.widgets.input_area import InputArea

    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        reg = app._command_registry
        cands = reg.complete("/s")
        await app._on_completion_requested(InputArea.CompletionRequested(cands))
        await pilot.pause()
        assert app._completion_menu is not None
        app._dismiss_completion()
        assert app._completion_menu is None
        # 再次 dismiss 是 no-op,不抛
        app._dismiss_completion()


@pytest.mark.asyncio
async def test_tab_completion_preserves_args():
    """Tab 补全命令名时保留已输参数:/he arg1 → /help arg1(不整体替换丢参数)。"""
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.ui.app import BirdApp
    from birdcode.ui.widgets.input_area import InputArea

    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/he arg1"
        await pilot.press("tab")
        assert ia.text == "/help arg1"


@pytest.mark.asyncio
async def test_tab_completion_case_insensitive():
    """Tab 补全大小写不敏感(与 Enter 执行一致):/HE → /help。"""
    from birdcode.agent.mock_provider import MockProvider
    from birdcode.ui.app import BirdApp
    from birdcode.ui.widgets.input_area import InputArea

    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/HE"
        await pilot.press("tab")
        assert ia.text == "/help "


# ---- F3 滑动窗口:候选多于窗口时选中项恒可见 ----


def test_completion_window_shows_all_when_few():
    """候选数 ≤ 窗口 → 全显示、无 …。"""
    m = CompletionMenu(_make_cands(5), window=8)
    r = m.render()
    assert "/c00  " in r and "/c04  " in r
    assert "…" not in r


def test_completion_window_truncates_without_ellipsis():
    """候选 > 窗口且 sel 在头部 → 只画首窗口、无 …(超出项靠 ↑↓ 滑动)。"""
    m = CompletionMenu(_make_cands(12), window=5)
    r = m.render()  # sel=0 → 窗口 [0:5)
    assert "/c00  " in r and "/c04  " in r
    assert "/c05  " not in r
    assert "…" not in r


def test_completion_window_slides_to_keep_selection_visible():
    """sel 移到中段 → 窗口滑动把它包进可视区,顶部项滑出。"""
    m = CompletionMenu(_make_cands(12), window=5)
    for _ in range(6):
        m.move(1)
    assert m.selected_name() == "/c06"
    r = m.render()
    assert "▶ /c06  " in r  # 选中项带标记、必可见
    assert "/c00  " not in r  # 已滑出顶部
    assert "…" not in r


def test_completion_window_at_last_item():
    """sel 在末尾 → 末窗口贴底。"""
    m = CompletionMenu(_make_cands(12), window=5)
    for _ in range(11):
        m.move(1)
    assert m.selected_name() == "/c11"
    r = m.render()
    assert "▶ /c11  " in r
    assert "/c07  " in r and "/c00  " not in r  # 末窗口 [7:12]
    assert "…" not in r


def test_completion_selection_always_visible_invariant():
    """不变式:任意 move(含回绕)后,选中项必带 ▶ 出现在 render() 输出里。"""
    m = CompletionMenu(_make_cands(30), window=8)
    for _ in range(60):
        m.move(1)
        assert f"▶ {m.selected_name()}  " in m.render()
    for _ in range(60):
        m.move(-1)
        assert f"▶ {m.selected_name()}  " in m.render()


# ---- 自动弹浮层:输入 "/" 实时过滤、空格/backspace 收起、Enter 执行、Tab 填名 ----


@pytest.mark.asyncio
async def test_slash_auto_shows_menu_in_composer():
    """输入 "/" 自动弹浮层(无需 Tab),挂在 #composer(StatusBar 下方)。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        assert app._completion_menu is None  # noqa: SLF001
        ia.text = "/"
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        # 挂在 composer 里(StatusBar 下方)
        assert app._completion_menu.parent is app.query_one("#composer")  # noqa: SLF001
        # 显示全部非 hidden 命令,且按 name 升序
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert len(names) >= 10
        assert names == sorted(names)


@pytest.mark.asyncio
async def test_typing_filters_menu_live():
    """/c → 只剩 /c 开头;/cl → 进一步收窄(复用同一浮层,不重挂)。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/c"
        await pilot.pause()
        menu = app._completion_menu  # noqa: SLF001
        assert menu is not None
        assert all(c.name.startswith("/c") for c in menu.candidates)
        assert any(c.name == "/clear" for c in menu.candidates)
        first_menu = menu
        ia.text = "/cl"
        await pilot.pause()
        assert app._completion_menu is first_menu  # noqa: SLF001 - 复用同一浮层
        assert all(c.name.startswith("/cl") for c in app._completion_menu.candidates)  # noqa: SLF001


@pytest.mark.asyncio
async def test_space_dismisses_menu():
    """键入空格(开始输参数)→ 浮层关闭。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/clear"
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        ia.text = "/clear "  # 空格 → 命令名段结束
        await pilot.pause()
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_backspace_removing_slash_dismisses_menu():
    """删掉 "/" → 浮层关闭、输入框沉底。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/c"
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        ia.text = ""  # 模拟退格删到空
        await pilot.pause()
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_enter_executes_selected_command():
    """浮层开时 Enter → 执行选中命令(直接提交其名),不留输入框。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/he"  # 浮层过滤到 /help
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        await pilot.press("enter")
        await pilot.pause()
        # /help 执行 → #scroll 挂上帮助文本
        texts = [str(s.content) for s in app.query("#scroll Static")]
        assert any("可用命令" in t for t in texts), texts
        # 输入框已清空、浮层已关
        assert ia.text == ""
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_tab_fills_selected_name_then_closed():
    """浮层开时 Tab → 填选中命令名 + 空格(不提交),浮层关闭、可继续输参数。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "/cl"
        await pilot.pause()
        await pilot.press("tab")
        assert ia.text == "/clear "
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_menu_open_raises_composer_input_slides_up():
    """浮层开 → composer 长高、输入框上移让位(下方放命令列表)。"""
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        composer = app.query_one("#composer")
        await pilot.pause()
        h_before = composer.size.height
        y_before = ia.region.y
        ia.text = "/"
        await pilot.pause(delay=0.05)
        assert app._completion_menu is not None  # noqa: SLF001
        assert composer.size.height > h_before  # composer 长高
        assert ia.region.y < y_before  # 输入框上移


# ---- 候选按 name 升序(即便传入乱序) ----


def test_completion_menu_sorts_by_name():
    cands = [
        Command(
            name="/zebra", description="z", usage="/zebra", type=CommandType.LOCAL, handler=_noop
        ),
        Command(
            name="/alpha", description="a", usage="/alpha", type=CommandType.LOCAL, handler=_noop
        ),
        Command(name="/mid", description="m", usage="/mid", type=CommandType.LOCAL, handler=_noop),
    ]
    m = CompletionMenu(cands)
    assert [c.name for c in m.candidates] == ["/alpha", "/mid", "/zebra"]


def test_completion_update_sorts_and_resets_sel():
    m = CompletionMenu(_make_cands(3), window=8)  # /c0,/c1,/c2
    m.move(1)  # sel=1
    m.update(
        [
            Command(name="/z", description="z", usage="/z", type=CommandType.LOCAL, handler=_noop),
            Command(name="/a", description="a", usage="/a", type=CommandType.LOCAL, handler=_noop),
        ]
    )
    assert [c.name for c in m.candidates] == ["/a", "/z"]
    assert m.selected_name() == "/a"  # sel 重置到 0(排序后首项)
