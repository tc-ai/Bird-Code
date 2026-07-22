from pathlib import Path

import pytest

from birdcode.agent.mock_provider import MockProvider
from birdcode.ui.app import BirdApp
from birdcode.ui.pure import FileCandidate, list_file_candidates, parse_at_token
from birdcode.ui.widgets.file_completion_menu import FileCompletionMenu
from birdcode.ui.widgets.input_area import InputArea


def _fc(name: str, is_dir: bool) -> FileCandidate:
    return FileCandidate(
        name=name + "/" if is_dir else name,
        full_path=name + "/" if is_dir else name,
        is_dir=is_dir,
    )


def test_menu_move_and_select():
    m = FileCompletionMenu([_fc("app.py", False), _fc("ui", True)])
    assert m.selected_candidate().name == "app.py"
    m.move(1)
    assert m.selected_candidate().name == "ui/"
    m.move(1)  # 回绕
    assert m.selected_candidate().name == "app.py"


def test_menu_render_marks_selected():
    m = FileCompletionMenu([_fc("app.py", False), _fc("ui", True)])
    r = m.render()
    assert "▶ app.py" in r
    assert "  ui/" in r


def test_menu_truncated_renders_hint():
    m = FileCompletionMenu([_fc("a", False)], truncated=True)
    assert "继续输入" in m.render()


def test_menu_update_resets_sel():
    m = FileCompletionMenu([_fc("a", False), _fc("b", False), _fc("c", False)])
    m.move(2)
    assert m.selected_candidate().name == "c"
    m.update([_fc("z", False)])
    assert m.selected_candidate().name == "z"
    assert m.candidates[0].name == "z"


def test_menu_empty_render_blank():
    assert FileCompletionMenu([]).render() == ""


def test_menu_window_selection_always_visible():
    # 不变式:任意 move 后选中项带 ▶ 出现在 render
    cands = [_fc(f"f{i:02d}", False) for i in range(30)]
    m = FileCompletionMenu(cands, window=8)
    for _ in range(60):
        m.move(1)
        assert f"▶ {m.selected_candidate().name}" in m.render()
    for _ in range(60):
        m.move(-1)
        assert f"▶ {m.selected_candidate().name}" in m.render()


def test_at_in_middle_of_line():
    # "读 @src" → @ 在列 2，光标在末尾列 6（"读 @src" 共 6 码点，CJK"读"算 1）
    assert parse_at_token("读 @src", 6) is not None
    at = parse_at_token("读 @src", 6)
    assert at.at_col == 2
    assert at.start_col == 3
    assert at.prefix == "src"


def test_at_at_line_start():
    at = parse_at_token("@src", 4)
    assert at.at_col == 0
    assert at.start_col == 1
    assert at.prefix == "src"


def test_at_preceded_by_non_space_is_none():
    # 邮箱 a@b：@ 前是字母 → 不触发
    assert parse_at_token("a@b", 3) is None


def test_space_after_at_closes_token():
    # @ 后含空格 → token 已闭合
    assert parse_at_token("@sr c", 5) is None


def test_multiple_at_picks_nearest():
    # "@a @b"：取最近一个 @（列 3）
    at = parse_at_token("@a @b", 5)
    assert at.at_col == 3
    assert at.prefix == "b"


def test_no_at_returns_none():
    assert parse_at_token("hello world", 11) is None


def test_empty_line_returns_none():
    assert parse_at_token("", 0) is None


def test_cursor_out_of_range_returns_none():
    assert parse_at_token("@src", -1) is None
    assert parse_at_token("@src", 99) is None


def test_just_at_sign_no_chars():
    # 单个 @，光标紧随其后 → prefix 为空（触发：列当前目录全部）
    at = parse_at_token("@", 1)
    assert at is not None
    assert at.prefix == ""
    assert at.start_col == 1


def _make_tree(tmp_path: Path) -> Path:
    """tmp/src 下:目录 ui/、文件 app.py / util.py / .hidden。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "ui").mkdir()
    (src / "app.py").write_text("x")
    (src / "util.py").write_text("y")
    (src / ".hidden").write_text("z")
    return src


def test_list_basic_dirs_first_then_files(tmp_path: Path) -> None:
    src = _make_tree(tmp_path)
    cands, trunc = list_file_candidates(src, "", "src")
    assert trunc is False
    names = [c.name for c in cands]
    assert names == ["ui/", "app.py", "util.py"]  # 目录在前,各自字母序


def test_list_full_path_uses_base_rel(tmp_path: Path) -> None:
    src = _make_tree(tmp_path)
    cands, _ = list_file_candidates(src, "", "src")
    by_name = {c.name: c.full_path for c in cands}
    assert by_name["ui/"] == "src/ui/"
    assert by_name["app.py"] == "src/app.py"


def test_list_prefix_filter_case_insensitive(tmp_path: Path) -> None:
    src = _make_tree(tmp_path)
    cands, _ = list_file_candidates(src, "U", "src")  # U → ui/ + util.py
    names = [c.name for c in cands]
    assert names == ["ui/", "util.py"]


def test_list_hidden_filtered(tmp_path: Path) -> None:
    src = _make_tree(tmp_path)
    cands, _ = list_file_candidates(src, "", "src")
    assert all(not c.name.startswith(".") for c in cands)


def test_list_empty_base_rel(tmp_path: Path) -> None:
    # base_rel="" → full_path 不带前缀
    base = tmp_path
    (base / "foo.py").write_text("x")
    cands, _ = list_file_candidates(base, "foo", "")
    assert cands[0].full_path == "foo.py"


def test_list_truncates_at_100(tmp_path: Path) -> None:
    base = tmp_path / "many"
    base.mkdir()
    for i in range(101):
        (base / f"f{i:03d}.py").write_text("x")
    cands, trunc = list_file_candidates(base, "", "many")
    assert trunc is True
    assert len(cands) == 100


def test_list_nonexistent_dir_returns_empty(tmp_path: Path) -> None:
    cands, trunc = list_file_candidates(tmp_path / "nope", "", "nope")
    assert cands == []
    assert trunc is False


def test_list_is_dir_flag(tmp_path: Path) -> None:
    src = _make_tree(tmp_path)
    cands, _ = list_file_candidates(src, "", "src")
    by_name = {c.name: c.is_dir for c in cands}
    assert by_name["ui/"] is True
    assert by_name["app.py"] is False


# ---- Task 4: InputArea 触发 @ 文件浮层（互斥路由、过滤、dismiss）----


@pytest.mark.asyncio
async def test_at_triggers_file_menu(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    (tmp_path / "pkg").mkdir()
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@"
        # TextArea 赋值 .text 不移动光标(停在 (0,0))→ @ token 要求光标在 @ 之后,
        # 否则 parse_at_token("@", 0)=None;手动把光标挪到行尾以模拟真实键入。
        ia.move_cursor((0, len(ia.text)))
        await pilot.pause()
        assert isinstance(app._completion_menu, FileCompletionMenu)  # noqa: SLF001
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert "app.py" in names
        assert "pkg/" in names


@pytest.mark.asyncio
async def test_at_prefix_filters(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alpha.py").write_text("x")
    (tmp_path / "beta.py").write_text("y")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@al"
        ia.move_cursor((0, len(ia.text)))  # 见上:TextArea 赋值不挪光标
        await pilot.pause()
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert names == ["alpha.py"]


@pytest.mark.asyncio
async def test_at_hidden_filtered(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".secret").write_text("x")
    (tmp_path / "app.py").write_text("y")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@"
        ia.move_cursor((0, len(ia.text)))  # 见上:TextArea 赋值不挪光标
        await pilot.pause()
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert ".secret" not in names
        assert "app.py" in names


@pytest.mark.asyncio
async def test_at_not_triggered_when_preceded_by_nonspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "a@b"
        ia.move_cursor((0, len(ia.text)))  # 见上:TextArea 赋值不挪光标
        await pilot.pause()
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_at_dismissed_when_prefix_gone(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@ap"
        ia.move_cursor((0, len(ia.text)))  # 见上:TextArea 赋值不挪光标
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        ia.text = "no at sign here"
        ia.move_cursor((0, len(ia.text)))  # 见上:TextArea 赋值不挪光标
        await pilot.pause()
        assert app._completion_menu is None  # noqa: SLF001


# ---- Task 5: @ 浮层键位 (Tab 展开目录 / Tab+Enter 插入文件 / ↑↓ 导航 / Esc 关闭) ----


@pytest.mark.asyncio
async def test_at_tab_expands_directory(monkeypatch, tmp_path):
    """Tab 选中目录 → 补全为 @pkg/ 并展开下一级(列出 pkg 内容)。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "inner.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@pk"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert ia.text == "@pkg/"
        # 展开后浮层重开,列出 pkg 内容
        assert isinstance(app._completion_menu, FileCompletionMenu)  # noqa: SLF001
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert "inner.py" in names


@pytest.mark.asyncio
async def test_at_tab_file_inserts_and_closes(monkeypatch, tmp_path):
    """Tab 选中文件 → 补全 @app.py + 空格(token 闭合),浮层关。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@ap"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert ia.text == "@app.py "
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_at_enter_inserts_path(monkeypatch, tmp_path):
    """Enter → 插入高亮项 full_path + 空格,关闭浮层。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@ap"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert ia.text == "@app.py "
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_at_arrow_moves_highlight(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    (tmp_path / "pkg").mkdir()
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        menu = app._completion_menu  # noqa: SLF001
        assert menu is not None
        first = menu.selected_candidate().name
        await pilot.press("down")
        await pilot.pause()
        assert app._completion_menu.selected_candidate().name != first  # noqa: SLF001


@pytest.mark.asyncio
async def test_at_escape_dismisses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@ap"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        assert app._completion_menu is not None  # noqa: SLF001
        app.action_interrupt()  # Esc
        assert app._completion_menu is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_file_menu_mounts_in_composer(monkeypatch, tmp_path):
    """@ 浮层挂在 #composer 末尾(StatusBar 下方)。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标
        await pilot.pause()
        assert app._completion_menu.parent is app.query_one("#composer")  # noqa: SLF001


@pytest.mark.asyncio
async def test_file_and_command_menus_are_mutually_exclusive(monkeypatch, tmp_path):
    """@ 浮层开时切到 / 命令 → 文件浮层被命令浮层替换(互斥)。"""
    from birdcode.ui.widgets.completion_menu import CompletionMenu

    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@"
        ia.move_cursor((0, len(ia.text)))
        await pilot.pause()
        assert isinstance(app._completion_menu, FileCompletionMenu)  # noqa: SLF001
        # 切到命令态:命令浮层替换文件浮层
        ia.text = "/he"
        ia.move_cursor((0, len(ia.text)))
        await pilot.pause()
        assert isinstance(app._completion_menu, CompletionMenu)  # noqa: SLF001
        assert not isinstance(app._completion_menu, FileCompletionMenu)  # noqa: SLF001


# ---- Task 7: 端到端(句中 @ 引用 + 多级目录导航) ----


@pytest.mark.asyncio
async def test_at_in_middle_of_sentence(monkeypatch, tmp_path):
    """句中引用:'读 @app.py' 的 @ 在句中(前导非空白为 CJK)也能触发并补全。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "读 @ap"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        assert isinstance(app._completion_menu, FileCompletionMenu)  # noqa: SLF001
        await pilot.press("tab")
        await pilot.pause()
        assert ia.text == "读 @app.py "


@pytest.mark.asyncio
async def test_at_nested_directory_navigation(monkeypatch, tmp_path):
    """@pkg/ → 展开 → @pkg/sub/ 多级目录导航。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "deep.py").write_text("x")
    app = BirdApp(MockProvider(delay=0.0))
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        ia = app.query_one(InputArea)
        ia.text = "@pk"
        ia.move_cursor((0, len(ia.text)))  # TextArea 赋值不挪光标,手动挪到行尾
        await pilot.pause()
        await pilot.press("tab")  # → @pkg/ 展开
        await pilot.pause()
        # pkg 下只有 sub/ → Tab 展开
        await pilot.press("tab")
        await pilot.pause()
        assert ia.text == "@pkg/sub/"
        names = [c.name for c in app._completion_menu.candidates]  # noqa: SLF001
        assert "deep.py" in names
