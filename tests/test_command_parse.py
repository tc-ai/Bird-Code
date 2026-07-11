from birdcode.ui.pure import parse_command


def test_non_slash_returns_none():
    assert parse_command("hello") is None
    assert parse_command("") is None
    assert parse_command("   ") is None


def test_plain_slash_returns_none():
    assert parse_command("/") is None
    assert parse_command("/ ") is None  # 只有斜杠+空格,无命令名


def test_name_lowercased_args_preserved():
    p = parse_command("/HELP")
    assert p is not None and p.name == "/help" and p.args == ""


def test_name_and_args_split_on_first_space():
    p = parse_command("/Profile GPT-4  extra")
    assert p is not None
    assert p.name == "/profile"
    assert p.args == "GPT-4  extra"  # 原文,大小写与多空格保留


def test_no_args_empty_string():
    p = parse_command("/clear")
    assert p is not None and p.name == "/clear" and p.args == ""
