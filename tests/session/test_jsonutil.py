from birdcode.session.jsonutil import read_json_dict


def test_read_json_dict_missing_returns_none(tmp_path):
    assert read_json_dict(tmp_path / "nope.json") is None


def test_read_json_dict_valid(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"k": 1}', encoding="utf-8")
    assert read_json_dict(p) == {"k": 1}


def test_read_json_dict_non_dict_returns_none(tmp_path):
    """合法 JSON 但非 dict(如 list)→ None(调用方预期 dict 结构)。"""
    p = tmp_path / "a.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_json_dict(p) is None


def test_read_json_dict_corrupt_returns_none(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{坏 json", encoding="utf-8")
    assert read_json_dict(p) is None


def test_read_json_dict_non_utf8_returns_none(tmp_path):
    """非 UTF-8 字节(崩溃转储/外部编辑)→ None,不抛 UnicodeDecodeError。

    CR #4 实证:此分支曾被 store._read_meta / subagent_meta.read_subagent_meta 各自
    漏掉(ValueError 旧只覆盖 JSONDecodeError),需两处并行修。集中到 read_json_dict 后,
    未来此类扩展只需改一处——本测试即该共享契约的回归守护。
    """
    p = tmp_path / "a.json"
    p.write_bytes(b"\xff\xfe\x00\x01bad bytes")
    assert read_json_dict(p) is None
