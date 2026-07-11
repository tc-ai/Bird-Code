# tests/test_migrate.py
"""一次性迁移测试:旧 sqlite(~/.birdcode/permissions.db)→ YAML。

用标准库 sqlite3 直接往临时 db 写 permissionrule 行(不定义 SQLModel 模型,
避免与 migrate.py 内联的 _OldRule 共享全局 SQLModel.metadata 造成表名冲突)。
migrate 读时用自己的 _OldRule(extend_existing)读回,断言生成 Tool(pattern) 规则。
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import yaml


def _write_old_rows(db_path: Path, rows: list[dict[str, object]]) -> None:
    """用原生 sqlite3 建 permissionrule 表并插入行(列名与旧 store.py 的表一致)。"""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE permissionrule ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "tool_name TEXT NOT NULL,"
            "path_pattern TEXT,"
            "action TEXT DEFAULT 'allow',"
            "created_at TEXT DEFAULT '')"
        )
        conn.executemany(
            "INSERT INTO permissionrule (tool_name, path_pattern, action, created_at) "
            "VALUES (:tool_name, :path_pattern, :action, :created_at)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_no_db_returns_zero(tmp_path):
    """db 不存在 → 返回 0,不抛。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    n = migrate_sqlite_to_yaml(db_path=tmp_path / "missing.db", out_path=tmp_path / "out.yaml")
    assert n == 0


def test_migrate_writes_tool_pattern_rules(tmp_path):
    """两行旧规则(allow + deny)→ YAML 含 Bash(git *) / Write(./src/**)。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [
            {"tool_name": "Bash", "path_pattern": "git *", "action": "allow", "created_at": "t1"},
            {
                "tool_name": "Write",
                "path_pattern": "./src/**",
                "action": "deny",
                "created_at": "t2",
            },
        ],
    )
    out = tmp_path / "permissions.yaml"

    n = migrate_sqlite_to_yaml(db_path=db, out_path=out)
    assert n == 2
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["allow"] == ["Bash(git *)"]
    assert data["deny"] == ["Write(./src/**)"]


def test_migrate_merges_into_existing_yaml(tmp_path):
    """已有 YAML 规则不丢失;迁移条目去重追加。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    out = tmp_path / "permissions.yaml"
    out.write_text("allow:\n  - Read(*)\ndeny: []\n", encoding="utf-8")
    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [{"tool_name": "Bash", "path_pattern": "git *", "action": "allow", "created_at": "t"}],
    )

    migrate_sqlite_to_yaml(db_path=db, out_path=out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["allow"] == ["Read(*)", "Bash(git *)"]  # 原有保留 + 追加


def test_migrate_null_pattern_becomes_star(tmp_path):
    """旧规则 path_pattern=NULL → 写成 Tool(*)。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [{"tool_name": "Write", "path_pattern": None, "action": "allow", "created_at": ""}],
    )
    out = tmp_path / "permissions.yaml"

    migrate_sqlite_to_yaml(db_path=db, out_path=out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["allow"] == ["Write(*)"]


def test_migrate_self_contained_no_store_import():
    """迁移模块源码不导入 store(后者已删除)。"""
    import birdcode.permission.migrate as m

    src = inspect.getsource(m)
    assert "from birdcode.permission.store" not in src
    assert "import birdcode.permission.store" not in src


def test_corrupt_db_returns_zero(tmp_path):
    """db 文件存在但不是合法 sqlite → best-effort 返回 0,不抛。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    db = tmp_path / "permissions.db"
    db.write_text("not a sqlite file", encoding="utf-8")
    n = migrate_sqlite_to_yaml(db_path=db, out_path=tmp_path / "out.yaml")
    assert n == 0


def test_migrate_unknown_action_defaults_to_allow(tmp_path):
    """旧 action 非 allow/deny(如脏数据)→ 归入 allow(安全侧:不丢规则)。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [{"tool_name": "Edit", "path_pattern": "*", "action": "weird", "created_at": ""}],
    )
    out = tmp_path / "permissions.yaml"

    migrate_sqlite_to_yaml(db_path=db, out_path=out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["allow"] == ["Edit(*)"]
    assert data["deny"] == []


def test_migrate_skips_when_no_new_rules(tmp_path):
    """db 规则已全在 YAML 中 → 不重写(保留用户手写注释/格式)。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    out = tmp_path / "permissions.yaml"
    original = "# my comment\nallow:\n  - Bash(git *)\ndeny: []\n"
    out.write_text(original, encoding="utf-8")
    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [{"tool_name": "Bash", "path_pattern": "git *", "action": "allow", "created_at": ""}],
    )
    n = migrate_sqlite_to_yaml(db_path=db, out_path=out)
    assert n == 0
    assert out.read_text(encoding="utf-8") == original  # 未重写,注释保留


def test_migrate_corrupt_yaml_not_clobbered(tmp_path):
    """用户 YAML 损坏 → 不覆盖(跳过迁移),保留原文件供用户修复。"""
    from birdcode.permission.migrate import migrate_sqlite_to_yaml

    out = tmp_path / "permissions.yaml"
    corrupt = "allow: [unclosed\n"  # 未闭合 flow 序列 → YAMLError
    out.write_text(corrupt, encoding="utf-8")
    db = tmp_path / "permissions.db"
    _write_old_rows(
        db,
        [{"tool_name": "Bash", "path_pattern": "git *", "action": "allow", "created_at": ""}],
    )
    n = migrate_sqlite_to_yaml(db_path=db, out_path=out)
    assert n == 0
    assert out.read_text(encoding="utf-8") == corrupt  # 未被覆盖
