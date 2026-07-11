# src/birdcode/permission/migrate.py
"""一次性迁移:旧 ~/.birdcode/permissions.db(sqlite)→ ~/.birdcode/permissions.yaml。

仅在 db 文件存在时触发;迁移后不删 db(留用户自行备份)。best-effort,失败只记日志。
自包含:内联旧表结构,不依赖 store.py(后者已在本次重构中删除)。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from birdcode.utils.logging import get_logger

log = get_logger("birdcode.permission.migrate")  # 进 debug.log


def migrate_sqlite_to_yaml(
    db_path: Path | None = None,
    *,
    out_path: Path | None = None,
) -> int:
    """返回迁移条数;db 不存在或不可读返回 0。

    db_path 默认 ~/.birdcode/permissions.db;out_path 默认 ~/.birdcode/permissions.yaml。
    out_path 显式传入仅用于测试(避免污染真实 home)。
    """
    db_path = db_path or (Path.home() / ".birdcode" / "permissions.db")
    if not db_path.exists():
        return 0
    try:
        from sqlmodel import Field, Session, SQLModel, create_engine, select

        class _OldRule(SQLModel, table=True):
            __tablename__ = "permissionrule"
            # extend_existing:函数可能被测试多次调用(每次重新定义本类),
            # 同名 Table 重入 SQLModel.metadata 时 extend 而非 raise。
            __table_args__ = {"extend_existing": True}
            id: int | None = Field(default=None, primary_key=True)
            tool_name: str
            path_pattern: str | None = None
            action: str = "allow"
            created_at: str = ""

        engine = create_engine(f"sqlite:///{db_path}")
        rows: list[_OldRule] = []
        with Session(engine) as s:
            rows = list(s.exec(select(_OldRule)).all())
    except Exception as exc:  # noqa: BLE001 - 迁移 best-effort,任何失败都不阻断启动
        log.warning("旧权限 db 迁移失败 %s: %s(跳过)", db_path, exc)
        return 0

    if not rows:
        # 旧 db 存在但无规则:不写任何文件(避免在用户 home 留下空的 permissions.yaml)。
        return 0

    data: dict[str, list[str]] = {"allow": [], "deny": []}
    for r in rows:
        action = r.action if r.action in ("allow", "deny") else "allow"
        pat = r.path_pattern or "*"
        data[action].append(f"{r.tool_name}({pat})")

    out = out_path if out_path is not None else (Path.home() / ".birdcode" / "permissions.yaml")
    existing: dict[str, list[str]] = {"allow": [], "deny": []}
    if out.exists():
        try:
            lo = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
            if isinstance(lo, dict):
                for k in ("allow", "deny"):
                    if isinstance(lo.get(k), list):
                        existing[k] = [str(x) for x in lo[k]]
        except yaml.YAMLError:
            # 用户 YAML 损坏:不覆盖(否则原有内容被仅含 db 规则的结构抹掉,且不可逆)。
            # 跳过本次迁移;db 不删,用户修好 YAML 后下次启动再迁。
            log.warning("user YAML 损坏,跳过迁移(不覆盖): %s", out)
            return 0
    added = 0
    for k in ("allow", "deny"):
        for e in data[k]:
            if e not in existing[k]:
                existing[k].append(e)
                added += 1
    if added == 0:
        # db 规则已全部存在于 YAML:不重写(避免 yaml.safe_dump 抹掉用户手写注释/格式)。
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log.info("迁移 %d 条旧权限规则 → %s", added, out)
    return added
