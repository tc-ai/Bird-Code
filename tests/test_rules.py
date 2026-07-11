from pathlib import Path

from birdcode.permission.rules import RuleSet, default_yaml_paths, parse_rule


def test_parse_rule():
    r = parse_rule("Bash(git *)", "allow")
    assert r is not None and r.tool == "bash" and r.pattern == "git *" and r.action == "allow"


def test_match_bash_command():
    rs = RuleSet(sources=[[parse_rule("Bash(git *)", "allow")]])  # type: ignore[list-item]
    assert rs.match("bash", "git status") == "allow"
    assert rs.match("bash", "npm run build") is None


def test_match_path_glob():
    rs = RuleSet(sources=[[parse_rule("Write(./src/**)", "allow")]])  # type: ignore[list-item]
    assert rs.match("write_file", "/abs/proj/src/x.py") == "allow"


def test_source_priority_session_over_user():
    session = [parse_rule("Bash(rm ./build/*)", "allow")]
    user = [parse_rule("Bash(rm *)", "deny")]
    rs = RuleSet(sources=[session, user])  # session 先 → 高优先
    assert rs.match("bash", "rm ./build/x") == "allow"
    assert rs.match("bash", "rm /etc/x") == "deny"


def test_intra_source_deny_wins():
    layer = [parse_rule("Bash(git *)", "allow"), parse_rule("Bash(git push *)", "deny")]
    rs = RuleSet(sources=[layer])  # type: ignore[list-item]
    assert rs.match("bash", "git push origin") == "deny"
    assert rs.match("bash", "git status") == "allow"


def test_load_yaml(tmp_path: Path):
    f = tmp_path / "p.yaml"
    f.write_text("allow:\n  - Bash(git *)\ndeny:\n  - Write(./secrets/**)\n", encoding="utf-8")
    rs = RuleSet.load(yaml_paths=[f])
    assert rs.match("bash", "git status") == "allow"
    assert rs.match("write_file", str(tmp_path / "secrets" / "k")) == "deny"


def test_default_yaml_paths_order():
    paths = default_yaml_paths()
    names = [p.name for p in paths]
    assert names == ["permissions.local.yaml", "permissions.yaml", "permissions.yaml"]


def test_bash_command_match_is_case_sensitive():
    """命令(bash)模式大小写敏感:git * 命中 git status,不命中 GIT STATUS。

    路径模式才沿用平台大小写语义;命令名各平台都区分大小写,否则 Win 上
    Bash(git *) 会误命中大写命令、与 Linux 行为相悖(移植性 bug)。
    """
    rs = RuleSet(sources=[[parse_rule("Bash(git *)", "allow")]])  # type: ignore[list-item]
    assert rs.match("bash", "git status") == "allow"
    assert rs.match("bash", "GIT STATUS") is None


def test_append_local_uses_explicit_path(tmp_path: Path):
    """append_local 的 path 参数:写到指定文件(而非硬编码 cwd)。

    gate 自定义 yaml_paths 时把 _local_path 传进来,保证「写入」与「读取」同一文件。
    """
    import yaml

    from birdcode.permission.rules import append_local

    p = tmp_path / "local.yaml"
    returned = append_local("Bash(git *)", "allow", path=p)
    assert returned == p
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["allow"] == ["Bash(git *)"]
    assert data["deny"] == []


def test_bash_pattern_with_dot_slash_is_command_not_path():
    """Bash(./deploy.sh *) 是命令模式(非路径):不做 ./ 归一,且大小写敏感。

    回归旧 path_glob 前缀启发式:会把 ./ 开头的 bash pattern 误判为路径 glob(平台大小写)。
    """
    rs = RuleSet(sources=[[parse_rule("Bash(./deploy.sh *)", "allow")]])  # type: ignore[list-item]
    assert rs.match("bash", "./deploy.sh --prod") == "allow"
    assert rs.match("bash", "./DEPLOY.SH --prod") is None  # 命令大小写敏感


def test_default_yaml_paths_anchors_to_given_root(tmp_path: Path):
    """default_yaml_paths(root) 把 local/project 锚定到给定 root(与 gate project_root 同源)。"""
    from birdcode.permission.rules import default_yaml_paths

    paths = default_yaml_paths(tmp_path)
    assert paths[0] == tmp_path / ".birdcode" / "permissions.local.yaml"
    assert paths[1] == tmp_path / ".birdcode" / "permissions.yaml"
    assert paths[2] == Path.home() / ".birdcode" / "permissions.yaml"


def test_mcp_rule_does_not_prefix_match_sibling():
    """回归(安全 #2):批准 mcp__srv__foo 绝不可经蛇形桥命中 mcp__srv__foo_bar /
    mcp__srv__foobar。旧实现 tn.startswith(tool+'_') 把 MCP 命名空间当单段前缀 →
    越权 auto-approve 同级写工具(HITL 被绕过)。"""
    r = parse_rule("mcp__srv__foo(*)", "allow")
    assert r is not None
    assert r.matches("mcp__srv__foo", None) is True
    assert r.matches("mcp__srv__foo_bar", None) is False  # 前缀重叠也不命中
    assert r.matches("mcp__srv__foobar", None) is False
    assert r.matches("mcp__other__foo", None) is False


def test_parse_rule_accepts_mcp_names_with_hyphen_and_dot():
    """回归(#4):server 名含 - 或 . 时(如 my-server、github.com),mcp__my-server__x
    须能解析,否则其 Session/Always 规则被 parse_rule 静默丢弃(HITL 每次重弹)。"""
    r1 = parse_rule("mcp__my-server__tool(*)", "allow")
    assert r1 is not None and r1.tool == "mcp__my-server__tool"
    r2 = parse_rule("mcp__github.com__x(*)", "allow")
    assert r2 is not None and r2.tool == "mcp__github.com__x"


def test_parse_rule_warns_on_invalid_mcp_rules(monkeypatch):
    """#2/#1:失效 MCP 规则须警告(非静默),避免用户误以为受保护。两类:① 路径范围
    (mcp__fs__write(/etc/**),subject None 不匹配路径);② server 级/裸前缀
    (mcp__evil_srv(*)、mcp__(*)——精确匹配下不命中任何工具)。规则仍加载(死规则)。"""
    import birdcode.permission.rules as R

    monkeypatch.setattr(R, "_WARNED_INVALID_MCP", set())  # 隔离去重状态
    warned: list[str] = []

    def fake_warning(msg, *args, **kwargs):
        warned.append(msg % args if args else msg)

    monkeypatch.setattr(R.log, "warning", fake_warning)
    # ① 路径范围
    r = parse_rule("mcp__fs__write(/etc/**)", "deny")
    assert r is not None and r.pattern == "/etc/**"
    assert any("/etc/**" in w for w in warned)
    # ② server 级(缺 tool 段,#1)
    warned.clear()
    parse_rule("mcp__evil_srv(*)", "deny")
    assert any("mcp__evil_srv" in w for w in warned)
    # ② 裸前缀(#1)
    warned.clear()
    parse_rule("mcp__(*)", "deny")
    assert warned
    # 合法工具级 * 不警告
    warned.clear()
    parse_rule("mcp__fs__write(*)", "allow")
    assert not warned


def test_parse_rule_dedups_repeated_invalid_mcp_warning(monkeypatch):
    """#2 reload 噪音:_reload_local 每次 HITL Always 都重读本地 YAML→重 parse,同一
    手写无效规则反复警告。按 (source, 原文) 去重,首次警告后跳过。"""
    import birdcode.permission.rules as R

    monkeypatch.setattr(R, "_WARNED_INVALID_MCP", set())
    warned: list[str] = []

    def fake_warning(msg, *args, **kwargs):
        warned.append(msg % args if args else msg)

    monkeypatch.setattr(R.log, "warning", fake_warning)
    for _ in range(5):
        R.parse_rule("mcp__srv__tool(/etc/**)", "deny", source="local.yaml")
    assert len([w for w in warned if "/etc/**" in w]) == 1  # 5 次 parse 只警告 1 次
    # 不同 source 仍警告(去重粒度是 (source, 原文))
    R.parse_rule("mcp__srv__tool(/etc/**)", "deny", source="other.yaml")
    assert len([w for w in warned if "/etc/**" in w]) == 2


def test_parse_rule_rejects_typo_with_dot_or_hyphen():
    """#3:正则 ./- 仅限 mcp__ 命名空间段。普通工具的 typo(delete-file、edit.file、
    Bash.git)须被当格式错误丢弃,而非静默接受为死规则(丢失反馈)。"""
    assert parse_rule("delete-file(*)", "allow") is None
    assert parse_rule("edit.file(*)", "allow") is None
    assert parse_rule("Bash.git(*)", "allow") is None
    # 对照:合法的普通工具名(含下划线)与 mcp__ 名仍解析
    assert parse_rule("write_file(*)", "allow") is not None
    assert parse_rule("mcp__my-server__tool(*)", "allow") is not None
