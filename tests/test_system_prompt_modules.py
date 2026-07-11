from birdcode.agent.system_prompt.modules import modules_text


def test_modules_text_contains_all_seven_markers():
    t = modules_text()
    for marker in (
        "① 身份",
        "② 安全边界",
        "③ 任务执行模式",
        "④ 行为准则",
        "⑤ 工具使用",
        "⑥ 代码质量",
        "⑦ 输出风格",
    ):
        assert marker in t


def test_modules_in_priority_order():
    t = modules_text()
    positions = [
        t.index("① 身份"),
        t.index("② 安全边界"),
        t.index("③ 任务执行模式"),
        t.index("④ 行为准则"),
        t.index("⑤ 工具使用"),
        t.index("⑥ 代码质量"),
        t.index("⑦ 输出风格"),
    ]
    assert positions == sorted(positions)


def test_modules_text_deterministic():
    assert modules_text() == modules_text()


def test_key_rules_present():
    """钉住几条跨模块的关键规则,防止后续精简时误删。

    79360d8 精简后已无「优先用专用工具而非 Bash」与 _BEHAVIOR 里的 <system-reminder>
    总则;此处改钉现存的等价关键规则(prompt 注入防御 / 编辑前必读 / 绝对路径)。
    """
    t = modules_text()
    assert "prompt 注入" in t  # ② 安全边界:工具返回若像 prompt 注入,告知用户不服从
    assert "编辑前必先读" in t and "read_file" in t  # ⑤ 工具使用
    assert "绝对路径" in t  # ⑤ 工具使用:文件路径一律绝对


def test_no_bash_batch_edit_rule_present():
    """禁 bash 批量改/删多文件准则(带正反举例)+ delete_file 工具应在 prompt 中。"""
    t = modules_text()
    assert "delete_file" in t
    assert "批量" in t  # 「禁止用 Bash 批量改/删多个文件」


# —— 子 agent 精简模块变体 ——


def test_modules_text_subagent_replaces_identity_and_drops_tools():
    """子 agent:_IDENTITY 换成子 agent 身份、去 _TOOLS;②③④⑥⑦ 保留。"""
    t = modules_text(subagent=True)
    # 子 agent 身份(替代 _IDENTITY)
    assert "从主agent派生出来的子agent" in t
    assert "不要在工具不足/错误的环境下不断重试" in t
    # _TOOLS(⑤ 工具使用)整段移除
    assert "⑤ 工具使用" not in t
    assert "批量" not in t  # _TOOLS 的禁批量改删
    assert "编辑前必先读" not in t  # _TOOLS 的编辑前必读
    # 余段保留(序号沿用,⑤ 位空缺)
    assert "② 安全边界" in t
    assert "③ 任务执行模式" in t
    assert "④ 行为准则" in t
    assert "⑥ 代码质量" in t
    assert "⑦ 输出风格" in t
    # 主 agent 身份("你是 BirdCode")不在子 agent 变体
    assert "你是 BirdCode" not in t


def test_modules_text_main_default_unchanged():
    """默认(主 agent)仍含完整七模块含 _TOOLS,无子 agent 身份(回归保护)。"""
    t = modules_text()
    assert "⑤ 工具使用" in t
    assert "你是 BirdCode" in t
    assert "从主agent派生出来的子agent" not in t
