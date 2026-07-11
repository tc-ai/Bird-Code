from birdcode.permission.blacklist import dangerous_hint, is_dangerous

_DANGEROUS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "rm -rf ./*",
    "find / -name x -delete",
    "chmod -R 000 /",
    "sudo apt install x",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "echo x > /dev/sda",
]
_BENIGN = ["ls -la", "git status", "echo hello", "rg pattern src/", "python -m pytest"]


def test_dangerous_commands_caught():
    for cmd in _DANGEROUS:
        hit, label = is_dangerous(cmd)
        assert hit, f"应拦截: {cmd}"


def test_benign_commands_pass():
    for cmd in _BENIGN:
        hit, _ = is_dangerous(cmd)
        assert not hit, f"误拦: {cmd}"


def test_hint_contains_label():
    hit, label = is_dangerous("rm -rf /")
    assert hit
    assert label in dangerous_hint(label)


def test_power_commands_caught():
    """关机/重启类命令硬拦(编码 agent 场景几乎无合法用途,且不可逆)。"""
    for cmd in ["shutdown now", "shutdown -h now", "reboot", "poweroff", "halt"]:
        hit, label = is_dangerous(cmd)
        assert hit, f"应拦截: {cmd}"
        assert label == "关机/重启"


def test_power_words_in_args_not_caught():
    """关机/重启词出现在参数/引号里不误拦(只拦命令动词位置)。"""
    for cmd in [
        'git commit -m "fix reboot handling"',
        "grep shutdown src/",
        "cat shutdown.log",
        "rg reboot src/",
    ]:
        hit, _ = is_dangerous(cmd)
        assert not hit, f"误拦: {cmd}"
