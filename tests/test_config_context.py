from birdcode.config.schema import AppConfig


def _cfg(**over) -> AppConfig:
    base = {
        "providers": {
            "p": {"protocol": "anthropic", "model": "m", "base_url": "u", "api_key": "k"}
        },
        "default": "p",
    }
    base.update(over)
    return AppConfig.model_validate(base)


def test_context_defaults():
    c = _cfg()
    assert c.context_window == 200_000
    assert c.compact_summary_reserve == 20_000
    assert c.compact_safety_margin == 13_000
    assert c.compact_tail_budget == 30_000


def test_autocompact_threshold_derived():
    c = _cfg()
    assert c.autocompact_threshold == 200_000 - 20_000 - 13_000  # 167_000


def test_threshold_follows_config_override():
    c = _cfg(context_window=100_000, compact_summary_reserve=10_000, compact_safety_margin=5_000)
    assert c.autocompact_threshold == 85_000
