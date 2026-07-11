# tests/test_logging.py
import logging


def test_get_logger_writes_to_file(tmp_path, monkeypatch):
    from birdcode.utils import logging as bc_logging

    monkeypatch.setattr(bc_logging, "_log_dir", tmp_path)
    log = bc_logging.get_logger("test")
    log.setLevel(logging.DEBUG)
    log.debug("hello-birdcode")

    # flush handlers
    for h in log.handlers:
        h.flush()

    files = list(tmp_path.glob("*.log"))
    assert files, "expected a log file to be created"
    assert "hello-birdcode" in files[0].read_text(encoding="utf-8")


def test_get_logger_does_not_add_stdout_handler(tmp_path, monkeypatch):
    from birdcode.utils import logging as bc_logging

    monkeypatch.setattr(bc_logging, "_log_dir", tmp_path)
    log = bc_logging.get_logger("uniq")
    assert not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in log.handlers
    )
