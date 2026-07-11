# tests/session/test_store_ctx.py
from birdcode.session.models import SessionContext
from birdcode.session.store import SessionStore


def test_ctx_accessor(tmp_path):
    ctx = SessionContext(session_id="s1", cwd=".", version="v", git_branch=None)
    s = SessionStore(ctx, tmp_path, root=tmp_path)
    assert s.ctx is ctx
    s.close()
