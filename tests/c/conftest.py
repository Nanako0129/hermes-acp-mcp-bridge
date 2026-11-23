import json
import sys
from pathlib import Path

import pytest

from antigravity_acp_mcp.acp_connection import AcpConnection

FAKE = str(Path(__file__).with_name("fake_acp.py"))


@pytest.fixture
def connect(tmp_path):
    """connect(behaviour, mux_env=None, initialize=True) -> started AcpConnection on the fake."""
    made = []

    def factory(behaviour=None, mux_env=None, initialize=True):
        cwd = tmp_path / "empty-cwd"
        cwd.mkdir(exist_ok=True)
        conn = AcpConnection(sys.executable, [FAKE, json.dumps(behaviour or {})], str(cwd), mux_env=mux_env)
        conn.start()
        made.append(conn)
        if initialize:
            conn.initialize(timeout=20)
        return conn

    yield factory
    for conn in made:
        conn.close()


@pytest.fixture
def planted_secrets(monkeypatch):
    names = ("DISCORD_BOT_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_TOKEN")
    for n in names:
        monkeypatch.setenv(n, "planted-" + n.lower())
    monkeypatch.setenv("ACP_MUX_SESSION_ID", "ambient-must-not-leak")
    monkeypatch.setenv("ACP_MUX_PAR", "/ambient/par")
    return names


def dump(conn):
    return conn.request("x/dump", {}, timeout=10)

