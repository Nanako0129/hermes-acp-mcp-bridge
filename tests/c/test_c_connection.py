import threading

import pytest

from antigravity_acp_mcp.acp_connection import AcpProcessExit, AcpRpcError

from conftest import dump


def test_early_update_before_session_new_reply_is_buffered_and_flushed(connect):
    conn = connect({"early_update": True})
    sid = conn.request("session/new", {"cwd": conn.cwd, "mcpServers": []}, timeout=10)["sessionId"]
    got = []
    conn.attach_session(sid, got.append, lambda p: None)
    assert [u["content"]["text"] for u in got] == ["early"]  # delivered in order, on attach


def test_stderr_flood_of_a_megabyte_does_not_block_child(connect):
    conn = connect({"flood_stderr": 1_200_000})  # initialize reply only comes after the flood
    assert conn.request("x/dump", {}, timeout=10)["cwd"] == conn.cwd
    assert len(conn._stderr) == 200  # 12000 lines were flooded; only the bounded tail is kept


def test_child_exit_surfaces_status_and_diagnostic_but_never_stderr_bytes(connect):
    secret_lines = ["boot ok", "token ya29.SECRET1234567", "Authorization: Bearer SECRET"]
    conn = connect({"exit_on": "session/new", "exit_code": 7, "stderr_lines": secret_lines})
    with pytest.raises(AcpProcessExit) as raised:
        conn.request("session/new", {"cwd": conn.cwd, "mcpServers": []}, timeout=10)
    text = str(raised.value)
    assert "status 7" in text and "server stderr suppressed" in text
    assert "google-access-token-prefix" in text and "bearer-token" in text
    for planted in ("ya29.SECRET", "Bearer SECRET", "boot ok"):
        assert planted not in text and planted not in repr(raised.value)
    assert raised.value.exit_status == 7
    with pytest.raises(AcpProcessExit):  # later requests fail the same typed way
        conn.request("x/dump", {}, timeout=10)


def test_rate_limit_stderr_sets_fixed_retry_class(connect):
    conn = connect({"exit_on": "session/new", "stderr_lines": ["Error 429, RESOURCE_EXHAUSTED"]})
    with pytest.raises(AcpProcessExit) as raised:
        conn.request("session/new", {}, timeout=10)
    assert raised.value.retry_class == "rate-limit" == conn.stderr_retry_class
    assert "RESOURCE_EXHAUSTED" not in str(raised.value)


def test_child_env_has_no_secrets_but_gets_explicit_mux_vars(connect, planted_secrets):
    conn = connect(mux_env={"ACP_MUX_AFFINITY": "conv-1", "ACP_MUX_AFFINITY_PIN": "1"})
    info = dump(conn)
    for name in planted_secrets:
        assert name not in info["env_names"], name
    assert info["mux"] == {"ACP_MUX_AFFINITY": "conv-1", "ACP_MUX_AFFINITY_PIN": "1"}
    assert info["cwd"] == conn.cwd


def test_unlisted_mux_variable_is_refused():
    from antigravity_acp_mcp.acp_connection import AcpConnection

    with pytest.raises(ValueError):
        AcpConnection("x", [], "/tmp", mux_env={"ACP_MUX_PAR": "/evil"})


def test_initialize_is_honest_and_unknown_agent_requests_get_32601(connect):
    conn = connect({"agent_request": "fs/read_text_file", "permission": None})
    sid = conn.request("session/new", {"cwd": conn.cwd, "mcpServers": []}, timeout=10)["sessionId"]
    conn.attach_session(sid, lambda u: None, lambda p: None)
    meta = conn.request("session/prompt", {"sessionId": sid, "prompt": []}, timeout=10)["_meta"]
    assert meta["agentRequestResponse"]["error"]["code"] == -32601
    init = dump(conn)["log"][0]["params"]
    assert init["clientInfo"]["name"] == "hermes-antigravity-acp-mcp"
    assert init["clientCapabilities"] == {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False}


def test_rpc_error_is_typed(connect):
    conn = connect()
    with pytest.raises(AcpRpcError) as raised:
        conn.request("x/error", {}, timeout=10)
    assert raised.value.code == -32000


def test_cancel_answers_pending_permissions_cancelled_then_notifies(connect):
    conn = connect({"permission": {"toolCall": {"toolCallId": "t1"}, "options": []}, "wait_cancel": True})
    sid = conn.request("session/new", {"cwd": conn.cwd, "mcpServers": []}, timeout=10)["sessionId"]
    seen = threading.Event()
    conn.attach_session(sid, lambda u: None, lambda p: seen.set())  # returns None: stays pending
    result = {}
    t = threading.Thread(target=lambda: result.update(conn.request("session/prompt", {"sessionId": sid, "prompt": []})))
    t.start()
    assert seen.wait(10)
    conn.cancel_session(sid)
    t.join(10)
    assert result["stopReason"] == "cancelled"
    log = dump(conn)["log"]
    answers = [m for m in log if m.get("id") == 900]
    assert answers[0]["result"] == {"outcome": {"outcome": "cancelled"}}
    methods = [m.get("method") for m in log]
    assert methods.index("session/cancel") > log.index(answers[0])


def test_close_reaps_the_child(connect):
    conn = connect()
    conn.close()
    assert conn._proc.poll() is not None
