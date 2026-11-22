"""Unit tests for StreamableHTTPMCPServer security and JSON-RPC dispatch."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from hermes_acp_mcp_bridge import StreamableHTTPMCPServer, openai_tools_to_mcp


def _post(url: str, payload: dict, *, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_openai_tools_conversion_and_auth_guards() -> None:
    specs = openai_tools_to_mcp(
        [
            {
                "type": "function",
                "function": {
                    "name": "ping_node",
                    "description": "Ping node",
                    "parameters": {"type": "object", "properties": {"host": {"type": "string"}}},
                },
            }
        ]
    )
    assert [s.name for s in specs] == ["ping_node"]

    def executor(name: str, args: dict) -> str:
        return f"{name}:{args.get('host')}"

    with StreamableHTTPMCPServer(
        tools=specs,
        executor=executor,
        policy_check=lambda name, args: (args.get("host") != "blocked", "denied by policy"),
    ) as srv:
        cfg = srv.acp_mcp_server_config()
        auth_header = {h["name"]: h["value"] for h in cfg["headers"]}

        # Missing auth -> 401
        with pytest.raises(urllib.error.HTTPError) as exc_unauth:
            _post(srv.url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert exc_unauth.value.code == 401

        # Foreign origin -> 403
        with pytest.raises(urllib.error.HTTPError) as exc_origin:
            _post(
                srv.url,
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={**auth_header, "Origin": "https://evil.example"},
            )
        assert exc_origin.value.code == 403

        # Declared tool works
        status, ok_res = _post(
            srv.url,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping_node", "arguments": {"host": "alpha"}},
            },
            headers=auth_header,
        )
        assert status == 200
        assert ok_res["result"]["isError"] is False
        assert ok_res["result"]["content"][0]["text"] == "ping_node:alpha"

        # Policy-blocked call returns isError=True without invoking executor
        _, pol_res = _post(
            srv.url,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "ping_node", "arguments": {"host": "blocked"}},
            },
            headers=auth_header,
        )
        assert pol_res["result"]["isError"] is True
        assert pol_res["result"]["content"][0]["text"] == "denied by policy"

        # Undeclared tool rejected with JSON-RPC -32602
        _, unk_res = _post(
            srv.url,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "shell_exec", "arguments": {}},
            },
            headers=auth_header,
        )
        assert unk_res["error"]["code"] == -32602
