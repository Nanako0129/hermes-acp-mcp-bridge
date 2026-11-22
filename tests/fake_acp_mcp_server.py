"""Fake ACP server that connects to Streamable HTTP MCP servers over loopback."""

from __future__ import annotations

import json
import sys
import urllib.request


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _mcp_post(server_cfg: dict, rpc_payload: dict) -> tuple[int, dict | None]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    for h in server_cfg.get("headers") or ():
        headers[h["name"]] = h["value"]
    req = urllib.request.Request(
        server_cfg["url"],
        data=json.dumps(rpc_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        raw = resp.read().decode("utf-8").strip()
        return resp.status, (json.loads(raw) if raw else None)


def main() -> None:
    state: dict = {
        "session_counter": 0,
        "sessions": {},
        "mcp_initialized": False,
        "tools": [],
        "req_seq": 1000,
    }
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        msg = json.loads(line)
        rid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "initialize":
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "mcpCapabilities": {"http": True, "sse": True},
                            "_meta": {"antigravity": {"systemPrompt": {"append": True}}},
                        },
                    },
                }
            )
        elif method in ("session/new", "session/resume"):
            sid = params.get("sessionId")
            if not sid:
                state["session_counter"] += 1
                sid = f"mcp-sess-{state['session_counter']}"
            mcp_servers = params.get("mcpServers") or []
            state["sessions"][sid] = {
                "mcpServers": mcp_servers,
                "meta": params.get("_meta") or {},
                "model": "",
                "method": method,
            }
            if mcp_servers and not state["mcp_initialized"]:
                cfg = mcp_servers[0]
                _, init_res = _mcp_post(
                    cfg,
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": "2025-03-26", "capabilities": {}},
                    },
                )
                assert init_res and "result" in init_res
                _mcp_post(cfg, {"jsonrpc": "2.0", "method": "notifications/initialized"})
                _, list_res = _mcp_post(
                    cfg, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
                )
                state["tools"] = (list_res or {}).get("result", {}).get("tools", [])
                state["mcp_initialized"] = True
            _emit({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": sid}})
        elif method == "session/set_model":
            sid = params["sessionId"]
            if sid in state["sessions"]:
                state["sessions"][sid]["model"] = params.get("modelId", "")
            _emit({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "session/prompt":
            sid = params["sessionId"]
            sess = state["sessions"][sid]
            cfg = (sess.get("mcpServers") or [None])[0]
            prompt_text = " ".join(
                b.get("text", "") for b in params.get("prompt") or () if b.get("type") == "text"
            )
            results: list[str] = []
            if cfg and state["tools"] and "MULTI_HOP" in prompt_text:
                tool_name = state["tools"][0]["name"]
                # Ask permission for the bridge tool first (must be auto-allowed)
                perm_id = state["req_seq"]
                state["req_seq"] += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": perm_id,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": sid,
                            "toolCall": {"toolCallId": "tc-1", "name": tool_name, "title": tool_name},
                            "options": [
                                {"optionId": "allow_once", "kind": "allow_once", "name": "Allow"},
                                {"optionId": "reject_once", "kind": "reject_once", "name": "Deny"},
                            ],
                        },
                    }
                )
                perm_reply = json.loads(sys.stdin.readline())
                assert perm_reply["result"]["outcome"]["optionId"] == "allow_once"

                # Also verify undeclared tool permission is rejected!
                deny_id = state["req_seq"]
                state["req_seq"] += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": deny_id,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": sid,
                            "toolCall": {
                                "toolCallId": "tc-bad",
                                "name": "undeclared_shell",
                                "title": "undeclared_shell",
                            },
                            "options": [
                                {"optionId": "allow_once", "kind": "allow_once", "name": "Allow"},
                                {"optionId": "reject_once", "kind": "reject_once", "name": "Deny"},
                            ],
                        },
                    }
                )
                deny_reply = json.loads(sys.stdin.readline())
                assert deny_reply["result"]["outcome"]["optionId"] == "reject_once"

                # Execute 3 sequential MCP tool hops inside THIS ONE session/prompt!
                for idx in range(1, 4):
                    _, call_res = _mcp_post(
                        cfg,
                        {
                            "jsonrpc": "2.0",
                            "id": 100 + idx,
                            "method": "tools/call",
                            "params": {"name": tool_name, "arguments": {"step": idx}},
                        },
                    )
                    text = call_res["result"]["content"][0]["text"]
                    results.append(text)
            sys_append = ((sess.get("meta") or {}).get("antigravity") or {}).get("systemPrompt", {}).get("append", "")
            _emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "sys": sys_append,
                                        "model": sess.get("model"),
                                        "hops": results,
                                        "prompt": prompt_text,
                                    }
                                ),
                            },
                        },
                    },
                }
            )
            _emit({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


if __name__ == "__main__":
    main()
