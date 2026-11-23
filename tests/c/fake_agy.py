"""Fake agy ACP server for conversation/facade tests: really calls the parked MCP endpoint over HTTP.

argv[1]: JSON behaviour {"script": [[step, ...] per prompt, ...], "log": path, "new_delay": seconds}
steps: {"call": {"name": str, "arguments": {...}}}  POST tools/call to the session's MCP server, wait for the answer
       {"text": str}                                 agent_message_chunk
       {"echo_last": true}                            agent_message_chunk with the last tool answer text
       {"silent": seconds}                            sleep without any output
       {"stderr": str}                                one line on stderr
       {"foreign": true}                              a tool_call update that is not ours (native tool)
Prompts are numbered across fake processes (counted from the log), so a respawned child continues the script.
A prompt with no script entry answers "ok". session/cancel ends the in-flight prompt with stopReason cancelled.
The log file gets one JSON line per event: {"ev": "prompt"|"new"|"answer"|"cancel", ...}.
"""

import json
import sys
import threading
import time
import urllib.request

B = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
LOG = B.get("log")
lock = threading.Lock()
state = {"sessions": {}, "n_prompt": 0, "inflight": None, "next_sid": 0}


def log(**kw):
    if LOG:
        with lock, open(LOG, "a") as fh:
            fh.write(json.dumps(kw) + "\n")


def emit(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def update(sid, payload):
    emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": payload}})


def mcp_call(server, name, arguments):
    headers = {h["name"]: h["value"] for h in server.get("headers", [])}
    headers.update({"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": arguments}}).encode()
    req = urllib.request.Request(server["url"], data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read() or b"{}")
    except Exception as exc:  # noqa: BLE001 - dropped connection / refused
        return {"error": type(exc).__name__}
    res = data.get("result") or {}
    text = "".join(c.get("text", "") for c in res.get("content", []))
    return {"text": text, "isError": bool(res.get("isError"))}


def run_prompt(rid, sid, text, steps, token):
    server = state["sessions"][sid]
    last = ""
    for step in steps:
        if token["cancelled"]:
            return
        if "call" in step:
            tcid = f"tc{time.monotonic_ns()}"
            update(sid, {"sessionUpdate": "tool_call", "toolCallId": tcid, "status": "in_progress",
                         "title": server["name"] + "_" + step["call"]["name"],
                         "_meta": {"is_mcp_tool_call": True,
                                   "mcp": {"server": server["name"], "tool": step["call"]["name"]}}})
            ans = mcp_call(server, step["call"]["name"], step["call"].get("arguments", {}))
            log(ev="answer", sid=sid, name=step["call"]["name"], **ans)
            last = ans.get("text", "")
            if token["cancelled"]:
                return
            update(sid, {"sessionUpdate": "tool_call_update", "toolCallId": tcid,
                         "status": "failed" if ans.get("isError") or ans.get("error") else "completed"})
        elif "text" in step:
            update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": step["text"]}})
        elif step.get("echo_last"):
            update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": last}})
        elif "silent" in step:
            time.sleep(step["silent"])
        elif step.get("foreign"):
            update(sid, {"sessionUpdate": "tool_call", "toolCallId": "native1", "status": "in_progress",
                         "title": "view_file"})
        elif "stderr" in step:
            sys.stderr.write(step["stderr"] + "\n")
            sys.stderr.flush()
    with lock:
        if token["done"]:
            return
        token["done"] = True
    emit({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


def main():
    for line in sys.stdin:
        msg = json.loads(line)
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            emit({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": 1, "agentCapabilities": {"mcpCapabilities": {"http": True}}}})
        elif method == "session/new":
            state["next_sid"] += 1
            sid = f"s{state['next_sid']}"
            state["sessions"][sid] = params["mcpServers"][0]
            log(ev="new", sid=sid, meta=params.get("_meta"))
            reply = {"jsonrpc": "2.0", "id": rid, "result": {"sessionId": sid}}
            if B.get("new_delay"):
                threading.Timer(float(B["new_delay"]), emit, args=(reply,)).start()
            else:
                emit(reply)
        elif method in ("session/set_model", "session/set_config_option"):
            emit({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "session/prompt":
            sid = params["sessionId"]
            text = params["prompt"][0]["text"]
            n = sum(1 for line in open(LOG) if '"ev": "prompt"' in line) if LOG and __import__("os").path.exists(LOG) else 0
            log(ev="prompt", sid=sid, n=n, text=text)
            script = B.get("script", [])
            steps = script[n] if n < len(script) else [{"text": "ok"}]
            token = {"cancelled": False, "done": False, "rid": rid}
            state["inflight"] = token
            threading.Thread(target=run_prompt, args=(rid, sid, text, steps, token), daemon=True).start()
        elif method == "session/cancel":
            token = state["inflight"]
            log(ev="cancel", sid=params.get("sessionId"))
            if token and not token["done"]:
                token["cancelled"] = True
                token["done"] = True
                emit({"jsonrpc": "2.0", "id": token["rid"], "result": {"stopReason": "cancelled"}})
        elif rid is not None and method:
            emit({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "nf"}})


main()
