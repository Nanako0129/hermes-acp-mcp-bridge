"""Session-pinned ACP bridge executing multi-hop tools over native HTTP MCP."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass, field
from typing import Any

from .mcp_server import (
    BridgeToolError,
    StreamableHTTPMCPServer,
    ToolCallRecord,
    ToolExecutor,
    ToolPolicyCheck,
    ToolSpec,
)

PermissionDecider = Callable[[dict[str, Any], set[str], str], str | None]


@dataclass
class BridgeTurnResult:
    """Result of a single-turn (potentially multi-hop) ACP MCP execution."""

    text: str
    thought_text: str
    session_id: str
    warm_reused: bool
    prompt_calls_count: int
    tool_calls_executed: list[ToolCallRecord] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    model: str = ""


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _supports_http_mcp(init_result: dict[str, Any]) -> bool:
    caps = init_result.get("agentCapabilities")
    if not isinstance(caps, dict):
        return False
    mcp_caps = caps.get("mcpCapabilities")
    return isinstance(mcp_caps, dict) and mcp_caps.get("http") is True


def _supports_system_prompt_append(init_result: dict[str, Any]) -> bool:
    caps = init_result.get("agentCapabilities")
    if not isinstance(caps, dict):
        return False
    meta = caps.get("_meta")
    if not isinstance(meta, dict):
        return False
    ag = meta.get("antigravity")
    if not isinstance(ag, dict):
        return False
    sp = ag.get("systemPrompt")
    return isinstance(sp, dict) and sp.get("append") is True


def _default_permission_decider(
    params: dict[str, Any],
    allowed_tools: set[str],
    server_name: str,
) -> str | None:
    """Auto-allow only permission prompts targeting declared bridge MCP tools."""
    tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
    title = str(tool_call.get("title") or "")
    raw_input = tool_call.get("rawInput") if isinstance(tool_call.get("rawInput"), dict) else {}
    tool_name = str(
        tool_call.get("name")
        or raw_input.get("tool")
        or raw_input.get("name")
        or ""
    ).strip()

    is_bridge_tool = (
        tool_name in allowed_tools
        or any(title == name or title.startswith(f"{server_name}: {name}") for name in allowed_tools)
        or (server_name and f"mcp__{server_name}__" in tool_name)
    )
    options = params.get("options") or []
    if is_bridge_tool:
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") in ("allow_once", "allow_always"):
                return str(opt.get("optionId") or "")
    for opt in options:
        if isinstance(opt, dict) and opt.get("kind") in ("reject_once", "reject_always"):
            return str(opt.get("optionId") or "")
    return None


@dataclass
class _WarmEntry:
    proc: subprocess.Popen[str]
    inbox: queue.Queue[str | None]
    reader_thread: threading.Thread
    init_result: dict[str, Any]
    mcp_server: StreamableHTTPMCPServer
    bound_session_id: str = ""
    meta_fingerprint: str = ""
    tools_fingerprint: str = ""
    effective_model: str = ""
    last_used: float = field(default_factory=time.monotonic)


class ACPMCPBridge:
    """Manages warm ACP subprocesses paired with in-process Streamable HTTP MCP servers."""

    def __init__(
        self,
        *,
        command: str,
        args: Sequence[str] = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        warm_ttl_seconds: float = 300.0,
        require_http_mcp: bool = True,
        server_name: str = "hermes-tools",
        permission_decider: PermissionDecider | None = None,
    ) -> None:
        self._command = command
        self._args = tuple(args)
        self._cwd = cwd or os.getcwd()
        self._env = dict(env or {})
        self._warm_ttl_seconds = warm_ttl_seconds
        self._require_http_mcp = require_http_mcp
        self._server_name = server_name
        self._permission_decider = permission_decider or _default_permission_decider
        self._lock = threading.Lock()
        self._warm: dict[str, _WarmEntry] = {}
        self._req_counter = 0

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _spawn_entry(
        self,
        *,
        tools: Sequence[ToolSpec],
        executor: ToolExecutor,
        policy_check: ToolPolicyCheck | None,
        timeout_seconds: float,
    ) -> _WarmEntry:
        mcp_server = StreamableHTTPMCPServer(
            tools=tools,
            executor=executor,
            policy_check=policy_check,
            server_name=self._server_name,
        ).start()
        child_env = {**os.environ, **self._env}
        try:
            proc = subprocess.Popen(
                [self._command, *self._args],
                cwd=self._cwd,
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception:
            mcp_server.close()
            raise

        inbox: queue.Queue[str | None] = queue.Queue()

        def _pump_stdout() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    inbox.put(line)
            finally:
                inbox.put(None)

        thread = threading.Thread(target=_pump_stdout, daemon=True)
        thread.start()
        entry = _WarmEntry(
            proc=proc,
            inbox=inbox,
            reader_thread=thread,
            init_result={},
            mcp_server=mcp_server,
        )
        try:
            entry.init_result = self._rpc(
                entry,
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {},
                    "clientInfo": {"name": "hermes-acp-mcp-bridge", "version": "0.1.0"},
                },
                timeout_seconds=timeout_seconds,
            )
            if self._require_http_mcp and not _supports_http_mcp(entry.init_result):
                raise BridgeToolError(
                    "ACP server does not advertise agentCapabilities.mcpCapabilities.http=true"
                )
        except Exception:
            self._close_entry(entry)
            raise
        return entry

    def _close_entry(self, entry: _WarmEntry) -> None:
        try:
            entry.mcp_server.close()
        finally:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                if entry.proc.poll() is None:
                    entry.proc.terminate()
                    try:
                        entry.proc.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        entry.proc.kill()
                        entry.proc.wait(timeout=1.5)

    def _acquire(
        self,
        conversation_key: str,
        *,
        resume_session_id: str,
        tools: Sequence[ToolSpec],
        executor: ToolExecutor,
        policy_check: ToolPolicyCheck | None,
        timeout_seconds: float,
    ) -> tuple[_WarmEntry, bool]:
        now = time.monotonic()
        with self._lock:
            entry = self._warm.pop(conversation_key, None)
        if entry is not None:
            alive = entry.proc.poll() is None
            fresh = (now - entry.last_used) <= self._warm_ttl_seconds
            session_ok = not resume_session_id or entry.bound_session_id == resume_session_id
            if alive and fresh and session_ok:
                while True:
                    try:
                        entry.inbox.get_nowait()
                    except queue.Empty:
                        break
                entry.mcp_server.update_tools(tools, executor=executor)
                entry.mcp_server.clear_records()
                return entry, True
            self._close_entry(entry)
        return (
            self._spawn_entry(
                tools=tools,
                executor=executor,
                policy_check=policy_check,
                timeout_seconds=timeout_seconds,
            ),
            False,
        )

    def _park(self, conversation_key: str, entry: _WarmEntry) -> None:
        entry.last_used = time.monotonic()
        evicted: _WarmEntry | None = None
        with self._lock:
            evicted = self._warm.get(conversation_key)
            self._warm[conversation_key] = entry
        if evicted is not None and evicted is not entry:
            self._close_entry(evicted)

    def _send(self, entry: _WarmEntry, payload: dict[str, Any]) -> None:
        assert entry.proc.stdin is not None
        entry.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        entry.proc.stdin.flush()

    def _rpc(
        self,
        entry: _WarmEntry,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        req_id = self._next_id()
        self._send(entry, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"ACP request timed out waiting for {method}")
            try:
                raw_line = entry.inbox.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if entry.proc.poll() is not None:
                    raise BridgeToolError(f"ACP server exited unexpectedly during {method}") from None
                continue
            if raw_line is None:
                raise BridgeToolError(f"ACP server closed stdout during {method}")
            line = raw_line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("id") == req_id:
                if "error" in msg and msg["error"] is not None:
                    err = msg["error"]
                    raise BridgeToolError(f"ACP {method} error: {err}")
                result = msg.get("result")
                return result if isinstance(result, dict) else {}
            if "id" in msg and "method" in msg:
                # Server -> client RPC (e.g. session/request_permission)
                if msg["method"] == "session/request_permission":
                    option_id = self._permission_decider(
                        msg.get("params") or {},
                        allowed_tools or set(),
                        self._server_name,
                    )
                    outcome = (
                        {"outcome": "selected", "optionId": option_id}
                        if option_id
                        else {"outcome": "cancelled"}
                    )
                    self._send(
                        entry,
                        {"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}},
                    )
                else:
                    self._send(
                        entry,
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32601, "message": "Method not found"},
                        },
                    )
                continue
            if on_notification is not None:
                on_notification(msg)

    def run_turn(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        executor: ToolExecutor,
        policy_check: ToolPolicyCheck | None = None,
        conversation_key: str = "default",
        resume_session_id: str = "",
        model: str = "",
        timeout_seconds: float = 120.0,
    ) -> BridgeTurnResult:
        """Execute one user turn (including all internal MCP tool hops) in a single ``session/prompt``."""
        started = time.monotonic()
        entry, from_pool = self._acquire(
            conversation_key,
            resume_session_id=resume_session_id,
            tools=tools,
            executor=executor,
            policy_check=policy_check,
            timeout_seconds=timeout_seconds,
        )
        ok_to_park = False
        try:
            supports_sys_append = _supports_system_prompt_append(entry.init_result)
            meta: dict[str, Any] = {}
            if supports_sys_append and system_prompt.strip():
                meta = {"antigravity": {"systemPrompt": {"append": system_prompt}}}

            tools_fp = _hash_payload([t.to_mcp_dict() for t in tools])
            meta_fp = _hash_payload(meta)
            mcp_servers = [entry.mcp_server.acp_mcp_server_config()] if tools else []

            warm_same_session = (
                from_pool
                and bool(entry.bound_session_id)
                and (not resume_session_id or entry.bound_session_id == resume_session_id)
                and entry.meta_fingerprint == meta_fp
                and entry.tools_fingerprint == tools_fp
            )

            if warm_same_session:
                session_id = entry.bound_session_id
            elif resume_session_id:
                res = self._rpc(
                    entry,
                    "session/resume",
                    {
                        "sessionId": resume_session_id,
                        "cwd": self._cwd,
                        "mcpServers": mcp_servers,
                        **({"_meta": meta} if meta else {}),
                    },
                    timeout_seconds=timeout_seconds,
                )
                session_id = str(res.get("sessionId") or resume_session_id)
                entry.bound_session_id = session_id
                entry.meta_fingerprint = meta_fp
                entry.tools_fingerprint = tools_fp
            else:
                res = self._rpc(
                    entry,
                    "session/new",
                    {
                        "cwd": self._cwd,
                        "mcpServers": mcp_servers,
                        **({"_meta": meta} if meta else {}),
                    },
                    timeout_seconds=timeout_seconds,
                )
                session_id = str(res.get("sessionId") or "")
                if not session_id:
                    raise BridgeToolError("ACP session/new did not return a sessionId")
                entry.bound_session_id = session_id
                entry.meta_fingerprint = meta_fp
                entry.tools_fingerprint = tools_fp

            if model and not (warm_same_session and entry.effective_model == model):
                self._rpc(
                    entry,
                    "session/set_model",
                    {"sessionId": session_id, "modelId": model},
                    timeout_seconds=timeout_seconds,
                )
                entry.effective_model = model

            effective_prompt = user_prompt
            if system_prompt.strip() and not supports_sys_append and not warm_same_session and not resume_session_id:
                effective_prompt = f"{system_prompt}\n\n{user_prompt}"

            text_parts: list[str] = []
            thought_parts: list[str] = []
            usage: dict[str, Any] = {}

            def _on_update(msg: dict[str, Any]) -> None:
                if msg.get("method") != "session/update":
                    return
                params = msg.get("params") or {}
                upd = params.get("update") or {}
                kind = upd.get("sessionUpdate")
                content = upd.get("content") or {}
                if kind == "agent_message_chunk" and content.get("type") == "text":
                    text_parts.append(str(content.get("text") or ""))
                elif kind == "agent_thought_chunk" and content.get("type") == "text":
                    thought_parts.append(str(content.get("text") or ""))
                elif kind == "usage_update":
                    usage.update(upd)

            prompt_res = self._rpc(
                entry,
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": effective_prompt}],
                },
                timeout_seconds=timeout_seconds,
                on_notification=_on_update,
                allowed_tools={t.name for t in tools},
            )
            if isinstance(prompt_res.get("usage"), dict):
                usage.update(prompt_res["usage"])

            records = entry.mcp_server.recorded_calls
            ok_to_park = True
            return BridgeTurnResult(
                text="".join(text_parts),
                thought_text="".join(thought_parts),
                session_id=session_id,
                warm_reused=warm_same_session,
                prompt_calls_count=1,
                tool_calls_executed=records,
                usage=usage,
                elapsed_seconds=time.monotonic() - started,
                model=entry.effective_model or model,
            )
        finally:
            if ok_to_park and entry.proc.poll() is None:
                self._park(conversation_key, entry)
            else:
                self._close_entry(entry)

    def shutdown(self) -> None:
        with self._lock:
            entries = list(self._warm.values())
            self._warm.clear()
        for entry in entries:
            self._close_entry(entry)
