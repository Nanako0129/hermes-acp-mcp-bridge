"""Ephemeral authenticated Streamable HTTP MCP server over loopback."""

from __future__ import annotations

import hmac
import http.server
import json
import secrets
import socketserver
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self
from urllib.parse import urlparse


class BridgeToolError(RuntimeError):
    """Raised when tool registration, authorization, or dispatch fails."""


@dataclass(frozen=True)
class ToolSpec:
    """Normalized MCP tool definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_mcp_dict(self) -> dict[str, Any]:
        schema = self.input_schema or {"type": "object", "properties": {}}
        if "type" not in schema:
            schema = {"type": "object", **schema}
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": schema,
        }


@dataclass(frozen=True)
class ToolCallRecord:
    """Observed execution telemetry for a single MCP ``tools/call``."""

    name: str
    arguments: dict[str, Any]
    result_text: str
    is_error: bool
    duration_seconds: float


ToolExecutor = Callable[[str, dict[str, Any]], Any]
ToolPolicyCheck = Callable[[str, dict[str, Any]], tuple[bool, str]]


def openai_tools_to_mcp(tools: Sequence[dict[str, Any]] | None) -> list[ToolSpec]:
    """Convert OpenAI-style ``tools`` payloads into MCP :class:`ToolSpec` items."""
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for entry in tools or ():
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if entry.get("type") == "function" else entry
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        params = fn.get("parameters")
        schema = dict(params) if isinstance(params, dict) else {"type": "object", "properties": {}}
        specs.append(
            ToolSpec(
                name=name,
                description=str(fn.get("description") or ""),
                input_schema=schema,
            )
        )
    return specs


def _normalize_tool_result(raw: Any) -> tuple[str, bool]:
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], bool):
        return _stringify(raw[0]), raw[1]
    if isinstance(raw, dict):
        is_err = bool(raw.get("error")) and len(raw) == 1
        return json.dumps(raw, ensure_ascii=False), is_err
    return _stringify(raw), False


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _is_loopback_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and parsed.hostname in (
        "127.0.0.1",
        "localhost",
        "::1",
    )


class _LoopbackHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamableHTTPMCPServer:
    """In-process loopback Streamable HTTP MCP server for one ACP bridge scope.

    Security invariants:
    - Binds strictly to ``127.0.0.1`` on an ephemeral OS-assigned port.
    - Requires ``Authorization: Bearer <per-instance-256-bit-token>``.
    - Rejects non-loopback ``Origin`` headers to block browser DNS rebinding.
    - Enforces an explicit allowlist of declared tools plus an optional policy hook.
    """

    PROTOCOL_VERSION = "2025-03-26"

    def __init__(
        self,
        *,
        tools: Sequence[ToolSpec],
        executor: ToolExecutor,
        policy_check: ToolPolicyCheck | None = None,
        server_name: str = "hermes-tools",
        server_version: str = "0.1.0",
        max_body_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.server_name = server_name
        self.server_version = server_version
        self._executor = executor
        self._policy_check = policy_check
        self._max_body_bytes = max_body_bytes
        self._token = secrets.token_urlsafe(32)
        self._tools: list[ToolSpec] = list(tools)
        self._tools_by_name: dict[str, ToolSpec] = {t.name: t for t in self._tools}
        self._lock = threading.Lock()
        self._records: list[ToolCallRecord] = []
        self._httpd: _LoopbackHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise BridgeToolError("MCP server has not been started")
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def recorded_calls(self) -> list[ToolCallRecord]:
        with self._lock:
            return list(self._records)

    def clear_records(self) -> None:
        with self._lock:
            self._records.clear()

    def update_tools(self, tools: Sequence[ToolSpec], executor: ToolExecutor | None = None) -> None:
        with self._lock:
            self._tools = list(tools)
            self._tools_by_name = {t.name: t for t in self._tools}
            if executor is not None:
                self._executor = executor

    def acp_mcp_server_config(self) -> dict[str, Any]:
        return {
            "type": "http",
            "name": self.server_name,
            "url": self.url,
            "headers": [{"name": "Authorization", "value": f"Bearer {self._token}"}],
        }

    def start(self) -> Self:
        if self._httpd is not None:
            return self
        bridge = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_empty(self, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _authorize(self) -> bool:
                origin = self.headers.get("Origin")
                if origin and not _is_loopback_origin(origin):
                    self._send_json(403, {"error": "forbidden origin"})
                    return False
                host = (self.headers.get("Host") or "").split(":")[0].lower()
                if host and host not in ("127.0.0.1", "localhost", "[::1]"):
                    self._send_json(403, {"error": "forbidden host"})
                    return False
                auth = self.headers.get("Authorization") or ""
                prefix = "Bearer "
                if not auth.startswith(prefix) or not hmac.compare_digest(
                    auth[len(prefix) :], bridge._token
                ):
                    self._send_json(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self) -> None:
                if not self._authorize():
                    return
                self._send_empty(405)

            def do_DELETE(self) -> None:
                if not self._authorize():
                    return
                self._send_empty(200)

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/mcp":
                    self._send_json(404, {"error": "not found"})
                    return
                if not self._authorize():
                    return
                try:
                    content_len = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    self._send_json(400, {"error": "invalid Content-Length"})
                    return
                if content_len <= 0 or content_len > bridge._max_body_bytes:
                    self._send_json(400, {"error": "invalid request body size"})
                    return
                raw = self.rfile.read(content_len)
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(
                        400,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        },
                    )
                    return
                if not isinstance(msg, dict):
                    self._send_json(
                        400,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"},
                        },
                    )
                    return

                req_id = msg.get("id")
                method = msg.get("method")
                if req_id is None:
                    # JSON-RPC notification (e.g. notifications/initialized)
                    self._send_empty(202)
                    return

                response = bridge._handle_rpc(req_id, method, msg.get("params"))
                self._send_json(200, response)

        self._httpd = _LoopbackHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"mcp-http-{self.server_name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _handle_rpc(self, req_id: Any, method: Any, params: Any) -> dict[str, Any]:
        if not isinstance(method, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.server_name, "version": self.server_version},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            with self._lock:
                tools_payload = [tool.to_mcp_dict() for tool in self._tools]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_payload}}
        if method == "tools/call":
            return self._handle_tool_call(req_id, params)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _handle_tool_call(self, req_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params"},
            }
        tool_name = str(params.get("name") or "").strip()
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Tool arguments must be an object"},
            }

        with self._lock:
            spec = self._tools_by_name.get(tool_name)
            executor = self._executor
        if spec is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Undeclared tool: {tool_name}"},
            }

        if self._policy_check is not None:
            allowed, reason = self._policy_check(tool_name, arguments)
            if not allowed:
                record = ToolCallRecord(
                    name=tool_name,
                    arguments=dict(arguments),
                    result_text=reason or f"Blocked by policy: {tool_name}",
                    is_error=True,
                    duration_seconds=0.0,
                )
                with self._lock:
                    self._records.append(record)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": record.result_text}],
                        "isError": True,
                    },
                }

        started = time.monotonic()
        try:
            raw_result = executor(tool_name, dict(arguments))
            text, is_error = _normalize_tool_result(raw_result)
        except Exception as exc:  # noqa: BLE001 - isolate arbitrary tool handler failures
            text, is_error = f"Tool execution error ({tool_name}): {exc}", True
        elapsed = time.monotonic() - started

        record = ToolCallRecord(
            name=tool_name,
            arguments=dict(arguments),
            result_text=text,
            is_error=is_error,
            duration_seconds=elapsed,
        )
        with self._lock:
            self._records.append(record)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()
