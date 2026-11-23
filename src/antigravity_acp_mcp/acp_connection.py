"""One ACP child process spoken to over NDJSON JSON-RPC 2.0 on stdio.

Security posture (PlanC components 1, H2, H3, L1):
- the child env is ``_build_subprocess_env()`` plus only the explicit mux control vars;
- stderr is untrusted: always drained, classified at capture time, summarised only by
  ``_acp_stderr_diagnostic`` -- raw stderr bytes never reach an exception or a log;
- request params (``session/new`` carries the MCP bearer) are never logged.
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ._upstream import _acp_stderr_diagnostic, _acp_stderr_retry_class, _build_subprocess_env

log = logging.getLogger(__name__)

MUX_ENV_KEYS = frozenset(
    {
        "ACP_MUX_AFFINITY",
        "ACP_MUX_SESSION_ID",
        "ACP_MUX_AFFINITY_PIN",
        "ACP_MUX_AFFINITY_TTL_SECONDS",
        "ACP_MUX_AFFINITY_MAX_ENTRIES",
    }
)
INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
    "clientInfo": {
        "name": "hermes-antigravity-acp-mcp",
        "title": "Hermes Antigravity ACP (MCP)",
        "version": "0.1.0",
    },
}
CANCELLED_OUTCOME = {"outcome": {"outcome": "cancelled"}}

_STDERR_KEEP_LINES = 200
_STDERR_LINE_CHARS = 4096
_UPDATE_BUFFER_LIMIT = 10_000

# on_permission(params) -> response result, or None to leave the request pending until
# ``cancel_session`` (or ``respond``) answers it.
PermissionHandler = Callable[[dict[str, Any]], dict[str, Any] | None]


class AcpError(RuntimeError):
    """Base class for ACP connection failures."""


class AcpProcessExit(AcpError):
    """The child exited. ``str()`` is the exit status plus the non-echoing stderr diagnostic."""

    def __init__(self, message: str, *, exit_status: int | None, retry_class: str | None) -> None:
        super().__init__(message)
        self.exit_status = exit_status
        self.retry_class = retry_class  # fixed class name or None, never child text


class AcpRpcError(AcpError):
    def __init__(self, code: Any, message: str) -> None:
        super().__init__(f"ACP error {code}: {message}")
        self.code = code


class _Pending:
    __slots__ = ("error", "event", "result")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: Exception | None = None


class AcpConnection:
    def __init__(
        self,
        command: str,
        args: Sequence[str],
        cwd: str,
        *,
        mux_env: Mapping[str, str] | None = None,
    ) -> None:
        unknown = set(mux_env or ()) - MUX_ENV_KEYS
        if unknown:
            raise ValueError(f"not a permitted mux control variable: {sorted(unknown)}")
        self._argv = [command, *args]
        self._cwd = cwd
        self._mux_env = dict(mux_env or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()  # guards the maps below; never held while calling out
        # Serialises session-update callbacks (order of chunks) without holding _lock. Callbacks must not block:
        # the stdout reader runs them, so a callback waiting on another lock would stall every pending response.
        self._dispatch_lock = threading.Lock()
        self._saw_rate_limit = False
        self._write_lock = threading.Lock()  # one NDJSON line at a time on stdin
        self._ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}
        self._updates: dict[str, collections.deque[dict[str, Any]]] = {}
        self._sessions: dict[str, tuple[Callable[[dict[str, Any]], None], PermissionHandler]] = {}
        self._pending_perms: dict[str, list[Any]] = {}
        self._stderr: collections.deque[str] = collections.deque(maxlen=_STDERR_KEEP_LINES)
        self._stderr_class: str | None = None
        self._exit_error: AcpProcessExit | None = None
        self._exited = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        env = _build_subprocess_env()
        env.update(self._mux_env)
        self._proc = subprocess.Popen(
            self._argv,
            cwd=self._cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._threads = [
            threading.Thread(target=self._read_stdout, name="acp-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name="acp-stderr", daemon=True),
        ]
        for t in self._threads:
            t.start()

    supports_images = False  # agentCapabilities.promptCapabilities.image from initialize

    def initialize(self, timeout: float | None = 30.0) -> dict[str, Any]:
        result = self.request("initialize", INITIALIZE_PARAMS, timeout=timeout)
        caps = (result or {}).get("agentCapabilities") or {} if isinstance(result, dict) else {}
        self.supports_images = bool((caps.get("promptCapabilities") or {}).get("image"))
        return result

    def close(self) -> None:
        """Reap the child: stdin EOF, then terminate, then kill."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        for stop in (None, proc.terminate, proc.kill):
            if stop:
                stop()
            try:
                proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                continue
        for t in self._threads:
            t.join(timeout=2.0)

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def stderr_retry_class(self) -> str | None:
        return self._stderr_class

    @property
    def saw_rate_limit(self) -> bool:
        """Sticky per child: any stderr line classified rate-limit (not only the first classified line)."""
        return self._saw_rate_limit

    @property
    def exited(self) -> bool:
        return self._exited.is_set()

    # -- client -> agent -----------------------------------------------------------

    def request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> Any:
        slot = _Pending()
        with self._lock:
            if self._exit_error is not None:
                raise self._exit_error
            rid = next(self._ids)
            self._pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        except OSError:
            self._exited.wait(5.0)  # the reader builds the typed error
            with self._lock:
                self._pending.pop(rid, None)
            raise self._exit_error or AcpError("ACP child stdin closed") from None
        if not slot.event.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"ACP request {method} timed out after {timeout}s")
        if slot.error is not None:
            raise slot.error
        return slot.result

    def notify(self, method: str, params: dict[str, Any]) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})
        except OSError:
            pass  # child gone: pending requests already fail with the typed exit error

    def respond(self, request_id: Any, result: Any) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except OSError:
            pass

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    # -- sessions ------------------------------------------------------------------

    def attach_session(
        self,
        session_id: str,
        on_update: Callable[[dict[str, Any]], None],
        on_permission: PermissionHandler,
    ) -> None:
        """Route one session's updates and permission requests; flush updates buffered so far.

        ``on_update`` runs on the stdout reader thread (and here for the flush) under the dispatch lock, never
        under the connection lock, so order is preserved: it must not block.
        """
        with self._dispatch_lock:
            with self._lock:
                self._sessions[session_id] = (on_update, on_permission)
                buffered = list(self._updates.pop(session_id, ()))
            for update in buffered:
                self._call_update(on_update, update)

    def detach_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._updates.pop(session_id, None)

    def cancel_session(self, session_id: str) -> None:
        """Answer every pending permission request ``cancelled``, then send session/cancel."""
        with self._lock:
            pending = self._pending_perms.pop(session_id, [])
        for request_id in pending:
            self.respond(request_id, CANCELLED_OUTCOME)
        self.notify("session/cancel", {"sessionId": session_id})

    # -- reader threads ------------------------------------------------------------

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict):
                self._dispatch(msg)
        self._on_exit()

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        # readline(limit) bounds a single huge line; the deque bounds the rest.
        for raw in iter(lambda: self._proc.stderr.readline(_STDERR_LINE_CHARS), b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            self._stderr.append(line)
            cls = _acp_stderr_retry_class([line])
            if self._stderr_class is None:
                self._stderr_class = cls
            if cls == "rate-limit":
                self._saw_rate_limit = True

    def _dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if method is None:  # response to our request
            with self._lock:
                slot = self._pending.pop(msg.get("id"), None)
            if slot is not None:
                if "error" in msg:
                    err = msg["error"] if isinstance(msg["error"], dict) else {}
                    slot.error = AcpRpcError(err.get("code"), str(err.get("message", "")))
                else:
                    slot.result = msg.get("result")
                slot.event.set()
        elif "id" in msg:
            self._handle_agent_request(msg["id"], method, msg.get("params") or {})
        elif method == "session/update":
            self._handle_update(msg.get("params") or {})

    def _handle_update(self, params: dict[str, Any]) -> None:
        sid = str(params.get("sessionId") or "")
        update = params.get("update")
        if not sid or not isinstance(update, dict):
            return
        with self._dispatch_lock:
            with self._lock:
                entry = self._sessions.get(sid)
                if entry is None:  # may precede the session/new response
                    buf = self._updates.setdefault(sid, collections.deque(maxlen=_UPDATE_BUFFER_LIMIT))
                    buf.append(update)
                    return
            self._call_update(entry[0], update)

    @staticmethod
    def _call_update(consumer: Callable[[dict[str, Any]], None], update: dict[str, Any]) -> None:
        try:
            consumer(update)
        except Exception:  # noqa: BLE001 - a consumer bug must not kill the reader thread
            log.exception("session update consumer failed")

    def _handle_agent_request(self, request_id: Any, method: str, params: dict[str, Any]) -> None:
        if method != "session/request_permission":
            self._send_error(request_id, -32601, f"Method not found: {method}")
            return
        sid = str(params.get("sessionId") or "")
        with self._lock:
            entry = self._sessions.get(sid)
        result: dict[str, Any] | None = CANCELLED_OUTCOME
        if entry is not None:
            try:
                result = entry[1](params)
            except Exception:  # noqa: BLE001 - a broken gate must fail closed, not hang the child
                log.exception("permission handler failed; answering cancelled")
                result = CANCELLED_OUTCOME
        if result is None:
            with self._lock:
                self._pending_perms.setdefault(sid, []).append(request_id)
        else:
            self.respond(request_id, result)

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        try:
            self._send(
                {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
            )
        except OSError:
            pass

    def _on_exit(self) -> None:
        proc = self._proc
        assert proc is not None
        try:
            status = proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            status = proc.wait()
        self._threads[1].join(timeout=2.0)  # let stderr finish so the diagnostic is complete
        error = AcpProcessExit(
            f"ACP child exited with status {status}; {_acp_stderr_diagnostic(list(self._stderr))}",
            exit_status=status,
            retry_class=self._stderr_class or _acp_stderr_retry_class(list(self._stderr)),
        )
        with self._lock:
            self._exit_error = error
            pending, self._pending = list(self._pending.values()), {}
        for slot in pending:
            slot.error = error
            slot.event.set()
        self._exited.set()
