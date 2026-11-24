"""PlanC components 5-8: per-conversation turn state, interrupt/deadline/abandonment, liveness.

One ``Conversation`` per Hermes conversation key (process-global registry). It owns one warm agy child (through
the account mux) and, per ACP session, one parked MCP endpoint. ``run()`` serves one ``chat.completions.create``:
it classifies the request with ``matcher.classify`` and either answers parked MCP calls and keeps reading the same
prompt (``resume_held``), sends a delta on the same idle session (``continue``), or retires the session and starts
a fresh one with a full text replay (``replay``). The plugin never executes a tool: a parked ``tools/call`` is
returned to Hermes as ``tool_calls`` and answered with the tool row Hermes sends next.

Concurrency model:
- Every ACP session has a generation number. Events carry it; anything from an older generation is dropped (a
  stale ``tools/call`` is answered ``isError`` at once and never handed to Hermes).
- ``self.lock`` guards state only and is never held while waiting on agy (spawn, session/new and the prompt run
  without it), so ``cancel()`` always gets it quickly.
- Callbacks from the connection reader / MCP threads never take ``self.lock`` except the deadline path, which is
  safe because those threads hold no connection lock.
Rule (plan component 4): a cancelled session is never prompted again; the next request starts a fresh session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import secrets
import shutil
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .acp_connection import AcpConnection, AcpError
from .matcher import (
    TurnView,
    IMAGE_PART_TYPES,
    _text,
    canonical_args,
    classify,
    explain_mismatch,
    fingerprint,
    images,
    project,
    trailing_tool_round,
    view_after,
)
from .parked_mcp import DEFAULT_DEADLINE_SECONDS, ParkedCall, ParkedMcpServer
from .permission import PermissionGate
from .session import AcpSession, new_session

logger = logging.getLogger(__name__)

STEP_DEADLINE_SECONDS = 90.0  # per model step (no ACP activity); no _await runs while Hermes executes a tool
COLLECT_WINDOW_SECONDS = 0.3  # gather near-simultaneous tools/call into one tool_calls (agy is sequential: S1b)
IDLE_TTL_SECONDS = 1800.0  # an idle conversation's warm agy child is reaped after this
GUARD_TEXT = "already executed; its result is in the conversation above"
NUDGE_TEXT = ("Your last reply was only {reply!r}, which looks cut off. Reply again with your complete answer to the "
              "user's latest request.")
GUARD_LOOP_TEXT = "The result of the last step is already in the conversation above; send 'continue' to proceed."
_SYS_MSG_RE = re.compile(r"<system-message>\n(.*?)\n</system-message>", re.DOTALL)
_SYS_MSG_DEDUPE_MIN_CHARS = 200
_SYS_MSG_OMITTED = "<system-message>(unchanged from earlier in this session)</system-message>"
_SYS_MSG_LATER = "<system-message>(same text is repeated in full later in this prompt)</system-message>"


def _sys_digests(text: str) -> set[str]:
    return {hashlib.sha256(m.group(1).encode("utf-8")).hexdigest() for m in _SYS_MSG_RE.finditer(text)
            if len(m.group(1)) >= _SYS_MSG_DEDUPE_MIN_CHARS}


def dedupe_system_messages(text: str, seen: set[str] | None, later: set[str] | frozenset[str] = frozenset()) -> str:
    """Replace large ``<system-message>`` blocks already sent in this ACP session with a one-line placeholder.

    ``later``: digests whose full copy appears later in the same prompt (replay renders ACTIVE_REQUEST and its
    progress before history, so the copy the model must act on stays complete and history copies point forward).
    Only called on outgoing text rendered for agy (never on matcher inputs), so prefix fingerprints stay exact.
    """
    if seen is None or "<system-message>" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if len(body) < _SYS_MSG_DEDUPE_MIN_CHARS:
            return m.group(0)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest in later:
            return _SYS_MSG_LATER
        if digest in seen:
            return _SYS_MSG_OMITTED
        seen.add(digest)
        return m.group(0)

    return _SYS_MSG_RE.sub(repl, text)


def degenerate(text: str) -> bool:
    """A final reply made only of punctuation (e.g. a lone ``}``): Gemini sometimes ends a long tool turn this way
    (seen 2026-11-23: 47 output tokens, content ``}``). Empty text is left to Hermes' own empty-reply handling."""
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= 8 and all(unicodedata.category(c).startswith("P") for c in stripped)


class ProviderError(RuntimeError):
    """Typed failure surfaced to Hermes; text never carries child stderr bytes."""


class CancelledByUser(ProviderError):
    """Hermes interrupted the request (``client.cancel()``); not retryable."""


class RateLimited(ProviderError):
    """agy's harness reported a 429 / RESOURCE_EXHAUSTED; not retried on another account."""


class StepTimeout(ProviderError):
    """No ACP activity for a whole model step."""


def mcp_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out, seen = [], set()
    for entry in tools or []:
        fn = entry.get("function") if isinstance(entry, dict) and entry.get("type") == "function" else entry
        if not isinstance(fn, dict) or not fn.get("name") or fn["name"] in seen:
            continue
        seen.add(fn["name"])
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"}
        out.append({"name": fn["name"], "description": str(fn.get("description") or ""), "inputSchema": params})
    return out


MAX_PROMPT_IMAGES = 8  # ponytail: newest N attached; Hermes already replays only its recent window (default 3)


class _Images:
    """Numbers inline images in render order for one ACP session (same bytes = same number, numbering continues
    across prompts on that session) and builds each prompt's blocks with the images it has not attached yet.

    ``text`` marks image positions with a per-instance random sentinel, never with the visible ``[image #n]``: a user,
    the model or a tool can type that string, and it must not decide where an image lands. ``blocks`` turns every
    sentinel into the visible marker and puts each newly attached image right after its own sentinel."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self._index: dict[str, int] = {}
        self._sent = 0
        self._tag = "\x1eIMG" + secrets.token_hex(6) + ":"
        self._sentinel = re.compile(re.escape(self._tag) + r"(\d+)\x1f")

    def text(self, content: Any) -> str:
        if not isinstance(content, list) or not images(content):
            return _text(content)
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") in IMAGE_PART_TYPES:
                img = images([part])
                if not img:
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    parts.append(f"[image at {str(url)[:500]} (not attached; use a vision tool to view it)]" if url
                                 else "[image]")
                    continue
                n = self._index.get(img[0][1])
                if n is None:
                    self.items.append(img[0])
                    n = self._index[img[0][1]] = len(self.items)
                parts.append(f"{self._tag}{n}\x1f")
            else:
                parts.append(str(part.get("text") or part.get("type") or ""))
        return "\n".join(parts)

    def blocks(self, text: str) -> list[dict[str, Any]]:
        """Prompt blocks: each newly attached image sits right after its own marker, so the prompt keeps its order
        and its closing instruction stays last (images appended at the end were read as the request)."""
        first = max(self._sent, len(self.items) - MAX_PROMPT_IMAGES)
        note = ""
        if first > self._sent:
            skipped = f"#{self._sent + 1}" + (f"-#{first}" if first > self._sent + 1 else "")
            note = f"(image {skipped} not attached: only the newest {MAX_PROMPT_IMAGES} are)\n\n"
        out: list[dict[str, Any]] = []
        buf, pos, attached = note, 0, set()
        for m in self._sentinel.finditer(text):
            n = int(m.group(1))
            buf += text[pos:m.start()] + f"[image #{n}]"
            pos = m.end()
            if first < n <= len(self.items) and n not in attached:
                out.append({"type": "text", "text": buf})
                mime, data = self.items[n - 1]
                out.append({"type": "image", "mimeType": mime, "data": data})
                attached.add(n)
                buf = ""
        buf += text[pos:]
        if buf or not out:
            out.append({"type": "text", "text": buf})
        self._sent = len(self.items)
        return out


def render_replay(messages: list[dict[str, Any]], imgs: _Images | None = None,
                  fence: str | None = None, seen_sys: set[str] | None = None) -> list[dict[str, Any]]:
    """Full replay for a fresh session as ACP prompt blocks; the system band rides at the top (no append).

    ``fence`` wraps tool output; a conversation passes its own (random once, then stable) so successive replays share
    a byte-identical prefix, which is meant to let the model provider's prefix cache hit (not measured)."""
    msgs = [m for m in messages if isinstance(m, dict)]
    fence = fence or "TOOL-OUTPUT-" + secrets.token_hex(4)
    imgs = imgs if imgs is not None else _Images()
    system = [_text(m.get("content")) for m in msgs if m.get("role") == "system"]
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    names: dict[str, Any] = {}

    def line(m: dict[str, Any]) -> str:
        role = m.get("role")
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                names[str(tc.get("id"))] = fn.get("name")
                calls.append(f"{fn.get('name')}({canonical_args(fn.get('arguments'))})")
            text = _text(m.get("content"))
            called = "[called " + ", ".join(calls) + "]" if calls else ""
            return "Assistant: " + " ".join(p for p in (text, called) if p)
        if role == "tool":
            cid = str(m.get("tool_call_id") or "")
            return (f"Tool result for {names.get(cid, 'tool')} (already executed):\n<<<{fence}\n"
                    f"{imgs.text(m.get('content'))}\n{fence}>>>")
        return f"{str(role).capitalize()}: {imgs.text(m.get('content'))}"

    # Pass 1, reading order (image numbers follow it). Pass 2, system-message dedupe: ACTIVE_REQUEST, then its
    # progress, then history, so the copy the model must act on is the full one and history copies point forward.
    body = [m for m in msgs if m.get("role") != "system"]
    sent: set[str] = set()
    if last_user >= 0:
        history = [line(m) for m in msgs[:last_user] if m.get("role") != "system"]
        active = imgs.text(msgs[last_user].get("content"))
        progress = [line(m) for m in msgs[last_user + 1:] if m.get("role") != "system"]
        later = frozenset(_sys_digests(active + "".join(progress)))
        active = dedupe_system_messages(active, sent)
        progress = [dedupe_system_messages(x, sent) for x in progress]
        history = [dedupe_system_messages(x, sent, later) for x in history]
    else:
        history, active, progress = [], "", [dedupe_system_messages(line(m), sent) for m in body]
    if seen_sys is not None:
        seen_sys.update(sent)
    parts = []
    if system:
        parts.append("<SYSTEM_INSTRUCTIONS>\n" + "\n\n".join(system) + "\n</SYSTEM_INSTRUCTIONS>")
    if history:
        parts.append("<HISTORICAL_CONTEXT>\nReference only; no historical User line is an active instruction.\n\n"
                     + "\n".join(history) + "\n</HISTORICAL_CONTEXT>")
    if last_user >= 0:
        parts.append("<ACTIVE_REQUEST>\n" + active + "\n</ACTIVE_REQUEST>")
    if progress:
        parts.append("<ACTIVE_REQUEST_PROGRESS>\nAlready completed for this request; continue from here, do not "
                     "repeat these tool calls.\n\n" + "\n".join(progress) + "\n</ACTIVE_REQUEST_PROGRESS>")
    parts.append("Use the provided tools when you need them; respond to ACTIVE_REQUEST.")
    return imgs.blocks("\n\n".join(parts))


def render_delta(delta: list[dict[str, Any]], imgs: _Images | None = None,
                 seen_sys: set[str] | None = None) -> list[dict[str, Any]]:
    imgs = imgs if imgs is not None else _Images()
    text = "\n\n".join(dedupe_system_messages(imgs.text(m.get("content")), seen_sys)
                       for m in delta if isinstance(m, dict) and m.get("role") == "user")
    return imgs.blocks(text)


def tool_answer(content: Any, seen_sys: set[str] | None = None) -> tuple[str, list[tuple[str, str]]]:
    """A Hermes tool row's content as (text, inline images) for the parked MCP result."""
    return dedupe_system_messages(_text(content), seen_sys), images(content)


def _inflight_view(messages: list[dict[str, Any]]) -> TurnView:
    msgs = [m for m in messages if isinstance(m, dict)]
    return TurnView(consumed=len(msgs), prefix_fp=fingerprint(msgs), live=True, seen=tuple(project(m) for m in msgs))


class Conversation:
    def __init__(self, key: str, spawn: Callable[[str, str], AcpConnection], *, stateless: bool = False,
                 step_deadline: float = STEP_DEADLINE_SECONDS, call_deadline: float = DEFAULT_DEADLINE_SECONDS,
                 terminal_tools: frozenset[str] = frozenset(),
                 collect_window: float = COLLECT_WINDOW_SECONDS,
                 initial_conn: AcpConnection | None = None, cwd: str | None = None, owns_cwd: bool = False):
        self.key = key
        self._spawn = spawn  # (key, cwd) -> started + initialized AcpConnection
        self.stateless = stateless
        self.step_deadline = step_deadline
        self.call_deadline = call_deadline
        self.terminal_tools = frozenset(terminal_tools)
        self.collect_window = max(0.0, float(collect_window))
        # Only a directory this conversation created is deleted on close; a caller's cwd (or initial_conn's) is not.
        # close() deletes the cwd only if this conversation created it, or the caller hands ownership over
        # (owns_cwd=True, e.g. a pool's pre-warmed temp dir); a caller's own workspace is never deleted.
        self.cwd = cwd or (initial_conn.cwd if initial_conn is not None else None)
        self._owns_cwd = self.cwd is None or owns_cwd
        if self.cwd is None:
            self.cwd = tempfile.mkdtemp(prefix="acp-mcp-cwd-")  # empty dedicated dir (H1)
        self.lock = threading.RLock()  # state only; never held while waiting on agy
        self._spawn_lock = threading.Lock()  # serialises child spawn (outside self.lock)
        self.events: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self.gen = 0
        self.retire_reason = ""
        self.conn: AcpConnection | None = initial_conn
        self.server: ParkedMcpServer | None = None
        self.session: AcpSession | None = None
        self.prompt_thread: threading.Thread | None = None
        self.tools_fp = ""
        self.model = ""
        self.view = TurnView()
        self.handed: list[ParkedCall] = []
        self.guard: set[tuple[str, str]] = set()
        self.guard_refusals = 0
        self.owner_token: threading.Event | None = None
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.usage: dict[str, int] | None = None
        self.images = _Images()  # image numbering of the current ACP session
        self._seen_sys_msgs: set[str] = set()  # sha256 of large <system-message> blocks already sent in this session
        self.fence = "TOOL-OUTPUT-" + secrets.token_hex(4)  # unguessable to tool output, stable across replays
        self.nudged = False  # one re-ask per user turn after a degenerate final reply
        self.last_activity = time.monotonic()
        self.last_used = time.monotonic()
        self.closed = False

    # ---- public ---------------------------------------------------------------------------------------------

    def run(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str | None,
            token: threading.Event | None = None, thinking: str | None = None) -> dict:
        """Serve one create. Returns {"text", "reasoning", "tool_calls": [ParkedCall], "finish_reason"}."""
        started = self.begin(messages, tools, model, token, thinking)
        return started if "guest" in started else self.finish(started)

    def begin(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str | None,
              token: threading.Event | None = None, thinking: str | None = None,
              only: tuple[str, ...] | None = None) -> dict:
        """Classify and claim this conversation for one create without waiting on agy.

        Returns a handle for ``finish()``, ``{"guest": True}``, or ``{"declined": kind}`` when ``only`` is given and
        the decision is not in it (no state is touched then). Once a handle is returned the view is live, so a
        concurrent caller classifying this conversation sees it busy (the gateway's branch pool relies on this).
        """
        token = token or threading.Event()
        if token.is_set():
            raise CancelledByUser("request cancelled")
        tools_fp = json.dumps(mcp_tools(tools), sort_keys=True)
        need_open = False
        with self.lock:
            decision = classify(messages, self.view, tools)
            why = ""
            if decision.kind == "continue" and tools_fp != self.tools_fp:
                decision.kind, why = "replay", "tool list changed"  # fixed per session
            elif decision.kind == "continue" and f"{model or ''}|{thinking or ''}" != self.model:
                decision.kind, why = "replay", "model or thinking level changed"
            if decision.kind == "resume_held" and self.session is not None and not self.handed and self.view.live:
                # The held calls were already answered and that step is still running: this is a duplicate of the
                # request being served. Never retire the live step for it (that would fail the original request):
                # a pool caller (``only=``) moves it to another branch, a direct caller gets an error.
                if only is not None:
                    return {"declined": "duplicate", "why": "held calls already answered; step still running"}
                raise ProviderError("duplicate tool-result request while the step it answers is still running")
            if decision.kind == "resume_held" and (self.session is None or not self.handed):
                decision.kind = "replay"  # no session, or the held calls were already answered
            if decision.kind == "replay" and self.view.consumed and not why:
                why = ("session retired (" + self.retire_reason + ")" if self.view.retired
                       else explain_mismatch(messages, list(self.view.seen)))
            if only is not None and decision.kind not in only:
                return {"declined": decision.kind, "why": why}
            self.last_used = time.monotonic()
            logger.info("acp-mcp turn: %s%s", decision.kind, f" ({why})" if why else "")
            if decision.kind == "guest":
                return {"guest": True}
            self.owner_token = token
            if decision.kind == "resume_held":
                gen = self.gen
                for call in self.handed:
                    call.answer(*tool_answer(decision.tool_results.get(call.id, ""), self._seen_sys_msgs))
                self.handed = []
                self.last_activity = time.monotonic()
                self.text, self.reasoning, self.usage = [], [], None
            elif decision.kind == "continue":
                gen = self.gen
                self.nudged = False
                self.guard, self.guard_refusals = set(), 0
                self.view = _inflight_view(messages)
                self._start_prompt(gen, render_delta(decision.delta, self.images, self._seen_sys_msgs))
            else:  # supersede / replay
                self._retire("superseded" if decision.kind == "supersede" else "replay")
                gen = self.gen
                self.nudged = False
                self.view = _inflight_view(messages)  # live while opening: a concurrent /btw is a guest (F2)
                need_open = True
        return {"gen": gen, "need_open": need_open, "messages": messages, "tools": tools, "model": model,
                "tools_fp": tools_fp, "token": token, "thinking": thinking}

    def finish(self, h: dict, on_open: Callable[[], None] | None = None) -> dict:
        """Open the session if ``begin`` asked for it, then wait for the step result. ``on_open`` runs once the open
        attempt is over (success or failure); the gateway uses it to release its child-cap slot."""
        try:
            if h["need_open"]:
                self._open_interruptibly(h["gen"], h["messages"], h["tools"], h["model"], h["tools_fp"], h["token"],
                                         h["thinking"])
        finally:
            if on_open is not None:
                on_open()
        return self._await(h["messages"], h["gen"], h["token"])

    def _open_interruptibly(self, gen: int, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                            model: str | None, tools_fp: str, token: threading.Event,
                            thinking: str | None = None) -> None:
        """Spawn/initialize/session-new can take tens of seconds: run them off-thread so cancel() returns within
        a poll interval (F1). A late-finishing open sees the generation moved on and cleans itself up."""
        done: dict[str, BaseException | None] = {}

        def opener() -> None:
            try:
                self._open_and_prompt(gen, messages, tools, model, tools_fp, token, thinking)
                done["error"] = None
            except BaseException as exc:  # noqa: BLE001 - re-raised on the create thread
                done["error"] = exc

        th = threading.Thread(target=opener, name=f"acp-open-{gen}", daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.2)
            if token.is_set():
                raise self._fail(gen, "cancelled", CancelledByUser("request cancelled"))
            if self.gen != gen and th.is_alive():
                raise ProviderError("superseded by a newer request")
        error = done.get("error")
        if error is not None:
            with self.lock:
                if self.gen == gen:
                    self._retire("open-failed")  # not live any more: reapable, no phantom guest window
            if isinstance(error, ProviderError):
                raise error
            raise ProviderError(f"could not open an Antigravity session: {type(error).__name__}: {error}") from None

    def cancel(self, token: threading.Event | None = None) -> None:
        """Hermes interrupt (request-scoped facade ``cancel()``): fast, never waits on agy."""
        if token is not None:
            token.set()
        with self.lock:
            if token is None or self.owner_token is token:
                self._retire("cancelled")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self._retire("close")
            conn, self.conn = self.conn, None
        try:
            if conn is not None:
                conn.close()
        finally:
            if self._owns_cwd:
                shutil.rmtree(self.cwd, ignore_errors=True)

    @property
    def live(self) -> bool:
        return self.view.live

    # ---- session lifecycle -----------------------------------------------------------------------------------

    def _connection(self) -> AcpConnection:
        with self._spawn_lock:
            with self.lock:
                conn = self.conn
            if conn is not None and not conn.exited:
                return conn
            if conn is not None:
                conn.close()
            conn = self._spawn(self.key, self.cwd)
            with self.lock:
                if not self.closed:
                    self.conn = conn
                    return conn
            conn.close()  # the conversation was closed while this child was spawning (e.g. a cancelled guest)
            raise CancelledByUser("conversation closed while starting Antigravity")

    def _open_and_prompt(self, gen: int, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                         model: str | None, tools_fp: str, token: threading.Event,
                         thinking: str | None = None) -> None:
        """Open a fresh session for generation ``gen`` without holding self.lock, then install it."""
        conn = self._connection()
        server = ParkedMcpServer(mcp_tools(tools), lambda call: self._on_call(gen, call), deadline=self.call_deadline,
                                 on_deadline=lambda call: self._on_deadline(gen, call)).start()
        gate = PermissionGate(server.server_name, server.tool_names,
                              on_foreign_tool_call=lambda _u: self.events.put((gen, "foreign", None)))
        try:
            session = new_session(conn, mcp_server=server.acp_mcp_server_config(), gate=gate, model=model,
                                  thinking=thinking)
        except Exception:
            server.close()
            raise
        abort: BaseException | None = None
        with self.lock:
            if self.gen != gen or token.is_set():  # cancelled or superseded while opening
                abort = CancelledByUser("request cancelled") if token.is_set() else ProviderError(
                    "superseded by a newer request")
            else:
                self.server, self.session = server, session
                self.tools_fp, self.model = tools_fp, f"{model or ''}|{thinking or ''}"
                self.guard, self.guard_refusals = trailing_tool_round(messages), 0
                self.view = _inflight_view(messages)
                self.images = _Images()
                self._seen_sys_msgs = set()
                blocks = render_replay(messages, self.images, self.fence, self._seen_sys_msgs)
                logger.info("acp-mcp open: model=%s thinking=%s images=%d", model or "-", thinking or "-",
                            sum(1 for b in blocks if b.get("type") == "image"))
                self._start_prompt(gen, blocks)
        if abort is not None:
            server.retire()
            server.close()
            session.close()  # outside self.lock: session/close may wait on agy
            raise abort

    def _start_prompt(self, gen: int, blocks: list[dict[str, Any]]) -> None:  # lock held
        session = self.session
        assert session is not None
        self.text, self.reasoning, self.usage = [], [], None
        self.last_activity = time.monotonic()

        def worker() -> None:
            try:
                self.events.put((gen, "end", session.prompt(blocks, timeout=None)))
            except Exception as exc:  # noqa: BLE001 - surfaced to the waiting create as a typed error
                self.events.put((gen, "error", exc))

        self.prompt_thread = threading.Thread(target=worker, name=f"acp-prompt-{gen}", daemon=True)
        self.prompt_thread.start()

    def _retire(self, reason: str) -> None:
        """Retire the current session without blocking (lock held). Always starts a new generation."""
        session, server, thread = self.session, self.server, self.prompt_thread
        self.session, self.server, self.prompt_thread = None, None, None
        self.gen += 1
        self.retire_reason = reason
        self.handed = []
        self.guard, self.guard_refusals = set(), 0
        self.view = replace(self.view, live=False, handed_out=(), retired=True)
        if session is None:
            return
        logger.info("acp-mcp: retiring session (%s)", reason)
        if server is not None:
            server.retire()
        try:
            session.cancel()
        except (AcpError, OSError):
            pass

        def release() -> None:  # M4: parked calls are released only after the prompt ended
            if thread is not None:
                thread.join(10.0)
            if server is not None:
                server.release_parked(reason, drop_connection=True)
                server.close()
            session.close()

        threading.Thread(target=release, name="acp-retire", daemon=True).start()

    # ---- callbacks (connection reader / MCP HTTP threads) -----------------------------------------------------

    def _on_call(self, gen: int, call: ParkedCall) -> None:
        if gen != self.gen:
            call.answer("session retired", is_error=True)  # straggler from a retired session (M1)
            return
        if (call.name, canonical_args(call.arguments)) in self.guard:
            self.guard_refusals += 1
            call.answer(GUARD_TEXT, is_error=True)  # replay guard: never handed to Hermes twice
            if self.guard_refusals >= 2:
                self.events.put((gen, "guard-loop", None))
            return
        self.last_activity = time.monotonic()
        self.events.put((gen, "call", call))

    def _on_deadline(self, gen: int, call: ParkedCall) -> None:  # before agy sees the isError (M4 order)
        with self.lock:
            if gen == self.gen:
                self._retire("deadline")

    # ---- waiting ---------------------------------------------------------------------------------------------

    def _drain_updates(self, gen: int) -> None:
        with self.lock:
            session = self.session if self.gen == gen else None
        if session is None:
            return
        while True:
            try:
                u = session.updates.get_nowait()
            except queue.Empty:
                return
            self.last_activity = time.monotonic()
            kind = u.get("sessionUpdate")
            content = u.get("content") or {}
            if kind == "agent_message_chunk" and content.get("type", "text") == "text":
                self.text.append(str(content.get("text") or ""))
            elif kind == "agent_thought_chunk" and content.get("type", "text") == "text":
                self.reasoning.append(str(content.get("text") or ""))
            elif kind == "usage_update":
                usage: dict[str, int] = dict(self.usage or {})
                for src_k, dst_k in (("used", "used"), ("size", "size"), ("inputTokens", "input_tokens"),
                                     ("outputTokens", "output_tokens"), ("input_tokens", "input_tokens"),
                                     ("output_tokens", "output_tokens")):
                    val = u.get(src_k)
                    if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                        usage[dst_k] = val
                if usage:
                    self.usage = usage

    def _fail(self, gen: int, reason: str, exc: ProviderError, *, drop_conn: bool = False) -> ProviderError:
        with self.lock:
            if self.gen == gen:
                self._retire(reason)
            conn = None
            if drop_conn or (self.conn is not None and self.conn.exited):
                conn, self.conn = self.conn, None
        if conn is not None:
            conn.close()  # a fresh child next time (clears a sticky rate-limit flag, a dead process)
        return exc

    def _await(self, messages: list[dict[str, Any]], gen: int, token: threading.Event) -> dict:
        while True:
            if token.is_set():
                raise self._fail(gen, "cancelled", CancelledByUser("request cancelled"))
            if self.gen != gen:
                raise ProviderError(f"Antigravity session ended ({self.retire_reason})")
            self._drain_updates(gen)
            conn = self.conn
            if conn is not None and conn.saw_rate_limit:
                raise self._fail(gen, "rate-limit", RateLimited("Antigravity reported 429 RESOURCE_EXHAUSTED; "
                                                               "not retried"), drop_conn=True)
            try:
                egen, kind, payload = self.events.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() - self.last_activity > self.step_deadline:
                    raise self._fail(gen, "step-timeout", StepTimeout(
                        f"no Antigravity activity for {self.step_deadline:.0f}s")) from None
                continue
            if egen != gen:
                if kind == "call":
                    payload.answer("session retired", is_error=True)
                continue
            if kind == "call":
                calls = [payload] if payload.state == "parked" else []
                until = time.monotonic() + self.collect_window
                while (left := until - time.monotonic()) > 0:
                    try:
                        g2, k2, p2 = self.events.get(timeout=left)
                    except queue.Empty:
                        break
                    if g2 == gen and k2 == "call":
                        if p2.state == "parked":
                            calls.append(p2)
                    elif g2 == gen:
                        self.events.put((g2, k2, p2))
                        break
                if not calls:
                    continue
                self._drain_updates(gen)
                with self.lock:
                    if self.gen != gen or token.is_set():
                        continue  # retired/cancelled meanwhile: the loop top raises
                    self.handed = calls
                    self.view = view_after(messages, handed_out=tuple(c.id for c in calls), live=True)
                    if self.stateless:
                        self._retire("stateless-handout")  # children continue through replay (component 9)
                    elif self.terminal_tools and all(c.name in self.terminal_tools for c in calls):
                        self._retire("terminal-handout")  # caller ends on this tool (e.g. SubagentHandback)
                return {"text": "".join(self.text), "reasoning": "".join(self.reasoning), "tool_calls": calls,
                        "finish_reason": "tool_calls", "usage": dict(self.usage) if self.usage else None}
            if kind == "end":
                self._drain_updates(gen)
                if payload.stop_reason == "cancelled":
                    raise self._fail(gen, "prompt-cancelled", CancelledByUser("prompt cancelled"))
                with self.lock:
                    if self.gen != gen:
                        continue
                    reply = "".join(self.text)
                    if not self.nudged and payload.stop_reason == "end_turn" and degenerate(reply):
                        self.nudged = True
                        logger.info("acp-mcp: degenerate final reply (%d chars); asking once more", len(reply))
                        self._start_prompt(gen, [{"type": "text", "text": NUDGE_TEXT.format(reply=reply.strip())}])
                        continue
                    self.guard, self.guard_refusals = set(), 0
                    self.view = view_after(messages)
                return {"text": "".join(self.text), "reasoning": "".join(self.reasoning), "tool_calls": [],
                        "finish_reason": payload.finish_reason, "usage": dict(self.usage) if self.usage else None}
            if kind == "guard-loop":
                self._drain_updates(gen)
                with self.lock:
                    if self.gen == gen:
                        self._retire("guard-loop")
                    self.view = replace(view_after(messages), retired=True)
                return {"text": GUARD_LOOP_TEXT, "reasoning": "", "tool_calls": [], "finish_reason": "stop",
                        "usage": dict(self.usage) if self.usage else None}
            if kind == "foreign":
                raise self._fail(gen, "native-tool", ProviderError("agy attempted a native tool; session cancelled"),
                                 drop_conn=True)
            # error
            raise self._fail(gen, "error", ProviderError(
                f"Antigravity ACP request failed: {type(payload).__name__}: {payload}"))


class Registry:
    """Process-global conversations keyed by the Hermes conversation key; idle ones are reaped."""

    def __init__(self, spawn: Callable[[str, str], AcpConnection], idle_ttl: float = IDLE_TTL_SECONDS):
        self._spawn = spawn
        self._idle_ttl = idle_ttl
        self._lock = threading.Lock()
        self._conversations: dict[str, Conversation] = {}

    def get(self, key: str) -> Conversation:
        if not key:
            return Conversation("", self._spawn, stateless=True)
        now = time.monotonic()
        with self._lock:
            idle = [k for k, c in self._conversations.items()
                    if k != key and not c.live and now - c.last_used > self._idle_ttl]
            reaped = [self._conversations.pop(k) for k in idle]
            conv = self._conversations.get(key)
            if conv is None:
                conv = self._conversations[key] = Conversation(key, self._spawn)
        for c in reaped:
            c.close()
        return conv

    def guest(self) -> Conversation:
        return Conversation("", self._spawn, stateless=True)

    def close_all(self) -> None:
        with self._lock:
            convs, self._conversations = list(self._conversations.values()), {}
        for conv in convs:
            conv.close()
