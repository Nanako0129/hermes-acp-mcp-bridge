"""Turn matcher for the suspended-prompt design (PlanC component 5) and the replay-guard seed (component 6).

Pure functions over Hermes's OpenAI-shaped message lists; no I/O. The facade owns state and acts on the
decision. Decisions:

- ``resume_held``: the tail is exactly the tool rows for the call ids already handed to Hermes (H) → answer the
  parked MCP calls with these contents and keep reading the same ACP prompt.
- ``continue``: the session is idle and not retired, the projected prefix matches what it has seen, and the delta
  starts a new user turn → send only the delta as a new ``session/prompt``.
- ``supersede``: a new user turn while a prompt is live (in flight or parked) → cancel it, then ``replay``.
- ``guest``: a positively recognised side request (``/btw`` fork) while a prompt is live → own stateless session.
- ``replay``: everything else (first request, retry of the same create, dropped call, steer after a tool row,
  compression rewrite, detached tool result after a cancel) → fresh session, full text replay. Always safe.

The prefix is compared on a projection that ignores assistant text (Hermes may transform the reply it stores) but
keeps assistant tool-call ids, user/system text and tool rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

BTW_FORK_MARKER = "The user asked a quick SIDE question with /btw"
TOOL_CALL_NAME = "tool_call"  # Hermes's deferred-tool wrapper (tools/tool_search_catalog.TOOL_CALL_NAME)


@dataclass(frozen=True)
class TurnView:
    """What the conversation's ACP session has already consumed."""

    consumed: int = 0  # number of Hermes messages represented in the session, incl. the pending assistant slot
    prefix_fp: str = ""  # fingerprint of project(messages[:consumed])
    seen: tuple = ()  # project(messages[:consumed]) for diagnostics only (explain_mismatch); never logged raw
    handed_out: tuple[str, ...] = ()  # H: call ids returned to Hermes as tool_calls, not yet answered
    live: bool = False  # a prompt is in flight or parked
    retired: bool = False  # the session was cancelled and must never be prompted again


@dataclass
class Decision:
    kind: str
    delta: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)  # call id -> raw tool content (resume_held)


IMAGE_PART_TYPES = ("image", "image_url", "input_image")
_DATA_URL = re.compile(r"^data:([\w.+-]+/[\w.+-]+)(?:;[^,;]+)*;base64,(.*)$", re.DOTALL)


def _image(part: dict[str, Any]) -> tuple[str, str] | None:
    """(mimeType, base64 data) of an inline image part; None for remote URLs and non-images."""
    if part.get("type") == "image" and part.get("data"):
        return str(part.get("mimeType") or part.get("mime_type") or "image/png"), str(part["data"])
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    m = _DATA_URL.match(str(url or ""))
    return (m.group(1), m.group(2)) if m else None


def images(content: Any) -> list[tuple[str, str]]:
    """Inline images of a message's content, in order."""
    if not isinstance(content, list):
        return []
    return [img for p in content if isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES
            and (img := _image(p))]


def _text(content: Any) -> str:
    """Text of a message's content; each image part becomes an ``[image]`` placeholder at its position."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append("[image]" if part.get("type") in IMAGE_PART_TYPES
                             else str(part.get("text") or part.get("type") or ""))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _image_ids(content: Any) -> tuple[str, ...]:
    return tuple(hashlib.sha256(data.encode()).hexdigest()[:16] for _, data in images(content))


def _tool_call_ids(message: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(tc.get("id") or "") for tc in (message.get("tool_calls") or []) if isinstance(tc, dict)
    )


def project(message: dict[str, Any]) -> tuple:
    role = str(message.get("role") or "")
    if role == "assistant":
        return ("assistant", tuple(sorted(_tool_call_ids(message))))
    content = message.get("content")
    if role == "tool":
        return ("tool", str(message.get("tool_call_id") or ""), _image_ids(content), _text(content))
    return (role, _image_ids(content), _text(content))


def fingerprint(messages: list[dict[str, Any]]) -> str:
    blob = json.dumps([project(m) for m in messages if isinstance(m, dict)], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def view_after(messages: list[dict[str, Any]], *, handed_out: tuple[str, ...] = (), live: bool = False) -> TurnView:
    """The view once a prompt step ends: Hermes will append one assistant message (text, or tool_calls = H)."""
    pending = {"role": "assistant", "tool_calls": [{"id": i} for i in handed_out]}
    seen = [*[m for m in messages if isinstance(m, dict)], pending]
    return TurnView(consumed=len(seen), prefix_fp=fingerprint(seen), handed_out=handed_out, live=live,
                    seen=tuple(project(m) for m in seen))


def is_guest_shape(messages: list[dict[str, Any]], tools: list[Any] | None) -> bool:
    """Positive match only: the ``/btw`` fork (tools denied, last user row starts with the fork prompt)."""
    if tools:
        return False
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return bool(users) and _text(users[-1].get("content")).lstrip().startswith(BTW_FORK_MARKER)


def _held_groups(row: dict[str, Any], handed_out: tuple[str, ...]) -> dict[str, list[str]] | None:
    """Map each handed-out call id to the row's call ids that answer it, or None when the row is not ours.

    Hermes core (``agent/tool_call_batches.expand_local_tool_batches``) splits one ``tool_call`` wrapper call
    carrying ``{"calls": [e1, e2, ...]}`` into N ``tool_call`` entries before persisting: the first keeps the parent
    id, the rest get new ids right after it. Without this, every batched call looked like a foreign row and forced
    a replay (measured on a live gateway 2026-11-27: 30 of 31 replays in one turn, 113 API calls, 646 s).
    """
    held = set(handed_out)
    groups: dict[str, list[str]] = {}
    current, current_name = None, None
    for tc in row.get("tool_calls") or []:
        if not isinstance(tc, dict):
            return None
        cid, name = str(tc.get("id") or ""), (tc.get("function") or {}).get("name")
        if cid in held and cid not in groups:
            current, current_name = groups.setdefault(cid, [cid]), name
        elif current is not None and current_name == name == TOOL_CALL_NAME:  # a split sibling
            current.append(cid)
        else:
            return None
    return groups if set(groups) == held else None


def _merge_contents(contents: list[Any]) -> Any:
    """One parked MCP result from the tool rows of a split batch, in call order; a single row passes through."""
    if len(contents) == 1:
        return contents[0]
    parts: list[Any] = []
    for k, content in enumerate(contents, 1):
        parts.append({"type": "text", "text": f"[result {k}/{len(contents)}]"})
        parts.extend(content if isinstance(content, list) else [{"type": "text", "text": _text(content)}])
    return parts


def classify(messages: list[dict[str, Any]], view: TurnView | None, tools: list[Any] | None = None) -> Decision:
    msgs = [m for m in messages if isinstance(m, dict)]
    if view is None or view.consumed == 0:
        return Decision("replay")
    if view.live and is_guest_shape(msgs, tools):
        return Decision("guest")
    if view.handed_out:
        last_call = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")),
                        default=-1)
        groups = _held_groups(msgs[last_call], view.handed_out) if last_call >= 0 else None
        if groups is not None:
            tail = msgs[last_call + 1:]
            ids = [str(m.get("tool_call_id") or "") for m in tail]
            row_ids = {c for g in groups.values() for c in g}
            if tail and all(m.get("role") == "tool" for m in tail) and len(ids) == len(set(ids)) \
                    and set(ids) == row_ids:
                by_id = {str(m["tool_call_id"]): m.get("content") for m in tail}
                return Decision("resume_held", tool_results={
                    held: _merge_contents([by_id[c] for c in g]) for held, g in groups.items()})
    if len(msgs) <= view.consumed or fingerprint(msgs[:view.consumed]) != view.prefix_fp:
        return Decision("replay")
    delta = msgs[view.consumed:]
    if not any(m.get("role") == "user" for m in delta):
        return Decision("replay", delta=delta)  # e.g. detached tool rows after a cancel, or a dropped call
    if view.live:
        return Decision("supersede", delta=delta)
    if view.retired or view.handed_out:
        return Decision("replay", delta=delta)
    return Decision("continue", delta=delta)


def explain_mismatch(messages: list[dict[str, Any]], seen: list[tuple]) -> str:
    """Why the prefix no longer matches: position, role and which part differs. Never returns content."""
    msgs = [m for m in messages if isinstance(m, dict)]
    for i, old in enumerate(seen):
        if i >= len(msgs):
            return f"history shorter than seen ({len(msgs)} < {len(seen)})"
        new = project(msgs[i])
        if new != old:
            if new[0] != old[0]:
                return f"index {i}: role {old[0]} -> {new[0]}"
            if new[0] == "assistant":
                return f"index {i}: assistant tool-call ids differ ({len(old[1])} -> {len(new[1])} calls)"
            if new[0] == "tool" and new[1] != old[1]:
                return f"index {i}: tool_call_id differs"
            if new[-2] != old[-2]:
                return f"index {i}: {new[0]} images differ ({len(old[-2])} -> {len(new[-2])})"
            old_t, new_t = old[-1], new[-1]
            common = next((k for k in range(min(len(old_t), len(new_t))) if old_t[k] != new_t[k]),
                          min(len(old_t), len(new_t)))
            return (f"index {i}: {new[0]} text differs (len {len(old_t)} -> {len(new_t)}, "
                    f"first difference at char {common})")
    return "prefix matches"


def canonical_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return arguments.strip()
    return json.dumps(arguments if arguments is not None else {}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def trailing_tool_round(messages: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Replay-guard seed: (name, canonical args) of the last assistant tool round that has tool rows after it."""
    msgs = [m for m in messages if isinstance(m, dict)]
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            answered = {str(t.get("tool_call_id") or "") for t in msgs[i + 1:] if t.get("role") == "tool"}
            seed = set()
            for tc in m["tool_calls"]:
                if isinstance(tc, dict) and str(tc.get("id") or "") in answered:
                    fn = tc.get("function") or {}
                    seed.add((str(fn.get("name") or ""), canonical_args(fn.get("arguments"))))
            return seed
        if m.get("role") == "user":
            return set()
    return set()
