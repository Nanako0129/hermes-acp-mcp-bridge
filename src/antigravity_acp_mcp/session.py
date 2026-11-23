"""ACP session layer (PlanC component 2): session/new with parked MCP server and native
tools disabled, model selection, prompt with stopReason mapping, cancel."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

from .acp_connection import AcpConnection, AcpError
from .permission import PermissionGate

# MEASURED on agy 1.5.2: ``_meta.antigravity.disabledTools`` is the key that works;
# ``_meta.agy.disabledTools`` did not visibly disable anything.
NATIVE_TOOLS = [
    "view_file", "view_file_outline", "view_code_item", "multimodal_view_file", "list_dir",
    "list_directory", "find_by_name", "grep_search", "code_search", "codebase_search",
    "write_to_file", "write_file", "edit_file", "replace_file_content", "multi_edit",
    "delete_directory", "run_command", "read_terminal", "read_url_content", "read_url",
    "search_web", "read_browser_page", "fetch_media", "generate_image", "execute_notebook",
    "invoke_subagent", "manage_subagents", "manage_task", "run_workflow", "skill_search",
    "trajectory_search",
    # S1: with the above disabled the model reached for agy's MCP-resource tools (permission-gated).
    "list_resources", "read_resource",
]  # fmt: skip

# ACP stopReason -> OpenAI-style finish_reason ("cancelled" has no OpenAI equivalent).
STOP_REASON_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
    "cancelled": "cancelled",
}


@dataclass(frozen=True)
class PromptResult:
    stop_reason: str
    finish_reason: str
    result: dict[str, Any]


class AcpSession:
    def __init__(self, conn: AcpConnection, session_id: str, gate: PermissionGate) -> None:
        self.conn = conn
        self.session_id = session_id
        self.gate = gate
        self.updates: queue.Queue[dict[str, Any]] = queue.Queue()  # session/update payloads

    def prompt(self, text: str, timeout: float | None = None) -> PromptResult:
        result = self.conn.request(
            "session/prompt",
            {"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]},
            timeout=timeout,
        )
        result = result if isinstance(result, dict) else {}
        stop = result.get("stopReason")
        if stop not in STOP_REASON_FINISH:
            raise AcpError(f"unknown ACP stopReason {stop!r}")
        return PromptResult(stop, STOP_REASON_FINISH[stop], result)

    def cancel(self) -> None:
        self.conn.cancel_session(self.session_id)

    def close(self) -> None:
        self.conn.detach_session(self.session_id)


def new_session(
    conn: AcpConnection,
    *,
    mcp_server: dict[str, Any],
    gate: PermissionGate,
    model: str | None = None,
) -> AcpSession:
    # Never log these params: mcp_server carries the bearer (L1).
    result = conn.request(
        "session/new",
        {
            "cwd": conn.cwd,
            "mcpServers": [mcp_server],
            "_meta": {"antigravity": {"disabledTools": NATIVE_TOOLS}},
        },
        timeout=60.0,
    )
    session = AcpSession(conn, str(result["sessionId"]), gate)

    def on_update(update: dict[str, Any]) -> None:
        gate.observe_update(update)
        session.updates.put(update)

    conn.attach_session(session.session_id, on_update, gate.decide)
    if model:
        select_model(conn, session.session_id, result, model)
    return session


def select_model(conn: AcpConnection, session_id: str, new_result: dict[str, Any], model: str) -> None:
    """set_config_option on the ``model`` option when it offers ``model``, else set_model."""
    for option in new_result.get("configOptions") or []:
        if not isinstance(option, dict) or "model" not in (option.get("category"), option.get("id")):
            continue
        if model in [o.get("value") for o in option.get("options") or [] if isinstance(o, dict)]:
            conn.request(
                "session/set_config_option",
                {"sessionId": session_id, "configId": option.get("id") or "model", "value": model},
                timeout=30.0,
            )
            return
    conn.request("session/set_model", {"sessionId": session_id, "modelId": model}, timeout=30.0)
