"""Permission gate (PlanC component 3, M2) and the detection-only native-tool tripwire (H1).

The gate allows a ``session/request_permission`` only for an MCP tool call to THIS
session's server and a tool in the active set, and only through an ``allow_once``
option. It never matches on titles and never picks ``allow_always``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from .acp_connection import CANCELLED_OUTCOME


def _our_meta(meta: Any, server_name: str, tool_names: Collection[str]) -> bool:
    if not isinstance(meta, dict) or meta.get("is_mcp_tool_call") is not True:
        return False
    mcp = meta.get("mcp")
    return (
        isinstance(mcp, dict)
        and mcp.get("server") == server_name
        and mcp.get("tool") in tool_names
    )


class PermissionGate:
    def __init__(
        self,
        server_name: str,
        tool_names: Collection[str],
        on_foreign_tool_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.server_name = server_name
        self.tool_names = frozenset(tool_names)
        self._on_foreign = on_foreign_tool_call
        self._ours: set[str] = set()  # toolCallIds first seen carrying our _meta

    def decide(self, params: dict[str, Any]) -> dict[str, Any]:
        """Response result for a ``session/request_permission`` request."""
        tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
        options = [o for o in params.get("options") or [] if isinstance(o, dict)]
        allowed = _our_meta(tool_call.get("_meta"), self.server_name, self.tool_names)
        for kind in ("allow_once", "reject_once") if allowed else ("reject_once",):
            for option in options:
                if option.get("kind") == kind and option.get("optionId"):
                    if kind == "allow_once" and tool_call.get("toolCallId"):
                        self._ours.add(str(tool_call["toolCallId"]))
                    return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
        return CANCELLED_OUTCOME

    def observe_update(self, update: dict[str, Any]) -> None:
        """Tripwire, best effort: native tools may emit no ``tool_call`` update at all."""
        kind = update.get("sessionUpdate")
        if kind not in ("tool_call", "tool_call_update"):
            return
        call_id = str(update.get("toolCallId") or "")
        if _our_meta(update.get("_meta"), self.server_name, self.tool_names):
            self._ours.add(call_id)
        elif kind == "tool_call" and call_id not in self._ours and self._on_foreign:
            self._on_foreign(update)
