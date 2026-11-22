"""Adapters connecting Hermes Core ``tools.registry`` to ``StreamableHTTPMCPServer``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def make_hermes_registry_executor(
    registry: Any,
    *,
    scope: str | None = None,
    context_kwargs: dict[str, Any] | None = None,
) -> Callable[[str, dict[str, Any]], Any]:
    """Wrap a Hermes ``tools.registry.ToolRegistry`` for MCP ``tools/call`` execution."""

    base_kwargs = dict(context_kwargs or {})

    def _execute(name: str, arguments: dict[str, Any]) -> Any:
        return registry.dispatch(name, arguments, scope=scope, **base_kwargs)

    return _execute
