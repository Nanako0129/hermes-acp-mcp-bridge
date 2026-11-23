"""Hermes model-provider registration (imported only by the built plugin's ``__init__.py``, never by tests)."""

from __future__ import annotations

import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_DEFAULT_COMMAND = os.path.expanduser("~/.local/bin/antigravity-acp-file-mux")


class AntigravityAcpMcpProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from .client import AntigravityAcpMcpClient

        if not client_kwargs.get("command"):
            client_kwargs["command"] = self.process_command
        if not client_kwargs.get("args"):
            client_kwargs["args"] = list(self.process_args)
        if not client_kwargs.get("base_url"):
            client_kwargs["base_url"] = self.base_url
        return AntigravityAcpMcpClient(**client_kwargs)


antigravity_acp_mcp = AntigravityAcpMcpProfile(
    name="antigravity-acp-mcp",
    aliases=("agy-mcp",),
    display_name="Antigravity ACP (MCP tools)",
    description="Google Antigravity via the official ACP server; Hermes tools over a parked loopback MCP endpoint",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://antigravity-mcp",
    auth_type="external_process",
    supports_vision=False,
    process_command=os.environ.get("HERMES_ANTIGRAVITY_ACP_MCP_COMMAND", "").strip() or _DEFAULT_COMMAND,
    process_args=(),
    process_command_env_vars=("HERMES_ANTIGRAVITY_ACP_MCP_COMMAND",),
    process_args_env_var="HERMES_ANTIGRAVITY_ACP_MCP_ARGS",
    fallback_models=(
        "gemini-4-hydrogen-medium", "gemini-4-hydrogen-high", "gemini-4-hydrogen-low", "gemini-4-hydrogen-xhigh",
        "gemini-4-argon-medium", "gemini-4-argon-high", "gemini-4-neon-medium", "gemini-4-neon-high",
    ),
)

register_provider(antigravity_acp_mcp)
