# hermes-acp-mcp-bridge

Native **Streamable HTTP MCP** bridge + session-pinned warm ACP transport for executing multi-hop agent tool calls inside a **single `session/prompt` round-trip**.

## Why this exists

Standard OpenAI-shaped ACP adapters simulate tool calling using text/XML `<tool_bridge>` prompts. Every single tool call ends the ACP turn, returns control to the caller, executes one local tool in `<50 ms`, and starts a brand-new `session/prompt` round-trip (re-incurring queue + TTFT + thinking tokens). An 8-tool task triggers 8 separate `session/prompt` turns.

Antigravity ACP (`v1.5.2+`) natively advertises:
- `agentCapabilities.mcpCapabilities.http: true`
- `agentCapabilities._meta.antigravity.systemPrompt.append: true`

This repo bridges those two capabilities:
1. Spawns an ephemeral, loopback-only (`127.0.0.1`), 256-bit-bearer-authenticated **Streamable HTTP MCP** server (`POST /mcp`) backed by your tool executor.
2. Passes `mcpServers: [{type: "http", ...}]` and `_meta.antigravity.systemPrompt.append` to ACP `session/new` / `session/resume`.
3. Handles `session/request_permission` fail-closed (auto-allowing only declared bridge MCP tools) while ACP drives multiple `tools/call` hops inside **one** `session/prompt`.
4. Keeps a session-pinned warm process pool so follow-up user turns skip `Popen`, `initialize`, `session/resume`, and redundant `session/set_model` when `sessionId`, `_meta`, and tool schema hash match.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from hermes_acp_mcp_bridge import ACPMCPBridge, ToolSpec

def run_tool(name: str, args: dict) -> str:
    if name == "read_status":
        return f"status ok for {args.get('target')}"
    return f"unknown tool: {name}"

bridge = ACPMCPBridge(command="agy", args=("acp",))
try:
    result = bridge.run_turn(
        user_prompt="Check status of service-a and service-b, then summarize.",
        system_prompt="Be concise.",
        tools=[
            ToolSpec(
                name="read_status",
                description="Read service health",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
            )
        ],
        executor=run_tool,
        conversation_key="demo-1",
    )
    print(result.text)
    print(f"prompt calls={result.prompt_calls_count}, tool hops={len(result.tool_calls_executed)}")
finally:
    bridge.shutdown()
```

See [docs/hermes-core-seam.md](docs/hermes-core-seam.md) for wiring this into Hermes Core (`AIAgent` / `tools.registry`).
