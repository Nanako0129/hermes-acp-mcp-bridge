# Hermes Core Integration Seam

`hermes-antigravity-acp` treats ACP as a stateless OpenAI `chat.completions.create()` drop-in, so Hermes Core (`agent/conversation_loop.py` + `agent/tool_executor.py`) owns the multi-turn tool loop:

```
AIAgent -> chat.completions.create() -> ACP session/prompt -> XML <tool_bridge> -> AIAgent executes 1 tool -> next chat.completions.create() -> ...
```

To collapse N tool hops into **one** `session/prompt` round-trip, Hermes Core needs an agent-level provider seam that delegates tool execution into `StreamableHTTPMCPServer`:

1. Convert the active turn's `tools` schemas via `openai_tools_to_mcp(api_kwargs["tools"])`.
2. Wrap Hermes's tool dispatcher (`make_hermes_registry_executor(tools.registry.registry, scope=..., context_kwargs=...)` or a thin wrapper around `AIAgent._execute_tool_calls`).
3. Call `bridge.run_turn(user_prompt=..., system_prompt=..., tools=..., executor=..., conversation_key=..., resume_session_id=...)` once instead of issuing N `chat.completions.create()` hops.
4. Persist the returned `BridgeTurnResult.tool_calls_executed` and `BridgeTurnResult.text` into the Hermes session transcript as usual.

Keeping this bridge in a standalone package allows iterating on the Hermes Core seam without altering `hermes-antigravity-acp`'s zero-core-patch plugin boundary.
