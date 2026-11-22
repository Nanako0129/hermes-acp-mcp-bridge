"""End-to-end tests for single-session/prompt multi-hop MCP execution and warm reuse."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hermes_acp_mcp_bridge import ACPMCPBridge, ToolSpec, make_hermes_registry_executor

FAKE_SERVER = str(Path(__file__).with_name("fake_acp_mcp_server.py"))


def test_single_prompt_multi_hop_and_warm_reuse() -> None:
    calls: list[tuple[str, dict]] = []

    def executor(name: str, args: dict) -> str:
        calls.append((name, args))
        return f"hop-{args['step']}-ok"

    bridge = ACPMCPBridge(command=sys.executable, args=(FAKE_SERVER,))
    try:
        tools = [ToolSpec(name="inspect_step", description="Inspect a step")]
        turn1 = bridge.run_turn(
            user_prompt="MULTI_HOP please",
            system_prompt="SYS_NATIVE",
            tools=tools,
            executor=executor,
            conversation_key="conv-1",
            model="fake-model-mcp",
        )
        assert turn1.prompt_calls_count == 1
        assert turn1.warm_reused is False
        assert len(turn1.tool_calls_executed) == 3
        assert [c[1]["step"] for c in calls] == [1, 2, 3]
        payload1 = json.loads(turn1.text)
        assert payload1["sys"] == "SYS_NATIVE"
        assert payload1["model"] == "fake-model-mcp"
        assert payload1["hops"] == ["hop-1-ok", "hop-2-ok", "hop-3-ok"]

        # Turn 2 reuses the warm process directly (no session/resume or extra set_model)
        turn2 = bridge.run_turn(
            user_prompt="MULTI_HOP again",
            system_prompt="SYS_NATIVE",
            tools=tools,
            executor=executor,
            conversation_key="conv-1",
            resume_session_id=turn1.session_id,
            model="fake-model-mcp",
        )
        assert turn2.warm_reused is True
        assert turn2.session_id == turn1.session_id
        assert len(turn2.tool_calls_executed) == 3
        assert len(calls) == 6
    finally:
        bridge.shutdown()


def test_hermes_registry_executor_adapter() -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def dispatch(self, name: str, args: dict, *, scope=None, **kwargs):
            self.seen.append((name, args, scope, kwargs))
            return {"ok": True, "tool": name}

    reg = FakeRegistry()
    exec_fn = make_hermes_registry_executor(reg, scope="cli", context_kwargs={"task_id": "t-1"})
    out = exec_fn("read_file", {"path": "README.md"})
    assert out == {"ok": True, "tool": "read_file"}
    assert reg.seen == [("read_file", {"path": "README.md"}, "cli", {"task_id": "t-1"})]


def test_permission_decider_matches_real_acp_mcp_metadata() -> None:
    from hermes_acp_mcp_bridge.acp_bridge import _default_permission_decider

    options = [
        {"optionId": "allow_always", "name": "Allow Always", "kind": "allow_always"},
        {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
        {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
    ]
    allowed_params = {
        "toolCall": {
            "toolCallId": "48a154",
            "kind": "other",
            "title": "hermes-tools_lookup_kv",
            "_meta": {"mcp": {"tool": "lookup_kv", "server": "hermes-tools"}, "is_mcp_tool_call": True},
        },
        "options": options,
    }
    assert _default_permission_decider(allowed_params, {"lookup_kv"}, "hermes-tools") == "allow"

    other_server = {
        "toolCall": {
            "title": "foreign_lookup_kv",
            "_meta": {"mcp": {"tool": "lookup_kv", "server": "foreign"}, "is_mcp_tool_call": True},
        },
        "options": options,
    }
    assert _default_permission_decider(other_server, {"lookup_kv"}, "hermes-tools") == "deny"
