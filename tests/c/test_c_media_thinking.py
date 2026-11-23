"""Images (ACP image blocks, MCP image content) and thinking level (reasoning_effort -> agy) end to end."""

import pytest
from test_c_conversation import SYS, TOOLS, asst, events, make_conv

from antigravity_acp_mcp.client import thinking_level
from antigravity_acp_mcp.conversation import MAX_PROMPT_IMAGES, degenerate, render_delta, render_replay
from antigravity_acp_mcp.matcher import classify, explain_mismatch, view_after
from antigravity_acp_mcp.permission import PermissionGate
from antigravity_acp_mcp.session import new_session

from conftest import dump

PNG_A, PNG_B = "QUFBQQ==", "QkJCQg=="
SERVER = {"type": "http", "name": "hermes-deadbeef", "url": "http://127.0.0.1:1/mcp/p", "headers": []}


def img(data, mime="image/png"):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def user_with(*parts):
    return {"role": "user", "content": [{"type": "text", "text": "look"}, *parts]}


def image_blocks(blocks):
    return [(b["mimeType"], b["data"]) for b in blocks if b["type"] == "image"]


def test_replay_attaches_inline_images_numbered_and_deduped():
    msgs = [SYS, user_with(img(PNG_A)), {"role": "assistant", "content": "seen"},
            user_with(img(PNG_B, "image/jpeg"), img(PNG_A), {"type": "image_url", "image_url": {"url": "https://x/y.png"}})]
    blocks = render_replay(msgs)
    text = blocks[0]["text"]
    assert "User: look\n[image #1]" in text  # history row
    assert "[image #2]\n[image #1]\n[image at https://x/y.png (not attached; use a vision tool" in text  # active row, same bytes = same number
    assert image_blocks(blocks) == [("image/png", PNG_A), ("image/jpeg", PNG_B)]
    assert [b["text"] for b in blocks[1:] if b["type"] == "text"] == ["image #1:", "image #2:"]


def test_replay_caps_attached_images_to_the_newest():
    datas = [f"SU1H{i:04d}" for i in range(MAX_PROMPT_IMAGES + 2)]
    blocks = render_replay([SYS, user_with(*[img(d) for d in datas])])
    assert [d for _, d in image_blocks(blocks)] == datas[2:]
    assert "(image #1-#2 not attached: only the newest 8 are)" in blocks[0]["text"]


def test_delta_carries_its_images_and_text_only_stays_one_block():
    assert image_blocks(render_delta([user_with(img(PNG_A))])) == [("image/png", PNG_A)]
    assert render_delta([{"role": "user", "content": "hi"}]) == [{"type": "text", "text": "hi"}]


def test_image_change_is_a_prefix_mismatch_reported_without_content():
    base = [SYS, user_with(img(PNG_A))]
    view = view_after(base)
    swapped = [SYS, user_with(img(PNG_B)), {"role": "assistant", "content": "x"}, {"role": "user", "content": "n"}]
    assert classify(swapped, view).kind == "replay"
    why = explain_mismatch(swapped, list(view.seen))
    assert why == "index 1: user images differ (1 -> 1)" and PNG_B not in why


def test_images_reach_agy_in_the_prompt_and_in_tool_results(tmp_path):
    conv, log = make_conv(tmp_path, [[{"call": {"name": "fetch", "arguments": {"k": "shot"}}}, {"text": "done"}]])
    try:
        msgs = [SYS, user_with(img(PNG_A))]
        r1 = conv.run(msgs, TOOLS, "m")
        msgs += [asst(r1["tool_calls"]),
                 {"role": "tool", "tool_call_id": r1["tool_calls"][0].id,
                  "content": [{"type": "text", "text": "screenshot"}, img(PNG_B, "image/webp")]}]
        assert conv.run(msgs, TOOLS, "m")["text"] == "done"
        ev = events(log)
        assert [e["images"] for e in ev if e["ev"] == "prompt"] == [["image/png"]]
        (answer,) = [e for e in ev if e["ev"] == "answer"]
        assert answer["text"] == "screenshot\n[image]" and answer["images"] == ["image/webp"]
    finally:
        conv.close()


@pytest.mark.parametrize(("effort", "level"), [
    ("none", "low"), ("minimal", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"),
    ("max", "xhigh"), ("ultra", "xhigh"), ("", None), (None, None), ("weird", None),
])
def test_effort_maps_to_thinking_level(monkeypatch, effort, level):
    monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_MCP_THINKING_LEVEL", raising=False)
    assert thinking_level(effort) == level


def test_env_overrides_effort(monkeypatch):
    monkeypatch.setenv("HERMES_ANTIGRAVITY_ACP_MCP_THINKING_LEVEL", "low")
    assert thinking_level("high") == "low"


LEGACY = ["gemini-4-hydrogen-low", "gemini-4-hydrogen-medium", "gemini-4-hydrogen-high", "gemini-4-hydrogen-xhigh",
          "gemini-3.8-flash-high"]
THOUGHT = [{"id": "thinking_level", "category": "thought_level",
            "options": [{"value": "low"}, {"value": "medium"}, {"value": "high"}]}]


def calls(conn):
    return [(m["method"], m["params"]) for m in dump(conn)["log"]
            if m.get("method") in ("session/set_model", "session/set_config_option")]


@pytest.mark.parametrize(("model", "thinking", "expected"), [
    # composite id with an offered sibling: the suffix is swapped, no extra option call
    ("gemini-4-hydrogen-medium", "xhigh", [("session/set_model", "gemini-4-hydrogen-xhigh")]),
    ("gemini-4-hydrogen-medium", "medium", [("session/set_model", "gemini-4-hydrogen-medium")]),
    # no exact sibling: nearest sibling, then thought_level set to the nearest level it offers
    ("gemini-3.8-flash-high", "xhigh", [("session/set_model", "gemini-3.8-flash-high"),
                                        ("session/set_config_option", "high")]),
    ("gemini-3.8-flash-high", "low", [("session/set_model", "gemini-3.8-flash-high"),
                                      ("session/set_config_option", "low")]),
    # not a composite id: the model as requested, then the option
    ("m", "medium", [("session/set_model", "m"), ("session/set_config_option", "medium")]),
    # no effort: the model id alone decides
    ("gemini-4-hydrogen-high", None, [("session/set_model", "gemini-4-hydrogen-high")]),
])
def test_thinking_selection(connect, model, thinking, expected):
    conn = connect({"legacy_models": LEGACY, "config_options": THOUGHT})
    new_session(conn, mcp_server=SERVER, gate=PermissionGate("hermes-deadbeef", set()), model=model, thinking=thinking)
    got = [(m, p.get("modelId") or p.get("value")) for m, p in calls(conn)]
    assert got == expected


def test_thinking_change_mid_conversation_opens_a_fresh_session(tmp_path):
    conv, log = make_conv(tmp_path, [[{"text": "one"}], [{"text": "two"}], [{"text": "three"}]])
    try:
        msgs = [SYS, {"role": "user", "content": "q"}]
        conv.run(msgs, TOOLS, "m", thinking="medium")
        msgs += [{"role": "assistant", "content": "one"}, {"role": "user", "content": "next"}]
        conv.run(msgs, TOOLS, "m", thinking="medium")  # same level: continue on the same session
        msgs += [{"role": "assistant", "content": "two"}, {"role": "user", "content": "more"}]
        conv.run(msgs, TOOLS, "m", thinking="high")
        sids = [e["sid"] for e in events(log) if e["ev"] == "prompt"]
        assert sids[0] == sids[1] != sids[2]
    finally:
        conv.close()


def test_prompt_drops_images_when_the_agent_lacks_the_capability(connect):
    conn = connect({})
    session = new_session(conn, mcp_server=SERVER, gate=PermissionGate("hermes-deadbeef", set()))
    session.prompt(render_delta([user_with(img(PNG_A))]), timeout=10)
    (params,) = [m["params"] for m in dump(conn)["log"] if m.get("method") == "session/prompt"]
    assert all(b["type"] == "text" for b in params["prompt"])


def test_prompt_keeps_images_when_the_agent_declares_them(connect):
    conn = connect({"prompt_images": True})
    session = new_session(conn, mcp_server=SERVER, gate=PermissionGate("hermes-deadbeef", set()))
    session.prompt(render_delta([user_with(img(PNG_A))]), timeout=10)
    (params,) = [m["params"] for m in dump(conn)["log"] if m.get("method") == "session/prompt"]
    assert [b["type"] for b in params["prompt"]] == ["text", "text", "image"]


def test_nearest_level_both_directions(connect):
    only_high = [{"id": "thinking_level", "category": "thought_level", "options": [{"value": "medium"}, {"value": "high"}]}]
    conn = connect({"config_options": only_high})
    new_session(conn, mcp_server=SERVER, gate=PermissionGate("hermes-deadbeef", set()), model="m", thinking="low")
    assert calls(conn)[-1] == ("session/set_config_option",
                               {"sessionId": "sess-1", "configId": "thinking_level", "value": "medium"})


def test_numbering_continues_across_prompts_on_one_session(tmp_path):
    conv, log = make_conv(tmp_path, [[{"text": "one"}], [{"text": "two"}]])
    try:
        msgs = [SYS, user_with(img(PNG_A))]
        conv.run(msgs, TOOLS, "m")
        msgs += [{"role": "assistant", "content": "one"}, user_with(img(PNG_B), img(PNG_A))]
        conv.run(msgs, TOOLS, "m")
        prompts = [e for e in events(log) if e["ev"] == "prompt"]
        assert prompts[1]["sid"] == prompts[0]["sid"]
        assert prompts[1]["text"] == "look\n[image #2]\n[image #1]"  # A keeps #1 and is not attached again
        assert prompts[1]["images"] == ["image/png"]
    finally:
        conv.close()


def test_data_url_with_parameters_is_inline():
    part = {"type": "image_url", "image_url": {"url": f"data:image/png;name=x.png;base64,{PNG_A}"}}
    assert image_blocks(render_delta([user_with(part)])) == [("image/png", PNG_A)]


def test_one_skipped_image_note_names_a_single_number():
    datas = [f"SU1H{i:04d}" for i in range(MAX_PROMPT_IMAGES + 1)]
    assert "(image #1 not attached: only the newest 8 are)" in render_replay([SYS, user_with(*[img(d) for d in datas])])[0]["text"]


@pytest.mark.parametrize(("text", "bad"), [("}", True), (" ]} ", True), ("...", True), ("", False), ("OK", False),
                                           ("\U0001F44D", False), ("好", False), ("}" * 9, False)])
def test_degenerate_reply_detection(text, bad):
    assert degenerate(text) is bad


def test_degenerate_final_reply_is_re_asked_once_on_the_same_session(tmp_path):
    conv, log = make_conv(tmp_path, [[{"call": {"name": "fetch", "arguments": {"k": "a"}}}, {"text": "}"}],
                                     [{"text": "the full answer"}]])
    try:
        msgs = [SYS, {"role": "user", "content": "q"}]
        r1 = conv.run(msgs, TOOLS, "m")
        msgs += [asst(r1["tool_calls"]), {"role": "tool", "tool_call_id": r1["tool_calls"][0].id, "content": "A"}]
        r2 = conv.run(msgs, TOOLS, "m")
        assert r2["text"] == "the full answer" and r2["finish_reason"] == "stop"
        prompts = [e for e in events(log) if e["ev"] == "prompt"]
        assert len(prompts) == 2 and prompts[1]["sid"] == prompts[0]["sid"] and "'}'" in prompts[1]["text"]
    finally:
        conv.close()


def test_second_degenerate_reply_is_returned_as_is(tmp_path):
    conv, log = make_conv(tmp_path, [[{"text": "}"}], [{"text": "}"}]])
    try:
        assert conv.run([SYS, {"role": "user", "content": "q"}], TOOLS, "m")["text"] == "}"
        assert len([e for e in events(log) if e["ev"] == "prompt"]) == 2
    finally:
        conv.close()
