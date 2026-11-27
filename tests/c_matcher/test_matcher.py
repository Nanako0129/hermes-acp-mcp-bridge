"""PlanC S3 matcher table on synthetic Hermes message tails (no user conversation data)."""

import json

from antigravity_acp_mcp.matcher import (
    BTW_FORK_MARKER,
    _text,
    TurnView,
    canonical_args,
    classify,
    trailing_tool_round,
    view_after,
)

SYS = {"role": "system", "content": "You are Hermes."}
U1 = {"role": "user", "content": "check a and b"}


def call(cid, name="fetch", args='{"k": 1}'):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def asst_calls(*cids):
    return {"role": "assistant", "content": "", "tool_calls": [call(c) for c in cids]}


def tool(cid, text="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": text}


def test_first_request_replays():
    assert classify([SYS, U1], None).kind == "replay"


def test_resume_held_single_and_parallel_any_order():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a", "call_b"), live=True)
    msgs = [*base, asst_calls("call_a", "call_b"), tool("call_b", "B"), tool("call_a", "A")]
    d = classify(msgs, view)
    assert d.kind == "resume_held"
    assert d.tool_results == {"call_a": "A", "call_b": "B"}


def split_row(*cids):
    """Hermes core's expand_local_tool_batches output: one tool_call parent split into N tool_call entries."""
    return {"role": "assistant", "content": "", "tool_calls": [call(c, name="tool_call") for c in cids]}


def test_resume_held_after_hermes_splits_a_tool_call_batch():
    base = [SYS, U1]
    view = view_after(base, handed_out=("p",), handed_sizes=(3,), live=True)
    msgs = [*base, split_row("p", "k1", "k2"), tool("k2", "C"), tool("p", "A"), tool("k1", "B")]
    d = classify(msgs, view)
    assert d.kind == "resume_held"
    assert _text(d.tool_results["p"]) == "[result 1/3]\nA\n[result 2/3]\nB\n[result 3/3]\nC"


def test_split_siblings_follow_their_own_parent():
    base = [SYS, U1]
    view = view_after(base, handed_out=("p", "q"), handed_sizes=(2, 2), live=True)
    row = split_row("p", "k1", "q", "k2")
    d = classify([*base, row, tool("p", "A"), tool("k1", "B"), tool("q", "C"), tool("k2", "D")], view)
    assert d.kind == "resume_held"
    assert _text(d.tool_results["q"]) == "[result 1/2]\nC\n[result 2/2]\nD"


def test_split_empty_child_result_stays_empty():
    base = [SYS, U1]
    view = view_after(base, handed_out=("p",), handed_sizes=(2,), live=True)
    d = classify([*base, split_row("p", "k1"), tool("p", ""), tool("k1", "B")], view)
    assert _text(d.tool_results["p"]) == "[result 1/2]\n\n[result 2/2]\nB"


def test_sibling_count_must_match_the_batch():
    base = [SYS, U1]
    view = view_after(base, handed_out=("p",), handed_sizes=(1,), live=True)  # one-entry batch: never split
    assert classify([*base, split_row("p", "x"), tool("p"), tool("x")], view).kind == "replay"
    view = view_after(base, handed_out=("p",), handed_sizes=(3,), live=True)  # Hermes deduped one entry
    assert classify([*base, split_row("p", "k1"), tool("p"), tool("k1")], view).kind == "resume_held"
    view = view_after(base, handed_out=("p",), handed_sizes=(3,), live=True)  # left whole (e.g. connector-only)
    assert classify([*base, split_row("p"), tool("p", "A")], view).kind == "resume_held"


def test_replay_guard_covers_a_split_batch():
    from antigravity_acp_mcp.matcher import is_guarded
    e1 = {"name": "session_search", "arguments": {"query": "a"}}
    e2 = {"name": "session_search", "arguments": '{"query": "b"}'}  # Hermes parses string arguments
    row = {"role": "assistant", "content": "", "tool_calls": [
        call("p", "tool_call", json.dumps({"calls": [e1]})),
        call("k1", "tool_call", json.dumps({"calls": [{"name": "session_search", "arguments": {"query": "b"}}]}))]}
    guard = trailing_tool_round([SYS, U1, row, tool("p"), tool("k1")])
    assert is_guarded("tool_call", {"calls": [e1, e2]}, guard)
    e3 = {"name": "session_search", "arguments": {"query": "c"}}
    assert not is_guarded("tool_call", {"calls": [e1, e3]}, guard)  # partly ran: e3 never did, let through
    assert not is_guarded("tool_call", {"calls": [e3]}, guard)
    # an unsplit one-entry batch re-issued in another spelling is still the same call
    one = {"role": "assistant", "content": "", "tool_calls": [
        call("q", "tool_call", json.dumps({"calls": [{"name": "x", "arguments": '{"q": 1}'}]}))]}
    guard = trailing_tool_round([SYS, U1, one, tool("q")])
    assert is_guarded("tool_call", {"name": "x", "arguments": {"q": 1}}, guard)


def test_guard_ignores_entries_of_a_batch_left_whole():
    from antigravity_acp_mcp.matcher import is_guarded
    entries = [{"name": "session_search", "arguments": {"query": str(i)}} for i in range(11)]
    whole = {"role": "assistant", "content": "", "tool_calls": [call("w", "tool_call", json.dumps({"calls": entries}))]}
    guard = trailing_tool_round([SYS, U1, whole, tool("w", "too many calls: 11 > max 10")])
    assert is_guarded("tool_call", {"calls": entries}, guard)  # the identical call: exact match, as before
    assert not is_guarded("tool_call", {"calls": entries[:6]}, guard)  # its entries never ran


def test_empty_text_part_is_empty():
    assert _text([{"type": "text", "text": ""}]) == ""
    assert _text([{"type": "resource"}]) == "resource"


def test_foreign_extra_call_still_replays():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)  # a plain call cannot gain siblings
    msgs = [*base, asst_calls("call_a", "x"), tool("call_a"), tool("x")]
    assert classify(msgs, view).kind == "replay"
    view = view_after(base, handed_out=("p",), live=True)  # split sibling without its tool row
    assert classify([*base, split_row("p", "k1"), tool("p")], view).kind == "replay"


def test_dropped_call_without_tool_row_replays():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a", "call_b"), live=True)
    msgs = [*base, asst_calls("call_a", "call_b"), tool("call_a")]  # dedupe/cap dropped call_b's row
    assert classify(msgs, view).kind == "replay"


def test_steer_row_after_tool_row_supersedes_live_prompt():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    msgs = [*base, asst_calls("call_a"), tool("call_a"), {"role": "user", "content": "also check c"}]
    assert classify(msgs, view).kind == "supersede"


def test_retry_of_same_create_replays():
    base = [SYS, U1]
    view = view_after(base, live=True)
    assert classify(base, view).kind == "replay"


def test_compression_rewrite_replays():
    base = [SYS, U1, {"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}]
    view = view_after(base[:2])
    rewritten = [SYS, {"role": "user", "content": "[summary of earlier turns]"}, {"role": "user", "content": "next"}]
    assert classify(rewritten, view).kind == "replay"


def test_new_user_turn_after_idle_continues_with_delta_ignoring_assistant_text():
    base = [SYS, U1]
    view = view_after(base)  # prompt ended with text; Hermes stores its own rendering of it
    msgs = [*base, {"role": "assistant", "content": "a and b are fine (transformed)"},
            {"role": "user", "content": "thanks, now c"}]
    d = classify(msgs, view)
    assert d.kind == "continue"
    assert [m["role"] for m in d.delta] == ["user"]


def test_new_user_turn_on_retired_session_replays():
    base = [SYS, U1]
    v = view_after(base)
    view = TurnView(v.consumed, v.prefix_fp, retired=True)
    msgs = [*base, {"role": "assistant", "content": "x"}, {"role": "user", "content": "next"}]
    assert classify(msgs, view).kind == "replay"


def test_detached_tool_rows_after_cancel_replay():
    base = [SYS, U1]
    v = view_after(base, handed_out=("call_a",))
    view = TurnView(v.consumed, v.prefix_fp, handed_out=(), live=False, retired=True)  # deadline cancelled it
    msgs = [*base, asst_calls("call_a"), tool("call_a")]
    assert classify(msgs, view).kind == "replay"


def test_btw_guest_while_live_does_not_touch_owned_turn():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    fork = [*base, {"role": "user", "content": BTW_FORK_MARKER + " ...\n\nwhat is a?"}]
    assert classify(fork, view, tools=[]).kind == "guest"
    assert classify(fork, view, tools=[{"type": "function"}]).kind != "guest"  # positive match only


def test_unknown_shape_never_continues():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    msgs = [*base, asst_calls("call_zz"), tool("call_zz")]
    assert classify(msgs, view).kind == "replay"


def test_trailing_tool_round_seed_and_canonical_args():
    msgs = [SYS, U1, asst_calls("call_a", "call_b"), tool("call_a")]
    assert trailing_tool_round(msgs) == {("fetch", canonical_args('{"k":1}'))}
    assert trailing_tool_round([SYS, U1]) == set()
    assert trailing_tool_round([SYS, U1, asst_calls("c1"), tool("c1"), {"role": "user", "content": "new"}]) == set()
    assert canonical_args('{"b": 2, "a": 1}') == canonical_args({"a": 1, "b": 2})


def test_explain_mismatch_reports_position_and_kind_without_content():
    from antigravity_acp_mcp.matcher import explain_mismatch
    base = [SYS, U1]
    view = view_after(base)
    edited = [SYS, {"role": "user", "content": "check a and b <memory-context>secret recall</memory-context>"},
              {"role": "assistant", "content": "x"}, {"role": "user", "content": "next"}]
    why = explain_mismatch(edited, list(view.seen))
    assert why.startswith("index 1: user text differs") and "secret" not in why
    assert classify(edited, view).kind == "replay"
