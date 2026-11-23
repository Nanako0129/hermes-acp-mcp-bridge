# Plan: Hermes Antigravity ACP provider, design C (suspended prompt)

Status: revision 4 — closing epoch after two plan-verifier REVISEs (simplified: one session per prompt chain, see component 4). Nothing below is approved or built.

## Envelope

**Outcome.** A new Hermes model-provider plugin, installed beside the current one as a separate provider
(working id `antigravity-acp-mcp`), that answers one Hermes user turn with ONE ACP `session/prompt`, however many
Hermes tools the model calls, while every tool still runs through Hermes's own unmodified tool pipeline
(approvals, guards, interrupts, persistence, display). Silent multi-minute hangs become a visible error in seconds.

**Seen working, by whom.** The user, on Discord against the test Hermes gateway: a turn that needs 2+ tools answers with
one ACP prompt (mux/agent log shows one `session/prompt` per turn) and Hermes's approval card still appears for a
gated tool. Latency per turn is compared with the current plugin on the same prompts.

**Measured facts this rests on (2026-11-23, agy_acp_server 1.5.2).**
- agy calls loopback HTTP MCP tools several times inside one prompt; sends `progressToken`, accepts SSE.
- An MCP `tools/call` has a hard 180 s deadline (`context deadline exceeded`); progress notifications do not extend it.
- `session/cancel` with a parked call ends the prompt `cancelled` at once; the same process and session stay usable.
- After an MCP timeout agy used its native `view_file` with no permission request.
- `systemPrompt.append` carrying Hermes's system band draws Google 429 on every prompt; the same band inside
  `session/prompt` does not. The append is not used, and no text is rewritten to avoid checks.

**Repository / placement (assumption, needs approval).** Build in the user's repo `Nanako0129/hermes-acp-mcp-bridge`
(reusing its `mcp_server.py` transport and security invariants; `acp_bridge.py`'s execute-in-HTTP-thread model is
replaced). Ship as its own plugin directory so the current `antigravity-acp` provider stays installed as fallback;
switching is a Hermes config change the user makes. The account mux (`antigravity-acp-file-mux`) is used unchanged as
the ACP command.

**Non-goals.** Hermes core changes (option B; branch `feat/acp-mcp-bridge` stays a bench harness).
`systemPrompt.append`. `fs/*` / `terminal/*` client methods (advertised false). `session/load` history replay,
list/fork/delete, elicitation, modes, plans. Changing the mux. Changing Hermes settings (approval timeout etc.)
without the user. Any retry on another account after a 429. Upstreaming to rhgo1749 (separate decision).

**Owners.** Main session: design, matcher logic, integration, acceptance. Executors only for well-specified
mechanical pieces after approval. Live tests on the gateway host that restart the gateway run from user-confirmed Orca scripts.

## Components

1. **AcpConnection** — one agy (mux) child per (conversation key), launched with cwd = an empty dedicated
   directory and the child environment built by the current plugin's `_build_subprocess_env` /
   `_acp_child_env_filter` allowlist unchanged (never `os.environ`; H2). NDJSON JSON-RPC; stdout and stderr reader
   threads; stderr is untrusted: classified at capture time (`_acp_stderr_retry_class`) and summarised only by
   `_acp_stderr_diagnostic` (non-reversible); exception and log text carry the class and fingerprint, never stderr
   bytes (H3). `session/update` buffered per sessionId (may precede the `session/new` reply); `-32601` for unknown
   agent→client requests; honest `clientInfo`; terminal false. fs read: decided by S1a — if agy routes native reads
   through `fs/read_text_file` when advertised, advertise it and answer deny-all (or reuse the current
   `_fs_read_text_file` policy) as a second layer; otherwise false. `session/new`/`resume` params are never logged
   (they carry the MCP bearer; L1).
2. **Session layer** — `session/new` (or capability-gated `session/resume` on the same lane) with `mcpServers=[ours]`
   and native tools disabled; model and thinking level via `session/set_config_option` when the session offers it,
   else `session/set_model`; full `stopReason` mapping (end_turn, max_tokens, max_turn_requests, refusal, cancelled);
   `session/cancel` also answers every pending `session/request_permission` with `cancelled`.
3. **Permission gate** — allow (`allow_once` only) only when `toolCall._meta.is_mcp_tool_call` is true,
   `_meta.mcp.server` equals this session's random server name (`hermes-<8 hex>`, M2) and `_meta.mcp.tool` is in
   the active turn's tool set; everything else `reject_once`/`cancelled`. Never `allow_always`, never by title;
   the current plugin's native-read auto-allow (`_safe_read_permission_option`) is not ported.
   **Native-tool tripwire (H1), detection only:** any `tool_call` update whose `_meta` is not our MCP server →
   `session/cancel`, kill the child, raise a typed error. It cannot prevent a read that already ran, so it never
   counts toward the S1a ship gate.
4. **Parked MCP server** — `mcp_server.py` security (127.0.0.1, ephemeral port, bearer, Host/Origin checks,
   declared-tool allowlist, Handler socket `timeout`) with the executor replaced by "park". **One server name,
   token and URL path per ACP session (M1).** **Rule: a cancelled session is never prompted again.** Every
   `session/cancel` (deadline, supersede, interrupt, reset) retires that session: its MCP endpoint refuses every
   later `tools/call` (`isError`, never parked) and is closed once the prompt reply arrives; the next prompt for that
   conversation starts a fresh session with a newly rotated name, token and path, seeded by full text replay. So a
   straggler call can only ever hit a retired endpoint. (Warm reuse still applies to the *process*, not the session.)
   Each parked call gets a plugin-minted uuid id (never the JSON-RPC id, never a provider prefix Hermes rewrites,
   `message_sanitization.py:620-648`). **In-prompt dedupe (M3):** a call whose name + canonical arguments equal a
   call already parked or in H waits for that call's answer instead of being handed out again.
5. **Turn registry + matcher** (process-global, keyed by the plugin's existing `_conversation_affinity`; Hermes
   caches and reuses clients, so state must not live on a client instance either way).
   **Turn ownership (L2).** Per affinity key at most one *owned* turn exists. Precedence for a create on a key
   with an owned turn: (1) tail = tool rows for H → *resume-held*; (2) adds a new user row → *supersede* (cancel
   the owned prompt, then treat as a new turn); (3) positively matches a known guest shape (`/btw` fork,
   post-turn review fork, title generation — the shapes that really share a key are measured first; the affinity
   already includes the write-origin lane, `client.py:1841`) → *guest*: stateless own session, never touches the
   owned turn; (4) anything else → *reset*.
   **Handed-out set.** The facade tracks the call ids it has already returned to Hermes as `tool_calls` (H), and
   a FIFO of calls that parked after that return (Q).
   **Collection rule for parallel calls.** When the first `tools/call` parks, wait up to 300 ms for more
   `tools/call` on the same prompt before returning; return all collected calls in one `tool_calls`.
   Each `create(messages, tools)` is classified:
   - *resume-held*: the message tail is exactly the tool rows for H (any order) → answer those parked calls, clear
     H; if Q is non-empty, return Q as the next `tool_calls` at once (no model latency), else keep reading the
     same prompt.
   - *continue*: nothing parked, session alive, history prefix matches what this session has seen → send only the new
     user/tool delta as a new `session/prompt`.
   - *reset*: anything else (Hermes retry of the same create, dropped call with no tool row, steer row after tool row,
     compression rewrite, restart, unknown session) → `session/cancel`, answer parked calls `isError`
     ("superseded"), fresh session, full text replay with the system band at the top of the first prompt.
   Reset is always safe; a log line per outcome measures how often the matcher falls back.
6. **Deadline fallback (gated on measurement S1c)** — at 170 s on a parked call: `session/cancel`, mark the turn
   "detached". When Hermes later sends the tool row, the matcher sees no parked call and a cancelled session →
   a fresh session (component 4 rule) seeded by full text replay that includes the tool result.
   **Replay guard (all text-replay paths: deadline, reset, delegated children).** Whenever a create is answered by
   text replay, the guard is seeded from the replayed history's trailing tool round: the last assistant
   `tool_calls` and their tool rows (name + canonical arguments). In that create, a `tools/call` matching a seeded
   signature is answered `isError` ("already executed; its result is in the conversation above") and never handed
   to Hermes. This makes double execution impossible on these paths regardless of model behaviour; it only blocks
   an identical repeat of the most recent round (a deliberate re-run needs different arguments or a new user turn).
   S1c measures the *cost*: how often the model re-issues and how many extra model steps that wastes. Stop: if the
   model loops (two guard refusals in one create), end the create with a visible message
   ("the last step's result is above; send 'continue'"). Delegated children use the same guard (seeded per create
   from their own replay). Tool output sent as text is fenced with a random nonce delimiter (L3, optional).
7. **Interrupt and abandonment.**
   - **In-flight create interrupted**: Hermes calls the facade's `cancel()` from the interrupt thread
     (`agent/client_lifecycle.py::_abort_request_slot_client` prefers `type(client).cancel`; the current plugin
     implements it for this reason). `cancel()` sends `session/cancel` and answers pending permission requests
     `cancelled`; parked HTTP calls are released only AFTER the prompt reply says `cancelled` (or their
     connections are closed without a result), so the model never sees an error on a live prompt (M4); the
     blocked create raises a non-retryable cancelled error within 5 s. Request-scoped: only the turn this client
     instance owns.
   - `close()` is lifecycle-only (error cleanup, eviction, teardown): it reaps owned children; it is not an
     interrupt signal.
   - **No create in flight** (Hermes is running the tool, or ended the turn: guardrail halt, persistence failure,
     dedupe/delegate cap drop, `/stop` during tool execution): no plugin method is called. The parked prompt is
     released by whichever comes first: the next create for that key that is not a resume (cancel the old
     prompt, then handle the new create), or the 170 s deadline. Bound: a stopped turn holds at most one idle
     parked prompt for ≤ 170 s and never executes anything; the next user message is never delayed by it.
   - The plugin never executes a tool.
8. **Liveness** — the guarantee is a **per-model-step inactivity deadline** (default 90 s): it restarts on every
   `session/update` and every answered parked call, and is suspended only while a call is parked with Hermes.
   On expiry: `session/cancel`, raise a typed provider error with the stderr tail. Best-effort speed-up: a stderr
   classifier for the agy harness's 429 line raises the same error at once. Real sample of that line (captured
   2026-11-23 by the mux into the host's `~/.hermes/cache/acp-mux-state.json`, field `detail`):
   `harness stderr: ERROR: logging before google.Init: E1123 … errorreport.go:273] Error 429, Message: Resource has
   been exhausted (e.g. check quota)., Status: RESOURCE_EXHAUSTED`. The same text is stored in agy's conversation DB
   steps (`~/.gemini/antigravity-acp/conversations/<id>.db`). No automatic retry on another account.
9. **Facade** — `chat.completions.create` returning OpenAI-shaped objects (sync and the existing dual-mode awaitable
   wrapper), `tool_calls` built from parked MCP calls (id, name, JSON arguments), text and reasoning from
   `agent_message_chunk` / `agent_thought_chunk`, usage from `usage_update`, `stream=True` returning chunks after the
   prompt step ends. Delegated children (empty affinity) run stateless: own session per create, no parking across
   creates (their tool calls fall back to the text path of component 6).

## Slices

| ID | Outcome | Prerequisites | Proving acceptance | Stop conditions / rollback |
|---|---|---|---|---|
| S1 | Unmeasured server behaviour measured (live, needs user go; **30 small prompts total** through the mux, harmless in-process tools and planted canaries only) | user go | Probe area: a scratch dir on the host with planted canary files (random content, nothing real) and a marker dir. "Native execution" = canary content in the reply or in agy's conversation DB, OR a native `tool_call_update` status `completed`, OR a marker file created. **S1a native tools (H1)**, 16 prompts: 2 prompts find which disable key (`_meta.agy.disabledTools` / `_meta.antigravity.disabledTools`) suppresses native `view_file` of an outside-cwd canary; then, for the working key, 7 probes (read canary outside cwd by absolute path, read canary inside cwd, list/search, write marker, `run_command` creating a marker, URL fetch, web search) × 2 conditions (fresh; right after an MCP `isError` with the guard's wording); also record whether `view_file` routes through `fs/read_text_file` when advertised. **S1b parallel**, 2 prompts. **S1c re-issue cost**, 10 prompts: 5 deadline-style (cancel at 20 s → fresh session, text replay) + 5 reset-style replays ending in a tool row; count re-issues and guard refusals. **S1d retry (M3)**, 2 prompts (5xx; dropped connection). **S1e token at rest (L1)**: counted after the runs, no extra prompts. Each record lists its canary/marker and the observed detection result. | **Stop S1 at the first** 429/RESOURCE_EXHAUSTED, auth or suspension error (no retry, no other account). **Ship nothing** unless a blocking mechanism (a disable key, or denied `fs` routing) leaves zero native executions in all 14 S1a matrix probes; the tripwire does not count. |
| S2 | Components 1–4 | S1a result (which disable key) | Fake-ACP tests: early `session/update` buffering; stderr flood (≥ 1 MB) does not block; child exit surfaces exit status + stderr tail; cancel answers pending permissions; permission gate rejects title-only matches, foreign servers and `allow_always`; child env has no secret-named variable (H2); stderr containing `ya29.…`/`Bearer …` never appears in `str(exc)` or logs (H3); a fake `tools/call` yields `tool_calls` and no handler runs. Live: 2-hop prompt through the mux, one `session/prompt`. | Any fake test needing a real-server behaviour not measured in S1 → stop and add it to S1. |
| S3 | Components 5–7 | S2 | Table-driven matcher tests on SYNTHETIC message tails built from Hermes's message schema (no user conversation data read): normal multi-tool; parallel calls (2 concurrent `tools/call` → one prompt, zero resets, both answered); late call into Q; Hermes retry of the same create; dropped call (no tool row); steer row after tool row; compression rewrite; new user turn; **guest create** (fork-shaped messages, same affinity, while a call is parked → stateless own session, main turn resumes, main session receives no `session/cancel`); interrupt via `type(client).cancel` while a create is in flight → create raises within 5 s, no orphan child or waiter. late `tools/call` after a reset is refused and never handed to Hermes (M1); duplicate `tools/call` (new JSON-RPC id, same signature) is not handed out twice (M3); cancel-before-release order (M4). cancel followed by the next prompt: a `tools/call` on the retired endpoint is refused and never handed out, while one from the fresh session parks and is returned as `tool_calls`; replay guard: a reset create and a delegated create whose replay ends in a tool row, fake agy re-issuing the same signature → `isError`, nothing handed to Hermes. Fake-ACP deadline test with a shortened clock. | Unknown shape not resetting → fix before S4. Live reset rate later > 5 % of creates → stop S5 rollout. |
| S4 | Components 8–9 | S3 | Fake server that goes silent after a *resume-held* answer → typed error within the step deadline; fake server emitting the real 429 sample line → typed error < 5 s, child reaped. Facade tests: sync, dual-mode awaitable, `stream=True`, Hermes provider-seam test shapes. | — none beyond failing tests |
| S5 | Installed beside the current provider on the gateway host | S1–S4 done; security dispositions closed; deployed mux verified to run `mcp_config_violation` (M2); deployed command has no capture wrapper (L1); open decisions (1)(2) answered | User-run Orca script installs the plugin dir and restarts the gateway; user switches the provider. Discord checks: (a) 2+ tool turn = one ACP prompt; (b) approval < 170 s continues the same prompt; (c) approval > 170 s with a gated test tool that appends one line to a marker file: marker line count = 1 and approval cards = 1 for that turn, turn completes or ends with the visible message; (d) `/stop` mid-create: reply within 5 s; `/stop` during a tool: next message answered normally, no second execution, stale prompt released ≤ 170 s; (e) latency vs current plugin on the same three prompts. | Rollback: switch provider back; script `--revert` removes the plugin dir. Any double execution → roll back at once. |

## Security review (pilotfish:security-reviewer, 2026-11-23) — dispositions

| ID | Finding | Disposition |
|---|---|---|
| H1 | agy native tools bypass Hermes; fs=false removes the current plugin's read-policy layer; S1a too narrow | FIX: widened S1a (all native tools × both keys × after MCP error), fs routing measured and deny-all if routed, native-tool tripwire, empty cwd; ship nothing without a measured block |
| H2 | child env not specified; bridge passes all of `os.environ` | FIX: reuse `_build_subprocess_env` allowlist + test (component 1, S2) |
| H3 | stderr tail in exceptions can reach Discord | FIX: capture-time classifier + `_acp_stderr_diagnostic` only; test (component 1, S2) |
| M1 | calls not bound to a session; late calls after reset | FIX: per-session server name/token/path, refuse after cancel; test (component 4, S3) |
| M2 | permission gate shape; fixed server name collision | FIX: `is_mcp_tool_call` + random per-session name + tool set, `allow_once` only; mux guard verified before S5 |
| M3 | agy retry of `tools/call` → double handout | FIX + measure: S1d; in-prompt signature dedupe; plugin-minted ids |
| M4 | error answers on a live prompt | FIX: release parked calls only after `cancelled`; test |
| L1 | bearer at rest / in logs | FIX: never log session params; S1e count; no capture wrapper in the deployed command |
| L2 | guest vs supersede contradiction | FIX: explicit precedence; guest only on positive match |
| L3 | tool output as user text | DEFER (same as current text bridge); optional nonce fence |
| L4 | mcp_server hardening (socket timeout, Host edge cases) | FIX the one-line socket timeout; rest DEFER |
| L5 | guard keying for delegated children | FIX in text (per create) |
| INFO | Hermes rewrites provider-prefixed call ids | FIX: plugin-minted uuid ids |

**Budget.** About 1,200 lines of plugin code plus tests (vs 9,500 in the current plugin). Live calls: S1 = 30 small prompts,
S2 ≈ 3, S5 = the user's own Discord use.

**Open decisions for the user.** (1) Repo placement above. (2) Whether Hermes's approval timeout should stay 300 s
(fallback path covers it) or be lowered. (3) Whether band-in-prompt is acceptable long term (it is the pre-#78
status quo and upstream's placement; Google may later check prompt text too).

## S1 evidence (2026-11-23, agy_acp_server 1.5.2 via the account mux on the gateway host; 30 prompts; no 429/auth/suspension signal)

Probe: a standalone S1 probe script (not in this repo); raw records kept locally. Canaries were random
strings in planted files; "native execution" = canary in reply/updates/agy conversation DB, a non-ours
`tool_call_update` completed, a marker file, or the example.com title in the reply.

- **S1a native tools.**
  - Control (no disable key): the model tried `view_file` on the outside-cwd canary; agy's own workspace policy
    denied it ("access to that path is denied by the workspace access policy"). No `tool_call` update was emitted
    for that native call → the tripwire cannot rely on updates.
  - `_meta.agy.disabledTools`: no visible effect (model reached for other tools).
  - `_meta.antigravity.disabledTools` (31 names): the model reports "no file-reading or shell-execution tools are
    available". **14/14 matrix probes** (read outside cwd, read inside cwd, list/search, write, run_command, URL
    fetch, web search) × (fresh, after an MCP `isError` with the guard wording) → **zero native executions**
    (no canary anywhere incl. agy DBs, no markers, no example.com title). **Ship gate met** with this key.
  - With tools disabled the model tried agy's MCP-resource tools `list_resources` / `read_resource`; both arrive as
    permission requests and the gate rejected them. Add both names to `disabledTools`.
- **S1b parallel.** 2/2 runs asked for 3 independent lookups "in parallel": agy issued them strictly one after
  another (next call ≈ 0.01 s after the previous answer). The 300 ms collection window is kept but rarely used.
- **S1c re-issue cost.** 10/10 (5 deadline-style cancel → fresh session text replay, 5 reset-style replay ending in
  a tool row): **0 re-issues**; every reply used the replayed result. The replay guard stays as a safety net.
- **S1d retry.** agy retries a failed `tools/call` once: after HTTP 500 (≈ 4 s later) and after a dropped connection
  (≈ 6.7 s later), same arguments, new JSON-RPC id. The in-prompt signature dedupe (component 4) is required.
- **S1e token at rest.** agy stores the `mcpServers` config incl. the bearer header in its conversation DBs
  (35 DB files under the lane's `conversations/` contained a probe token). Mitigation as planned: one token, port
  and path per session, server closed when the session ends, so a stored token points at a closed port.
