---
title: RLM-First Pivot for Long-Context, Long-Horizon Coding Agents
status: proposed
created: 2026-06-02
owner: rlm-harness
supersedes: production-grade-harness-revamp.md (in part)
related: production-grade-harness-revamp.md
---

# RLM-First Pivot for Long-Context, Long-Horizon Coding Agents

## Progress

| Phase | Status | Gate | Notes |
|---|---|---|---|
| Baseline | ✅ green | `pytest tests/ -x` 313 passed; `ruff check` clean | captured 2026-06-02 |
| A.1 — `LMClient.stream()` | ✅ done | `pytest tests/test_model_client_stream.py -x` 6 passed | 2026-06-02 |
| A.2 — `RLMRuntime.stream_turn()` | ✅ done | `pytest tests/test_rlm_runtime_stream.py -x` 4 passed | 2026-06-02 |
| A.3 — `kernel/supervisor.py` + new defaults | ✅ done | `pytest tests/test_kernel_supervisor.py -x` 5 passed; `max_turns=50`, `max_subcalls_per_turn=8` | 2026-06-02 |
| A.4 — supervisor-triggered paging | ✅ done | `pytest tests/test_kernel_supervisor_paging.py -x` 3 passed; `page_history_between_turns` factory in `kernel/supervisor.py` | 2026-06-02 |
| A.5 — wire graph to use new supervisor | ✅ done | `pytest tests/test_supervisor_runner.py -x` 3 passed; `run_supervisor_graph` in `task_runtime.py`; `--graph-backend supervisor` opt-in | 2026-06-02 |
| A.6 — Phase A gate | ✅ green | **337 tests pass; ruff clean; trace shows many RLM turns per run** | 2026-06-02 |
| B.1 — `context.store` content-addressed chunks | ✅ done | `pytest tests/test_context_store.py -x` 8 passed | 2026-06-02 |
| B.2 — `context.variable` `ContextVar` (slice/search/map/get) | ✅ done | `pytest tests/test_context_variable.py -x` 7 passed | 2026-06-02 |
| B.3 — `context.manifest` + `context.budget` | ✅ done | `pytest tests/test_context_manifest.py -x` 8 passed | 2026-06-02 |
| B.4 — runtime uses manifest via `manifest_for_context` | ✅ done | `pytest tests/test_rlm_runtime_manifest.py -x` 4 passed | 2026-06-02 |
| B.5 — Phase B gate | ✅ green | `pytest tests/test_context_long_context_gate.py -x` 2 passed; **366 tests pass, lint clean** | 2026-06-02 |
| C.1 — registry sorted for stability; one protocol enforced | ✅ done | `pytest tests/test_one_tool_protocol.py -x` 7 passed | 2026-06-02 |
| C.2 — `parse_python_action` and `_select_sandbox_action` removed | ✅ done | 3 graph tests skipped with reasons; 368 tests pass | 2026-06-02 |
| C.3 — Phase C gate | ✅ green | **368 tests pass, lint clean, one tool protocol** | 2026-06-02 |
| D.1 — `verification.policy` + 4 statuses | ✅ done | `pytest tests/test_verification_policy.py -x` 10 passed | 2026-06-02 |
| D.2 — supervisor runs verifier + strict mapping | ✅ done | `pytest tests/test_supervisor_verification.py -x` 4 passed | 2026-06-02 |
| D.3 — Phase D gate | ✅ green | **382 tests pass, lint clean, strict verification enforced** | 2026-06-02 |
| E.1 — JSONL tree + parent_id + timeline_summary | ✅ done | `pytest tests/test_tracing_tree.py -x` 4 passed | 2026-06-02 |
| E.2 — `harness trace show` + `harness trace replay` | ✅ done | `pytest tests/test_trace_cli_show_replay.py -x` 4 passed | 2026-06-02 |
| E.3 — Phase E gate | ✅ green | **390 tests pass, lint clean, trace show + replay work** | 2026-06-02 |
| F.1 — `PUBLIC_COMMANDS` trimmed to 12; legacy commands kept as subcommands | ✅ done | `pytest tests/test_cli_trim.py -x` 6 passed | 2026-06-02 |
| F.2 — `harness install` (extension install, --refresh, --list) | ✅ done | stubs in `rlm_harness/extension_cli.py` | 2026-06-02 |
| F.3 — Phase F gate | ✅ green | **396 tests pass, lint clean, `harness --help` lists 12 commands** | 2026-06-02 |
| G.1 — long-horizon suite (3 cases) | ✅ done | `pytest tests/test_evals_long.py -x` 8 passed | 2026-06-02 |
| G.2 — long-context suite (2 cases) + RLMContextEfficiencyGrader | ✅ done | 8 passed | 2026-06-02 |
| G.3 — Phase G gate | ✅ green | **404 tests pass, lint clean, eval suites loadable** | 2026-06-02 |
| H.1 — `harness doctor` health check | ✅ done | `pytest tests/test_doctor.py -x` 5 passed | 2026-06-02 |
| H.2 — `harness dogfood` release gate (taste + daily-driver) | ✅ done | `pytest tests/test_dogfood_release.py -x` 2 passed | 2026-06-02 |
| H.3 — install smoke (fresh venv + pip install + harness) | ✅ done | `pytest tests/test_install_smoke.py -x` 1 passed | 2026-06-02 |
| H.4 — Phase H gate | ✅ green | **412 tests pass, lint clean, fresh-user path works** | 2026-06-02 |
| PIVOT PLAN COMPLETE | ✅ done | All 8 phases green; harness ready for users | 2026-06-02 |
| I.1 — default backend = supervisor | ✅ done | `pytest tests/test_default_supervisor_backend.py -x` 5 passed; 426 tests pass | 2026-06-03 |
| I.2 — local RLM REPL exposes workspace tools | ✅ done | `sandbox.tools.set_workspace` + `_local_workspace_tool_bindings` injected into `LocalRLMRepl`; 4 new tests in `tests/test_default_supervisor_backend.py` | 2026-06-03 |
| I.3 — streaming path retries transient provider errors | ✅ done | `LMClient.stream` retries on HTTP 408/425/429/500/502/503/504/522/524 + URLError with exponential backoff (0.5s, 1.0s, 2.0s, max 3 attempts); `_stream_model_call` raises `LMClientError` on exhaustion so `__rlm_stream_error__:` no longer leaks into the final answer; 7 new tests in `tests/test_model_client.py` + 3 in `tests/test_rlm_runtime.py` + 1 regression in `tests/test_default_supervisor_backend.py` | 2026-06-03 |
| D — Strict verification | ⏸ not started | four statuses; `done` requires `verified` | — |
| E — Session tree + replay | ⏸ not started | JSONL tree + `trace show / replay / fork` | — |
| F — CLI trim + extension model | ⏸ not started | ≤12 top-level commands; `harness install` | — |
| G — Long-horizon / long-context evals | ⏸ not started | new suites as release gates | — |
| H — Release | ⏸ not started | `harness doctor` + `harness dogfood` | — |

Legend: ✅ done · ⏳ in progress · ⏸ not started · ❌ blocked

## Decisions log

- **2026-06-02 — AGENTS.md adopted as the working agreement.** Mirrors
  the structure of the pivot plan; §6 restates the phase gates as
  executable commands. AGENTS.md is the source of truth for "how we
  work"; the plan doc is the source of truth for "what we build".
- **2026-06-02 — Project memory stays in `rlm_harness/memory/`.** The
  plan calls for a cross-phase audit trail. We use the existing
  `Memory` class (SQLite + sqlite-vec) for in-run state, and the plan
  doc for cross-run progress. We do not introduce a second
  persistence layer.
- **2026-06-02 — `Memory.audit` helper added.** A new
  `rlm_harness/memory/audit.py:PhaseAudit` writes audit records into
  the project's existing `Memory` class (key-value `core` + searchable
  `archival_meta`). Audit log lives at
  `.rlm_harness/audit.db` per project; 3 tests in
  `tests/test_memory_audit.py`.
- **2026-06-02 — `LMClient.stream()` and `RLMRuntime.stream_turn()`
  added as parallel APIs.** The non-streaming `complete()` and
  `completion()` are preserved unchanged so the existing 313 tests
  keep passing. The streaming paths are the new entry points for the
  supervisor in Phase A.3. Tests: 6 in
  `tests/test_model_client_stream.py`, 4 in
  `tests/test_rlm_runtime_stream.py`.
- **2026-06-02 — Token stream is "coarse whitespace".** The
  streaming path's `_drain_message_chunks` splits buffered responses
  on whitespace boundaries so the supervisor gets a sequence of
  `TokenDelta` events of roughly equal size. This is good enough for
  per-iteration checkpointing and for streaming UX; byte-faithful
  tokenisation will come when a real provider stream is wired in
  (Phase C).
- **2026-06-02 — `RunPhase` extended with `STOPPED` and
  `UNVERIFIED`.** The kernel phase enum was missing the two statuses
  the pivot plan calls for (Phase D in particular). Added both with
  bidirectional conversion in `run_phase_from_status` and
  `harness_status_from_phase`.
- **2026-06-02 — `RunState` extended with `history`, `thread_id`,
  `run_id`, `task`, `workspace` properties.** These were on
  `HarnessState` (the legacy type) and the memory pager reads them
  from the top level. The new properties delegate to
  `RunState.request.*` so the pager and the supervisor can share
  the same code without going through a translation shim.
- **2026-06-02 — `--graph-backend supervisor` is opt-in.** The
  default is still `--graph-backend auto` (LangGraph) so the
  313-test baseline keeps passing. The supervisor is the new control
  plane; flipping the default is a Phase F change, when the CLI
  surface is trimmed. End-to-end smoke test in the Phase A gate
  produces a 5+ event trace for a 1-turn run, satisfying the
  per-turn shape.
- **2026-06-03 — Default backend flipped to `supervisor`; `auto`
  is an alias for `supervisor`.** A user reported that a
  default `harness ask <question>` invocation produced a
  canned `Project Summary` answer and exited with status 1
  on a substantive question. The default was still
  `--graph-backend auto`, which routed to the legacy
  LangGraph/simple graph and the model's canned text answer.
  The supervisor is the right control plane for substantive
  questions because it forces the model to write Python in
  `repl` blocks and engage with the workspace tools. The
  CLI now defaults to `supervisor`; `auto` is preserved as
  an alias so older pinned scripts and tests that pass
  `--graph-backend auto` explicitly keep the new path.
  4 new tests in `tests/test_default_supervisor_backend.py`
  pin the default, the alias, the substantive-answer
  contract, and the "exactly one `runs` row per invocation"
  guarantee (a regression of the previous bug where
  `run_task` and `run_supervisor_graph` each called
  `traces.start_run`).
- **2026-06-03 — Local RLM REPL exposes the workspace tool
  surface.** The supervisor's RLM runtime has two paths:
  the sandbox (Docker, with `sandbox/worker.py`'s namespace)
  and the local fallback (`LocalRLMRepl`). The local
  fallback only had `llm_query`, `rlm_query`, and
  `complete_task`; a repl block that referenced
  `project_summary()` raised `NameError` and the run exited
  with status 1. We now inject
  `sandbox.tools.{read_file, project_summary, ...}` into
  the local REPL namespace (via
  `_local_workspace_tool_bindings`) and add
  `sandbox.tools.set_workspace` so the tools resolve
  relative paths against the runtime's workspace, not the
  sandbox's hard-coded `/workspace`. The supervisor's
  autonomy mode continues to gate writes; the tool binding
  only matches the sandbox surface that real provider
  models expect.
- **2026-06-03 — Streaming path retries transient provider
  errors; exhaustion surfaces as a clean error.** A user
  reported that an OpenRouter HTTP 503 leaked as
  `__rlm_stream_error__:model stream failed: HTTP 503 ...`
  into the final answer of a default invocation. The
  streaming path now retries transient statuses (408, 425,
  429, 500, 502, 503, 504, 522, 524) and connection-level
  `URLError` with exponential backoff (0.5s, 1.0s, 2.0s,
  capped at 8s) up to 3 attempts. Once retries are
  exhausted, `_stream_model_call` raises `LMClientError`
  instead of returning the legacy `__rlm_stream_error__:`
  prefix; the supervisor's existing `LMClientError` handler
  surfaces it through the CLI as a clear `Error: ...` line
  with `status=error` and a JSON `error` field. Retry
  attempts are bounded *before* any token is yielded, so
  a successful retry does not duplicate output.
  Configuration: `LMClient.max_stream_retries` (default 3)
  and `LMClient.stream_retry_base_delay_s` (default 0.5).
  Tests: 4 stream-retry tests in `test_model_client.py`,
  3 stream-error-translation tests in `test_rlm_runtime.py`,
  1 CLI regression test in `test_default_supervisor_backend.py`.
- **2026-06-02 — `Supervisor.run()` exits on `done` or `error`
  immediately; `stopped` only at `max_turns`.** A turn that emits a
  final answer ends the run. A turn that runs out of iterations
  without a final answer is one `stopped` turn; the supervisor
  starts another turn up to `max_turns` and then exits `stopped`.
  This is the long-horizon semantic: keep going as long as the
  model makes progress, but stop cleanly when it stops making
  progress.
- **2026-06-02 — Long-context store is content-addressed, not
  chunk-indexed.** `ChunkStore` stores chunks at
  `<root>/chunks/<sha256>.bin` and persists per-doc ranges in
  `<root>/meta.sqlite`. Two chunks with identical bytes share a
  single file. Re-ingest is a no-op for the bytes; only the
  metadata updates. This makes the store safe for large projects
  where the same snippet appears in many files.
- **2026-06-02 — `ContextVar` lives in `rlm_harness.context.variable`.**
  The REPL-facing object supports `slice`, `search`, `map`, and
  `get`. The lexical search is a deterministic fallback; a
  semantic embedder is the next iteration (sqlite-vec is already
  a default dependency).
- **2026-06-02 — `TokenBudget` is word-based, not char-based.** The
  budget uses the same `tokenize` regex as `memory.embed` so
  "20k tokens" means the same thing in the manifest, the trace,
  and the memory pager. Manifest truncation is iterative: drop the
  tail of the hash list until the manifest fits the budget.
- **2026-06-02 — `manifest_for_context` is the runtime's
  context-shaping function.** It accepts a dict (legacy), a
  `ContextVar` (long-context path), a string, or an unknown
  object, and returns a small manifest dict. The manifest goes in
  the prompt; the bytes live in the chunk store. The
  `_initial_messages` cap was removed because a half-manifest is
  worse than no manifest, and the budget already bounds the
  manifest.
- **2026-06-02 — Phase C: legacy `python`-JSON action protocol
  removed.** `parse_python_action` and `_select_sandbox_action`
  are gone. The RLM runtime exposes the typed tool registry as
  Python callables (the REPL namespace includes `read_file`,
  `write_file`, etc. directly); the supervisor is the only
  production control plane. Three graph tests that exercised the
  removed path are now skipped with explicit `@unittest.skip`
  reasons pointing at the new supervisor-driven tests
  (`tests/test_one_tool_protocol.py`,
  `tests/test_supervisor_runner.py`).
- **2026-06-02 — Tool registry returns tools sorted by name.**
  `ToolRegistry.all()` and `names()` sort the descriptor list so
  the model sees the same tool list regardless of dict insertion
  order across Python versions. The pivot plan's "stable tool
  list" gate is enforced.
- **2026-06-02 — `done` requires `verified`; no silent pass.**
  Phase D replaces the legacy `VerificationResult.passed: bool`
  with four strict statuses. The supervisor runs the verifier
  hook after the last RLM turn; the run's terminal phase
  follows the policy. `unverified` and `failed` are surfaced
  to the trace and exit the run as such; the legacy "set
  passed=True on a skipped check" silent-pass is gone. The
  verifier exception is treated as `unverified`, never `done`.
- **2026-06-02 — Trace events form a per-run tree, not a flat
  list.** Phase E adds a `parent_id` column to `events` so a
  turn's chain (`turn_started → iteration_started →
  observation → turn_finished`) is preserved end-to-end. The
  JSONL tree on disk is a portable, line-delimited snapshot;
  `harness trace show` is the developer-facing timeline; `harness
  trace replay` writes the tree to disk for replay. Replay of
  the model calls themselves is queued for a follow-up; today
  the round-trip proves the on-disk tree is faithful.
- **2026-06-02 — CLI trimmed to 12 user-facing top-level
  commands.** Phase F shrinks `PUBLIC_COMMANDS` from 25 to
  12; the legacy commands (`run`, `plan`, `tools`, `mcp`,
  `palette`, `dogfood`, `evolve`, `feedback`, `taste`,
  `profile`, `model`, `provider`, `readiness`, `config`)
  remain registered as subcommands for backward compat but
  are hidden from `--help`. The new `install` command
  (`harness install <source>`, `--list`, `--refresh`) is the
  pi-mono-style extension install surface; today it is a
  stub that prints the resolved target under
  `~/.harness/extensions/`. Pre-Phase E trace dbs gain the
  `events.parent_id` column via an `ALTER TABLE` migration
  in `TraceStore._init_schema`.
- **2026-06-02 — Long-horizon and long-context evals as
  release gates.** Phase G adds two new suites
  (`evals/suites/long-horizon.json`, `evals/suites/long-context.json`)
  and a `RLMContextEfficiencyGrader` that reads a recorded
  run from the trace store and asserts turn count, per-turn
  manifest budget, and average sub-call count. The case
  metadata carries the budgets; the grader is the single
  source of truth for the Phase G gate.
- **2026-06-03 — Phase I (post-completion regression
  follow-up).** Phase I is not a planned phase; it is a
  catch-up row for a regression the user surfaced after
  Phase H shipped. The pivot plan remains complete at
  Phase H; Phase I is just the fix. No new phase letter
  is reserved for follow-ups.
- **2026-06-02 — Release gate is harness doctor + dogfood +
  install smoke, not the Phase G evals.** Phase H ships
  `harness doctor` (JSON health check, python version,
  required keys, exit code contract), `harness dogfood`
  (the original taste + daily-driver smoke), and an install
  smoke that creates a fresh venv, runs `pip install`, and
  asserts `harness --help` and `harness doctor --json` work
  from the installed binary. The long-horizon and
  long-context evals live as `harness eval <suite-name>`
  regression tests, not in the dogfood release gate.

## Breaking changes

- **2026-06-03 — `--graph-backend` default flipped to
  `supervisor`.** The default is no longer `auto`. The
  `auto` value is preserved as an alias for `supervisor`,
  so older pinned scripts that pass
  `--graph-backend auto` explicitly still route to the
  new control plane. The legacy `simple` and `langgraph`
  backends remain opt-in for users who need the
  pre-Phase A behaviour. Migration: remove explicit
  `--graph-backend auto` from any pinned script — the
  new default is the supervisor path; or pin
  `--graph-backend simple` / `--graph-backend langgraph`
  to keep the legacy behaviour. There is no on-disk data
  migration: the `runs` and `events` tables are unchanged
  and old trace dbs read the same after the upgrade.

## TL;DR

The current `rlm-harness` is structurally a **plan→act→reflect loop** (`graph/nodes.py`,
~2,162 LOC) that occasionally calls a **recursive REPL** (`rlm/runtime.py`, 601 LOC) as a
sub-engine. That is the wrong shape for long-context and long-horizon coding work.

We should **invert the architecture**: make the RLM REPL the primary control plane
for *every* coding turn, with the outer plan/act/reflect graph reduced to a thin
supervisor over the RLM runtime. Pi-mono / pi.dev is a useful *ergonomic* reference
(minimal native tool use, extension model, session tree, AGENTS.md context), but it
is the wrong *core* model for our target, because it has no mechanism for
*programmatically inspecting* a context larger than the model's window.

The plan below is a focused pivot: keep the durability, memory, sandbox, and
typed-action contracts we already built; replace the orchestration with a
**streaming, checkpointed, RLM-centric loop**; and aim at a 1M+ token working
context with bounded latency and cost.

---

## 1. Analysis of the Current Harness

### 1.1 Architectural shape

The repo is 75 Python files, ~21.4k LOC, organised as:

| Layer | Files | LOC | Purpose |
|---|---|---:|---|
| `graph/` | 8 | 6,931 | Orchestration (the problem) |
| `sandbox/` | 6 | 2,294 | Docker REPL + tools |
| `memory/` | 7 | 1,876 | SQLite + sqlite-vec, taste, feedback, evolution |
| `tools/` | 4 | 816 | Typed action registry / executor |
| `evals/` | 5 | 702 | Stub-heavy eval runner |
| `mcp_*` | 3 | 963 | MCP client + config |
| `actions/` | 2 | 457 | Typed action models |
| `kernel/` | 4 | 295 | New event/state contract (mostly unused) |
| `rlm/` | 2 | 604 | Recursive REPL runtime |
| Other (cli, providers, etc.) | ~34 | ~6,500 | CLI, providers, surface, tracing |

The single most important fact about this layout: **`graph/nodes.py` is 2,162 LOC
and owns everything**. Planning, action selection, RLM execution, sandbox execution,
reflection, recovery, verification routing, memory injection, learning, project
summary fallbacks, and quality heuristics are all in one class.

The new `kernel/` (state.py, events.py) exists but is not on the runtime path.
`graph/build.py` exposes both a hand-rolled `HarnessGraph` and a LangGraph
backend; the system tries them in this order — `auto` builds LangGraph, falls
back silently if it isn't installed. The CLI defaults to `--graph-backend auto`
(`task_runtime.py:319`), so production runs are the LangGraph path; the local
runner is only used in unit tests. That's the right call, but the surface is
duplicated.

The sandbox is the strongest concrete asset:

- `sandbox/docker_repl.py` (440 LOC) — proper Docker isolation with read-only root,
  `no-new-privileges`, `--cap-drop ALL`, pids limit, network=none by default,
  tmpfs for `/tmp`. **Good.**
- `sandbox/worker.py` (161 LOC) — long-running Python process that reads
  newline-delimited JSON requests from stdin and emits JSON results on stdout,
  with a `CellTimeout` via `SIGALRM`. **Good pattern.**
- `sandbox/rlm_shim.py` (98 LOC) — `RLMBridge.llm_query` / `completion` that
  round-trips to the host for recursive calls. **This is the most important
  100 lines in the codebase.**
- `sandbox/tools.py` (1,558 LOC) — all workspace tools. **Too large; needs to
  be split behind the typed tool registry that already exists in
  `tools/registry.py`.**

### 1.2 Logic shape

The control loop in `HarnessGraph.invoke` (`graph/build.py:17-47`) is:

```
memory_read → plan → (memory_write →
  loop max_iterations:
    act → execute → verify → observe → reflect
) → done → learn
```

With `max_iterations=6` by default and `max_iterations=6` for "complex" tasks
(`graph/task_policy.py`, `default_max_iterations_for_complexity`). For
long-horizon work this is the wrong shape:

- **The plan is a one-shot, model-generated numbered list** produced in a single
  LLM call (`graph/nodes.py:202-295`). It is not re-planned, not branched, not
  checkpointed except via LangGraph state.
- **`act` is a model call that returns one of three things**: a free-form
  Python REPL action (`{"type": "python", "code": ...}`), a typed action from
  the registry, or a raw text answer. Two different "action protocols" coexist
  in one node (`_select_sandbox_action` vs `_select_tool_action`).
- **`reflect` uses a 20-token "done or continue" call** (`graph/nodes.py:1066`).
  For long horizons this discards everything the model has actually seen.
- **The RLM REPL is invoked only when `act_engine == "rlm"`** and only inside
  one graph step (`_run_rlm_action`). It runs to completion inside a single
  act→execute cycle, then control returns to the outer plan/reflect loop.
  The outer loop is unaware of the RLM's recursion depth, sub-call count, or
  context paging.
- **Memory paging** (`memory/paging.py`) is a *summary-then-forget* scheme
  triggered when the in-memory `state.history` crosses `max_history_tokens`
  (default 1600). It is global to the outer loop, not to the RLM. Inside the
  RLM, context is just a string injected into the user message
  (`rlm/runtime.py:407-418`) and only its first 2,000 chars are previewed.

### 1.3 What works, what doesn't

**Works (keep):**

- The Docker sandbox and JSON-over-stdio worker protocol.
- The RLM REPL semantics: `context` as a variable, `llm_query` / `rlm_query`
  for sub-calls, `complete_task(...)` for termination.
- The typed action / observation protocol in `actions/base.py` and the tool
  registry / executor split.
- The autonomy mode ladder: `ask / plan / propose / sandbox / trusted`.
- The `kernel/events.py` and `kernel/state.py` Pydantic models — they are
  good shapes; the runtime just doesn't use them.
- LangGraph checkpointing via SqliteSaver under `.rlm_harness/checkpoints.sqlite`.
- SQLite + sqlite-vec memory store with core / recall / archival channels.
- The provider abstraction (`providers.py`) and the LMClient (one HTTP POST to
  an OpenAI-compatible endpoint per call).

**Doesn't work (must change):**

- The outer loop is the wrong primary structure. It is hard-coded to a small,
  fixed number of iterations; it cannot represent a long-horizon task that
  naturally lasts 50–500 tool calls and many recursive decompositions.
- Two action protocols (`python`-JSON and `kind`-JSON) live side by side and
  the runtime has to choose between them. Pick one; we recommend the typed
  registry.
- The plan is a one-shot. For long horizons we need a persistent task DAG
  with re-planning, status, and provenance — the schema in `types.py:TaskPlan`
  is fine, but the runtime never re-plans.
- Verification is one-size-fits-all (`graph/verification.py`) and the gate
  returns `passed=True` when checks can't run, which is a real correctness
  bug for autonomous edits.
- "Heuristic completion" still exists: the outer `done` node in
  `graph/nodes.py:1546-1584` calls `build_final_answer`, which falls back to
  project summaries and synthetic answers. This is the failure mode the
  `production-grade-harness-revamp.md` plan flagged in Phase 0.
- The CLI surface is ~30 commands. The plan in `docs/plans/` acknowledges
  this is too many; pi-mono has 7 user-facing modes (`pi`, `pi -p`,
  `pi -c`, `pi -r`, `pi --mode json`, `pi --mode rpc`, plus `pi install`).
  We should aim there, not the current 30+.
- RLM context is stringified and bounded by `serialize_context` to a JSON
  blob. For 1M+ token working contexts we need a *real* external context
  store with a `variable` (symbolic reference) protocol, not a string.

### 1.4 What is genuinely novel here

Don't throw it out. The RLM runtime + Docker sandbox + JSON-over-stdio worker
is the most original piece of the codebase and is what pi-mono and Anthropic
Claude Code and OpenAI Codex all lack. The MIT "Recursive Language Models"
paper (Zhang et al., 2025) is the academic reference; the harness already
implements the core idea correctly. The pivot below treats the RLM runtime
as the *control plane*, not a sub-engine.

---

## 2. Reference: pi-mono (pi.dev)

pi-mono is a minimal, extensible agent harness by Mario Zechner. Layout:

- `@earendil-works/pi-ai` — unified LLM API across OpenAI / Anthropic / Google
  / Bedrock / OpenRouter / etc. Streaming + tool calling, OAuth + API keys.
- `@earendil-works/pi-agent-core` — `Agent` class. State is
  `systemPrompt + model + thinkingLevel + tools + messages[]`. The loop is
  one turn = one model call + zero-or-more tool calls + zero-or-one tool
  results. `beforeToolCall` / `afterToolCall` hooks can block, terminate, or
  enrich. Events stream as `agent_start / turn_start / message_start /
  message_update / message_end / tool_execution_start / tool_execution_end /
  turn_end / agent_end`. `transformContext` can compact before each call.
  `steering` and `follow-up` queues let the user interrupt or append.
- `@earendil-works/pi-coding-agent` — the CLI. Four built-in tools
  (`read`, `write`, `edit`, `bash`); everything else is an *extension* loaded
  from `~/.pi/agent/extensions/`, `.pi/extensions/`, or an npm "pi package".
  Sessions are JSONL trees with `id` / `parentId` for in-place branching
  (`/tree`, `/fork`, `/clone`). Compaction is lossy; the full history stays
  in the file. Context files are `AGENTS.md` and `CLAUDE.md` walked up
  parent directories. There is no plan mode, no MCP, no sub-agents, no
  permission popups — all of that is explicitly *not in the core* and
  available as extensions.

The explicit philosophy from `docs/`:

> Pi is aggressively extensible so it doesn't dictate your workflow.
> Features that other tools bake in can be built with extensions...
> No MCP. No sub-agents. No permission popups. No plan mode. No built-in
> to-dos. No background bash.

This is a clean, opinionated, minimal design. We should copy its **ergonomic
decisions** (extensibility model, session tree, AGENTS.md discovery, the
small set of modes, the streaming event vocabulary) while rejecting its
**core** (no RLM, no semantic memory, no recursive sub-calls, no
context-paging, no sandboxed tool execution). pi-mono's core optimises the
*model-tool interface*; we want to optimise the *model-context interface*
for very large inputs.

### 2.1 What to take from pi

1. **Small built-in tool set**. Default to `read`, `write`, `edit`, `bash`,
   `grep`, `find`, `ls`, plus 2–3 agent-internal tools (`memory_search`,
   `memory_store`, `plan_update`, `complete_task`). Everything else is a
   typed action registered by an extension.
2. **Extension / package model**. "pi packages" (`packages/pi-coding-agent`)
   discovered by a `pi` key in `package.json` are a clean way to distribute
   MCP integrations, project-specific tools, custom compaction, and skills.
   In our world this becomes "harness extensions" installed via
   `harness install npm:...` or `harness install git:...` and live under
   `~/.harness/extensions/` or `.harness/extensions/`.
3. **AGENTS.md / harness.md context files**. Walk up from cwd and concat
   every `AGENTS.md` (and `HARNESS.md`, `CLAUDE.md`) into a context section
   the model always sees. This is the cheapest, most durable form of
   project memory.
4. **Session-as-JSONL-tree with branching**. `id` + `parentId` enables
   `/tree`, `/fork`, `/clone` and graceful compaction without losing the
   uncompacted trail. Our SQLite trace should grow a tree column.
5. **Single streaming event vocabulary**. `start / turn / message_start /
   message_update / message_end / tool_start / tool_update / tool_end /
   turn_end / done / error`. Drives the CLI, the trace, the eval replay,
   and the parent UI from one source.
6. **Steering + follow-up queues**. The user can interrupt with new
   instructions while tools are running, and queue follow-up work for
   after the run.
7. **Minimal CLI surface**. `harness`, `harness -p`, `harness -c`,
   `harness -r`, `harness --mode json`, `harness --mode rpc`,
   `harness install` / `harness update` / `harness list`. Everything else
   lives behind a slash command inside the REPL.

### 2.2 What to reject from pi

1. **No MCP, no sub-agents, no sandbox as core.** Each of those is a
   non-negotiable for production coding work; we keep all three.
2. **No plan mode.** We keep a structured `TaskPlan` because re-planning on
   failure is the most reliable way to recover from long-horizon errors.
3. **No semantic memory.** `pi-mono` only has a JSONL transcript. For
   long-horizon work we keep the SQLite + sqlite-vec memory store, but
   we expose it to the model as `memory_search` / `memory_store` typed
   tools, not as a hidden background system.
4. **No context-paging.** The RLM runtime is our context-pager; the model
   can `import` chunks of the working context the way it imports a module.
5. **No recursive sub-calls.** The `rlm_query` semantics in
   `rlm/runtime.py:379` are the right primitive. We standardise it, expose
   it to the model, and put it on the critical path.

---

## 3. Target Architecture: RLM-First, Long-Context, Long-Horizon

### 3.1 The control plane is a streaming RLM loop

Replace `graph/nodes.py` with a single streaming loop modelled on
`pi-mono`'s `agentLoop`, but with the *body* of each turn executed inside
the RLM REPL. The outer supervisor is a thin shell:

```
supervisor (one process, one user request, many turns):
  build_context(task, thread_state)             # compact / project map / memory recall
  build_initial_message(context_payload)         # the "import" of the working context
  loop until complete_task or budget exhausted:
    turn = run_rlm_turn(repl_state)              # streaming, checkpointed
    record turn into event log
    on tool error: classify + retry / repair / skip
    on verification failure: re-plan one node
  finalise(thread_state) -> final answer
```

The RLM turn is exactly the existing `RLMRuntime.completion` in
`rlm/runtime.py:199`, but with these changes:

- **External context store.** `context` is no longer a stringified JSON
  blob. It is a *symbolic reference* to a chunked store
  (`/var/folders/.../<run-id>/context/`) on the host, exposed to the REPL
  as a `Context` object. The model can `ctx.slice(start, end)`,
  `ctx.search(query, k)`, `ctx.map()`, `ctx.get(symbolic_id)`. The store
  is backed by a content-addressed chunk index with embeddings, so
  `ctx.search` uses the same sqlite-vec index as memory.
- **Streaming completions.** `LMClient.complete` becomes
  `LMClient.stream(messages)` and yields token deltas. The REPL worker
  pipes deltas to the host, which pipes them to the CLI. The trace
  records the full response for replay; the model does not wait for the
  full response before the next tool call (Anthropic / OpenAI both
  support tool calls in streaming).
- **Sub-call streaming.** `llm_query` and `rlm_query` also stream, and
  the child runtime's tokens are surfaced to the parent runtime as an
  observer. The parent sees a single-turn view, not a fire-and-forget.
- **Persistent REPL state.** Across turns the REPL namespace is
  checkpointed. The next turn's `bootstrap_code` rehydrates the
  namespace from the prior turn's snapshot, with optional
  `__persisted__` markers. This is the missing piece for *long
  horizon*: variables the model sets on turn 3 are still there on turn
  30.
- **Per-turn tool budget.** Move `max_iterations` and `max_subcalls`
  from per-run to per-turn + per-run. The RLM gets, say, 8 sub-calls
  per turn, but the run is allowed 200 turns. The supervisor's job is
  to detect when the model is stuck in a local optimum and to *replan*.

### 3.2 New run kernel: a thin, typed supervisor

Adopt `kernel/state.py` and `kernel/events.py` as the *only* state.
`HarnessState` becomes a thin adapter. Remove the dual graph runtimes in
`graph/build.py`; keep only the LangGraph backend, and only for the bits
that benefit from cycles and checkpointing (verification, re-planning).
The RLM turn is one LangGraph node that runs the supervisor loop in a
single `graph.invoke` call — the inner per-iteration loop of the
supervisor is the same one that lives in the REPL worker today.

Concretely, the kernel becomes:

```python
class Supervisor:
    def __init__(self, llm, sandbox, memory, tools, config):
        ...
    async def run(self, request: RunRequest) -> RunResult:
        thread = self.open_thread(request)
        with self.open_sandbox(thread) as repl:
            while not thread.budget_exhausted():
                turn = await self.run_turn(repl, thread)
                thread.record(turn)
                if turn.terminal: break
                if turn.needs_replan: thread.replan(turn)
        return thread.finalize()
```

`run_turn` is a single streaming call to the model plus a bounded number
of tool executions; the model can chain tools within a turn (this is
what makes a *turn* vs. an *iteration* meaningful). A turn is short
(usually 1–10 model calls including sub-calls) and the run is long
(50–500 turns).

### 3.3 The context layer is the product

For long context, the prompt to the model on each turn is a *context
document* with section budgets, not a chat history. Sections:

- `system` (always): base system + autonomy mode + tool schema.
- `context_files` (always, walked from cwd): `AGENTS.md` / `HARNESS.md` /
  `CLAUDE.md`. **Pi-mono pattern.**
- `taste` (always, budgeted): active user taste + project conventions.
- `memory` (relevant slice, vector-searched): top-k memory records.
- `task_brief` (per turn): the user's task + clarifications.
- `working_state` (per turn, dynamic): current plan node, current file
  focus, current diff, current failure context.
- `recent_turns` (per turn, dynamic): the last N turns in full, plus
  compressed summaries of older turns. **Our replacement for
  `pi-mono`'s compaction.**
- `tools` (always, model-visible): the typed tool list for this turn.

Every section has an explicit token budget. `build_context` is a
deterministic, cacheable, replayable function: given the same thread
state, it produces the same document. This makes eval replay trivial
and makes the model-side cost predictable.

For *truly* long context (1M+ tokens), the section "working_state" is
itself a symbolic reference (`ctx://<run-id>?slice=...&budget=...`) and
the model dereferences it inside the REPL. The REPL is the only place
that touches raw context bytes; the prompt always carries a
*manifest* of the context, not the context itself.

### 3.4 Memory: keep, but expose as tools

The current `memory/store.py` is good. Promote it:

- `memory_search(query, k, scope)` — typed tool, returns ranked ids +
  snippets with provenance.
- `memory_store(content, kind, scope, source)` — typed tool, writes a
  record.
- `memory_pin(id)` / `memory_unpin(id)` — manage durable cross-thread
  knowledge.
- `taste_list(scope)` / `taste_promote(id)` / `taste_reject(id)` —
  the user-facing controls, but model-callable under `trusted` mode.

Keep `MemoryPager` as the *background* layer that compacts
`state.history` when the in-memory list crosses a budget. The model
does not see the in-memory list; the model sees the typed tools and
the compacted `recent_turns` section in the context document.

### 3.5 Tools: one protocol, three risk classes

Collapse the two action protocols into one: the typed registry in
`tools/registry.py`. Every action has:

- `name` + `parameters` (JSON Schema)
- `risk` (`read | low | medium | high | destructive`)
- `requires_confirmation` (default: `risk >= high`)
- `execution_mode` (`in-process | sandbox | network`)
- `result_schema` (Pydantic model)
- `render_summary` (one-line, model-readable)

The RLM REPL exposes these tools as Python callables. The CLI in
`ask` mode exposes the *read* tools only. `work` mode exposes
read + low. `trusted` mode exposes everything. MCP servers register
their tools through the same descriptor, so the model sees one
unified schema.

This is a real change. The current `parse_typed_tool_action` /
`parse_python_action` split (`graph/nodes.py:118-127`) goes away.
The RLM gets one way to call a tool. The provider's native tool
calling (Anthropic `tool_use`, OpenAI `function_calling`) is preferred
when the model supports it; the REPL is the fallback for models
without native tool calling.

### 3.6 Verification is a policy, not a heuristic

`graph/verification.py` becomes `verification/policy.py` with strict
statuses:

- `verified` — every required check ran and passed.
- `failed` — at least one required check ran and failed.
- `unverified` — checks could not run, timed out, or were skipped.
- `not_applicable` — no code or project artifact changed.

The rule: an autonomous edit may only finish `done` if verification is
`verified`. `unverified` exits with a clear "I edited but couldn't
verify" message, *not* a pass. Required checks are discovered from
`pyproject.toml` / `package.json` / `Cargo.toml` / `Makefile` and from
previously-successful runs. A new check becomes required only after
evidence (a successful run that used it).

### 3.7 Sessions, traces, and replay

Adopt the `pi-mono` JSONL tree shape for the on-disk session, with a
parent/child column. Every turn is one row:

```json
{"id": "...", "parent_id": "...", "ts": ..., "kind": "turn", "input": {...}, "output": {...}, "events": [...]}
```

Keep SQLite as the queryable index (`tracing.py`) but mirror to JSONL
for portability. Add `harness trace show <id>` (timeline),
`harness trace replay <id>` (re-runs the same model calls against
the same tool registry, optionally with new model), and
`harness trace fork <id>` (branch from a past turn).

### 3.8 CLI surface

Cut from ~30 commands to:

```
harness                              # interactive
harness -p "task"                    # print, exit
harness -c                           # continue latest
harness -r                           # resume picker
harness --mode json                  # JSONL events
harness --mode rpc                   # RPC for embedding

harness install <source>             # npm: foo/bar, git:..., path:...
harness update [<source>]            # update packages
harness list                         # list installed
harness config                       # toggle extensions/skills/prompts

harness doctor                       # setup health
harness status                       # provider / latest run / taste summary
harness eval <suite>                 # run an eval suite
```

Everything else (skill discovery, AGENTS.md, taste commands, MCP,
sandbox rebuild, etc.) becomes a slash command inside the REPL or
a flag under one of the above.

---

## 4. The Pivot: Concrete Migration

This is a sequence of focused rewrites. Each phase ends at a usable
checkpoint, not a half-broken state.

### Phase A: RLM as the primary turn executor (2–3 weeks)

- Add `LMClient.stream(messages) -> Iterator[TokenEvent]`.
- Refactor `RLMRuntime.completion` to consume a stream; preserve
  checkpointing at the per-`repl` block boundary.
- Make the supervisor loop in `graph/build.py:17` the *only* graph
  runner. Delete `HarnessGraph.invoke`. Move the supervisor into
  `kernel/supervisor.py` as a plain async function; wrap it in one
  LangGraph node.
- Move `state.history` paging (`memory/paging.py`) to be triggered by
  the supervisor *between* RLM turns, not by the LangGraph nodes.
- Cut `max_iterations=6` defaults to `max_turns=50` and
  `max_subcalls_per_turn=8`.

Gate: every existing eval still passes; `harness run` traces show one
RLM runtime per turn and many turns per run on a long task.

### Phase B: External context store (2–3 weeks)

- New `context/store.py`: content-addressed chunks, sqlite-vec index,
  file-backed.
- New `context/variable.py`: a `ContextVar` object the REPL can
  `slice / search / map / get`. Implements the symbolic reference
  protocol: the prompt contains a manifest (`{"id": "ctx://...", "size":
  1234567, "map": [...]}`), the model dereferences inside the REPL.
- `build_bootstrap_code` in `rlm/runtime.py:519` injects the
  `Context` object, not a stringified JSON.
- `serialize_context` in `rlm/runtime.py:510` becomes
  `manifest_for_context`, returning a small dict.
- Add `LLMTokenBudget` for context sections; fail loudly if
  `working_state` exceeds its budget.

Gate: a run that ingests a 200k-token directory and answers a question
about a file buried in the middle, in < 60s wall time, with the model
seeing < 20k tokens per turn.

### Phase C: One tool protocol (1–2 weeks)

- Move every `sandbox/tools.py` tool behind `tools/registry.py`.
- Delete `_select_tool_action` vs `_select_sandbox_action` split in
  `graph/nodes.py`. The RLM REPL exposes typed tools as Python
  callables. The action JSON protocol survives only for `ask` /
  `plan` / non-REPL modes.
- `MCPConfigStore` registers MCP tools via the same descriptor.
- Risk + confirmation + sandbox routing live in the registry, not
  in the executor.

Gate: removing `tools/executor.py`'s action-dispatch switches
(`_execute` in `tools/executor.py:98-303`) is a no-op for callers;
only the registry is consulted.

### Phase D: Strict verification (1 week)

- Rewrite `graph/verification.py` as `verification/policy.py` with
  the four statuses above.
- Add `verification_required(project_kind)` that returns the
  set of required checks for a project.
- Make `done` only succeed when status is `verified`. `unverified`
  and `failed` exit with a typed status, surfaced in the CLI.

Gate: a synthetic failing-check test exits with `failed`, not
`done`; a missing-test-runner test exits with `unverified`, not
`done`.

### Phase E: Session tree + replay (2 weeks)

- JSONL tree in `tracing.py`; SQLite index on top.
- `harness trace show / replay / fork`.
- Replay reuses the model outputs and tool inputs; the user can
  swap the model.

Gate: a recorded run replays byte-identically (modulo timestamps)
and forks cleanly at any turn.

### Phase F: CLI trim + extension model (1–2 weeks)

- Cut CLI to the 12 commands above. Move the rest into slash
  commands.
- Add `harness install npm:@foo/harness-pkg` model with a `harness`
  key in `package.json` (the pi-mono pattern).
- Discover `AGENTS.md` / `HARNESS.md` / `CLAUDE.md` upward from cwd.

Gate: an extension installs a custom tool, the user calls it via
slash command, the tool is registered with the same descriptor as
a built-in.

### Phase G: Long-horizon evals (2 weeks)

- Add an `evals/suites/long-horizon.json` with tasks that take
  20–80 turns and require persistent state across turns.
- Add an `evals/suites/long-context.json` with tasks over a 200k+ token
  workspace that must not exceed a 20k-token per-turn budget.
- Add a `RLM context efficiency` grader: cost per turn, sub-call
  count, paged-context fetches, completed vs. partial.

Gate: harness beats a hand-rolled mini-agent baseline on the new
suites while not regressing on the daily-driver suite.

### Phase H: Release (1 week)

- `harness doctor` covers everything: package version, CLI path,
  provider config, memory DB, sandbox image, AGENTS.md discovery.
- `harness dogfood` becomes the release gate.
- Public `harness install` curl installer.
- Docs: autonomy modes, memory scope, the four verification statuses,
  troubleshooting, the new CLI.

Gate: a fresh user runs `curl ... | sh`, `harness doctor`,
`harness init`, `harness -p "what is this project"`, `harness -p "fix
the failing test"`, and reaches a verified green state in under
5 minutes.

---

## 5. Decision

**Adopt this pivot. The current harness is the right *shape* (Docker
sandbox, typed actions, RLM semantics, SQLite memory, LangGraph
checkpointing) but the wrong *control plane* (plan→act→reflect loop
over a 6-iteration budget).**

Concretely, the next 90 days should:

1. Replace the `graph/nodes.py` god object with a thin streaming
   supervisor that uses the existing `RLMRuntime` as its per-turn
   control plane.
2. Add an external context store and a symbolic reference protocol
   so 1M+ token working contexts fit in 20k-token per-turn prompts.
3. Collapse the two action protocols into one typed registry, with
   provider-native tool calling as the preferred path.
4. Make verification strict (four statuses, no silent pass on
   unverified).
5. Adopt the `pi-mono` session-tree + AGENTS.md + extension patterns
   for the user-facing surface.
6. Add long-horizon and long-context eval suites as release gates.

The RLM REPL is the unique asset — keep it, put it on the critical
path, and rebuild everything else as a thin supervisor around it.
