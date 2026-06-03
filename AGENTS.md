# AGENTS.md — rlm-harness

This file is the working agreement for every agent (human or AI) that touches
`rlm-harness`. It is **load-bearing**: read it before any non-trivial change.
If a change conflicts with this file, update the file in the same PR.

The product truth lives in `docs/plans/rlm-first-pivot.md`. That plan is the
north star. This file describes **how** we execute against it.

---

## 1. The product, in one paragraph

`rlm-harness` is a **local-first, RLM-first coding agent harness** for
**long-context, long-horizon** software work. Its control plane is a
**streaming, checkpointed, recursive REPL** (the "RLM runtime"). The outer
supervisor is a thin shell that owns the plan, the budget, the
verification policy, and the memory store. We are *not* a JSON-action
agent harness. We are *not* a model that lives inside its own chat
history. We are a runtime that lets a model *programmatically* inspect,
paginate, and recurse over a working context that can be much larger
than the model's window.

---

## 2. The plan

- **North star:** `docs/plans/rlm-first-pivot.md` (status: in execution).
- **Prior plan:** `docs/plans/production-grade-harness-revamp.md` (status:
  superseded in part; keep the diagnostic sections).
- Phases (from the pivot plan, renumbered for execution):
  1. **A — RLM as primary turn executor.** Streaming completions,
     supervisor loop, per-turn budgets, per-RLM-turn checkpointing.
  2. **B — External context store.** Content-addressed chunks, symbolic
     references, manifest-based prompt.
  3. **C — One tool protocol.** Collapse `python`-JSON and `kind`-JSON
     into the typed registry; provider-native tool calling when
     supported.
  4. **D — Strict verification.** Four statuses; `done` requires
     `verified`.
  5. **E — Session tree + replay.** JSONL tree on disk, SQLite index,
     `harness trace show / replay / fork`.
  6. **F — CLI trim + extension model.** ~12 user-facing commands;
     `harness install npm:foo` for extensions.
  7. **G — Long-horizon / long-context evals.** New suites as release
     gates.
  8. **H — Release.** `harness doctor` covers install; `harness dogfood`
     is the release gate; public curl installer.

Each phase ends at a *gate* — a concrete, runnable test or command that
must pass before the next phase starts. Gates are listed in the pivot
plan; they are restated in §6 below.

---

## 3. The way of working

### 3.1 Definition of done (every PR)

- [ ] Tests added for any new behaviour; existing tests still pass.
- [ ] `pytest tests/ -x` is green from a clean checkout.
- [ ] `ruff check rlm_harness tests` is clean.
- [ ] No silent fallbacks. If a feature is conditional, the
      conditional is surfaced in the CLI / API, not hidden behind
      try/except.
- [ ] No new public function/CLI flag without a docstring + a test.
- [ ] Any breaking change to the public CLI is called out in
      `docs/plans/rlm-first-pivot.md` under "Breaking changes".

### 3.2 How we touch the codebase

- **The RLM REPL is the control plane.** Do not add orchestration
  logic in `graph/nodes.py` that bypasses the REPL. The REPL is the
  place where the model inspects context, recurses, and emits
  structured actions.
- **Two action protocols are not allowed.** The split between
  `python`-JSON and `kind`-JSON actions is being collapsed in Phase C.
  Until that lands, *new* code paths must use the typed registry in
  `rlm_harness/tools/registry.py`. Do not add new `parse_python_action`
  call sites.
- **One memory.** Project memory (`rlm_harness/memory/store.py`) is the
  source of truth for taste, recall, archival, and audit. New state
  goes into the schema, not into ad-hoc `state.scratch` dicts.
- **One trace.** All observability flows through `rlm_harness/tracing.py`
  (the SQLite-backed `TraceStore`). New event types are Pydantic
  models in `rlm_harness/kernel/events.py`, registered via
  `AnyRunEvent`. Do not invent a parallel logging system.
- **No silent verification.** If verification cannot run, the run
  exits with `unverified`, not `done`. If verification failed, the run
  exits with `failed`, not `done`. This rule is enforced in Phase D;
  do not preempt it by relaxing checks to "passed" in earlier phases.

### 3.3 How we work on a phase

1. **Read the gate.** Every phase in the pivot plan ends with a
   "Gate" line. That gate is the acceptance test. Read it first.
2. **Write the test that proves the gate.** A new test in
   `tests/test_<phase>.py`, or an updated assertion in an existing
   test file. The test should fail *before* the implementation and
   pass *after*.
3. **Implement the smallest change that turns the test green.**
4. **Run the full suite.** `pytest tests/ -x` must be green. If it
   isn't, fix it before claiming the phase done.
5. **Run lint.** `ruff check rlm_harness tests`.
6. **Update the plan doc.** Append a row to the progress table in
   `docs/plans/rlm-first-pivot.md`. Note any decisions that
   diverged from the original plan.
7. **Commit.** One commit per phase gate, with a message of the form
   `phase A: <short description>`. Do not bundle phases.

### 3.4 How we work on long-running / cross-phase work

- **No half-broken state.** A phase that does not pass its gate does
  not get merged. The branch stays in a working pre-phase state.
- **Out-of-order is fine, but loud.** If a later phase is partially
  done before an earlier one lands, that is acceptable as long as the
  earlier gate still passes. Mark the out-of-order work in the
  plan's progress table with the *actual* phase letter, not the
  next one.
- **Reversible by default.** A phase that is hard to roll back
  (e.g., a schema migration in `memory/store.py`) needs a downgrade
  path documented in the same PR.

### 3.5 What we never do

- We do not mirror vault briefs into the repo as markdown. The repo
  has its own plan, in `docs/plans/`. (We keep our own opinions.)
- We do not silently change the default `max_iterations`, autonomy
  mode, or verification status. Defaults are part of the contract.
- We do not introduce a new persistence layer (Redis, file-based
  state, etc.) without first justifying why `memory/store.py` cannot
  hold it.
- We do not bypass the typed action registry. Even "internal" tools
  go through the registry.
- We do not auto-merge behaviour changes without an eval case or
  user approval. The eval case lives in `rlm_harness/evals/suites/`.

---

## 4. Architecture rules (cheat sheet)

```
+------------------------------------------------------------------+
| User: harness / harness -p / harness -c / harness -r / rpc       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| Supervisor (rlm_harness/kernel/supervisor.py)                    |
|   - one supervisor per run                                       |
|   - owns RunState, budget, plan DAG, verification policy         |
|   - emits typed events to TraceStore                             |
+------------------------------------------------------------------+
                              |
                              v  (per turn)
+------------------------------------------------------------------+
| RLM Runtime (rlm_harness/rlm/runtime.py)                         |
|   - one runtime per turn                                        |
|   - streaming completion                                         |
|   - sub-calls: llm_query (direct) / rlm_query (recursive)        |
|   - context is a symbolic reference, not a string                |
|   - REPL state checkpointed between turns                        |
+------------------------------------------------------------------+
        |                       |                       |
        v                       v                       v
+----------------+   +-------------------+   +-------------------+
| Tool Registry  |   | Memory (SQLite +  |   | Sandbox (Docker)  |
| (typed actions)|   | sqlite-vec)       |   | REPL worker       |
+----------------+   +-------------------+   +-------------------+
```

Module map (canonical location for each concern):

| Concern | Module |
|---|---|
| Run lifecycle, supervisor, plan | `rlm_harness/kernel/` |
| Streaming completions, sub-calls, REPL | `rlm_harness/rlm/` |
| External context store | `rlm_harness/context/` (added Phase B) |
| Typed actions, tool registry, executor | `rlm_harness/tools/`, `rlm_harness/actions/` |
| Verification policy | `rlm_harness/verification/` (added Phase D) |
| Memory (taste, recall, archival) | `rlm_harness/memory/` |
| Sandbox (Docker, REPL worker) | `rlm_harness/sandbox/` |
| Trace, replay, session tree | `rlm_harness/trace/` (split out in Phase E) |
| Provider LLM client | `rlm_harness/model_client.py` |
| CLI | `rlm_harness/cli.py` + per-command CLIs in `rlm_harness/*_cli.py` |
| Evals | `rlm_harness/evals/` |

---

## 5. Test discipline

- **One test file per source file** where the source file has public
  behaviour. `rlm_harness/foo/bar.py` → `tests/test_bar.py` (or
  `tests/test_foo_bar.py` for `rlm_harness/foo/bar.py`).
- **Prefer Pydantic-typed fixtures over ad-hoc dicts.** Test inputs
  are Pydantic models whenever the production code is.
- **No mocking the model.** When a test needs a model, use a
  `ScriptedRuntimeClient` (see `tests/test_rlm_runtime.py`) or a
  stub that returns deterministic `Completion` objects.
- **No mocking the filesystem when an in-memory backend is
  available.** The `Memory` class supports `:memory:`; the trace
  store should too. The sandbox has a local fallback in
  `RLMRuntime._completion_with_local_repl`.
- **A failing eval is a release blocker.** Evals in
  `rlm_harness/evals/suites/` are part of `pytest tests/`.

---

## 6. Phase gates (executable form)

These are the gates from `docs/plans/rlm-first-pivot.md`, restated as
the commands a developer runs to confirm a phase is done.

```bash
# Phase A — RLM as primary turn executor
pytest tests/test_rlm_runtime.py -x
pytest tests/test_graph.py -x
pytest tests/test_kernel_contracts.py -x
# Trace from a long task should show many RLM turns per run.
harness run "add a no-op docstring to a function in this project" \
  --provider stub --model stub --max-turns 50 --max-subcalls-per-turn 8
harness trace show <run-id>    # expect: many turn rows, not 6

# Phase B — External context store
pytest tests/test_context_store.py -x
# 200k-token directory, < 60s, model sees < 20k tokens per turn
harness -p "what does main.py in this 200k-token project do?" \
  --max-turns 30
# Manifest size in trace < 20k tokens; full context size = raw.

# Phase C — One tool protocol
pytest tests/test_tools_registry.py -x
pytest tests/test_tool_executor.py -x
# Removing tools/executor.py's _execute switch is a no-op.

# Phase D — Strict verification
pytest tests/test_verification_policy.py -x
# Failing check exits with status=failed, status=unverified if
# checks cannot run, status=verified only if all pass.

# Phase E — Session tree + replay
pytest tests/test_tracing.py -x
harness trace show <run-id>     # tree shape
harness trace replay <run-id>   # byte-identical (modulo timestamps)

# Phase F — CLI trim + extension model
pytest tests/test_cli_config.py -x
# `harness --help` lists <= 12 top-level commands.

# Phase G — Long-horizon / long-context evals
pytest tests/test_evals.py -x
# Without these suites, Phase G is not done.

# Phase H — Release
bash scripts/install.sh     # clean install
harness doctor              # all green
harness dogfood             # all green

# Phase I — Default backend = supervisor (regression follow-up)
pytest tests/test_default_supervisor_backend.py -x
# A default `harness ask <question>` invocation must route through
# the supervisor (the new control plane) and exit with status=done.
# `--graph-backend auto` is an alias for `supervisor`; `simple` /
# `langgraph` remain opt-in for users who need the legacy paths.
```

---

## 7. Communication conventions

- **Progress is recorded in the plan doc.** Append a row to the
  progress table in `docs/plans/rlm-first-pivot.md` after every
  phase gate.
- **Decisions that diverge from the plan are recorded in the same
  doc, under "Decisions log".** No silent divergence.
- **No long brief content in chat.** The plan doc is the source of
  truth; chat is for questions, answers, and short status.

---

## 8. Anti-patterns (do not)

- ❌ A new "act engine" with a third action protocol.
- ❌ A new persistence layer that bypasses `memory/store.py`.
- ❌ A graph node that owns multiple responsibilities.
- ❌ A "fallback" that silently changes semantics.
- ❌ A test that only checks the happy path.
- ❌ A default that "felt right" without a docstring explaining why.
- ❌ A "TODO" with no owner or phase attached.
