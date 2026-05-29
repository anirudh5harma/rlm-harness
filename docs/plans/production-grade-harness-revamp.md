---
title: Production-Grade Coding Agent Harness Revamp
status: active
created: 2026-05-29
owner: rlm-harness
---

# Production-Grade Coding Agent Harness Revamp

## Problem Frame

The harness has the right ambition, but the current shape is still closer to a
patched local coding assistant than a production-grade coding-agent runtime.
Recent fixes improved project summaries, progress markers, LangGraph packaging,
and taste learning, but the failure mode the user called out is deeper: the
agent can produce a decent first answer and then degrade because orchestration,
tool use, verification, memory, and final response quality are not governed by a
single durable contract.

This plan is intentionally not an implementation patch. It is the execution map
for turning the project into a daily driver that downloaded users can run
locally, that learns their preferences over time, and that improves through
reviewable self-evolution instead of silently mutating itself.

## External References

Primary and project references used:

- [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): distinguishes workflows from agents, recommends starting simple, increasing complexity only when needed, and optimizing tool interfaces as much as prompts.
- [Anthropic, Claude Code best practices](https://code.claude.com/docs/en/best-practices): emphasizes persistent project instructions, context management, verification, permission modes, hooks, subagents, and checkpoints for agentic coding.
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): positions LangGraph as the runtime for durable execution, streaming, human-in-the-loop, persistence, and memory.
- [OpenAI Agents SDK guide](https://developers.openai.com/api/docs/guides/agents): frames production agents as code-owned orchestration with tools, state, approvals, guardrails, and observability.
- [OpenAI Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/): treats traces as a comprehensive record of LLM calls, tool calls, handoffs, guardrails, and custom events.
- [SWE-agent ACI docs](https://swe-agent.com/0.7/background/aci/): shows that repository-level coding agents improve substantially when the agent-computer interface is designed around LM-friendly commands and feedback formats.
- [OpenHands agent architecture](https://docs.openhands.dev/sdk/arch/agent): uses a stateless, event-driven reasoning/action loop with action events, observation events, confirmation checks, and security validation.
- [OpenHands event architecture](https://docs.openhands.dev/sdk/arch/events): treats action and observation events as the bridge between runtime execution and model-visible conversation history.
- [Aider repository map](https://aider.chat/docs/repomap.html): uses a compact, graph-ranked map of the repository to give the model whole-codebase context within a token budget.
- [Aider linting and testing](https://aider.chat/docs/usage/lint-test.html): feeds lint/test failures back into the agent loop and expects non-zero exits to become repair signals.
- [Aider git integration](https://aider.chat/docs/git.html): keeps AI changes visible, undoable, and reviewable through git.
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli): local terminal agent that reads, edits, and runs code in a selected directory, with regular releases and install/upgrade flow.

## Current Architecture Read

What exists now:

- `pyproject.toml` now packages `langgraph` and `langgraph-checkpoint-sqlite` as default dependencies. This is the right direction. LangGraph should not be an accidental optional feature for users.
- `rlm_harness/graph/build.py` has both a local graph runner and a LangGraph-backed runner with checkpointing. This gives us a migration path, but it also means there are two orchestration surfaces.
- `rlm_harness/types.py` has useful state types, but the run contract is too small for a production coding agent. It lacks first-class events, artifacts, permission decisions, changed files, verification contracts, and resumable step state.
- `rlm_harness/graph/nodes.py` is the biggest risk. It contains planning, action selection, RLM execution, sandbox execution, reflection, recovery, verification routing, memory injection, learning, finalization, project-summary fallbacks, and quality heuristics in one large node implementation.
- `rlm_harness/rlm/runtime.py` lets the model drive Python REPL blocks and call `complete_task`. This is powerful, but it is currently mixed with orchestration responsibility and fragile final-answer extraction.
- `rlm_harness/sandbox/docker_repl.py` has strong ingredients: Docker isolation, read-only root, resource limits, workspace mount, and optional no-network execution. It needs a clearer capability protocol and image/version freshness checks.
- `rlm_harness/sandbox/tools.py` offers practical primitives: read, write, patch, shell, git, project summary/audit, pending changes, and completion. It has become too broad and should be split behind a typed tool registry.
- `rlm_harness/memory/*` has local project/user memory, taste records, feedback, and evolution proposals. This is a real advantage, but promotion into behavior needs evidence, confidence, scope, rollback, and eval gates.
- `rlm_harness/graph/verification.py` discovers common checks, but verification must become stricter. Skipped, errored, or unavailable checks cannot be treated like success for production editing tasks.
- `rlm_harness/tracing.py` stores runs and events in SQLite, but the trace is not yet rich enough for replay, debugging, evaluation, or user-facing "what happened?" inspection.
- `rlm_harness/evals/*` and `rlm_harness/dogfood.py` provide starter proof loops. They need to grow from output string checks into trajectory, safety, verification, preference, and long-horizon coding evals.
- `README.md` describes the intended daily-driver surface well, but some claims are aspirational relative to runtime reliability.

## What We Are Doing Right

- Local-first design: user profile and project memory live locally, which is the right privacy boundary for downloaded users.
- LangGraph is now packaged by default, which supports durable execution rather than best-effort loops.
- Docker sandboxing exists and already has resource and network constraints.
- The tool surface includes useful developer primitives instead of only high-level workflows.
- Risky file changes can be queued as proposals.
- Taste learning, feedback, and evolution proposals exist as separate concepts.
- Readiness, dogfood, and eval commands create an install-to-proof path.
- The CLI is beginning to separate progress output from final answer output, which matters for scripting.

## What Is Wrong

- The harness lacks a typed action/observation/event contract. The model can emit JSON Python actions or REPL code, but the runtime does not have a durable, inspectable protocol for every decision and side effect.
- `Nodes` is a god object. Production agents need separable orchestration, context building, action selection, tool execution, verification, memory, and final response assembly.
- Completion is still partly heuristic. A production harness should rely on explicit completion events and typed statuses, with quality checks as validators rather than the primary control plane.
- RLM REPL code is being used as both agent interface and executor. It should become one execution strategy behind a stable action protocol.
- Memory and taste can influence behavior without enough governance. The harness needs evidence, confidence, scope, expiry, conflict resolution, and eval-backed promotion.
- Verification is too permissive. For coding tasks, "could not verify" must be visible as unverified, not passed.
- Trace data is too shallow. We need replayable event streams with prompt/context/tool/action/observation/verification/artifact metadata, plus sensitive-data controls.
- Eval coverage is too narrow and too stub-heavy. We need real provider runs, trajectory grading, failure replay, and preference adherence tests.
- Packaging does not yet guarantee that the globally installed CLI, Python package, sandbox image, bundled evals, and local repo are in sync.

## North Star

Harness should be a local, production-grade coding-agent runtime with these
properties:

1. A downloaded user can install it, run `harness` inside any git repo, and get useful read-only help immediately.
2. Editing tasks run through scoped autonomy modes: ask-only, propose-only, sandboxed auto-edit, and trusted local edit.
3. Every run is durable, resumable, inspectable, and replayable.
4. Every model-driven action is typed, risk-scored, permission-gated, executed through a tool registry, and recorded as an event.
5. Every code change has an explicit verification status: verified, failed, or unverified.
6. User taste and project conventions compound over time, but only through scoped, evidence-backed records that can be inspected, rejected, and regression-tested.
7. Self-evolution produces proposals, eval cases, prompt rules, and verification policies. It does not silently rewrite core behavior.
8. The harness optimizes the agent-computer interface, not just the prompt.

## Target Architecture

### 1. Run Kernel

LangGraph becomes the default and canonical run kernel. The local runner can
remain temporarily as a test fixture or fallback, but product behavior should
compile through LangGraph.

New modules:

- `rlm_harness/kernel/state.py`
- `rlm_harness/kernel/events.py`
- `rlm_harness/kernel/run.py`
- `rlm_harness/kernel/langgraph_app.py`

Core models:

- `RunRequest`: user task, workspace, autonomy mode, provider/model, thread id, resume id.
- `RunContext`: project map, selected files, memory context, taste context, capabilities, budget.
- `RunState`: current phase, plan, event cursor, changed files, artifacts, verification, final response.
- `RunEvent`: append-only event base class.
- `CompletionEvent`: explicit done/partial/blocked/failed signal.

LangGraph nodes should become small:

- `context_build`
- `plan`
- `select_action`
- `authorize`
- `execute`
- `observe`
- `verify`
- `reflect`
- `learn`
- `finalize`

Each node consumes and produces typed state and events. No node should own the
whole runtime.

### 2. Action and Observation Protocol

Replace free-form JSON/Python action control with a typed action protocol.

New modules:

- `rlm_harness/actions/base.py`
- `rlm_harness/actions/workspace.py`
- `rlm_harness/actions/shell.py`
- `rlm_harness/actions/git.py`
- `rlm_harness/actions/memory.py`
- `rlm_harness/actions/completion.py`

Action types:

- `ReadFile`
- `ListFiles`
- `SearchCode`
- `ApplyPatch`
- `WriteFile`
- `RunShell`
- `GitStatus`
- `GitDiff`
- `ProposeChange`
- `ApplyPendingChange`
- `RecordMemory`
- `CompleteTask`

Observation types:

- `TextObservation`
- `FileObservation`
- `CommandObservation`
- `PatchObservation`
- `VerificationObservation`
- `PermissionObservation`
- `ErrorObservation`

Rules:

- Actions are model-visible as simple, stable tool schemas.
- Observations are compact, structured, and written for the model to recover from.
- All paths passed to tools are normalized and displayed as absolute workspace paths where useful, following the SWE-agent lesson that interface details matter.
- `CompleteTask` is the only successful completion signal.
- Reflection can reject completion, but it cannot invent completion without a `CompleteTask` action.

### 3. Tool Registry and Capability Handshake

`rlm_harness/sandbox/tools.py` should become implementation detail, not the
system boundary.

New modules:

- `rlm_harness/tools/registry.py`
- `rlm_harness/tools/descriptors.py`
- `rlm_harness/tools/risk.py`
- `rlm_harness/tools/render.py`

Every tool descriptor includes:

- name
- action model
- observation model
- scope
- side-effect class
- risk level
- whether it needs sandbox
- whether it needs confirmation
- timeout policy
- output summarizer
- test fixture

Capability handshake:

- Host and sandbox exchange protocol version.
- Sandbox declares available tool implementations and image version.
- Host refuses stale or incompatible sandbox images unless the user explicitly opts into degraded mode.

### 4. Sandbox Runtime

Keep Docker, but make it a runtime provider behind an interface.

New modules:

- `rlm_harness/runtime/base.py`
- `rlm_harness/runtime/docker.py`
- `rlm_harness/runtime/local.py`
- `rlm_harness/runtime/protocol.py`

Runtime modes:

- read-only local inspection
- sandboxed read/write with network off by default
- sandboxed network allowlist for package install or web tasks
- trusted local edit mode for advanced users

Each runtime must report:

- workspace mount
- effective user
- network mode
- writable paths
- installed tool versions
- sandbox image digest
- protocol version

### 5. Context Builder

The harness needs an intentional context layer, not scattered prompt fragments.

New modules:

- `rlm_harness/context/project_map.py`
- `rlm_harness/context/selection.py`
- `rlm_harness/context/tokens.py`
- `rlm_harness/context/render.py`

Inputs:

- git status and recent commits
- README and key config files
- dependency manifests
- test commands
- symbol/repo map
- user taste records
- project conventions
- approved evolution rules
- active run history summary

Design:

- Use an aider-style compact repo map.
- Use token budgets per context section.
- Separate always-on context from on-demand skills/conventions.
- Never dump 300 files into the response or prompt as the primary project summary.
- The summary path should answer: what this is, how it works, where to start, and what the next safe action is.

### 6. Planner, Executor, and Reviewer Agents

Use one orchestrator with role-specialized prompts rather than a single prompt
that tries to do everything.

Roles:

- `Planner`: scopes the task, selects files, chooses verification strategy.
- `Implementer`: proposes or executes actions.
- `Verifier`: runs checks and interprets failures.
- `Reviewer`: checks scope, taste, and final response quality.
- `Learner`: extracts preference/convention evidence after the run.

This can still be a single model call per phase. The point is clean contracts,
not agent theater.

### 7. Verification Policy

Verification becomes a policy engine with strict statuses.

New modules:

- `rlm_harness/verification/policy.py`
- `rlm_harness/verification/discovery.py`
- `rlm_harness/verification/runner.py`
- `rlm_harness/verification/result.py`

Statuses:

- `verified`: checks ran and passed.
- `failed`: at least one required check failed.
- `unverified`: checks could not run, timed out, or were skipped.
- `not_applicable`: no code or project artifact changed.

Rules:

- Editing code without verification is allowed only if surfaced as unverified.
- Verification discovery learns project commands, but learned commands require evidence from successful runs.
- A final answer for code edits must include changed files and verification status.
- Failed verification loops back into repair until budget or user scope stops it.

### 8. Memory, Taste, and Self-Evolution Governance

Keep local profile memory, but formalize promotion.

Memory channels:

- user taste: style, workflow, autonomy preferences
- project conventions: commands, architecture, file ownership, patterns
- run episodes: summaries, decisions, failures, successes
- evolution proposals: prompt rules, eval cases, verification policies, tooling ideas

Each memory record needs:

- scope: user, project, workspace, repo, branch
- source: explicit user statement, observed successful run, feedback, eval failure
- evidence run id
- confidence
- status: pending, active, rejected, expired
- created and last used timestamps
- conflict group

Promotion policy:

- Explicit user preferences can become active user taste immediately unless destructive or unsafe.
- Project conventions inferred from commands become active only after successful verification evidence.
- Prompt rules and verification policies become pending evolution proposals first.
- Any self-evolution that affects future code edits needs an eval case or user approval.

Runtime use:

- The context builder selects a small, relevant taste slice.
- The final response should mention when a new durable preference was learned.
- `harness profile` and `harness evolve` remain the user control plane.

### 9. Trace, Replay, and Observability

Tracing should become the spine of debugging and improvement.

New modules:

- `rlm_harness/trace/schema.py`
- `rlm_harness/trace/store.py`
- `rlm_harness/trace/replay.py`
- `rlm_harness/trace/redaction.py`
- `rlm_harness/trace/report.py`

Events to record:

- user task
- context sections and token counts
- plan
- model request metadata
- action selected
- authorization decision
- tool execution
- observation summary
- file artifacts and diffs
- verification checks
- memory changes
- final answer

Commands:

- `harness trace list`
- `harness trace show <run-id>`
- `harness trace replay <run-id>`
- `harness trace export <run-id>`

Sensitive data:

- Default local traces can include full data.
- Exported traces redact secrets, environment values, and long file contents unless explicitly allowed.

### 10. CLI and User Experience

The CLI should feel like a calm daily driver, not a science project.

Modes:

- `harness ask "..."`: read-only answer.
- `harness plan "..."`: plan only, no edits.
- `harness work "..."`: scoped editing with current default autonomy mode.
- `harness continue [task]`: continue the latest thread without copying an id.
- `harness verify`: run learned project verification.
- `harness trace`: inspect runs.
- `harness status`: show provider, latest run, taste, and evolution status.
- `harness taste`: inspect and manage taste.
- `harness profile`: compatibility alias for taste records.
- `harness evolve`: inspect and approve self-evolution.
- `harness doctor/readiness`: setup health.

Interactive behavior:

- Keep compact progress markers on stderr.
- Keep final answer on stdout.
- Show approvals as clear diffs or commands, not hidden prompts.
- Give users an explicit run id for anything that changed files.
- Never pretend skipped verification is success.

### 11. Packaging and Downloaded-User Scope

The harness must work for users who install it, not just the repo checkout.

Required packaging guarantees:

- Python package includes bundled eval suites and schemas.
- `langgraph` and SQLite checkpoint support are default dependencies.
- `harness readiness` verifies package version, CLI path, sandbox image version, Docker availability, provider config, and memory DB health.
- Installer prevents stale global CLI confusion by printing the resolved `harness` path and version.
- Sandbox image is versioned and rebuildable with `harness sandbox build`.
- Every local profile stays under the user's home directory by default.
- Project memory stays under the project workspace by default.

## Implementation Phases

### Phase 0: Freeze and Characterize

Goal: stop random patching and capture real failures.

Tasks:

- Add a failure corpus under `evals/failures/` or `tests/fixtures/failures/`.
- Capture the current bad interaction: "what is this project about and what must I do next" followed by a continuation task that fails.
- Add trace snapshots for summary, edit, verify, taste-learning, and self-evolution tasks.
- Define acceptance criteria for "intelligent response" rather than relying on string fragments.

Files:

- `rlm_harness/evals/runner.py`
- `rlm_harness/evals/suites/daily-driver.json`
- `EVALS.md`
- new `docs/plans/production-grade-harness-revamp.md`

Gate:

- We can reproduce current failure classes before refactoring.

### Phase 1: Run Contract and Event Model

Goal: make every run inspectable and durable.

Tasks:

- Add Pydantic models for run request, run state, events, actions, observations, artifacts, permissions, and completion.
- Migrate `HarnessState` into the new kernel state or wrap it with compatibility.
- Write serialization tests and golden event fixtures.
- Add explicit `CompleteTask` action and `CompletionEvent`.

Files:

- `rlm_harness/types.py`
- new `rlm_harness/kernel/*`
- new `rlm_harness/actions/*`
- `tests/test_kernel_events.py`

Gate:

- A run can be represented as a typed append-only event sequence.

### Phase 2: LangGraph as Canonical Kernel

Goal: remove split-brain orchestration.

Tasks:

- Move graph construction into `rlm_harness/kernel/langgraph_app.py`.
- Keep local runner only for focused unit tests while product runs use LangGraph.
- Persist checkpoints under `.rlm_harness/checkpoints.sqlite`.
- Add resume tests.
- Ensure missing LangGraph is a readiness/install error, not a silent fallback for installed users.

Files:

- `rlm_harness/graph/build.py`
- `rlm_harness/graph/checkpoint.py`
- new `rlm_harness/kernel/langgraph_app.py`
- `tests/test_graph.py`
- `tests/test_readiness.py`

Gate:

- A run can be interrupted after an action and resumed from checkpoint with the same event history.

### Phase 3: Tool Registry and Runtime Protocol

Goal: make tool use stable, scoped, and sandbox-aware.

Tasks:

- Introduce tool descriptors and risk metadata.
- Split `rlm_harness/sandbox/tools.py` into tool implementations grouped by domain.
- Add host/sandbox protocol version handshake.
- Add sandbox image version checks to readiness.
- Convert project summary/audit into tools with typed outputs.

Files:

- `rlm_harness/sandbox/tools.py`
- `rlm_harness/sandbox/docker_repl.py`
- `rlm_harness/sandbox/worker.py`
- new `rlm_harness/tools/*`
- new `rlm_harness/runtime/*`
- `tests/test_sandbox.py`
- `tests/test_tools_registry.py`

Gate:

- The model sees a stable tool list, and every execution records an action event and observation event.

### Phase 4: Context Builder and Project Map

Goal: give the model the right context without flooding it.

Tasks:

- Build a compact project map from files, manifests, tests, and symbols.
- Add token budgets for repo map, memory, taste, recent trace, and selected files.
- Replace ad hoc project-summary fallback with context-backed project orientation.
- Add context rendering tests for Python, Node, Rust, and mixed repos.

Files:

- new `rlm_harness/context/*`
- `rlm_harness/sandbox/tools.py`
- `rlm_harness/graph/nodes.py`
- `tests/test_context_builder.py`

Gate:

- The summary question answers what the project is, how it works, and what to do next without listing arbitrary file counts as the main content.

### Phase 5: Planner, Executor, Reviewer Split

Goal: decompose the god object.

Tasks:

- Extract planner, action selector, executor adapter, verifier, reviewer, learner, and finalizer services.
- Make RLM REPL an executor strategy, not the orchestration contract.
- Add reviewer checks for stale summaries, scope drift, incomplete verification, and user-taste mismatch.
- Remove heuristic completion paths once explicit completion is stable.

Files:

- `rlm_harness/graph/nodes.py`
- `rlm_harness/rlm/runtime.py`
- new `rlm_harness/agents/planner.py`
- new `rlm_harness/agents/executor.py`
- new `rlm_harness/agents/reviewer.py`
- `tests/test_graph.py`
- `tests/test_rlm_runtime.py`

Gate:

- `Nodes` becomes a thin adapter. Role services can be unit-tested without running the full graph.

### Phase 6: Verification and Safety

Goal: make autonomous edits trustworthy.

Tasks:

- Replace permissive verification with strict statuses.
- Add required checks per changed file type.
- Add authorization policies for destructive shell, dependency changes, prompt changes, and outside-workspace writes.
- Add approval UX for pending changes and commands.
- Add scope-creep tests.

Files:

- `rlm_harness/graph/verification.py`
- new `rlm_harness/verification/*`
- `rlm_harness/sandbox/tools.py`
- `rlm_harness/cli.py`
- `tests/test_verification_policy.py`
- `tests/test_safety_policy.py`

Gate:

- Any code-edit final answer says verified, failed, or unverified with evidence.

### Phase 7: Taste Learning and Self-Evolution Governance

Goal: make personalization useful without making the agent erratic.

Tasks:

- Add evidence, confidence, expiry, conflict group, and last-used fields to taste/evolution records.
- Add memory selection by task relevance and token budget.
- Add user-visible learning summaries.
- Require eval case or user approval before broad prompt-rule activation.
- Add rollback for taste records and evolution proposals.

Files:

- `rlm_harness/memory/profile.py`
- `rlm_harness/memory/evolution.py`
- `rlm_harness/memory/schema.sql`
- `rlm_harness/evals/runner.py`
- `rlm_harness/cli.py`
- `tests/test_memory.py`
- `tests/test_evals.py`

Gate:

- A learned preference changes future behavior in evals, can be inspected, and can be rejected.

### Phase 8: Trace, Replay, and Debug UX

Goal: make every bad run debuggable.

Tasks:

- Expand trace schema to typed events.
- Add replay from saved model outputs and tool observations.
- Add `harness trace show` with compact timeline.
- Add redacted export for issue reports.
- Connect eval failures to trace ids and evolution proposals.

Files:

- `rlm_harness/tracing.py`
- new `rlm_harness/trace/*`
- `rlm_harness/cli.py`
- `tests/test_tracing.py`

Gate:

- Given a run id, a developer can see what context the model saw, what action it chose, what happened, and why the final answer was produced.

### Phase 9: Daily-Driver Release Track

Goal: ship this as something downloaded users can trust.

Tasks:

- Expand `harness dogfood` into a release gate.
- Add install smoke tests for clean virtualenv and existing global install.
- Add sandbox image build/pull verification.
- Add real-provider nightly evals where credentials are available.
- Add documentation for autonomy modes, memory scope, safety guarantees, and troubleshooting.

Files:

- `rlm_harness/dogfood.py`
- `scripts/install.sh`
- `README.md`
- `EVALS.md`
- `tests/test_dogfood.py`
- `tests/test_cli_config.py`

Gate:

- A fresh user can install, configure, run readiness, ask project questions, perform a small edit, verify it, inspect the trace, and see learned preferences without using the repo checkout.

## Acceptance Scenarios

Minimum scenarios before calling the revamp production-ready:

- Read-only orientation: from an unknown repo, answer "what is this project and what should I do next?" with a human summary, architecture sketch, and safe next actions.
- Small bug fix: identify a failing test, edit the smallest relevant files, run focused verification, and summarize changes.
- Verification failure repair: introduce or encounter a failing check, loop once or more, and either fix it or stop with a clear failed/unverified status.
- Dirty worktree: detect user changes, avoid overwriting them, and separate harness changes from existing changes.
- Preference learning: user says "I prefer terse final answers with verification first"; next run reflects it and `harness profile` shows the record.
- Preference conflict: user changes their mind; old taste is rejected or superseded.
- Self-evolution proposal: a repeated failure creates a pending eval or prompt-rule proposal, not an unreviewed behavior mutation.
- Resume: interrupt a run after planning or after an action, resume from checkpoint, and continue without duplicating side effects.
- Sandbox stale image: readiness detects mismatch and tells the user how to rebuild.
- Trace replay: a bad run can be replayed enough to reproduce the control-flow decision that caused it.

## Immediate Next Work

Start with Phase 0 and Phase 1 only.

The first implementation branch should not touch the final-answer fallback or
project-summary heuristics again. It should add the run/event/action contract and
failure corpus, then prove the current behavior against that corpus. Once the
contract exists, future fixes become architectural improvements instead of
another layer of patches.

Recommended first PR:

- Add `rlm_harness/kernel/events.py` and `rlm_harness/actions/base.py`.
- Add golden failure fixtures for the unintelligent-summary path.
- Add trace/event tests that fail under current behavior.
- Do not change runtime behavior until those tests describe the desired contract.
