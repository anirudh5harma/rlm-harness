# Harness

Recursive coding-agent harness with a global `harness` CLI.

## Install

Requires Python 3.10+, git, Docker, and an API key from your chosen provider.

```bash
curl -fsSL https://raw.githubusercontent.com/anirudh5harma/rlm-harness/main/scripts/install.sh | sh
```

If your shell cannot find `harness` after install, open a new terminal or run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## First run

Bootstrap Harness from a project directory:

```bash
harness init --provider openrouter --api-key <key>
```

`init` saves provider configuration, scans the project for style and verification
conventions, summarizes configured MCP servers, and prints readiness checks. You
can still run the pieces individually:

```bash
harness readiness
harness /provider
harness /model
harness taste scan
```

Start Harness from any project directory:

```bash
harness
```

Harness will not silently run real tasks on the built-in `stub` provider. Fresh
installs are guided to `harness init` first; pass `--provider stub` only for an
intentional smoke test.

After a useful run, teach it your taste:

```bash
harness feedback add "Liked the concise summary and verification line." --rating good
```

If readiness shows Docker warnings, you can still try a non-sandboxed run with
`harness --no-sandbox "summarize this project"` while you fix Docker.

## Run experience

Interactive runs show a single updating status line on stderr while the final
answer stays on stdout for piping and scripts. In a real terminal Harness uses a
spinner-style status line; in plain streams it falls back to carriage-return
updates instead of printing a new line for every graph step. Visible accents use
light cyan when the terminal supports color.

`--json`, `--stream`, and `--quiet` suppress the status line. Set
`HARNESS_PROGRESS=on` or `HARNESS_PROGRESS=off` to override auto-detection, and
`HARNESS_COLOR=on` or `HARNESS_COLOR=off` to override color.

## Common commands

```bash
harness                              # interactive mode
harness "fix the failing tests"      # run one coding task (the primary shape)
harness "what is this project?" --ask   # read-only answer, no edits
harness "how should we fix it?" --plan  # read-only implementation plan
harness -c "do the next step"        # continue the latest thread
harness -r <thread-id> "next step"   # resume a specific thread
harness history list                 # show recent runs
harness history show <run-id>        # timeline of a run
harness status                       # current state and recommended next actions
harness init --provider openrouter --api-key <key>
harness doctor                       # check local setup and sandbox
harness mcp list                     # inspect configured MCP servers
harness eval daily-driver --provider stub --model stub
harness update                       # upgrade the managed install
```

Harness keeps compatibility aliases where they are useful (`ask`,
`plan`, `run`, `work`, `trace`, `taste`, `evolve`, `feedback`,
`provider`, `model`, `readiness`, `config`, `dogfood`, `tools`).
They still work and are reachable inside interactive mode as slash
commands (`/ask`, `/taste`, ...); they are just no longer advertised
in `harness --help`. Run `harness commands --all` to see every
registered alias. The primary shape is Harness-native: one `harness
"task"` verb with `--ask` / `--plan` modes, `--continue` / `--resume`
for thread flow, and `history` for inspecting runs. Use `harness
status` as the daily-driver handoff: it summarizes provider/API key
state, latest thread, taste, evolution, MCP configuration, storage
paths, and a short `next` list.

MCP servers are configured in `~/.harness/mcp.json` (or `HARNESS_MCP_CONFIG`)
with transport, auth, and purpose metadata. Local stdio MCPs run as
subprocesses with bounded newline-delimited JSON-RPC; hosted HTTP/SSE MCPs use
negotiated protocol/session headers and env-backed auth. Use
`harness mcp setup` for guided setup, `harness mcp tools <name>` to smoke-test
auth, `harness mcp trust/untrust <name>` to control autonomous use, and
`harness mcp enable/disable <name>` to scope what Harness can see during
workflows.

Supported provider shortcuts include `openrouter`, `openai`, `groq`, `together`,
`fireworks`, `deepinfra`, `opencode-go`, `custom`, and `stub`.

Configuration is saved in `~/.harness/config.json`. User-wide taste is saved in
`~/.harness/profile.db`; project memory, traces, and LangGraph checkpoints stay
under the current workspace's `.rlm_harness/` directory by default.

## Taste learning

Harness learns durable preferences and project conventions as it runs. Explicit
phrases like "I prefer concise final answers" are promoted into the user profile,
successful verification commands are remembered for the current project, and the
first memory-enabled run in a workspace automatically scans project style
conventions into project memory. The next planning/editing step receives that
taste as context before it acts.

Use `harness taste` to inspect active records, `harness taste context` to see
the exact preference block future runs receive, `harness taste learn ...` to
teach it directly, and `harness taste approve/reject <id>` to manage pending
records. `harness taste scan` inspects the current workspace on demand and stores
evidence-backed project style and verification conventions in project memory.
It looks at common project signals such as `pyproject.toml`, `package.json`,
`.editorconfig`, Prettier config, package-manager metadata, and sampled source
formatting.
Normal memory-enabled runs do the same bootstrap automatically; use
`--no-style-scan` to opt out for a run. `harness profile` remains available as a
compatibility alias.

## Feedback learning

Use `harness feedback add ...` after a run to teach Harness what to repeat or
avoid. Positive feedback like "Liked concise summaries" is recorded as feedback,
promoted into active taste, and proposed as future response guidance. Negative
feedback stays reviewable by default and can generate eval-case proposals,
especially when you pass `--run-id <id>` to connect feedback to a trace.

Use `harness feedback list` to inspect feedback history. Feedback is stored in
the same user/project memory split as taste, so downloaded users can build their
own local profile without sharing it.

## Self-evolution proposals

Harness separates learning from self-evolution. It can automatically create
pending proposals when it sees useful evidence, such as an explicit user
preference, a successful verification command, or a stopped run that should
become a regression eval. Approved proposals are injected into future runs as
runtime guidance; pending and rejected proposals remain inspectable but do not
steer behavior.

Use `harness evolve` to list pending proposals, `harness evolve approve <id>` to
activate one, and `harness evolve propose ...` when you want to teach a new
prompt rule, verification policy, eval case, or tooling idea manually.

## Taste regression evals

Eval cases can seed isolated taste records and approved evolution rules before a
Harness run, then grade both workspace tests and user-visible output. This makes
personalization measurable: a case can prove that learned preferences affect the
next response without touching your real `~/.harness/profile.db`.

Run the starter suite with:

```bash
harness eval taste-regression --no-sandbox --record-failures
```

When `--record-failures` is set, failed eval cases create pending project
evolution proposals so repeated quality gaps become reviewable work instead of
being lost in a test log.

For broader dogfooding, run the bundled daily-driver suite:

```bash
harness eval daily-driver --provider stub --model stub --record-failures
```

It exercises workspace inspection, a Python test fix, and learned summary style.

To run the full local proof loop, use:

```bash
harness dogfood --provider stub --model stub
```

`dogfood` reports readiness, runs the bundled taste and daily-driver evals, and
checks that feedback can become taste plus a reviewable evolution proposal. Add
`--strict-readiness` when you want setup blockers to fail the command. Add
`--install-smoke` to verify a fresh virtualenv install can run the installed
`harness` CLI and bundled eval resources.

## Daily-driver guardrails

Harness can apply ordinary workspace edits directly, but it also exposes proposal
tools for risky edits, dependency changes, and prompt or policy changes. Those
tools queue a diff for review before applying it. Destructive shell commands are
blocked by default unless explicit approval is provided to the tool call.

After code edits, Harness runs focused verification and discovers common
project-native commands from `pyproject.toml`, `package.json`, `Makefile`, and
`justfile`. Successful commands are learned as project conventions so future runs
start with better local taste.
