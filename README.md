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

Check whether Harness is ready for daily coding work:

```bash
harness readiness
```

Choose a provider and save your API key:

```bash
harness /provider
```

Then choose a model:

```bash
harness /model
```

Start Harness from any project directory:

```bash
harness
```

After a useful run, teach it your taste:

```bash
harness feedback add "Liked the concise summary and verification line." --rating good
```

If readiness shows Docker warnings, you can still try a non-sandboxed run with
`harness --no-sandbox "summarize this project"` while you fix Docker.

## Run experience

Interactive runs show a compact progress rail on stderr while the final answer
stays on stdout for piping and scripts. The rail uses simple markers such as
`[command]`, `[setup]`, `[plan]`, `[work]`, `[check]`, and `[done]`; commands and
important values are highlighted in cyan/blue when the terminal supports color.

`--json`, `--stream`, and `--quiet` suppress the progress rail. Set
`HARNESS_PROGRESS=on` or `HARNESS_PROGRESS=off` to override auto-detection, and
`HARNESS_COLOR=on` or `HARNESS_COLOR=off` to override color.

## Common commands

```bash
harness                              # interactive mode
harness "fix the failing tests"       # run one task
harness /provider openrouter --api-key <key>
harness /model qwen/qwen3.7-max
harness /config
harness readiness                     # check first-run and daily-driver setup
harness dogfood                       # run readiness, eval, and feedback proof checks
harness profile                      # show learned taste and project conventions
harness profile learn "Prefer small, reviewable diffs." --active
harness evolve                       # review proposed prompt/policy/eval improvements
harness feedback add "Liked the concise summary." --rating good
harness doctor
```

Supported provider shortcuts include `openrouter`, `openai`, `groq`, `together`,
`fireworks`, `deepinfra`, `opencode-go`, `custom`, and `stub`.

Configuration is saved in `~/.harness/config.json`. User-wide taste is saved in
`~/.harness/profile.db`; project memory, traces, and LangGraph checkpoints stay
under the current workspace's `.rlm_harness/` directory by default.

## Taste learning

Harness learns durable preferences and project conventions as it runs. Explicit
phrases like "I prefer concise final answers" are promoted into the user profile,
successful verification commands are remembered for the current project, and the
next run receives that taste as context before planning or editing.

Use `harness profile` to inspect active records, `harness profile learn ...` to
teach it directly, and `harness profile approve/reject <id>` to manage pending
records.

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
