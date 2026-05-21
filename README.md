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

## Common commands

```bash
harness                              # interactive mode
harness "fix the failing tests"       # run one task
harness /provider openrouter --api-key <key>
harness /model openai/gpt-4o-mini
harness /config
harness doctor
```

Supported provider shortcuts include `openrouter`, `openai`, `groq`, `together`, `fireworks`, `deepinfra`, `opencode-go`, `custom`, and `stub`.

Configuration is saved in `~/.harness/config.json`.
