# Harness

Local recursive coding-agent harness with a direct global CLI.

## Install with curl

Requirements: Python 3.10+, git, Docker, and an OpenAI-compatible API key for real model runs.

```bash
curl -fsSL https://raw.githubusercontent.com/anirudh5harma/rlm-harness/main/scripts/install.sh | sh
```

The installer creates an isolated app environment under `~/.local/share/harness` and links the global command at `~/.local/bin/harness`.

If `~/.local/bin` is not already on your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Configure

Select the model:

```bash
harness /model openai/gpt-4o-mini
```

Connect your provider and API key:

```bash
harness /provider openai-compatible --api-key "<your-api-key>"
```

Optional custom OpenAI-compatible base URL:

```bash
harness /provider openai-compatible --base-url "https://openrouter.ai/api/v1" --api-key "<your-api-key>"
```

Check configuration:

```bash
harness /config
```

Configuration is saved to `~/.harness/config.json` with file mode `0600`. Environment variables still override saved config: `HARNESS_PROVIDER`, `HARNESS_MODEL`, `HARNESS_BASE_URL`, `HARNESS_API_KEY`.

## Use anywhere

Start interactive mode in any directory:

```bash
harness
# harness> /model openai/gpt-4o-mini
# harness> /provider openai-compatible --api-key <key>
# harness> fix the failing tests
# harness> /quit
```

Run one task directly:

```bash
harness "fix the failing tests"
```

Offline smoke test:

```bash
harness "List files in workspace" --provider stub --json
```

Useful commands:

```bash
harness /model [model-name]
harness /provider [stub|openai-compatible] [--api-key key] [--base-url url]
harness /config
harness doctor
harness trace list
harness trace report <run-id>
```

## Install from a fork or branch

```bash
curl -fsSL https://raw.githubusercontent.com/anirudh5harma/rlm-harness/main/scripts/install.sh \
  | HARNESS_REPO_URL="https://github.com/you/rlm-harness.git" HARNESS_REF="main" sh
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,graph]'
python -m ruff check .
python -m unittest
```
