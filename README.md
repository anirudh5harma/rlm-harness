# Harness

Local recursive coding-agent harness with a direct global CLI.

## Install with curl

Requirements: Python 3.10+, git, Docker, and a provider API key for real model runs.

```bash
curl -fsSL https://raw.githubusercontent.com/anirudh5harma/rlm-harness/main/scripts/install.sh | sh
```

The installer creates an isolated app environment under `~/.local/share/harness` and links the global command at `~/.local/bin/harness`.

If `~/.local/bin` is not already on your PATH, the installer adds it to your shell profile and prints the one-line export you can run immediately:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Configure

Choose your provider. Running this with no provider lists popular options and prompts you to choose:

```bash
harness /provider
```

Or set one directly:

```bash
harness /provider openrouter --api-key "<your-api-key>"
```

Then list/select models for that provider:

```bash
harness /model
```

Or set a model directly:

```bash
harness /model openai/gpt-4o-mini
```

Popular providers include `openrouter`, `openai`, `groq`, `together`, `fireworks`, `deepinfra`, `opencode-go`, `custom`, and `stub`.

Optional custom base URL:

```bash
harness /provider custom --base-url "http://127.0.0.1:8080/v1" --api-key "<your-api-key>"
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
# harness> /provider
# harness> /model
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
harness /provider [provider] [--api-key key] [--base-url url]
harness /model [model-name]
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
