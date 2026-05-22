# Harness evaluation and observability

Harness evals are local-first and deterministic:

1. Create a small workspace fixture.
2. Run `harness` on the case prompt.
3. Run an objective validation command.
4. Record pass/fail, score, latency, stdout/stderr, workspace path, metadata, and timestamps.
5. Optionally upload the completed local report to LangSmith.

## Write an eval file

```yaml
name: smoke
cases:
  - id: fix-python-test
    prompt: Fix the failing tests and keep the implementation minimal.
    test_command: python -m unittest
    files:
      mathlib.py: 'def add(a, b): return a - b\n'
      test_mathlib.py: "import unittest\nfrom mathlib import add\nclass T(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
```

## Run it

```bash
harness eval evals.yaml
```

Useful flags:

```bash
harness eval evals.yaml --output results.json
harness eval evals.yaml --json
harness eval evals.yaml --work-root .harness-evals/work
```

## Upload to LangSmith

LangSmith is optional and only receives the already-computed local report; it does not replace the local grader.

```bash
harness eval evals.yaml \
  --output results.json \
  --langsmith-upload \
  --langsmith-dataset rlm-harness-evals \
  --langsmith-experiment local-run
```

Without `LANGSMITH_API_KEY`, upload skips cleanly unless `--langsmith-required` is set.
