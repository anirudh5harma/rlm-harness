# Harness evals

Harness includes a lightweight evaluation runner for repo-level coding tasks.

## Long-horizon suites

Create a YAML or JSON suite with files, a prompt, and a validation command:

```yaml
name: long-horizon-smoke
cases:
  - id: fix-python-test
    prompt: Fix the failing tests and keep the implementation minimal.
    test_command: python -m unittest
    files:
      mathlib.py: 'def add(a, b): return a - b\n'
      test_mathlib.py: "import unittest\nfrom mathlib import add\nclass T(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
```

Run it:

```bash
harness eval long-horizon suite.yaml --work-root .harness-evals/work
```

## SWE-bench-style manifests

Provide JSONL records with at least:

```json
{"instance_id":"repo__issue-1","repo":"owner/repo","base_commit":"abc123","problem_statement":"Fix the bug","test_command":"python -m pytest tests/test_bug.py"}
```

Optional `clone_url` causes the runner to clone and checkout the repo before running Harness.

```bash
harness eval swe-bench swebench.jsonl --limit 10 --work-root .harness-evals/swe
```

The runner reports pass rate, per-case status, latency, harness output, grader output, and metadata. Use `--json` or `--output results.json` for machine-readable results.
