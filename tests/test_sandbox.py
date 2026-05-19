import subprocess
import tempfile
import unittest
from pathlib import Path

from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig

IMAGE = "rlm-harness-sandbox:test"


def docker_available():
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


@unittest.skipUnless(docker_available(), "Docker daemon is not available")
class DockerREPLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DockerREPL.build_image(image=IMAGE)

    def sandbox(self, workspace):
        return DockerREPL(
            SandboxConfig(
                image=IMAGE,
                workspace=Path(workspace),
                default_timeout_s=2,
                start_timeout_s=10,
            )
        )

    def rlm_sandbox(self, workspace, subcall_config=None):
        return DockerREPL(
            SandboxConfig(
                image=IMAGE,
                workspace=Path(workspace),
                default_timeout_s=2,
                start_timeout_s=10,
            ),
            completion_client=LMClient(provider="stub"),
            subcall_config=subcall_config,
        )

    def test_executes_python_and_returns_stdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.sandbox(temp_dir) as repl:
                result = repl.execute("print(2 + 2)")

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "4")
        self.assertEqual(result.stderr, "")

    def test_namespace_persists_across_exec_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.sandbox(temp_dir) as repl:
                first = repl.execute("counter = 41")
                second = repl.execute("counter += 1\nprint(counter)")

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(second.stdout.strip(), "42")

    def test_captures_stderr_and_exception_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.sandbox(temp_dir) as repl:
                result = repl.execute(
                    "import sys\n"
                    "print('before')\n"
                    "print('warn', file=sys.stderr)\n"
                    "raise ValueError('bad')"
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "error")
        self.assertIn("before", result.stdout)
        self.assertIn("warn", result.stderr)
        self.assertIn("ValueError: bad", result.stderr)

    def test_timeout_marks_result_without_killing_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.sandbox(temp_dir) as repl:
                timeout = repl.execute("while True:\n    pass", timeout_s=1)
                after = repl.execute("print('alive')", timeout_s=1)

        self.assertFalse(timeout.ok)
        self.assertTrue(timeout.timed_out)
        self.assertEqual(timeout.status, "timeout")
        self.assertTrue(after.ok)
        self.assertEqual(after.stdout.strip(), "alive")

    def test_workspace_mount_allows_scoped_file_io(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "input.txt").write_text("hello", encoding="utf-8")
            with self.sandbox(workspace) as repl:
                result = repl.execute(
                    "from pathlib import Path\n"
                    "text = Path('/workspace/input.txt').read_text()\n"
                    "Path('/workspace/output.txt').write_text(text + ' sandbox')\n"
                    "print(Path('/workspace/output.txt').read_text())"
                )

            output = (workspace / "output.txt").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "hello sandbox")
        self.assertEqual(output, "hello sandbox")

    def test_container_cleanup_on_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repl = self.sandbox(temp_dir)
            name = repl.container_name
            with repl:
                result = repl.execute("print('cleanup')")

            completed = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertTrue(result.ok)
        self.assertEqual(completed.stdout.strip(), "")

    def test_rlm_completion_round_trip_from_sandbox_to_host_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.rlm_sandbox(temp_dir) as repl:
                result = repl.execute(
                    "answer = rlm.completion('summarize this', 'context text')\nprint(answer)"
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.subcalls, 1)
        self.assertGreater(result.tokens_used, 0)
        self.assertIn("Stub response for task:", result.stdout)
        self.assertIn("Query:\nsummarize this", result.stdout)

    def test_rlm_completion_depth_limit_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.rlm_sandbox(temp_dir, RLMSubcallConfig(max_depth=0)) as repl:
                result = repl.execute("print(rlm.completion('too deep', 'context', depth_hint=1))")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "error")
        self.assertIn("exceeds max depth", result.stderr)

    def test_rlm_completion_requires_host_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.sandbox(temp_dir) as repl:
                result = repl.execute("print(rlm.completion('query', 'context'))")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "error")
        self.assertIn("not enabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
