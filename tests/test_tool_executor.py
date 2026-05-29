import tempfile
import unittest
from pathlib import Path

from rlm_harness.actions import (
    ApplyPatchAction,
    ApplyPendingChangeAction,
    ClearPendingChangesAction,
    CommandObservation,
    CompleteTaskAction,
    DataObservation,
    FileObservation,
    ObservationStatus,
    PermissionObservation,
    ProjectSummaryAction,
    ProposeChangeAction,
    ReadFileAction,
    ReadFirstExistingAction,
    RunShellAction,
    TextObservation,
    WriteFileAction,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.tools import ToolExecutor


class ToolExecutorTests(unittest.TestCase):
    def test_read_file_action_returns_file_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# Harness\n", encoding="utf-8")

            observation = ToolExecutor(workspace).execute(ReadFileAction(path="README.md"))

        self.assertIsInstance(observation, FileObservation)
        self.assertEqual(observation.path, "README.md")
        self.assertEqual(observation.content, "# Harness\n")

    def test_write_file_requires_confirmation_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "notes.txt"
            action = WriteFileAction(path="notes.txt", content="hello\n")
            executor = ToolExecutor(workspace)

            denied = executor.execute(action)
            written = executor.execute(action, approved=True)
            content = target.read_text(encoding="utf-8")

        self.assertIsInstance(denied, PermissionObservation)
        self.assertEqual(denied.decision, "needs_confirmation")
        self.assertIsInstance(written, TextObservation)
        self.assertEqual(content, "hello\n")

    def test_ask_mode_denies_write_even_when_approved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            action = WriteFileAction(path="notes.txt", content="hello\n")

            observation = ToolExecutor(workspace, autonomy=AutonomyMode.ASK).execute(
                action,
                approved=True,
            )

        self.assertIsInstance(observation, PermissionObservation)
        self.assertEqual(observation.decision, "denied")
        self.assertIn("read-only", observation.reason)

    def test_proposal_and_apply_pending_change_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            executor = ToolExecutor(workspace)
            try:
                proposal = executor.execute(
                    ProposeChangeAction(
                        path="app.py",
                        content="print('new')\n",
                        reason="test",
                    )
                )
                change_id = proposal.data["id"]
                denied = executor.execute(ApplyPendingChangeAction(change_id=change_id))
                applied = executor.execute(
                    ApplyPendingChangeAction(change_id=change_id),
                    approved=True,
                )
                content = target.read_text(encoding="utf-8")
            finally:
                executor.execute(ClearPendingChangesAction())

        self.assertIsInstance(proposal, DataObservation)
        self.assertTrue(proposal.data["approval_required"])
        self.assertIsInstance(denied, PermissionObservation)
        self.assertIsInstance(applied, TextObservation)
        self.assertEqual(content, "print('new')\n")

    def test_run_shell_maps_return_codes_to_command_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ToolExecutor(Path(temp_dir))

            ok = executor.execute(RunShellAction(command="printf ok"))
            failed = executor.execute(RunShellAction(command="exit 7"))

        self.assertIsInstance(ok, CommandObservation)
        self.assertEqual(ok.status, ObservationStatus.OK)
        self.assertEqual(ok.stdout, "ok")
        self.assertEqual(failed.status, ObservationStatus.ERROR)
        self.assertEqual(failed.exit_code, 7)

    def test_apply_patch_reports_changed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "app.py").write_text("print('old')\n", encoding="utf-8")
            patch = (
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1 +1 @@\n"
                "-print('old')\n"
                "+print('new')\n"
            )

            observation = ToolExecutor(workspace).execute(ApplyPatchAction(diff=patch))

        self.assertEqual(observation.kind, "patch")
        self.assertEqual(observation.changed_files, ["app.py"])
        self.assertEqual(observation.diff_summary, "patch applied")

    def test_destructive_shell_command_returns_error_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            observation = ToolExecutor(Path(temp_dir)).execute(
                RunShellAction(command="rm -rf .")
            )

        self.assertEqual(observation.kind, "error")
        self.assertIn("destructive", observation.message)

    def test_project_summary_action_uses_workspace_context_and_restores_it(self):
        old_workspace = sandbox_tools.WORKSPACE
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Example\n\nA tiny Python package.\n",
                encoding="utf-8",
            )
            (workspace / "pyproject.toml").write_text(
                '[project]\nname = "example"\ndescription = "Tiny package."\n',
                encoding="utf-8",
            )

            observation = ToolExecutor(workspace).execute(ProjectSummaryAction())

        self.assertIs(sandbox_tools.WORKSPACE, old_workspace)
        self.assertIsInstance(observation, TextObservation)
        self.assertIn("Project Summary", observation.text)
        self.assertIn("example", observation.text.lower())

    def test_read_first_existing_uses_matched_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            observation = ToolExecutor(workspace).execute(
                ReadFirstExistingAction(paths=["missing.txt", "pyproject.toml"])
            )

        self.assertIsInstance(observation, FileObservation)
        self.assertEqual(observation.path, "pyproject.toml")
        self.assertEqual(observation.content, "[project]\n")

    def test_complete_task_action_returns_text_observation(self):
        observation = ToolExecutor(Path.cwd()).execute(
            CompleteTaskAction(summary="All set.", verification="pytest")
        )

        self.assertIsInstance(observation, TextObservation)
        self.assertEqual(observation.text, "All set.")
        self.assertEqual(observation.summary, "success")


if __name__ == "__main__":
    unittest.main()
