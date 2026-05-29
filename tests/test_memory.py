import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from rlm_harness import cli
from rlm_harness.memory import (
    EvolutionProposal,
    EvolutionProposalManager,
    EvolutionProposalStore,
    FeedbackRecord,
    FeedbackStore,
    Memory,
    MemoryError,
    MemoryPagingConfig,
    MemoryValidationError,
    TasteProfileManager,
    TasteProfileStore,
    TasteRecord,
    infer_evolution_from_feedback,
    infer_taste_from_feedback,
)
from rlm_harness.memory.embed import HashingEmbedder
from rlm_harness.types import HarnessState


class FixedClock:
    def __init__(self):
        self.value = 1_700_000_000

    def __call__(self):
        self.value += 1
        return self.value


class MemoryTests(unittest.TestCase):
    def test_migrations_are_idempotent_and_enable_wal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            Memory(path).close()
            Memory(path).close()

            with sqlite3.connect(path) as connection:
                version_rows = connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                vec_table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'archival_vec'"
                ).fetchall()
                fallback_table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'archival_embedding'"
                ).fetchall()

            self.assertEqual(version_rows, [(1,)])
            self.assertEqual(journal_mode, "wal")
            self.assertEqual(vec_table, [("archival_vec",)])
            self.assertEqual(fallback_table, [])

    def test_concurrent_initialization_against_fresh_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            errors = []

            def open_and_write(index):
                try:
                    memory = Memory(path)
                    try:
                        memory.core_set(f"key-{index}", f"value-{index}")
                    finally:
                        memory.close()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=open_and_write, args=(index,)) for index in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            memory = Memory(path)
            try:
                self.assertEqual(memory.vector_backend, "sqlite-vec")
                self.assertEqual(memory.core_get("key-5"), "value-5")
            finally:
                memory.close()

    def test_core_memory_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            memory = Memory(path)
            item = memory.core_set("repo.package-manager", "pip")
            memory.close()

            reopened = Memory(path)
            try:
                self.assertEqual(reopened.core_get("repo.package-manager"), "pip")
                reopened_item = reopened.core_item("repo.package-manager")
                self.assertEqual(reopened_item.updated_at, item.updated_at)
            finally:
                reopened.close()

    def test_recall_append_and_page_by_thread_query_and_recency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clock = FixedClock()
            memory = Memory(Path(temp_dir) / "memory.db", now=clock)
            try:
                memory.recall_append("thread-a", "user", "inspect the memory package")
                memory.recall_append("thread-a", "assistant", "implemented sqlite core storage")
                memory.recall_append("thread-a", "tool", "ran ruff check")
                memory.recall_append("thread-b", "user", "unrelated other task")

                recent = memory.recall_page("thread-a", k=2)
                queried = memory.recall_page("thread-a", query="sqlite storage", k=1)
            finally:
                memory.close()

            self.assertEqual([event.role for event in recent], ["tool", "assistant"])
            self.assertEqual(queried[0].content, "implemented sqlite core storage")

    def test_archival_search_is_semantic_persistent_and_filterable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            memory = Memory(path, embedder=HashingEmbedder(dim=64))
            try:
                memory.archival_add(
                    "fact",
                    "The harness uses SQLite memory with durable migrations.",
                    source_thread="thread-a",
                )
                memory.archival_add(
                    "pattern",
                    "The sandbox executes isolated Python code.",
                    source_thread="thread-b",
                )
            finally:
                memory.close()

            reopened = Memory(path, embedder=HashingEmbedder(dim=64))
            try:
                results = reopened.archival_search("sqlite durable memory", k=2)
                filtered = reopened.archival_search(
                    "sqlite durable memory",
                    k=2,
                    kind="pattern",
                )
            finally:
                reopened.close()

            self.assertEqual(results[0].memory.kind, "fact")
            self.assertIn("SQLite memory", results[0].memory.content)
            self.assertEqual([result.memory.kind for result in filtered], ["pattern"])

    def test_validates_bad_inputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = Memory(Path(temp_dir) / "memory.db")
            try:
                with self.assertRaises(MemoryValidationError):
                    memory.core_set("", "value")
                with self.assertRaises(MemoryValidationError):
                    memory.recall_append("thread", "invalid-role", "content")
                with self.assertRaises(MemoryValidationError):
                    memory.archival_add("fact", "content", metadata={"bad": object()})
            finally:
                memory.close()

    def test_memory_paging_config_validates_positive_limits(self):
        with self.assertRaises(ValueError):
            MemoryPagingConfig(max_history_tokens=0)

    def test_corrupt_database_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            path.write_bytes(b"not a sqlite database")

            with self.assertRaises(MemoryError):
                Memory(path)

    def test_cli_memory_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "memory.db")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(["mem", "--memory-db", path, "pin", "repo", "rlm"])
            self.assertEqual(exit_code, 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(["mem", "--memory-db", path, "get", "repo"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue().strip(), "rlm")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "mem",
                        "--memory-db",
                        path,
                        "archive-add",
                        "fact",
                        "SQLite archival memories are searchable.",
                    ]
                )
            self.assertEqual(exit_code, 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    ["mem", "--memory-db", path, "search", "SQLite searchable", "--limit", "1"]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("SQLite archival memories", stdout.getvalue())

    def test_taste_profile_records_are_typed_deduped_and_rendered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = Memory(Path(temp_dir) / "profile.db")
            try:
                store = TasteProfileStore(memory)
                first = store.add(
                    TasteRecord.create(
                        scope="user",
                        kind="preference",
                        text="Prefer concise final answers.",
                        confidence=0.8,
                    )
                )
                second = store.add(
                    TasteRecord.create(
                        scope="user",
                        kind="preference",
                        text="Prefer concise final answers.",
                        confidence=0.95,
                    )
                )
                context = store.render_context()
            finally:
                memory.close()

        self.assertEqual(first.id, second.id)
        self.assertEqual(second.confidence, 0.95)
        self.assertIn("User taste", context)
        self.assertIn("Prefer concise final answers", context)

    def test_cli_profile_learn_and_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "profile.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "profile",
                        "--profile-db",
                        path,
                        "learn",
                        "Prefer minimal diffs.",
                        "--active",
                    ]
                )
            self.assertEqual(exit_code, 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(["profile", "--profile-db", path])

        self.assertEqual(exit_code, 0)
        self.assertIn("Prefer minimal diffs", stdout.getvalue())

    def test_taste_profile_manager_routes_project_records_to_project_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            user_memory = Memory(Path(temp_dir) / "user.db")
            project_memory = Memory(Path(temp_dir) / "project.db")
            try:
                state = HarnessState(
                    task="fix tests",
                    workspace=temp_dir,
                    thread_id="thread",
                    run_id="run",
                    scratch={
                        "verification_result": {
                            "checks": [
                                {
                                    "check_type": "pytest",
                                    "passed": True,
                                    "command": "python -m pytest tests/test_memory.py",
                                }
                            ]
                        }
                    },
                )
                TasteProfileManager(user_memory, project_memory).learn_from_state(state)
                user_records = TasteProfileStore(user_memory).records(scope="project")
                project_records = TasteProfileStore(project_memory).records(scope="project")
            finally:
                user_memory.close()
                project_memory.close()

        self.assertEqual(user_records, [])
        self.assertEqual(len(project_records), 1)
        self.assertIn("python -m pytest", project_records[0].text)

    def test_evolution_proposals_are_deduped_and_rendered_after_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = Memory(Path(temp_dir) / "profile.db")
            try:
                store = EvolutionProposalStore(memory)
                first = store.add(
                    EvolutionProposal.create(
                        scope="user",
                        kind="prompt_rule",
                        title="Prefer concise responses",
                        body="Prefer concise final answers.",
                        rationale="The user explicitly asked for concise answers.",
                    )
                )
                second = store.add(
                    EvolutionProposal.create(
                        scope="user",
                        kind="prompt_rule",
                        title="Duplicate concise response rule",
                        body="Prefer concise final answers.",
                        rationale="Repeated evidence should update the same proposal.",
                    )
                )
                approved = store.approve(first.id)
                context = store.render_context()
            finally:
                memory.close()

        self.assertEqual(first.id, second.id)
        self.assertIsNotNone(approved)
        self.assertIn("Approved evolution rules", context)
        self.assertIn("Prefer concise final answers", context)

    def test_evolution_manager_routes_project_proposals_to_project_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            user_memory = Memory(Path(temp_dir) / "user.db")
            project_memory = Memory(Path(temp_dir) / "project.db")
            try:
                manager = EvolutionProposalManager(user_memory, project_memory)
                manager.add(
                    EvolutionProposal.create(
                        scope="project",
                        kind="verification_policy",
                        title="Run pytest",
                        body="Run `python -m pytest -q` for verification.",
                        rationale="The command passed for this project.",
                    )
                )
                user_proposals = EvolutionProposalStore(user_memory).proposals(scope="project")
                project_proposals = EvolutionProposalStore(project_memory).proposals(
                    scope="project"
                )
            finally:
                user_memory.close()
                project_memory.close()

        self.assertEqual(user_proposals, [])
        self.assertEqual(len(project_proposals), 1)
        self.assertIn("pytest", project_proposals[0].body)

    def test_cli_evolve_propose_list_and_approve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = str(Path(temp_dir) / "profile.db")
            memory_path = str(Path(temp_dir) / "memory.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "evolve",
                        "--profile-db",
                        profile_path,
                        "--memory-db",
                        memory_path,
                        "propose",
                        "--title",
                        "Prefer compact summaries",
                        "--body",
                        "Prefer compact summaries in final responses.",
                        "--rationale",
                        "The user keeps asking for concise answers.",
                    ]
                )
            self.assertEqual(exit_code, 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "evolve",
                        "--profile-db",
                        profile_path,
                        "--memory-db",
                        memory_path,
                        "list",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            proposals = json.loads(stdout.getvalue())
            self.assertEqual(len(proposals), 1)
            proposal_id = proposals[0]["id"]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "evolve",
                        "--profile-db",
                        profile_path,
                        "--memory-db",
                        memory_path,
                        "approve",
                        proposal_id,
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("approved", stdout.getvalue())

    def test_feedback_records_are_deduped_and_infer_taste(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = Memory(Path(temp_dir) / "profile.db")
            try:
                store = FeedbackStore(memory)
                first = store.add(
                    FeedbackRecord.create(
                        scope="user",
                        rating="good",
                        comment="Liked concise final answers.",
                        run_id="run-1",
                    )
                )
                second = store.add(
                    FeedbackRecord.create(
                        scope="user",
                        rating="positive",
                        comment="Liked concise final answers.",
                        run_id="run-1",
                    )
                )
                taste = infer_taste_from_feedback(first)
                proposals = infer_evolution_from_feedback(first)
            finally:
                memory.close()

        self.assertEqual(first.id, second.id)
        self.assertEqual(first.rating, "positive")
        self.assertEqual(len(taste), 1)
        self.assertEqual(taste[0].status, "active")
        self.assertIn("concise final answers", taste[0].text)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, "prompt_rule")

    def test_negative_feedback_creates_prompt_and_eval_proposals(self):
        feedback = FeedbackRecord.create(
            scope="user",
            rating="negative",
            comment="Avoid huge final answers.",
            run_id="run-2",
        )

        taste = infer_taste_from_feedback(feedback)
        proposals = infer_evolution_from_feedback(feedback)

        self.assertEqual(len(taste), 1)
        self.assertEqual(taste[0].status, "pending")
        self.assertIn("huge final answers", taste[0].text)
        self.assertEqual([proposal.kind for proposal in proposals], ["prompt_rule", "eval_case"])

    def test_cli_feedback_add_learns_taste_and_lists_feedback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = str(Path(temp_dir) / "profile.db")
            memory_path = str(Path(temp_dir) / "memory.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "feedback",
                        "--profile-db",
                        profile_path,
                        "--memory-db",
                        memory_path,
                        "add",
                        "Liked concise summaries.",
                        "--rating",
                        "good",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("taste=1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "feedback",
                        "--profile-db",
                        profile_path,
                        "--memory-db",
                        memory_path,
                        "list",
                        "--json",
                    ]
                )
            with Memory(Path(profile_path)) as memory:
                taste = TasteProfileStore(memory).records()

        self.assertEqual(exit_code, 0)
        records = json.loads(stdout.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["rating"], "positive")
        self.assertIn("concise summaries", taste[0].text)


if __name__ == "__main__":
    unittest.main()
