import contextlib
import io
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from rlm_harness import cli
from rlm_harness.memory import Memory, MemoryError, MemoryValidationError
from rlm_harness.memory.embed import HashingEmbedder


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


if __name__ == "__main__":
    unittest.main()
