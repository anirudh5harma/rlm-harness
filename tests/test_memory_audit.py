import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness.memory import Memory
from rlm_harness.memory.audit import PhaseAudit


class PhaseAuditTests(unittest.TestCase):
    def test_open_creates_memory_and_persists_phase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.db"
            audit = PhaseAudit.open(audit_path)
            audit.set_phase("A", gate="pytest tests/test_rlm_runtime.py -x")
            self.assertTrue(audit_path.exists())
            with Memory(audit_path) as memory:
                fresh = PhaseAudit(memory)
                self.assertEqual(fresh.current_phase(), "A")
                self.assertEqual(
                    fresh.current_gate(), "pytest tests/test_rlm_runtime.py -x"
                )
                self.assertNotEqual(fresh.current_phase(), "unknown")

    def test_record_writes_archival_entry_searchable_by_kind(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = PhaseAudit.open(Path(temp_dir) / "audit.db")
            audit.set_phase("A")
            archive_id = audit.record(
                kind="decision",
                content=json.dumps({"decision": "stream by default", "why": "latency"}),
                metadata={"phase": "A"},
            )
            self.assertGreater(archive_id, 0)
            results = audit.memory.archival_search(
                "stream latency decision", k=5, kind="decision"
            )
            self.assertTrue(
                any(result.memory.id == archive_id for result in results)
            )

    def test_record_never_raises_on_pathological_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = PhaseAudit.open(Path(temp_dir) / "audit.db")
            # set is not JSON-serializable; the audit helper must swallow
            # the failure rather than crash the run.
            archive_id = audit.record(
                kind="test",
                content="ok",
                metadata={"bad": {1, 2, 3}},  # type: ignore[arg-type]
            )
            self.assertEqual(archive_id, -1)


if __name__ == "__main__":
    unittest.main()
