import tempfile
import unittest
from pathlib import Path

from rlm_harness.tracing import TraceStore


class TraceStoreTests(unittest.TestCase):
    def test_records_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("task", temp_dir)
            traces.event(run_id, "kind", {"value": 1}, node="node")
            report = traces.render_report(run_id)
            self.assertIn("Trace report", report)
            self.assertIn('"value": 1', report)


if __name__ == "__main__":
    unittest.main()
