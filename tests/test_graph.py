import tempfile
import unittest
from pathlib import Path

from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import Nodes, parse_numbered_plan
from rlm_harness.model_client import LMClient
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState


class GraphTests(unittest.TestCase):
    def test_parse_numbered_plan(self):
        self.assertEqual(parse_numbered_plan("1. Inspect\n2. Act"), ["Inspect", "Act"])

    def test_stub_graph_reaches_done(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("test task", temp_dir)
            state = HarnessState(
                task="test task",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            graph = build_graph(Nodes(LMClient(provider="stub"), traces), backend="simple")
            final_state = graph.invoke(state)
            self.assertEqual(final_state.status, "done")
            self.assertTrue(final_state.final_answer)
            self.assertIn("Trace report", traces.render_report(run_id))


if __name__ == "__main__":
    unittest.main()
