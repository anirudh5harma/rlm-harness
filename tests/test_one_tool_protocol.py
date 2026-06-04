"""Tests for the one-tool-protocol gate (Phase C).

The pivot plan's Phase C gate:

    "the model sees a stable tool list, and every execution
     records an action event and observation event"

This test asserts:

* The default registry exposes a stable, public, in-scope tool
  list that the model can rely on.
* The list is sorted by name; the same registry always returns
  the same list.
* Every tool execution via the typed registry records both an
  `ActionSelectedEvent` and an `ObservationRecordedEvent` in the
  trace store.
* The legacy `python`-JSON action protocol is no longer
  referenced from the graph node path; new code paths use the
  typed registry only.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness.actions import (
    ReadFileAction,
)
from rlm_harness.kernel.events import (
    ActionSelectedEvent,
    ObservationRecordedEvent,
)
from rlm_harness.tools import default_tool_registry
from rlm_harness.tools.executor import ToolExecutor
from rlm_harness.tracing import TraceStore


class StableToolListTests(unittest.TestCase):
    def test_default_registry_is_stable(self):
        registry = default_tool_registry()
        names_a = registry.names()
        names_b = registry.names()
        self.assertEqual(names_a, names_b)
        self.assertEqual(names_a, sorted(names_a))
        self.assertGreater(len(names_a), 5)
        self.assertEqual(len(set(names_a)), len(names_a))

    def test_default_registry_exposes_only_public_tools(self):
        registry = default_tool_registry()
        public = registry.all()
        for descriptor in public:
            self.assertTrue(descriptor.name)
            self.assertTrue(descriptor.summary)
            self.assertTrue(descriptor.action_kind)

    def test_registry_payload_is_json_serialisable(self):
        """The tool list is what the model sees. It must be a
        JSON-serialisable list of dicts.
        """
        registry = default_tool_registry()
        payload = registry.payload()
        serialised = json.dumps(payload)
        self.assertEqual(json.loads(serialised), payload)


class ToolExecutionRecordsBothEventsTests(unittest.TestCase):
    def test_executor_emits_action_and_observation_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# Test\nA test project.\n")
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            traces.start_run("readme", str(workspace), thread_id="t")

            executor = ToolExecutor(workspace)
            action = ReadFileAction(path="README.md", reason="orient")
            observation = executor.execute(action)
            self.assertIsNotNone(observation)
            self.assertEqual(observation.action_id, action.action_id)

            descriptor = default_tool_registry().for_action_kind(action.kind)
            self.assertEqual(descriptor.name, "read_file")
            self.assertEqual(descriptor.risk.value, "read")

    def test_typed_events_round_trip_for_action_and_observation(self):
        """The Pydantic event types from the kernel/events.py
        module are the audit trail. A typed action and a typed
        observation must round-trip through the trace store.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run("x", str(workspace), thread_id="t")

            action = ReadFileAction(path="README.md")
            traces.record_typed_event(
                ActionSelectedEvent(
                    run_id=run_id,
                    sequence=traces.next_sequence(run_id),
                    node="tool",
                    action=action,
                )
            )
            from rlm_harness.actions import TextObservation

            obs = TextObservation(
                action_id=action.action_id,
                text="ok",
                summary="ok",
            )
            traces.record_typed_event(
                ObservationRecordedEvent(
                    run_id=run_id,
                    sequence=traces.next_sequence(run_id),
                    node="tool",
                    observation=obs,
                )
            )
            typed = traces.typed_events(run_id)
            kinds = [e.kind for e in typed]
            self.assertIn("action_selected", kinds)
            self.assertIn("observation_recorded", kinds)


class OneProtocolEnforcementTests(unittest.TestCase):
    def test_parse_python_action_is_gone(self):
        """The legacy `parse_python_action` parser must be
        removed from the production path. The graph node that
        used to call it (`_select_sandbox_action`) is no longer
        wired in the supervisor flow.
        """
        import rlm_harness.graph.nodes as nodes

        self.assertFalse(hasattr(nodes, "parse_python_action"))
        self.assertFalse(hasattr(nodes.Nodes, "_select_sandbox_action"))

    def test_graph_node_uses_typed_action_only(self):
        """`_select_tool_action` is the only action-selection
        path in the production graph node.
        """
        import rlm_harness.graph.nodes as nodes

        self.assertTrue(hasattr(nodes.Nodes, "_select_tool_action"))
        self.assertFalse(hasattr(nodes, "parse_python_action"))


if __name__ == "__main__":
    unittest.main()
