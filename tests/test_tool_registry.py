import unittest

from rlm_harness.actions import (
    ApplyPendingChangeAction,
    ProjectSummaryAction,
    ProposeChangeAction,
    parse_action,
)
from rlm_harness.graph.nodes import render_tool_list
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.tools import default_tool_registry, render_tool_catalog


class ToolRegistryTests(unittest.TestCase):
    def test_registry_covers_existing_sandbox_tool_names(self):
        registry = default_tool_registry()

        self.assertTrue(set(sandbox_tools.tool_names()).issubset(set(registry.names())))

    def test_registry_describes_risk_and_confirmation_policy(self):
        registry = default_tool_registry()

        write_file = registry.get("write_file")
        apply_pending = registry.get("apply_pending_change")
        read_file = registry.get("read_file")

        self.assertEqual(write_file.risk.value, "high")
        self.assertTrue(write_file.requires_confirmation)
        self.assertEqual(apply_pending.action_kind, "apply_pending_change")
        self.assertTrue(apply_pending.requires_confirmation)
        self.assertEqual(read_file.risk.value, "read")
        self.assertFalse(read_file.requires_confirmation)

    def test_public_registry_excludes_runtime_compatibility_actions(self):
        registry = default_tool_registry()

        self.assertNotIn("python_repl", registry.names())
        self.assertIn("python_repl", registry.names(include_internal=True))
        self.assertNotIn(
            "rlm_task",
            {tool["name"] for tool in registry.public_payload()},
        )

    def test_rendered_catalog_is_grouped_for_humans(self):
        catalog = render_tool_catalog()

        self.assertIn("Harness tools", catalog)
        self.assertIn("workspace", catalog)
        self.assertIn("mcp", catalog)
        self.assertIn("project_summary [read]", catalog)
        self.assertIn("mcp_call_tool [medium]", catalog)
        self.assertIn("apply_pending_change [high] (confirmation)", catalog)

    def test_graph_prompt_tool_list_uses_registry(self):
        tool_list = render_tool_list()

        self.assertIn("project_summary", tool_list)
        self.assertIn("complete_task", tool_list)
        self.assertNotIn("record_memory", tool_list)
        self.assertNotIn("python_repl", tool_list)

    def test_action_contract_accepts_registry_action_kinds(self):
        self.assertIsInstance(
            parse_action({"kind": "project_summary", "max_files": 20}),
            ProjectSummaryAction,
        )
        self.assertIsInstance(
            parse_action(
                {
                    "kind": "propose_file_change",
                    "path": "README.md",
                    "content": "# Harness\n",
                    "reason": "example",
                }
            ),
            ProposeChangeAction,
        )
        self.assertIsInstance(
            parse_action({"kind": "apply_pending_change", "change_id": "change_1"}),
            ApplyPendingChangeAction,
        )
        self.assertEqual(
            parse_action(
                {
                    "kind": "mcp_call_tool",
                    "server": "github",
                    "tool_name": "get_issue",
                    "arguments": {"id": 1},
                }
            ).kind,
            "mcp_call_tool",
        )


if __name__ == "__main__":
    unittest.main()
