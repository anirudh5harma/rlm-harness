import tempfile
import unittest
from pathlib import Path

from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import (
    RLM_SYSTEM_PROMPT,
    RLMRuntime,
    build_bootstrap_code,
    find_repl_blocks,
    stopped_final_answer,
)


class ScriptedRuntimeClient(LMClient):
    def __init__(self, responses):
        super().__init__(provider="stub")
        self.responses = list(responses)
        self.prompts = []

    def complete(self, messages, max_tokens=512, temperature=0.2):
        self.prompts.append([m.content for m in messages])
        if not self.responses:
            return super().complete(messages, max_tokens=max_tokens, temperature=temperature)
        content = self.responses.pop(0)
        from rlm_harness.types import Completion

        return Completion(content=content, model="scripted", provider="test", latency_ms=0)


class RLMRuntimeTests(unittest.TestCase):
    def test_find_repl_blocks_extracts_multiple_blocks(self):
        text = (
            "before\n```repl\na = 1\n```\nmiddle\n```python\nignored\n```\n```repl\nprint(a)\n```"
        )
        self.assertEqual(find_repl_blocks(text), ["a = 1", "print(a)"])

    def test_runtime_iterates_until_answer_ready(self):
        client = ScriptedRuntimeClient(
            [
                (
                    "Need to inspect.\n"
                    "```repl\n"
                    "print('seen context:', context)\n"
                    "answer['content'] = 'done: ' + context\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RLMRuntime(
                client, workspace=Path(temp_dir), max_iterations=3, sandbox_enabled=False
            )
            result = runtime.completion("summarize", context="abc")
        self.assertEqual(result.final_answer, "done: abc")
        self.assertEqual(result.status, "done")
        self.assertEqual(result.iterations, 1)
        self.assertIn("seen context: abc", result.observations[0].stdout)

    def test_runtime_exposes_llm_query_in_local_repl(self):
        client = ScriptedRuntimeClient(
            [
                (
                    "```repl\n"
                    "response = llm_query('What is this?', context='tiny')\n"
                    "answer['content'] = response\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                "sub answer",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RLMRuntime(
                client, workspace=Path(temp_dir), max_iterations=2, sandbox_enabled=False
            )
            result = runtime.completion("root", context="ctx")
        self.assertEqual(result.final_answer, "sub answer")
        self.assertEqual(result.subcalls, 1)

    def test_bootstrap_preserves_structured_context_and_ctx_alias(self):
        namespace = {}
        exec(build_bootstrap_code({"task": "what is this project", "plan": ["inspect"]}), namespace)

        self.assertIsInstance(namespace["context"], dict)
        self.assertEqual(namespace["context"]["task"], "what is this project")
        self.assertIs(namespace["ctx"], namespace["context"])

    def test_stopped_final_answer_does_not_return_raw_repl_blocks(self):
        answer = stopped_final_answer(
            [
                (
                    "```repl\n"
                    "print(context.keys())\n"
                    "```\n"
                    "```repl\n"
                    "answer['content'] = 'summary'\n"
                    "```"
                )
            ],
            [],
        )

        self.assertNotIn("```repl", answer)
        self.assertNotIn("print(context.keys())", answer)
        self.assertIn("stopped before producing a final answer", answer)

    def test_rlm_prompt_routes_project_overview_to_summary_tool(self):
        self.assertIn("project_summary", RLM_SYSTEM_PROMPT)
        self.assertIn("project_audit", RLM_SYSTEM_PROMPT)
        self.assertIn("Do not answer by\nprinting raw source code", RLM_SYSTEM_PROMPT)
        self.assertIn("file inventory", RLM_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
