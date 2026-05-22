import tempfile
import unittest
from pathlib import Path

from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import RLMRuntime, find_repl_blocks


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


if __name__ == "__main__":
    unittest.main()
