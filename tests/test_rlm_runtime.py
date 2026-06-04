import tempfile
import unittest
from pathlib import Path

from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.rlm.runtime import (
    RLM_STREAM_ERROR_PREFIX,
    RLM_SYSTEM_PROMPT,
    RLMObservation,
    RLMRuntime,
    _raise_on_stream_error,
    build_bootstrap_code,
    find_repl_blocks,
    format_observation,
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
        self.assertEqual(find_repl_blocks(text), ["a = 1", "ignored", "print(a)"])

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

    def test_observation_format_truncates_large_streams(self):
        observation = RLMObservation(
            code="print('x')",
            stdout="a" * 30_000,
            status="ok",
        )

        rendered = format_observation(observation)

        self.assertLess(len(rendered), 14_000)
        self.assertIn("truncated", rendered)
        self.assertIn("narrow the query", rendered)

    def test_observation_format_shares_budget_across_streams(self):
        observation = RLMObservation(
            code="print('x')",
            stdout="a" * 30_000,
            stderr="b" * 30_000,
            status="error",
        )

        rendered = format_observation(observation)

        self.assertLess(len(rendered), 12_000)
        self.assertIn("STDOUT:", rendered)
        self.assertIn("STDERR:", rendered)
        self.assertIn("STDOUT note: truncated", rendered)
        self.assertIn("STDERR note: truncated", rendered)
        self.assertIn("narrow the query", rendered)

    def test_rlm_prompt_routes_project_overview_to_summary_tool(self):
        self.assertIn("project_summary", RLM_SYSTEM_PROMPT)
        self.assertIn("project_audit", RLM_SYSTEM_PROMPT)
        self.assertIn("Do not answer by\nprinting raw source code", RLM_SYSTEM_PROMPT)
        self.assertIn("file inventory", RLM_SYSTEM_PROMPT)
        self.assertIn("friendly, ordinary English", RLM_SYSTEM_PROMPT)


class StreamErrorTranslationTests(unittest.TestCase):
    """The streaming path retries transient errors. When retries
    are exhausted, the runtime must surface the failure as an
    ``LMClientError`` so the supervisor's error path takes over
    — the user must not see ``__rlm_stream_error__:...`` in
    their final answer.
    """

    def test_stream_error_prefix_translates_to_lmclient_error(self):
        with self.assertRaises(LMClientError) as ctx:
            _raise_on_stream_error(
                f"{RLM_STREAM_ERROR_PREFIX}HTTP 503 Service Unavailable (after 3 attempts)"
            )
        self.assertIn("HTTP 503", str(ctx.exception))
        self.assertIn("after 3 attempts", str(ctx.exception))

    def test_non_stream_error_passes_through(self):
        # Anything without the prefix is a normal model
        # response and must not be turned into an exception.
        _raise_on_stream_error("Hello, world.")
        _raise_on_stream_error("")

    def test_streaming_model_call_raises_on_provider_error(self):
        """When the model's stream produces an error event
        (after retries are exhausted), the runtime must raise
        an ``LMClientError`` instead of returning a prefixed
        string the supervisor would otherwise forward to the
        user as the final answer.
        """
        from rlm_harness.types import TokenEvent

        class ErrorStreamingClient(LMClient):
            def __init__(self):
                super().__init__(provider="stub", model="error")

            def stream(self, messages, max_tokens=512, temperature=0.2):
                yield TokenEvent(type="start", model="error", provider="stub")
                yield TokenEvent(
                    type="error",
                    error="HTTP 503 Service Unavailable (after 3 attempts)",
                    provider="stub",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            runtime = RLMRuntime(
                ErrorStreamingClient(),
                workspace=workspace,
                max_iterations=1,
                sandbox_enabled=False,
            )
            with self.assertRaises(LMClientError) as ctx:
                list(runtime.stream_turn("hi", context={"task": "hi"}))
        self.assertIn("HTTP 503", str(ctx.exception))
        # The error message must NOT carry the legacy
        # ``__rlm_stream_error__:`` prefix the user used to
        # see in their final answer.
        self.assertFalse(str(ctx.exception).startswith(RLM_STREAM_ERROR_PREFIX))


if __name__ == "__main__":
    unittest.main()
