"""Tests for the streaming RLM runtime (Phase A.2).

The streaming RLM runtime yields one event per model call, observation,
sub-call, and final answer. It is what the supervisor in Phase A.3
consumes. The non-streaming `RLMRuntime.completion()` is preserved as
a thin wrapper that buffers the stream into a final `RLMResult`.

The streaming events are:

    TurnStarted(query, context)         — at the start of a turn
    IterationStarted(iteration)         — for each model call
    TokenDelta(delta)                   — for each token delta
    IterationFinished(response, blocks) — for each model call
    ObservationRecorded(observation)    — after each REPL block
    SubcallStarted / SubcallFinished    — for llm_query / rlm_query
    TurnFinished(result)                — at the end of a turn
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import (
    RLMRuntime,
    TurnFinished,
    TurnStarted,
)


class ScriptedStreamClient(LMClient):
    """A scripted LMClient whose `stream()` yields scripted chunks."""

    def __init__(self, scripted_responses: list[list[str]]):
        super().__init__(provider="stub", model="scripted")
        # Each model call consumes the next entry; each entry is a list
        # of deltas the stream yields before the `finish` event.
        self._queue = list(scripted_responses)
        self.calls: list[list[str]] = []

    def stream(self, messages, max_tokens=512, temperature=0.2):
        from rlm_harness.types import TokenEvent

        deltas = self._queue.pop(0) if self._queue else ["ok"]
        self.calls.append(deltas)
        yield TokenEvent(type="start", model=self.model, provider=self.provider)
        for delta in deltas:
            yield TokenEvent(
                type="delta", delta=delta, model=self.model, provider=self.provider
            )
        yield TokenEvent(
            type="finish",
            model=self.model,
            provider=self.provider,
            usage={"prompt_tokens": 1, "completion_tokens": len(deltas)},
            finish_reason="stop",
        )

    def complete(self, messages, max_tokens=512, temperature=0.2):
        # Tests should not call this; the stream is the entry point.
        raise AssertionError("ScriptedStreamClient should be used via stream()")


class StreamingRLMTests(unittest.TestCase):
    def test_turn_event_sequence_for_single_iteration(self):
        client = ScriptedStreamClient(
            [
                [
                    "Need to inspect.\n",
                    "```repl\n",
                    "print('seen:', context)\n",
                    "answer['content'] = 'done: ' + str(context)\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RLMRuntime(
                client, workspace=Path(temp_dir), max_iterations=2, sandbox_enabled=False
            )
            events = list(runtime.stream_turn("summarize", context="abc"))
        kinds = [type(event).__name__ for event in events]
        self.assertEqual(kinds[0], "TurnStarted")
        self.assertEqual(kinds[-1], "TurnFinished")
        # We expect at least one IterationStarted, one IterationFinished,
        # one ObservationRecorded, and at least one TokenDelta.
        self.assertIn("IterationStarted", kinds)
        self.assertIn("IterationFinished", kinds)
        self.assertIn("ObservationRecorded", kinds)
        self.assertIn("TokenDelta", kinds)
        finish = events[-1]
        self.assertIsInstance(finish, TurnFinished)
        self.assertEqual(finish.result.status, "done")
        self.assertEqual(finish.result.final_answer, "done: abc")
        start = events[0]
        self.assertIsInstance(start, TurnStarted)
        self.assertEqual(start.query, "summarize")
        self.assertEqual(start.context_preview, "abc")

    def test_token_deltas_concatenate_to_response(self):
        client = ScriptedStreamClient(
            [
                [
                    "```repl\n",
                    "answer['content'] = 'final'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RLMRuntime(
                client, workspace=Path(temp_dir), max_iterations=1, sandbox_enabled=False
            )
            events = list(runtime.stream_turn("q", context="c"))
        deltas = [
            event.delta
            for event in events
            if getattr(event, "delta", None)
        ]
        joined = "".join(deltas)
        self.assertIn("```repl", joined)
        self.assertIn("answer['ready'] = True", joined)

    def test_completion_and_stream_turn_agree_on_final_answer(self):
        """Both entry points reach the same final answer for the
        same scripted input.
        """
        from rlm_harness.model_client import LMClient

        with tempfile.TemporaryDirectory() as temp_dir:
            stream_client = ScriptedStreamClient(
                [
                    [
                        "```repl\n",
                        "answer['content'] = 'final'\n",
                        "answer['ready'] = True\n",
                        "```",
                    ]
                ]
            )
            stream_runtime = RLMRuntime(
                stream_client,
                workspace=Path(temp_dir),
                max_iterations=1,
                sandbox_enabled=False,
            )
            events = list(stream_runtime.stream_turn("q", context="c"))
            stream_result = next(e for e in events if isinstance(e, TurnFinished)).result

            # The non-streaming path uses the existing stub provider.
            # We do not assert the iteration count matches exactly —
            # stub paths may differ — only that the final answer does.
            stub_client = LMClient(provider="stub")
            # Force the stub to return the same REPL block.
            stub_client._stub_rlm_response = lambda text: (  # type: ignore[attr-defined]
                "```repl\n"
                "answer['content'] = 'final'\n"
                "answer['ready'] = True\n"
                "```"
            )
            stub_runtime = RLMRuntime(
                stub_client,
                workspace=Path(temp_dir),
                max_iterations=1,
                sandbox_enabled=False,
            )
            stub_result = stub_runtime.completion("q", context="c")
        self.assertEqual(stream_result.final_answer, stub_result.final_answer)

    def test_stream_turn_emits_observation_recorded(self):
        client = ScriptedStreamClient(
            [
                [
                    "```repl\n",
                    "print('hello')\n",
                    "answer['content'] = 'ok'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RLMRuntime(
                client, workspace=Path(temp_dir), max_iterations=1, sandbox_enabled=False
            )
            events = list(runtime.stream_turn("q", context="c"))
        from rlm_harness.rlm.runtime import ObservationRecorded

        observations = [e for e in events if isinstance(e, ObservationRecorded)]
        self.assertEqual(len(observations), 1)
        self.assertIn("hello", observations[0].observation.stdout)


if __name__ == "__main__":
    unittest.main()
