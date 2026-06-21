"""Tests for the streaming UX layer (visible output, diff display,
error cleanup, and token tracking).

These verify the user-facing improvements added after the
competitive audit:
- ``clean_final_answer`` strips tracebacks from final answers
- ``RunConsole.on_turn_event`` renders streaming events
- ``_extract_diff_from_code`` detects file-edit operations
- ``streaming_output_enabled`` respects HARNESS_PROGRESS
"""
from __future__ import annotations

import io
import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from rlm_harness.kernel.supervisor import clean_final_answer
from rlm_harness.rlm.runtime import (
    ObservationRecorded,
    RLMObservation,
    RLMResult,
    TokenDelta,
    TurnFinished,
)
from rlm_harness.task_runtime import (
    RunConsole,
    _extract_diff_from_code,
    _first_error_line,
    streaming_output_enabled,
)


class CleanFinalAnswerTests(unittest.TestCase):
    def test_strips_python_traceback_and_keeps_error_line(self):
        text = (
            "Some preamble\n"
            "Traceback (most recent call last):\n"
            "  File \"<string>\", line 1, in <module>\n"
            "  File \"/foo/bar.py\", line 10, in baz\n"
            "    raise RuntimeError('tests failed')\n"
            "RuntimeError: tests failed\n"
        )
        cleaned = clean_final_answer(text)
        self.assertNotIn("Traceback (most recent call last)", cleaned)
        self.assertNotIn("File \"<string>\"", cleaned)
        self.assertIn("RuntimeError: tests failed", cleaned)

    def test_preserves_clean_text(self):
        text = "The project uses Python and has a clean CLI."
        self.assertEqual(clean_final_answer(text), text)

    def test_strips_rlm_stream_error_prefix(self):
        text = "__rlm_stream_error__: provider returned 503"
        cleaned = clean_final_answer(text)
        self.assertNotIn("__rlm_stream_error__", cleaned)
        self.assertIn("provider returned 503", cleaned)

    def test_handles_none_and_empty(self):
        self.assertIsNone(clean_final_answer(None))
        self.assertEqual(clean_final_answer(""), "")


class StreamingOutputEnabledTests(unittest.TestCase):
    def test_disabled_when_harness_progress_is_on(self):
        with patch.dict(os.environ, {"HARNESS_PROGRESS": "on"}, clear=False):
            self.assertFalse(streaming_output_enabled(io.StringIO()))

    def test_disabled_when_harness_progress_is_off(self):
        with patch.dict(os.environ, {"HARNESS_PROGRESS": "off"}, clear=False):
            self.assertFalse(streaming_output_enabled(io.StringIO()))

    def test_disabled_for_non_tty(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(streaming_output_enabled(io.StringIO()))


class ExtractDiffFromCodeTests(unittest.TestCase):
    def test_detects_write_file(self):
        code = "write_file('mathlib.py', 'def add(a, b):\\n    return a + b\\n')"
        result = _extract_diff_from_code(code, "")
        self.assertIn("mathlib.py", result)
        self.assertIn("[edit]", result)

    def test_detects_apply_patch(self):
        code = "apply_patch(diff)"
        stdout = "--- a/mathlib.py\n+++ b/mathlib.py\n@@ -1,2 +1,2 @@\n"
        result = _extract_diff_from_code(code, stdout)
        self.assertIn("[patch]", result)
        self.assertIn("mathlib.py", result)

    def test_detects_propose_file_change(self):
        code = "propose_file_change('config.yaml', 'new: true')"
        result = _extract_diff_from_code(code, "")
        self.assertIn("[proposed]", result)
        self.assertIn("config.yaml", result)

    def test_returns_empty_for_non_edit_code(self):
        code = "print(project_summary())"
        self.assertEqual(_extract_diff_from_code(code, "summary text"), "")


class FirstErrorLineTests(unittest.TestCase):
    def test_extracts_last_error_line(self):
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"<string>\", line 1, in <module>\n"
            "  File \"/foo.py\", line 5, in bar\n"
            "    raise ValueError('bad value')\n"
            "ValueError: bad value\n"
        )
        result = _first_error_line(stderr)
        self.assertEqual(result, "ValueError: bad value")

    def test_handles_empty_stderr(self):
        self.assertEqual(_first_error_line(""), "unknown error")


class RunConsoleStreamingTests(unittest.TestCase):
    def _make_console(self, *, streaming: bool = True) -> RunConsole:
        args = Namespace(
            json_output=False,
            quiet=False,
            stream=False,
            no_memory=True,
        )
        console = RunConsole(args, stream=io.StringIO())
        console.enabled = True
        console.streaming = streaming
        return console

    def test_json_output_suppresses_streaming(self):
        args = Namespace(json_output=True, quiet=False, stream=False, no_memory=True)
        console = RunConsole(args, stream=io.StringIO())
        # on_turn_event should be a no-op when json_output is set
        console.on_turn_event(TokenDelta(delta="hello"))
        self.assertEqual(console.stream.getvalue(), "")

    def test_token_delta_writes_to_stream(self):
        console = self._make_console(streaming=True)
        console.on_turn_event(TokenDelta(delta="hello "))
        console.on_turn_event(TokenDelta(delta="world"))
        self.assertIn("hello", console.stream.getvalue())
        self.assertIn("world", console.stream.getvalue())

    def test_non_streaming_console_does_not_write_tokens(self):
        console = self._make_console(streaming=False)
        console.on_turn_event(TokenDelta(delta="hello"))
        self.assertEqual(console.stream.getvalue(), "")

    def test_observation_renders_edit_summary(self):
        console = self._make_console(streaming=True)
        obs = RLMObservation(
            code="write_file('mathlib.py', 'content')",
            stdout="wrote mathlib.py (42 bytes)",
            status="ok",
        )
        console.on_turn_event(ObservationRecorded(observation=obs, iteration=1))
        output = console.stream.getvalue()
        self.assertIn("[edit]", output)
        self.assertIn("mathlib.py", output)

    def test_observation_renders_clean_error(self):
        console = self._make_console(streaming=True)
        obs = RLMObservation(
            code="read_file('missing.py')",
            stderr=(
                "Traceback (most recent call last):\n"
                "  File ...\n"
                "ToolError: not a file: missing.py"
            ),
            status="error",
        )
        console.on_turn_event(ObservationRecorded(observation=obs, iteration=1))
        output = console.stream.getvalue()
        self.assertIn("[tool error]", output)
        self.assertIn("ToolError: not a file: missing.py", output)
        self.assertNotIn("Traceback", output)

    def test_turn_finished_shows_token_count(self):
        console = self._make_console(streaming=True)
        result = RLMResult(
            final_answer="done",
            status="done",
            iterations=2,
            tokens_used=1500,
        )
        console.on_turn_event(TurnFinished(result=result))
        output = console.stream.getvalue()
        self.assertIn("[done]", output)
        self.assertIn("1500", output)


if __name__ == "__main__":
    unittest.main()
