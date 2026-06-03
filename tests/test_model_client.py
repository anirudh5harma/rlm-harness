import json
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from rlm_harness.model_client import LMClient, LMClientError, is_project_audit_prompt
from rlm_harness.rlm.runtime import find_repl_blocks
from rlm_harness.types import Msg


class FakeResponse:
    status = 200

    def __init__(self, content: str = "hello"):
        body = json.dumps(
            {
                "model": "test-model",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )
        self._buffer = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self, amt: int = -1):
        """Behave like a real HTTPResponse.

        With `amt == -1` (default), return the entire buffered body.
        With `amt > 0`, return up to `amt` bytes and drain the buffer,
        so the next call returns `b""` and the streaming loop can
        terminate.
        """
        if amt is None or amt < 0:
            data, self._buffer = self._buffer, b""
            return data
        data = self._buffer[:amt]
        self._buffer = self._buffer[amt:]
        return data


class LMClientTests(unittest.TestCase):
    def test_openai_compatible_round_trip(self):
        seen_requests = []

        def fake_urlopen(request, timeout):
            seen_requests.append((request, timeout))
            return FakeResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="test-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
            )
            completion = client.complete(
                [Msg(role="user", content="Reply with exactly: hello")],
                max_tokens=12,
                temperature=0,
            )

        self.assertEqual(completion.content, "hello")
        self.assertEqual(completion.model, "test-model")
        self.assertEqual(completion.prompt_tokens, 3)
        self.assertEqual(completion.completion_tokens, 1)
        request, timeout = seen_requests[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(timeout, 120)
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(payload["max_tokens"], 12)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["messages"][0]["content"], "Reply with exactly: hello")
        self.assertEqual(request.headers["Authorization"], "Bearer token")
        self.assertIn("rlm-harness/0.1", request.headers["User-agent"])

    def test_openai_compatible_requests_json_mode_for_action_prompts(self):
        seen_payloads = []

        def fake_urlopen(request, timeout):
            seen_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse('{"kind":"project_summary"}')

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="test-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
            )
            completion = client.complete(
                [
                    Msg(
                        role="user",
                        content=(
                            "Return one typed action JSON object.\n"
                            "Task: what is this project?"
                        ),
                    )
                ],
                max_tokens=12,
                temperature=0,
            )

        self.assertEqual(completion.content, '{"kind":"project_summary"}')
        self.assertEqual(
            seen_payloads[0]["response_format"],
            {"type": "json_object"},
        )

    def test_openai_compatible_retries_without_json_mode_when_unsupported(self):
        seen_payloads = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            seen_payloads.append(payload)
            if "response_format" in payload:
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    hdrs={},
                    fp=BytesIO(b'{"error":{"message":"unsupported parameter: response_format"}}'),
                )
            return FakeResponse('{"kind":"project_summary"}')

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="test-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
            )
            completion = client.complete(
                [
                    Msg(
                        role="user",
                        content=(
                            "Return only valid JSON for this action.\n"
                            "Task: what is this project?"
                        ),
                    )
                ],
                max_tokens=12,
                temperature=0,
            )

        self.assertEqual(completion.content, '{"kind":"project_summary"}')
        self.assertEqual(len(seen_payloads), 2)
        self.assertIn("response_format", seen_payloads[0])
        self.assertNotIn("response_format", seen_payloads[1])

    def test_http_error_includes_provider_detail(self):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Forbidden",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"model access denied"}}'),
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="test-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
            )
            with self.assertRaisesRegex(
                LMClientError,
                "HTTP 403 Forbidden: model access denied",
            ):
                client.complete([Msg(role="user", content="hello")])

    def test_stub_project_question_uses_project_summary_tool(self):
        client = LMClient(provider="stub")

        completion = client.complete(
            [
                Msg(
                    role="user",
                    content=(
                        "Return only valid JSON for this action.\n"
                        "Task: what is this project"
                    ),
                )
            ]
        )

        payload = json.loads(completion.content)
        self.assertEqual(payload["type"], "python")
        self.assertEqual(payload["code"], "print(project_summary())")

    def test_stub_project_gap_question_uses_project_audit_tool(self):
        client = LMClient(provider="stub")

        completion = client.complete(
            [
                Msg(
                    role="user",
                    content=(
                        "Return only valid JSON for this action.\n"
                        "Task: find any logical and technical gaps in this project"
                    ),
                )
            ]
        )

        payload = json.loads(completion.content)
        self.assertEqual(payload["type"], "python")
        self.assertIn("project_audit()", payload["code"])
        self.assertIn("rlm.completion", payload["code"])

    def test_stub_handles_project_typo_and_next_steps(self):
        self.assertTrue(
            is_project_audit_prompt("what is this porject about and what must be done next")
        )
        client = LMClient(provider="stub")

        completion = client.complete(
            [
                Msg(
                    role="user",
                    content=(
                        "Return only valid JSON for this action.\n"
                        "Task: what is this porject about and what must be done next"
                    ),
                )
            ]
        )

        payload = json.loads(completion.content)
        self.assertIn("project_audit()", payload["code"])

    def test_stub_rlm_fallback_generates_valid_python_for_multiline_context(self):
        client = LMClient(provider="stub")
        completion = client.complete(
            [
                Msg(
                    role="user",
                    content=(
                        "Query:\n"
                        "what is this porject about and what must be done next\n\n"
                        "The context is available in the REPL as variable `context`."
                    ),
                )
            ]
        )

        blocks = find_repl_blocks(completion.content)
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], "<stub-rlm>", "exec")


class StreamRetryTests(unittest.TestCase):
    """Streaming-path retry on transient provider errors.

    HTTP 503 from OpenRouter (and similar transient 5xx/4xx
    codes) should not surface as the final answer. The streaming
    path retries with exponential backoff before giving up.
    """

    def _http_503(self, request):
        return urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs={},
            fp=BytesIO(b'{"error":{"message":"overloaded"}}'),
        )

    def test_stream_retries_on_503_and_succeeds(self):
        seen: list[int] = []

        def fake_urlopen(request, timeout):
            seen.append(len(seen) + 1)
            if len(seen) < 2:
                raise self._http_503(request)
            return FakeResponse("hi from attempt")

        client = LMClient(
            provider="openai-compatible",
            model="test-model",
            base_url="http://127.0.0.1:8080/v1",
            api_key="token",
            max_stream_retries=3,
            stream_retry_base_delay_s=0.0,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            events = list(
                client.stream(
                    [Msg(role="user", content="hello")],
                    max_tokens=64,
                    temperature=0.0,
                )
            )

        kinds = [event.type for event in events]
        # The first attempt raises HTTPError; the second
        # succeeds. The contract is `start → 0+ delta → finish`.
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "finish")
        self.assertNotIn("error", kinds)
        self.assertEqual(seen, [1, 2])

    def test_stream_exhausts_retries_and_yields_error_event(self):
        seen: list[int] = []

        def fake_urlopen(request, timeout):
            seen.append(len(seen) + 1)
            raise self._http_503(request)

        client = LMClient(
            provider="openai-compatible",
            model="test-model",
            base_url="http://127.0.0.1:8080/v1",
            api_key="token",
            max_stream_retries=3,
            stream_retry_base_delay_s=0.0,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            events = list(
                client.stream(
                    [Msg(role="user", content="hello")],
                    max_tokens=64,
                    temperature=0.0,
                )
            )

        kinds = [event.type for event in events]
        # All 3 attempts fail; the final event is an `error`
        # event whose message includes the attempt count so the
        # user can tell what happened.
        self.assertEqual(kinds[-1], "error")
        error_event = events[-1]
        self.assertIn("HTTP 503", error_event.error)
        self.assertIn("after 3 attempts", error_event.error)
        self.assertEqual(seen, [1, 2, 3])

    def test_stream_does_not_retry_non_transient_status(self):
        """HTTP 403 (auth) is not transient — the client should
        fail immediately, not waste retries on a permanent
        error.
        """
        seen: list[int] = []

        def fake_urlopen(request, timeout):
            seen.append(len(seen) + 1)
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Forbidden",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"bad key"}}'),
            )

        client = LMClient(
            provider="openai-compatible",
            model="test-model",
            base_url="http://127.0.0.1:8080/v1",
            api_key="bad",
            max_stream_retries=3,
            stream_retry_base_delay_s=0.0,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            events = list(
                client.stream(
                    [Msg(role="user", content="hello")],
                    max_tokens=64,
                    temperature=0.0,
                )
            )

        self.assertEqual(seen, [1])
        self.assertEqual(events[-1].type, "error")
        self.assertIn("HTTP 403", events[-1].error)
        # No "(after N attempts)" suffix on non-transient errors.
        self.assertNotIn("after ", events[-1].error)

    def test_stream_retries_on_url_error(self):
        """Connection-level failures (URLError) are also
        transient; the client should retry and eventually give
        up.
        """
        seen: list[int] = []

        def fake_urlopen(request, timeout):
            seen.append(len(seen) + 1)
            raise urllib.error.URLError("connection refused")

        client = LMClient(
            provider="openai-compatible",
            model="test-model",
            base_url="http://127.0.0.1:8080/v1",
            api_key="token",
            max_stream_retries=2,
            stream_retry_base_delay_s=0.0,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            events = list(
                client.stream(
                    [Msg(role="user", content="hello")],
                    max_tokens=64,
                    temperature=0.0,
                )
            )

        self.assertEqual(seen, [1, 2])
        self.assertEqual(events[-1].type, "error")
        self.assertIn("connection refused", events[-1].error)
        self.assertIn("after 2 attempts", events[-1].error)


if __name__ == "__main__":
    unittest.main()
