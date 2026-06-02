"""Tests for the streaming LMClient API (Phase A.1).

The streaming API is a thin wrapper over the existing OpenAI-compatible
HTTP path plus a stub fallback. It must:

* yield a `start` event first;
* yield one or more `delta` events whose concatenation equals the
  non-streaming content;
* yield exactly one `finish` event with usage;
* raise `LMClientError` for HTTP errors (the caller catches).
"""
import json
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.types import Msg


class StreamingFakeResponse:
    """Fake urllib response that returns a streaming SSE body."""

    status = 200

    def __init__(self, body: str = ""):
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self, amt: int = -1):
        if amt == -1:
            data, self._body = self._body, b""
            return data
        data = self._body[:amt]
        self._body = self._body[amt:]
        return data


class StreamingClientTests(unittest.TestCase):
    def test_stub_stream_yields_start_delta_finish(self):
        client = LMClient(provider="stub", model="stub")
        events = list(
            client.stream(
                [Msg(role="user", content="hi")],
                max_tokens=20,
                temperature=0,
            )
        )
        kinds = [event.type for event in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "finish")
        self.assertIn("delta", kinds[1:-1])
        content = "".join(event.delta for event in events if event.type == "delta")
        self.assertTrue(content)

    def test_stub_stream_finish_event_carries_model_and_provider(self):
        client = LMClient(provider="stub", model="my-model")
        finish = next(
            event
            for event in client.stream([Msg(role="user", content="hi")])
            if event.type == "finish"
        )
        self.assertEqual(finish.model, "my-model")
        self.assertEqual(finish.provider, "stub")
        self.assertIsNotNone(finish.usage)

    def test_openai_compatible_stream_parses_sse_chunks(self):
        sse_body = (
            'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        )

        def fake_urlopen(request, timeout):
            return StreamingFakeResponse(sse_body)

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="sse-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key="token",
            )
            events = list(
                client.stream(
                    [Msg(role="user", content="hello")],
                    max_tokens=20,
                    temperature=0,
                )
            )
        deltas = [event.delta for event in events if event.type == "delta"]
        self.assertEqual("".join(deltas), "hello")
        finish = next(event for event in events if event.type == "finish")
        self.assertEqual(finish.model, "sse-model")
        self.assertEqual(finish.usage, {"prompt_tokens": 4, "completion_tokens": 2})

    def test_openai_compatible_stream_sets_stream_flag_in_payload(self):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            return StreamingFakeResponse(
                'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="x",
                base_url="http://127.0.0.1:8080/v1",
            )
            list(client.stream([Msg(role="user", content="hi")]))
        self.assertTrue(seen["payload"].get("stream"))

    def test_openai_compatible_stream_surfaces_http_error(self):
        """HTTP errors become an `error` event in the stream.

        The streaming contract is: `stream()` never raises; it yields
        an `error` event for the caller to translate. `complete()`
        does the translation. The test exercises both paths.
        """
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "Server Error",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"upstream down"}}'),
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            client = LMClient(
                provider="openai-compatible",
                model="x",
                base_url="http://127.0.0.1:8080/v1",
            )
            events = list(client.stream([Msg(role="user", content="hi")]))
        errors = [event for event in events if event.type == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("upstream down", errors[0].error)

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(LMClientError):
                client.complete([Msg(role="user", content="hi")])

    def test_complete_and_stream_share_stub_provider(self):
        """The two paths are independent; both work for the stub."""
        client = LMClient(provider="stub", model="wrap-model")
        completion = client.complete([Msg(role="user", content="hi")])
        self.assertEqual(completion.model, "wrap-model")
        self.assertTrue(completion.content)

        events = list(client.stream([Msg(role="user", content="hi")]))
        self.assertEqual(events[0].type, "start")
        self.assertEqual(events[-1].type, "finish")
        content = "".join(event.delta for event in events if event.type == "delta")
        self.assertEqual(content, completion.content)


if __name__ == "__main__":
    unittest.main()
