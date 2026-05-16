import json
import unittest
from unittest.mock import patch

from rlm_harness.model_client import LMClient
from rlm_harness.types import Msg


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "test-model",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ).encode("utf-8")


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
        self.assertEqual(payload["messages"][0]["content"], "Reply with exactly: hello")
        self.assertEqual(request.headers["Authorization"], "Bearer token")


if __name__ == "__main__":
    unittest.main()
