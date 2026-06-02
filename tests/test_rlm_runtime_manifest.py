"""Tests for the manifest-aware context path in `RLMRuntime`
(Phase B.4).

The runtime's `_initial_messages` builds a small manifest from
whatever the supervisor passes. For a legacy dict context, the
manifest is the dict itself. For a `ContextVar` (long-context
path), the manifest is `ctx.map()`. The prompt carries the
manifest, not the bytes.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness.context.store import ChunkStore
from rlm_harness.context.variable import ContextVar
from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import (
    RLMRuntime,
    manifest_for_context,
)


class ScriptedStreamClient(LMClient):
    def __init__(self, scripted):
        super().__init__(provider="stub", model="scripted")
        self._queue = list(scripted)

    def stream(self, messages, max_tokens=512, temperature=0.2):
        from rlm_harness.types import TokenEvent

        deltas = self._queue.pop(0) if self._queue else ["ok"]
        yield TokenEvent(type="start", model=self.model, provider=self.provider)
        for d in deltas:
            yield TokenEvent(
                type="delta", delta=d, model=self.model, provider=self.provider
            )
        yield TokenEvent(
            type="finish",
            model=self.model,
            provider=self.provider,
            usage={"prompt_tokens": 1, "completion_tokens": len(deltas)},
            finish_reason="stop",
        )

    def complete(self, messages, max_tokens=512, temperature=0.2):
        chunks: list[str] = []
        for event in self.stream(messages, max_tokens=max_tokens, temperature=temperature):
            if event.type == "delta":
                chunks.append(event.delta)
            if event.type == "finish":
                from rlm_harness.types import Completion

                return Completion(
                    content="".join(chunks),
                    model=event.model or self.model,
                    provider=event.provider or self.provider,
                    latency_ms=0,
                )
        from rlm_harness.types import Completion

        return Completion(
            content="".join(chunks), model=self.model, provider=self.provider, latency_ms=0
        )


class ManifestForContextTests(unittest.TestCase):
    def test_manifest_for_dict_returns_dict(self):
        manifest = manifest_for_context({"task": "x", "files": ["a", "b"]})
        self.assertEqual(manifest["task"], "x")
        self.assertEqual(manifest["files"], ["a", "b"])

    def test_manifest_for_context_var_uses_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghij")
            ctx = ContextVar(store, "doc-1")
            manifest = manifest_for_context(ctx)
            self.assertEqual(manifest["doc_id"], "doc-1")
            self.assertEqual(manifest["chunk_count"], 1)

    def test_manifest_for_string_serialises_to_raw(self):
        manifest = manifest_for_context("hello world")
        # Strings are returned by `serialize_context` as-is (no
        # JSON quoting); the manifest's `_raw` is the literal
        # string.
        self.assertEqual(manifest["_raw"], "hello world")

    def test_runtime_initial_messages_carry_manifest(self):
        """The first user message includes the manifest, not the
        raw bytes. The model can dereference the manifest via
        `ctx.map()` inside the REPL.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghij" * 5)  # 50 bytes / 10 = 5 chunks
            ctx = ContextVar(store, "doc-1")
            client = ScriptedStreamClient(
                [
                    "```repl\nprint(ctx.map())\n```",
                ]
            )
            runtime = RLMRuntime(
                client,
                workspace=Path(temp_dir),
                max_iterations=1,
                sandbox_enabled=False,
            )
            # The runtime's `_initial_messages` builds the
            # manifest; the first user message carries it.
            messages = runtime._initial_messages("summarize", ctx)
            user_message = next(
                msg for msg in messages if msg.role == "user"
            )
            self.assertIn("Context manifest:", user_message.content)
            # The manifest serialised is bounded by the budget.
            # 5 chunks × 64 hex chars + small fields = ~400 chars.
            self.assertLess(len(user_message.content), 1_000)
            # The manifest is JSON; parse it and confirm shape.
            manifest_text = user_message.content.split("Context manifest:\n")[-1]
            parsed = json.loads(manifest_text)
            self.assertEqual(parsed["doc_id"], "doc-1")
            self.assertEqual(parsed["chunk_count"], 5)
            self.assertIn("content_hashes", parsed)


if __name__ == "__main__":
    unittest.main()
