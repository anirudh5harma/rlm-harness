"""Tests for the manifest and token budget (Phase B.3).

The manifest is what the model sees in its prompt. It is a small
JSON-friendly dict describing the working context without carrying
its bytes. The token budget bounds the manifest's contribution to
the prompt; if the manifest would exceed the budget, the store is
queried and only a slice of the chunk-hash list is included.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness.context.budget import TokenBudget
from rlm_harness.context.manifest import (
    build_manifest,
    build_manifest_for_doc,
    estimate_tokens,
)
from rlm_harness.context.store import ChunkStore


class EstimateTokensTests(unittest.TestCase):
    def test_estimate_tokens_uses_word_tokeniser(self):
        # The tokeniser is word-based to match the rest of the
        # harness (`memory.embed.tokenize`). Words are
        # `[A-Za-z0-9_]+` runs.
        self.assertEqual(estimate_tokens(""), 1)
        self.assertEqual(estimate_tokens("hello"), 1)
        self.assertEqual(estimate_tokens("hello world"), 2)
        self.assertEqual(estimate_tokens("hello, world!"), 2)
        self.assertEqual(estimate_tokens("one two three four five"), 5)


class TokenBudgetTests(unittest.TestCase):
    def test_budget_enforces_max_tokens(self):
        budget = TokenBudget(max_tokens=10)
        self.assertTrue(budget.fits(8))
        self.assertFalse(budget.fits(15))

    def test_budget_truncates_oversize_field(self):
        budget = TokenBudget(max_tokens=5)
        truncated = budget.truncate("abcdefghijklmnop" * 10, field="content")
        self.assertLess(len(truncated), 200)
        # The truncation marker is part of the field.
        self.assertIn("truncated", truncated)


class BuildManifestTests(unittest.TestCase):
    def test_manifest_for_doc_includes_doc_id_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            manifest = build_manifest_for_doc(store, "doc-1")
            self.assertEqual(manifest["doc_id"], "doc-1")
            self.assertEqual(manifest["chunk_count"], 2)
            self.assertEqual(len(manifest["content_hashes"]), 2)

    def test_manifest_serialised_stays_within_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=8_000)
            big = b"x" * 800_000
            store.ingest("doc-big", big)
            manifest = build_manifest_for_doc(store, "doc-big")
            serialised = json.dumps(manifest)
            tokens = estimate_tokens(serialised)
            # The pivot plan's per-turn budget is 20k tokens; the
            # manifest is well under that.
            self.assertLess(tokens, 20_000)
            # And the manifest is the *whole* hash list (no
            # truncation needed for a 200k-token document).
            self.assertEqual(len(manifest["content_hashes"]), 100)

    def test_manifest_truncates_hashes_when_oversize(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=1)
            # 200 chunks of 1 byte each → 200 hashes × 64 chars = 12.8k
            # characters; well over a tight budget.
            big = bytes(range(200))
            store.ingest("doc-tight", big)
            manifest = build_manifest_for_doc(
                store, "doc-tight", token_budget=200
            )
            serialised = json.dumps(manifest)
            tokens = estimate_tokens(serialised)
            # Truncation should bring the manifest under budget.
            self.assertLessEqual(tokens, 250)
            # Truncation must be flagged so the supervisor knows
            # to enrich the manifest on demand.
            self.assertTrue(manifest.get("truncated", False))

    def test_build_manifest_handles_empty_doc(self):
        manifest = build_manifest_for_doc(
            ChunkStore(Path("/tmp")), "missing-doc"
        )
        self.assertEqual(manifest["doc_id"], "missing-doc")
        self.assertEqual(manifest["chunk_count"], 0)

    def test_build_manifest_dispatches_to_doc_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            manifest = build_manifest({"doc-1": store}, "doc-1")
            self.assertEqual(manifest["doc_id"], "doc-1")
            self.assertEqual(manifest["chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()
