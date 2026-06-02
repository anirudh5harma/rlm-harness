"""Tests for the `ContextVar` object (Phase B.2).

`ContextVar` is the surface the model dereferences inside the REPL.
It supports:

* `slice(start, end)` — return bytes for a byte range across the
  whole document. The model can read a slice without paying the
  cost of loading the entire context.
* `search(query, k)` — return the top-k chunk hits ranked by
  semantic similarity. Uses sqlite-vec when available, falls back
  to a deterministic lexical score for the stub path.
* `map()` — return the manifest as a small dict. The manifest is
  what the prompt carries.
* `get(content_hash)` — return bytes for a single chunk.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.context.store import ChunkStore
from rlm_harness.context.variable import ContextVar


class ContextVarTests(unittest.TestCase):
    def test_slice_returns_bytes_for_byte_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            ctx = ContextVar(store, "doc-1")
            self.assertEqual(ctx.slice(0, 5), b"abcde")
            self.assertEqual(ctx.slice(3, 7), b"defg")
            self.assertEqual(ctx.slice(10, 16), b"klmnop")
            self.assertEqual(ctx.slice(0, 100), b"abcdefghijklmnop")

    def test_slice_clamps_to_document_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"short")
            ctx = ContextVar(store, "doc-1")
            self.assertEqual(ctx.slice(0, 1000), b"short")

    def test_search_returns_relevant_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=64)
            # Ingest a single document with multiple chunks.
            content = (
                b"the quick brown fox\n"
                b"jumps over the lazy dog\n"
                b"completely unrelated content"
            )
            store.ingest("doc-1", content)
            ctx = ContextVar(store, "doc-1")
            hits = ctx.search("quick fox", k=5)
            # Only the chunk that contains query terms is returned;
            # unrelated chunks score 0 and are filtered out.
            self.assertEqual(len(hits), 1)
            # The "quick brown fox" chunk should be the top hit.
            self.assertIn(b"quick brown fox", hits[0].content)

    def test_search_uses_lexical_fallback_when_no_embedder(self):
        """Without a sqlite-vec embedder, the search falls back to
        a deterministic lexical score. The test asserts the chunks
        are still ranked correctly: a chunk that contains the
        query terms beats one that does not.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Pad the gap between the two segments so they fall
            # into distinct chunks.
            store = ChunkStore(Path(temp_dir), chunk_chars=20)
            padding = b"x" * 40
            content = b"alpha beta gamma" + padding + b"delta epsilon zeta"
            store.ingest("doc-1", content)
            ctx = ContextVar(store, "doc-1")
            hits = ctx.search("alpha beta", k=5)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn(b"alpha beta gamma", hits[0].content)
            # The padding chunk has no query terms and is filtered
            # out, so the second segment is not in the top hits.
            self.assertNotIn(b"delta epsilon zeta", hits[0].content)

    def test_map_returns_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            ctx = ContextVar(store, "doc-1")
            manifest = ctx.map()
            self.assertEqual(manifest["doc_id"], "doc-1")
            self.assertEqual(manifest["chunk_count"], 2)
            self.assertEqual(len(manifest["content_hashes"]), 2)

    def test_get_returns_chunk_bytes_by_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghij")
            ctx = ContextVar(store, "doc-1")
            manifest = ctx.map()
            first_hash = manifest["content_hashes"][0]
            self.assertEqual(ctx.get(first_hash), b"abcdefghij")

    def test_manifest_stays_bounded_for_large_document(self):
        """A 200k-token document should yield a manifest of at most
        ~5k tokens (the gate's 20k-token budget is generous; the
        manifest is the only thing the prompt carries).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=8_000)
            # 200k tokens ~ 800k bytes (rough avg 4 bytes/token)
            big = b"x" * 800_000
            store.ingest("doc-big", big)
            ctx = ContextVar(store, "doc-big")
            manifest = ctx.map()
            # 100 chunks × 64 hex chars + small fields ~ 7k chars
            self.assertEqual(manifest["chunk_count"], 100)
            # The manifest, when serialised, is well under 20k chars.
            import json

            serialised = json.dumps(manifest)
            self.assertLess(len(serialised), 20_000)


if __name__ == "__main__":
    unittest.main()
