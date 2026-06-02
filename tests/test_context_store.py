"""Tests for the content-addressed context store (Phase B.1).

The store is the foundation of the long-context layer. It splits a
working context into fixed-size chunks, hashes each chunk, dedups by
content hash, and persists chunks to disk with metadata in SQLite.
A later phase adds a sqlite-vec index for search.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

from rlm_harness.context.store import ChunkStore


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ChunkStoreTests(unittest.TestCase):
    def test_ingest_splits_large_content_into_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=100)
            content = ("hello world. " * 50).encode("utf-8")  # 650 bytes
            chunk_ids = store.ingest("doc-1", content)
            self.assertGreater(len(chunk_ids), 5)
            # Each chunk is at most chunk_chars.
            for chunk in store.iter_chunks("doc-1"):
                self.assertLessEqual(len(chunk.content), 100)

    def test_chunk_ids_are_content_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=64)
            content = b"abcdefghij" * 20  # 200 bytes
            chunk_ids = store.ingest("doc-1", content)
            for chunk_id, expected in zip(
                chunk_ids, _split(content, 64), strict=False
            ):
                self.assertEqual(chunk_id, _hash(expected))

    def test_duplicate_chunks_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=64)
            content = b"x" * 200
            store.ingest("doc-1", content)
            # Re-ingest the same content; chunk count on disk
            # should not grow because the hashes match.
            disk_path = store.chunk_path(_hash(b"x" * 64))
            self.assertTrue(disk_path.exists())
            first_size = disk_path.stat().st_size
            store.ingest("doc-1", content)
            self.assertEqual(disk_path.stat().st_size, first_size)
            # Metadata still records both ranges, but the bytes
            # live in one place on disk.
            chunks = list(store.iter_chunks("doc-1"))
            self.assertEqual(len(chunks), 4)  # 200 bytes / 64 = 4

    def test_get_chunk_returns_bytes_by_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=64)
            store.ingest("doc-1", b"hello world")
            chunk = store.get_chunk(_hash(b"hello world"))
            self.assertEqual(chunk, b"hello world")

    def test_get_chunk_missing_hash_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=64)
            self.assertIsNone(store.get_chunk("0" * 64))

    def test_iter_chunks_yields_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            ordered = list(store.iter_chunks("doc-1"))
            self.assertEqual([c.content for c in ordered], list(_split(b"abcdefghijklmnop", 10)))

    def test_manifest_lists_chunks_for_a_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghijklmnop")
            manifest = store.manifest("doc-1")
            self.assertEqual(manifest["doc_id"], "doc-1")
            self.assertEqual(manifest["chunk_count"], 2)
            self.assertEqual(manifest["total_bytes"], 16)
            self.assertEqual(manifest["chunk_chars"], 10)

    def test_chunk_store_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(Path(temp_dir), chunk_chars=10)
            store.ingest("doc-1", b"abcdefghij")
            store.close()
            store2 = ChunkStore(Path(temp_dir), chunk_chars=10)
            chunks = list(store2.iter_chunks("doc-1"))
            self.assertEqual([c.content for c in chunks], [b"abcdefghij"])


def _split(content: bytes, size: int) -> list[bytes]:
    return [content[i : i + size] for i in range(0, len(content), size)]


if __name__ == "__main__":
    unittest.main()
