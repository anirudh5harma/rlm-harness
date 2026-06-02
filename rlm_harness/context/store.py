"""External context store for long-context working memory (Phase B).

The store is the foundation of the long-context layer. A *working
context* (a 200k+ token directory, a long log file, a multi-file
project) is split into fixed-size chunks, hashed, and persisted to
disk. A small *manifest* (doc_id, chunk count, total bytes, the
ordered list of content hashes) is what the model sees in its
prompt. The model dereferences chunks inside the REPL via the
`ContextVar` object (Phase B.2).

Design:

* **Content-addressed.** Each chunk is stored at
  `<store_root>/chunks/<sha256>.bin`. Two chunks with identical
  bytes share a single file on disk. Re-ingesting the same document
  does not grow the store.
* **SQLite metadata.** A `<store_root>/meta.sqlite` records per-doc
  chunk ranges, ordering, and the (doc_id, content_hash) mapping.
  A sqlite-vec virtual table for chunk embeddings is added in
  Phase B.2.
* **In-memory + disk.** The store caches chunk bytes in an LRU; the
  first read after a restart hydrates from disk. The cache is
  intentionally simple; the per-doc manifests are always in memory.
* **Stateless public API.** `ingest`, `iter_chunks`, `get_chunk`,
  `manifest`, `close`. The store does not own the embedding model;
  `ContextVar.search` is the caller that adds embeddings.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CHUNKS_DIRNAME = "chunks"
META_FILENAME = "meta.sqlite"
DEFAULT_CHUNK_CHARS = 4_000


@dataclass(frozen=True)
class Chunk:
    """A single chunk belonging to a document."""

    doc_id: str
    content_hash: str
    start: int
    end: int
    content: bytes

    @property
    def size(self) -> int:
        return len(self.content)


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ChunkStore:
    """A content-addressed store for chunked text.

    The store is safe to use from a single thread. It holds an
    internal lock for the SQLite write path; the read path is
    unprotected because SQLite handles concurrent readers.
    """

    def __init__(
        self,
        root: Path,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        meta_path: Optional[Path] = None,
    ):
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_chars = chunk_chars
        self.chunks_dir = self.root / CHUNKS_DIRNAME
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = meta_path or (self.root / META_FILENAME)
        self._lock = threading.RLock()
        self._cache: dict[str, bytes] = {}
        self._init_meta()

    # --- public API ----------------------------------------------------

    def chunk_path(self, content_hash: str) -> Path:
        return self.chunks_dir / f"{content_hash}.bin"

    def ingest(self, doc_id: str, content: bytes) -> list[str]:
        """Split `content` into chunks and persist. Returns the
        ordered list of content hashes.
        """
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        chunks = self._split(content, self.chunk_chars)
        with self._lock:
            self._delete_existing_ranges(doc_id)
            ordered_hashes: list[str] = []
            offset = 0
            for chunk_bytes in chunks:
                content_hash = _hash_bytes(chunk_bytes)
                self._persist_chunk_bytes(content_hash, chunk_bytes)
                self._record_range(doc_id, content_hash, offset, offset + len(chunk_bytes))
                ordered_hashes.append(content_hash)
                offset += len(chunk_bytes)
        return ordered_hashes

    def iter_chunks(self, doc_id: str) -> Iterator[Chunk]:
        """Yield chunks for `doc_id` in offset order."""
        rows = self._ranges_for(doc_id)
        for content_hash, start, end in rows:
            content = self._load_chunk_bytes(content_hash)
            if content is None:
                continue
            yield Chunk(
                doc_id=doc_id,
                content_hash=content_hash,
                start=start,
                end=end,
                content=content,
            )

    def get_chunk(self, content_hash: str) -> Optional[bytes]:
        """Return the bytes for `content_hash`, or None if absent."""
        if not content_hash:
            return None
        cached = self._cache.get(content_hash)
        if cached is not None:
            return cached
        path = self.chunk_path(content_hash)
        if not path.exists():
            return None
        data = path.read_bytes()
        self._cache[content_hash] = data
        return data

    def manifest(self, doc_id: str) -> dict:
        """Build a small manifest for `doc_id`."""
        rows = self._ranges_for(doc_id)
        total_bytes = sum(end - start for _, start, end in rows)
        return {
            "doc_id": doc_id,
            "chunk_count": len(rows),
            "chunk_chars": self.chunk_chars,
            "total_bytes": total_bytes,
            "content_hashes": [h for h, _, _ in rows],
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # --- internals -----------------------------------------------------

    def _init_meta(self) -> None:
        self._connection = sqlite3.connect(
            str(self.meta_path), check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunk_ranges (
                    doc_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, start, end)
                );
                CREATE INDEX IF NOT EXISTS chunk_ranges_doc
                    ON chunk_ranges(doc_id, start);
                """
            )
            self._connection.commit()

    def _split(self, content: bytes, size: int) -> list[bytes]:
        return [content[i : i + size] for i in range(0, len(content), size)]

    def _persist_chunk_bytes(self, content_hash: str, chunk_bytes: bytes) -> None:
        path = self.chunk_path(content_hash)
        if path.exists():
            self._cache[content_hash] = chunk_bytes
            return
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(chunk_bytes)
        tmp_path.rename(path)
        self._cache[content_hash] = chunk_bytes

    def _load_chunk_bytes(self, content_hash: str) -> Optional[bytes]:
        cached = self._cache.get(content_hash)
        if cached is not None:
            return cached
        path = self.chunk_path(content_hash)
        if not path.exists():
            return None
        data = path.read_bytes()
        self._cache[content_hash] = data
        return data

    def _record_range(
        self, doc_id: str, content_hash: str, start: int, end: int
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO chunk_ranges
                  (doc_id, content_hash, start, end)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, content_hash, start, end),
            )
            self._connection.commit()

    def _delete_existing_ranges(self, doc_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM chunk_ranges WHERE doc_id = ?", (doc_id,)
            )
            self._connection.commit()

    def _ranges_for(self, doc_id: str) -> list[tuple[str, int, int]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT content_hash, start, end
                FROM chunk_ranges
                WHERE doc_id = ?
                ORDER BY start
                """,
                (doc_id,),
            ).fetchall()
        return [(str(r["content_hash"]), int(r["start"]), int(r["end"])) for r in rows]


__all__ = [
    "CHUNKS_DIRNAME",
    "DEFAULT_CHUNK_CHARS",
    "META_FILENAME",
    "Chunk",
    "ChunkStore",
]
