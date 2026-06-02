"""The `ContextVar` object the REPL exposes (Phase B.2).

`ContextVar` is the surface the model dereferences inside the REPL.
It wraps a `ChunkStore` and a `doc_id`, and supports four
operations:

* `slice(start, end)` — return bytes for a byte range across the
  whole document. Cheap because the store only loads the chunks
  that overlap the range.
* `search(query, k)` — return the top-k chunk hits. The default
  implementation uses a deterministic lexical score so the harness
  works without an embedder; a sqlite-vec-backed search is added
  when an embedder is supplied.
* `map()` — return the manifest as a small dict. This is what
  fits in the prompt.
* `get(content_hash)` — return bytes for a single chunk.

The object is intentionally narrow. It does not own the embedder
or the trace; those are wired in by the supervisor (Phase B.4).
"""
from __future__ import annotations

import re
from typing import Optional

from rlm_harness.context.store import Chunk, ChunkStore


def _tokenise(text: str) -> set[str]:
    return {tok for tok in re.findall(r"\w+", text.lower()) if tok}


def _lexical_score(query_tokens: set[str], content: bytes) -> float:
    if not query_tokens:
        return 0.0
    text = content.decode("utf-8", errors="replace").lower()
    content_tokens = set(re.findall(r"\w+", text))
    if not content_tokens:
        return 0.0
    overlap = query_tokens.intersection(content_tokens)
    return len(overlap) / len(query_tokens)


class ContextVar:
    """A symbolic reference to a chunked document.

    The model receives a `ContextVar` in the REPL bootstrap; the
    prompt only carries the manifest, never the bytes. The model
    dereferences slices, hashes, and search hits at runtime.
    """

    def __init__(self, store: ChunkStore, doc_id: str):
        self.store = store
        self.doc_id = doc_id

    # --- public API ---------------------------------------------------

    def slice(self, start: int, end: int) -> bytes:
        """Return the byte range `[start, end)`. Out-of-bounds
        indices are clamped to the document size.
        """
        if start < 0:
            start = 0
        if end <= start:
            return b""
        chunks = list(self.store.iter_chunks(self.doc_id))
        if not chunks:
            return b""
        total_size = max(chunk.end for chunk in chunks)
        if start >= total_size:
            return b""
        end = min(end, total_size)
        out = bytearray()
        for chunk in chunks:
            if chunk.end <= start:
                continue
            if chunk.start >= end:
                break
            local_start = max(0, start - chunk.start)
            local_end = min(chunk.size, end - chunk.start)
            out.extend(chunk.content[local_start:local_end])
        return bytes(out)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Top-k chunks for `query` ranked by relevance.

        The default implementation uses a deterministic lexical
        score. A semantic embedder can be supplied later; the
        contract is "ordered by relevance, descending".
        """
        if k <= 0:
            return []
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []
        chunks = list(self.store.iter_chunks(self.doc_id))
        scored = [
            (_lexical_score(query_tokens, chunk.content), chunk) for chunk in chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:k] if _ > 0]

    def map(self) -> dict:
        """Return the manifest as a small dict.

        The manifest is what the prompt carries. It includes the
        doc_id, chunk count, total bytes, and the ordered list of
        content hashes — never the bytes themselves.
        """
        return self.store.manifest(self.doc_id)

    def get(self, content_hash: str) -> Optional[bytes]:
        """Return bytes for a single chunk by content hash."""
        return self.store.get_chunk(content_hash)

    def __repr__(self) -> str:
        manifest = self.map()
        return (
            f"ContextVar(doc_id={self.doc_id!r}, "
            f"chunks={manifest['chunk_count']}, "
            f"bytes={manifest['total_bytes']})"
        )


__all__ = ["ContextVar"]
