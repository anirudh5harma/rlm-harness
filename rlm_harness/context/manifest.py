"""Manifest generation for the long-context layer (Phase B.3).

The manifest is what the model sees in its prompt. It is a small
JSON-friendly dict describing the working context without carrying
its bytes. The supervisor emits a manifest per turn; the prompt
template places the manifest under the `working_state` section.

The default budget is generous: 20k tokens, matching the
pivot-plan gate. When the manifest would exceed the budget, the
hash list is truncated and `truncated=True` is set. The
supervisor resolves the rest on demand via `ContextVar.search`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from rlm_harness.context.budget import TokenBudget, estimate_tokens
from rlm_harness.context.store import ChunkStore

DEFAULT_TOKEN_BUDGET = 20_000


def build_manifest_for_doc(
    store: ChunkStore,
    doc_id: str,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict:
    """Build a manifest for a single doc.

    The manifest always carries:
    * `doc_id`
    * `chunk_count`
    * `chunk_chars` (the chunk size used to ingest)
    * `total_bytes`
    * `content_hashes` (the ordered list of chunk hashes)
    * `truncated` (True if the hash list was clipped to fit budget)

    The hash list is the only part that grows with the document
    size. For a 200k-token document at 8k-byte chunks (~100
    chunks), the list is ~6.4k characters; well under the default
    20k-token budget.
    """
    raw = store.manifest(doc_id)
    budget = TokenBudget(max_tokens=token_budget)
    hashes = list(raw.get("content_hashes") or [])
    truncated = False
    base = {
        "doc_id": raw["doc_id"],
        "chunk_count": raw["chunk_count"],
        "chunk_chars": raw["chunk_chars"],
        "total_bytes": raw["total_bytes"],
        "content_hashes": hashes,
    }
    # Iteratively shrink the hash list until the manifest fits.
    while estimate_tokens(json.dumps(base)) > budget.max_tokens and len(hashes) > 1:
        # Drop the last quarter of the remaining hashes. This
        # keeps the head of the document visible (where the
        # project README / entry point usually live) and lets the
        # supervisor pull the rest via `ContextVar.search`.
        keep = max(1, (len(hashes) * 3) // 4)
        hashes = hashes[:keep]
        truncated = True
        base["content_hashes"] = hashes
    base["truncated"] = truncated
    return base


def build_manifest(
    stores: Mapping[str, ChunkStore],
    doc_id: str,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict:
    """Dispatch helper. Looks up `doc_id` in `stores` and delegates
    to `build_manifest_for_doc`. Returns a minimal manifest if
    `doc_id` is unknown.
    """
    store = stores.get(doc_id)
    if store is None:
        return {
            "doc_id": doc_id,
            "chunk_count": 0,
            "chunk_chars": 0,
            "total_bytes": 0,
            "content_hashes": [],
            "truncated": False,
        }
    return build_manifest_for_doc(store, doc_id, token_budget=token_budget)


__all__ = [
    "DEFAULT_TOKEN_BUDGET",
    "build_manifest",
    "build_manifest_for_doc",
]
