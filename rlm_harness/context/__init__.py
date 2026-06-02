"""Context package (Phase B).

The context layer is the long-context surface of the harness. A
working context (a 200k+ token directory, a long log file, a
multi-file project) is split into chunks, hashed, and persisted
to disk. The model sees only a *manifest* in its prompt; it
dereferences chunks inside the REPL via `ContextVar`.

This package owns:

* `context.store` — content-addressed chunk storage (Phase B.1).
* `context.variable` — the `ContextVar` object the REPL exposes
  (Phase B.2). Supports `slice`, `search`, `map`, `get`.
* `context.manifest` — manifest generation (Phase B.3).
* `context.budget` — token budget per context section (Phase B.3).

The package is intentionally additive: nothing in the rest of the
harness imports from here until Phase B.4 wires the supervisor to
pass a `ContextVar` into the REPL.
"""

from rlm_harness.context.budget import TokenBudget, estimate_tokens
from rlm_harness.context.manifest import (
    DEFAULT_TOKEN_BUDGET,
    build_manifest,
    build_manifest_for_doc,
)
from rlm_harness.context.store import (
    DEFAULT_CHUNK_CHARS,
    Chunk,
    ChunkStore,
)
from rlm_harness.context.variable import ContextVar

__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_TOKEN_BUDGET",
    "Chunk",
    "ChunkStore",
    "ContextVar",
    "TokenBudget",
    "build_manifest",
    "build_manifest_for_doc",
    "estimate_tokens",
]
