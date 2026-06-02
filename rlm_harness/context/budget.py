"""Token budget for context sections (Phase B.3).

A token budget is a per-section cap on what the prompt carries.
For long-context work, the manifest is the only thing the prompt
carries; the bytes live in the chunk store. The budget enforces
the per-turn ceiling (`<20k` tokens for the working state) and
truncates gracefully when the manifest would otherwise exceed it.

The model never sees a manifest that violates its budget. If the
hash list would be too long, the manifest is truncated and flagged
with `truncated: True` so the supervisor can resolve the rest of
the chunks on demand (via `ContextVar.search`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Word-based tokenisation matches the rest of the harness
# (`memory.embed.tokenize`). Using the same definition everywhere
# means a budget of 20_000 tokens means the same thing in
# `memory.paging`, the trace, and the manifest.
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    return max(1, len(_WORD_RE.findall(text)))


@dataclass(frozen=True)
class TokenBudget:
    """A per-section cap. `fits(n)` answers yes/no; `truncate(s)`
    shrinks a string until it fits, with a marker.
    """

    max_tokens: int

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def fits(self, tokens: int) -> bool:
        return int(tokens) <= self.max_tokens

    def truncate(self, text: str, *, field: str = "value") -> str:
        """Shrink `text` until it fits under `max_tokens`. The
        returned string includes a marker explaining the
        truncation.
        """
        budget_chars = self.max_tokens * 4
        if len(text) <= budget_chars:
            return text
        head = budget_chars // 2
        tail = budget_chars - head - 80
        if tail < 0:
            tail = 0
        omitted = max(0, len(text) - head - tail)
        marker = (
            f"\n\n... [truncated {omitted} chars from {field}; "
            "the supervisor can resolve the rest on demand] ...\n\n"
        )
        return text[:head] + marker + (text[-tail:] if tail else "")


__all__ = ["TokenBudget", "estimate_tokens"]
