"""Audit log helper — uses the project's own Memory class.

Records phase progress and decisions to the project's SQLite memory
store. This is *not* a separate persistence layer; it uses the same
schema, the same `Memory` class, and the same `:memory:` fallback as
the in-run memory. It is just a structured write API for the
"what phase are we on / what decision was made" kind of record.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rlm_harness.memory.store import Memory


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PhaseAudit:
    """Write structured phase progress records to Memory.

    Records are written into two channels:

    * `core` (key-value) — one row per (phase, event). Always readable
      by key. Used for: "what phase are we on", "what gate are we
      aiming at", "what's the current commit hash".
    * `archival_meta` (searchable) — one row per decision, summary, or
      test result. Vector-searchable across runs. Used for: decisions
      log, gate test outputs, run summaries.

    The audit log lives in the same DB as the run's working memory
    because (a) we do not introduce a second persistence layer, and
    (b) a per-project memory DB is the right scope: it travels with
    the workspace, not with the global harness install.
    """

    CORE_PHASE = "audit.current_phase"
    CORE_GATE = "audit.current_gate"
    CORE_LAST_UPDATED = "audit.last_updated"

    def __init__(self, memory: Memory):
        self.memory = memory

    @classmethod
    def open(cls, path: Path) -> PhaseAudit:
        """Open (or create) a memory DB and return an audit helper."""
        return cls(Memory(path))

    def set_phase(self, phase: str, gate: str = "") -> None:
        self.memory.core_set(self.CORE_PHASE, phase)
        if gate:
            self.memory.core_set(self.CORE_GATE, gate)
        self.memory.core_set(self.CORE_LAST_UPDATED, _now_iso())

    def record(
        self,
        kind: str,
        content: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        source_thread: str = "audit",
    ) -> int:
        """Record a structured event to the audit log."""
        payload = dict(metadata or {})
        payload.setdefault("recorded_at", _now_iso())
        try:
            archive = self.memory.archival_add(
                kind=kind,
                content=content,
                source_thread=source_thread,
                metadata=payload,
            )
            return int(archive.id)
        except Exception:
            return -1

    def current_phase(self) -> str:
        return self.memory.core_get(self.CORE_PHASE) or "unknown"

    def current_gate(self) -> str:
        return self.memory.core_get(self.CORE_GATE) or ""

    def summary(self) -> dict[str, Any]:
        phase = self.current_phase()
        gate = self.current_gate()
        last = self.memory.core_get(self.CORE_LAST_UPDATED) or ""
        return {"phase": phase, "gate": gate, "last_updated": last}


def write_decision(phase: str, decision: str, rationale: str) -> int:
    """Convenience for code paths that do not have a Memory handle.

    Writes to `.rlm_harness/audit.db` under the project root. Returns
    the archival id, or -1 if the write was skipped.
    """
    audit_path = Path.cwd() / ".rlm_harness" / "audit.db"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = PhaseAudit.open(audit_path)
    audit.set_phase(phase)
    payload = json.dumps(
        {"decision": decision, "rationale": rationale}, sort_keys=True
    )
    return audit.record(
        kind="decision",
        content=payload,
        metadata={"phase": phase},
    )
