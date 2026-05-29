from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from rlm_harness.memory.profile import TasteRecord, normalize_text, validate_nonempty
from rlm_harness.memory.store import Memory, MemoryValidationError
from rlm_harness.types import HarnessState

EVOLUTION_PROPOSALS_KEY = "evolution.proposals.v1"
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
VALID_PROPOSAL_STATUSES = {PENDING, APPROVED, REJECTED}
VALID_PROPOSAL_SCOPES = {"user", "project"}
VALID_PROPOSAL_KINDS = {"prompt_rule", "verification_policy", "eval_case", "tooling"}


@dataclass(frozen=True)
class EvolutionProposal:
    id: str
    scope: str
    kind: str
    title: str
    body: str
    rationale: str
    status: str = PENDING
    source_run: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0

    @classmethod
    def create(
        cls,
        scope: str,
        kind: str,
        title: str,
        body: str,
        rationale: str,
        status: str = PENDING,
        source_run: Optional[str] = None,
        evidence: Optional[dict[str, Any]] = None,
        now: Optional[int] = None,
    ) -> EvolutionProposal:
        timestamp = int(now or time.time())
        return cls(
            id=f"evolve-{uuid.uuid4().hex[:12]}",
            scope=validate_proposal_scope(scope),
            kind=validate_proposal_kind(kind),
            title=validate_nonempty("title", title),
            body=validate_nonempty("body", body),
            rationale=validate_nonempty("rationale", rationale),
            status=validate_proposal_status(status),
            source_run=source_run,
            evidence=evidence or {},
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvolutionProposal:
        return cls(
            id=str(payload["id"]),
            scope=validate_proposal_scope(str(payload["scope"])),
            kind=validate_proposal_kind(str(payload["kind"])),
            title=validate_nonempty("title", str(payload["title"])),
            body=validate_nonempty("body", str(payload["body"])),
            rationale=validate_nonempty("rationale", str(payload["rationale"])),
            status=validate_proposal_status(str(payload.get("status", PENDING))),
            source_run=payload.get("source_run"),
            evidence=dict(payload.get("evidence") or {}),
            created_at=int(payload.get("created_at") or 0),
            updated_at=int(payload.get("updated_at") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "rationale": self.rationale,
            "status": self.status,
            "source_run": self.source_run,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.scope, self.kind, normalize_text(self.body))


class EvolutionProposalStore:
    """Reviewable self-evolution proposals backed by the Memory core store."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def proposals(
        self,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[EvolutionProposal]:
        if scope is not None:
            scope = validate_proposal_scope(scope)
        if status is not None:
            status = validate_proposal_status(status)
        if kind is not None:
            kind = validate_proposal_kind(kind)

        proposals = self._load_proposals()
        filtered = []
        for proposal in proposals:
            if scope is not None and proposal.scope != scope:
                continue
            if status is not None and proposal.status != status:
                continue
            if kind is not None and proposal.kind != kind:
                continue
            filtered.append(proposal)
        return sorted(filtered, key=lambda p: (p.status != PENDING, -p.updated_at))

    def add(self, proposal: EvolutionProposal) -> EvolutionProposal:
        proposals = self._load_proposals()
        existing_index = self._find_existing(proposals, proposal)
        now = int(time.time())
        if existing_index is not None:
            existing = proposals[existing_index]
            merged = EvolutionProposal(
                id=existing.id,
                scope=existing.scope,
                kind=existing.kind,
                title=proposal.title or existing.title,
                body=existing.body,
                rationale=proposal.rationale or existing.rationale,
                status=existing.status,
                source_run=proposal.source_run or existing.source_run,
                evidence={**existing.evidence, **proposal.evidence},
                created_at=existing.created_at,
                updated_at=now,
            )
            proposals[existing_index] = merged
            self._save_proposals(proposals)
            return merged

        proposals.append(proposal)
        self._save_proposals(proposals)
        return proposal

    def approve(self, proposal_id: str) -> Optional[EvolutionProposal]:
        return self._set_status(proposal_id, APPROVED)

    def reject(self, proposal_id: str) -> Optional[EvolutionProposal]:
        return self._set_status(proposal_id, REJECTED)

    def render_context(self, max_proposals: int = 8) -> str:
        approved = self.proposals(status=APPROVED)[:max_proposals]
        if not approved:
            return ""
        lines = [f"- {proposal.body} ({proposal.kind})" for proposal in approved]
        return "Approved evolution rules:\n" + "\n".join(lines)

    def _load_proposals(self) -> list[EvolutionProposal]:
        raw = self.memory.core_get(EVOLUTION_PROPOSALS_KEY)
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MemoryValidationError("evolution proposals are not valid JSON") from exc
        if not isinstance(payload, list):
            raise MemoryValidationError("evolution proposals must be a list")
        return [EvolutionProposal.from_dict(item) for item in payload if isinstance(item, dict)]

    def _save_proposals(self, proposals: list[EvolutionProposal]) -> None:
        payload = [proposal.to_dict() for proposal in sorted(proposals, key=lambda p: p.created_at)]
        self.memory.core_set(EVOLUTION_PROPOSALS_KEY, json.dumps(payload, sort_keys=True))

    def _set_status(self, proposal_id: str, status: str) -> Optional[EvolutionProposal]:
        proposals = self._load_proposals()
        for index, proposal in enumerate(proposals):
            if proposal.id == proposal_id:
                updated = EvolutionProposal(
                    id=proposal.id,
                    scope=proposal.scope,
                    kind=proposal.kind,
                    title=proposal.title,
                    body=proposal.body,
                    rationale=proposal.rationale,
                    status=validate_proposal_status(status),
                    source_run=proposal.source_run,
                    evidence=proposal.evidence,
                    created_at=proposal.created_at,
                    updated_at=int(time.time()),
                )
                proposals[index] = updated
                self._save_proposals(proposals)
                return updated
        return None

    @staticmethod
    def _find_existing(
        proposals: list[EvolutionProposal],
        candidate: EvolutionProposal,
    ) -> Optional[int]:
        for index, proposal in enumerate(proposals):
            if proposal.dedupe_key == candidate.dedupe_key:
                return index
        return None


class EvolutionProposalManager:
    """Routes user-wide and project-local self-evolution proposals."""

    def __init__(
        self,
        user_memory: Optional[Memory],
        project_memory: Optional[Memory] = None,
    ):
        self.user_store = EvolutionProposalStore(user_memory) if user_memory is not None else None
        self.project_store = (
            EvolutionProposalStore(project_memory) if project_memory is not None else None
        )

    def proposals(
        self,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[EvolutionProposal]:
        stores = [
            store
            for store in (self.user_store, self.project_store)
            if store is not None
        ]

        proposals: list[EvolutionProposal] = []
        for store in stores:
            proposals.extend(store.proposals(scope=scope, status=status, kind=kind))
        return sorted(proposals, key=lambda p: (p.status != PENDING, -p.updated_at))

    def add(self, proposal: EvolutionProposal) -> Optional[EvolutionProposal]:
        store = self._store_for_scope(proposal.scope)
        if store is None:
            return None
        return store.add(proposal)

    def approve(self, proposal_id: str) -> Optional[EvolutionProposal]:
        for store in (self.user_store, self.project_store):
            if store is None:
                continue
            proposal = store.approve(proposal_id)
            if proposal is not None:
                return proposal
        return None

    def reject(self, proposal_id: str) -> Optional[EvolutionProposal]:
        for store in (self.user_store, self.project_store):
            if store is None:
                continue
            proposal = store.reject(proposal_id)
            if proposal is not None:
                return proposal
        return None

    def propose_from_state(
        self,
        state: HarnessState,
        learned: list[TasteRecord],
    ) -> list[EvolutionProposal]:
        proposals: list[EvolutionProposal] = []
        for proposal in infer_evolution_proposals(state, learned):
            stored = self.add(proposal)
            if stored is not None:
                proposals.append(stored)
        return proposals

    def render_context(self, max_proposals: int = 8) -> str:
        proposals = self.proposals(status=APPROVED)[:max_proposals]
        if not proposals:
            return ""
        sections = []
        for scope in ("user", "project"):
            scoped = [proposal for proposal in proposals if proposal.scope == scope]
            if not scoped:
                continue
            title = "Approved user evolution" if scope == "user" else "Approved project evolution"
            lines = [f"- {proposal.body} ({proposal.kind})" for proposal in scoped]
            sections.append(f"{title}:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    def _store_for_scope(self, scope: str) -> Optional[EvolutionProposalStore]:
        if scope == "project":
            return self.project_store or self.user_store
        return self.user_store or self.project_store


def infer_evolution_proposals(
    state: HarnessState,
    learned: list[TasteRecord],
) -> list[EvolutionProposal]:
    proposals: list[EvolutionProposal] = []
    for record in learned:
        proposal = proposal_from_taste_record(record)
        if proposal is not None:
            proposals.append(proposal)

    if state.status in {"error", "stopped"}:
        proposals.append(
            EvolutionProposal.create(
                scope="project",
                kind="eval_case",
                title="Add regression eval for incomplete run",
                body=(
                    "Add or update an eval that reproduces this task when a run ends "
                    f"with status `{state.status}`."
                ),
                rationale="Failed or stopped runs reveal real daily-driver capability gaps.",
                source_run=state.run_id,
                evidence={
                    "task": state.task[:500],
                    "status": state.status,
                    "last_error": str(state.scratch.get("last_error") or "")[:500],
                },
            )
        )
    return proposals


def proposal_from_taste_record(record: TasteRecord) -> Optional[EvolutionProposal]:
    if record.kind in {"preference", "constraint"}:
        return EvolutionProposal.create(
            scope="user",
            kind="prompt_rule",
            title="Promote learned user taste into response guidance",
            body=record.text,
            rationale="The user expressed this preference directly during a run.",
            source_run=record.source_run,
            evidence={"taste_record_id": record.id, **record.evidence},
        )
    if record.kind == "verification_command":
        return EvolutionProposal.create(
            scope="project",
            kind="verification_policy",
            title="Promote successful verification command",
            body=record.text,
            rationale="This command passed and should be considered during future verification.",
            source_run=record.source_run,
            evidence={"taste_record_id": record.id, **record.evidence},
        )
    return None


def validate_proposal_scope(scope: str) -> str:
    scope = scope.strip().lower()
    if scope not in VALID_PROPOSAL_SCOPES:
        raise MemoryValidationError(
            f"scope must be one of: {', '.join(sorted(VALID_PROPOSAL_SCOPES))}"
        )
    return scope


def validate_proposal_status(status: str) -> str:
    status = status.strip().lower()
    if status not in VALID_PROPOSAL_STATUSES:
        raise MemoryValidationError(
            f"status must be one of: {', '.join(sorted(VALID_PROPOSAL_STATUSES))}"
        )
    return status


def validate_proposal_kind(kind: str) -> str:
    kind = re.sub(r"\s+", "_", kind.strip().lower())
    if kind not in VALID_PROPOSAL_KINDS:
        raise MemoryValidationError(
            f"kind must be one of: {', '.join(sorted(VALID_PROPOSAL_KINDS))}"
        )
    return kind
