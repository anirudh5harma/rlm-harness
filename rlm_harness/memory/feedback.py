from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from rlm_harness.memory.evolution import EvolutionProposal
from rlm_harness.memory.profile import (
    ACTIVE,
    PENDING,
    TasteRecord,
    clean_fragment,
    normalize_text,
    validate_nonempty,
    validate_scope,
)
from rlm_harness.memory.store import Memory, MemoryValidationError

FEEDBACK_RECORDS_KEY = "feedback.records.v1"
VALID_RATINGS = {"positive", "negative", "neutral"}


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    scope: str
    rating: str
    comment: str
    run_id: Optional[str] = None
    thread_id: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0

    @classmethod
    def create(
        cls,
        scope: str,
        rating: str,
        comment: str,
        run_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        evidence: Optional[dict[str, Any]] = None,
        now: Optional[int] = None,
    ) -> FeedbackRecord:
        return cls(
            id=f"feedback-{uuid.uuid4().hex[:12]}",
            scope=validate_scope(scope),
            rating=validate_rating(rating),
            comment=validate_nonempty("comment", comment),
            run_id=run_id,
            thread_id=thread_id,
            evidence=evidence or {},
            created_at=int(now or time.time()),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeedbackRecord:
        return cls(
            id=str(payload["id"]),
            scope=validate_scope(str(payload["scope"])),
            rating=validate_rating(str(payload["rating"])),
            comment=validate_nonempty("comment", str(payload["comment"])),
            run_id=payload.get("run_id"),
            thread_id=payload.get("thread_id"),
            evidence=dict(payload.get("evidence") or {}),
            created_at=int(payload.get("created_at") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "rating": self.rating,
            "comment": self.comment,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }

    @property
    def dedupe_key(self) -> tuple[str, str, str, str]:
        return (
            self.scope,
            self.rating,
            self.run_id or "",
            normalize_text(self.comment),
        )


class FeedbackStore:
    """User feedback records backed by the existing Memory core store."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def records(
        self,
        scope: Optional[str] = None,
        rating: Optional[str] = None,
    ) -> list[FeedbackRecord]:
        if scope is not None:
            scope = validate_scope(scope)
        if rating is not None:
            rating = validate_rating(rating)
        records = self._load_records()
        filtered = []
        for record in records:
            if scope is not None and record.scope != scope:
                continue
            if rating is not None and record.rating != rating:
                continue
            filtered.append(record)
        return sorted(filtered, key=lambda record: -record.created_at)

    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        records = self._load_records()
        existing = self._find_existing(records, record)
        if existing is not None:
            return records[existing]
        records.append(record)
        self._save_records(records)
        return record

    def _load_records(self) -> list[FeedbackRecord]:
        raw = self.memory.core_get(FEEDBACK_RECORDS_KEY)
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MemoryValidationError("feedback records are not valid JSON") from exc
        if not isinstance(payload, list):
            raise MemoryValidationError("feedback records must be a list")
        return [FeedbackRecord.from_dict(item) for item in payload if isinstance(item, dict)]

    def _save_records(self, records: list[FeedbackRecord]) -> None:
        payload = [record.to_dict() for record in sorted(records, key=lambda r: r.created_at)]
        self.memory.core_set(FEEDBACK_RECORDS_KEY, json.dumps(payload, sort_keys=True))

    @staticmethod
    def _find_existing(
        records: list[FeedbackRecord],
        candidate: FeedbackRecord,
    ) -> Optional[int]:
        for index, record in enumerate(records):
            if record.dedupe_key == candidate.dedupe_key:
                return index
        return None


def infer_taste_from_feedback(
    feedback: FeedbackRecord,
    active: bool = False,
) -> list[TasteRecord]:
    records: list[TasteRecord] = []
    status = ACTIVE if active or feedback.rating == "positive" else PENDING
    evidence = {
        "feedback_id": feedback.id,
        "rating": feedback.rating,
        "run_id": feedback.run_id,
        **feedback.evidence,
    }

    for pattern, kind, template in (
        (
            r"\b(?:do more of|more of|liked|love|good:?)\s+([^.;\n]+)",
            "preference",
            "Prefer {value}.",
        ),
        (r"\b(?:prefer|please prefer)\s+([^.;\n]+)", "preference", "Prefer {value}."),
        (
            r"\b(?:do less of|less of|disliked|bad:?)\s+([^.;\n]+)",
            "constraint",
            "Avoid {value}.",
        ),
        (r"\b(?:avoid|do not|don't|never)\s+([^.;\n]+)", "constraint", "Avoid {value}."),
    ):
        for match in re.finditer(pattern, feedback.comment, flags=re.IGNORECASE):
            value = clean_fragment(match.group(1))
            if value:
                records.append(
                    TasteRecord.create(
                        scope=feedback.scope,
                        kind=kind,
                        text=template.format(value=value),
                        confidence=0.9 if feedback.rating != "neutral" else 0.7,
                        status=status,
                        source_run=feedback.run_id,
                        evidence=evidence,
                    )
                )

    return records


def infer_evolution_from_feedback(feedback: FeedbackRecord) -> list[EvolutionProposal]:
    proposals: list[EvolutionProposal] = []
    evidence = {
        "feedback_id": feedback.id,
        "rating": feedback.rating,
        "run_id": feedback.run_id,
        **feedback.evidence,
    }
    if feedback.rating == "negative":
        proposals.append(
            EvolutionProposal.create(
                scope=feedback.scope,
                kind="prompt_rule",
                title="Address negative user feedback",
                body=f"Avoid repeating this issue in future runs: {feedback.comment}",
                rationale="The user gave negative feedback on a completed harness interaction.",
                source_run=feedback.run_id,
                evidence=evidence,
            )
        )
        if feedback.run_id:
            proposals.append(
                EvolutionProposal.create(
                    scope="project",
                    kind="eval_case",
                    title="Add regression eval from user feedback",
                    body=(
                        f"Add an eval case for run `{feedback.run_id}` that prevents "
                        f"this feedback from recurring: {feedback.comment}"
                    ),
                    rationale="Run-linked negative feedback is high-signal eval material.",
                    source_run=feedback.run_id,
                    evidence=evidence,
                )
            )
    elif feedback.rating == "positive":
        for record in infer_taste_from_feedback(feedback, active=False):
            proposals.append(
                EvolutionProposal.create(
                    scope=record.scope,
                    kind="prompt_rule",
                    title="Preserve positively rated behavior",
                    body=record.text,
                    rationale="The user gave positive feedback and described behavior to repeat.",
                    source_run=feedback.run_id,
                    evidence={**evidence, "taste_text": record.text},
                )
            )
    return proposals


def validate_rating(rating: str) -> str:
    rating = rating.strip().lower()
    aliases = {
        "good": "positive",
        "great": "positive",
        "positive": "positive",
        "up": "positive",
        "+1": "positive",
        "bad": "negative",
        "poor": "negative",
        "negative": "negative",
        "down": "negative",
        "-1": "negative",
        "ok": "neutral",
        "okay": "neutral",
        "neutral": "neutral",
    }
    normalized = aliases.get(rating, rating)
    if normalized not in VALID_RATINGS:
        raise MemoryValidationError(
            f"rating must be one of: {', '.join(sorted(VALID_RATINGS))}"
        )
    return normalized
