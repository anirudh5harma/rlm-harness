from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from rlm_harness.memory.store import Memory, MemoryValidationError
from rlm_harness.types import HarnessState

TASTE_RECORDS_KEY = "taste.records.v1"
ACTIVE = "active"
PENDING = "pending"
REJECTED = "rejected"
VALID_STATUSES = {ACTIVE, PENDING, REJECTED}
VALID_SCOPES = {"user", "project"}


@dataclass(frozen=True)
class TasteRecord:
    id: str
    scope: str
    kind: str
    text: str
    confidence: float = 0.75
    status: str = ACTIVE
    source_run: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0

    @classmethod
    def create(
        cls,
        scope: str,
        kind: str,
        text: str,
        confidence: float = 0.75,
        status: str = ACTIVE,
        source_run: Optional[str] = None,
        evidence: Optional[dict[str, Any]] = None,
        now: Optional[int] = None,
    ) -> TasteRecord:
        timestamp = int(now or time.time())
        return cls(
            id=f"taste-{uuid.uuid4().hex[:12]}",
            scope=validate_scope(scope),
            kind=validate_nonempty("kind", kind),
            text=validate_nonempty("text", text),
            confidence=max(0.0, min(1.0, float(confidence))),
            status=validate_status(status),
            source_run=source_run,
            evidence=evidence or {},
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TasteRecord:
        return cls(
            id=str(payload["id"]),
            scope=validate_scope(str(payload["scope"])),
            kind=validate_nonempty("kind", str(payload["kind"])),
            text=validate_nonempty("text", str(payload["text"])),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.75)))),
            status=validate_status(str(payload.get("status", ACTIVE))),
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
            "text": self.text,
            "confidence": self.confidence,
            "status": self.status,
            "source_run": self.source_run,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.scope, self.kind, normalize_text(self.text))


class TasteProfileStore:
    """Typed taste/profile records backed by the existing Memory core store."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def records(
        self,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[TasteRecord]:
        if scope is not None:
            scope = validate_scope(scope)
        if status is not None:
            status = validate_status(status)
        records = self._load_records()
        filtered = []
        for record in records:
            if scope is not None and record.scope != scope:
                continue
            if status is not None and record.status != status:
                continue
            if kind is not None and record.kind != kind:
                continue
            filtered.append(record)
        return sorted(filtered, key=lambda r: (r.status != ACTIVE, -r.confidence, -r.updated_at))

    def add(self, record: TasteRecord) -> TasteRecord:
        records = self._load_records()
        existing_index = self._find_existing(records, record)
        now = int(time.time())
        if existing_index is not None:
            existing = records[existing_index]
            merged = TasteRecord(
                id=existing.id,
                scope=existing.scope,
                kind=existing.kind,
                text=existing.text,
                confidence=max(existing.confidence, record.confidence),
                status=ACTIVE if ACTIVE in {existing.status, record.status} else record.status,
                source_run=record.source_run or existing.source_run,
                evidence={**existing.evidence, **record.evidence},
                created_at=existing.created_at,
                updated_at=now,
            )
            records[existing_index] = merged
            self._save_records(records)
            return merged

        records.append(record)
        self._save_records(records)
        return record

    def learn_from_state(self, state: HarnessState) -> list[TasteRecord]:
        learned = []
        for record in infer_taste_records(state):
            learned.append(self.add(record))
        return learned

    def approve(self, record_id: str) -> Optional[TasteRecord]:
        return self._set_status(record_id, ACTIVE)

    def reject(self, record_id: str) -> Optional[TasteRecord]:
        return self._set_status(record_id, REJECTED)

    def render_context(self, max_records: int = 16) -> str:
        active_records = self.records(status=ACTIVE)[:max_records]
        if not active_records:
            return ""

        sections: list[str] = []
        for scope in ("user", "project"):
            scoped = [record for record in active_records if record.scope == scope]
            if not scoped:
                continue
            title = "User taste" if scope == "user" else "Project conventions"
            lines = [
                f"- {record.text} ({record.kind}, confidence {record.confidence:.2f})"
                for record in scoped
            ]
            sections.append(f"{title}:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    def _load_records(self) -> list[TasteRecord]:
        raw = self.memory.core_get(TASTE_RECORDS_KEY)
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MemoryValidationError("taste profile records are not valid JSON") from exc
        if not isinstance(payload, list):
            raise MemoryValidationError("taste profile records must be a list")
        return [TasteRecord.from_dict(item) for item in payload if isinstance(item, dict)]

    def _save_records(self, records: list[TasteRecord]) -> None:
        payload = [record.to_dict() for record in sorted(records, key=lambda r: r.created_at)]
        self.memory.core_set(TASTE_RECORDS_KEY, json.dumps(payload, sort_keys=True))

    def _set_status(self, record_id: str, status: str) -> Optional[TasteRecord]:
        records = self._load_records()
        for index, record in enumerate(records):
            if record.id == record_id:
                updated = TasteRecord(
                    id=record.id,
                    scope=record.scope,
                    kind=record.kind,
                    text=record.text,
                    confidence=record.confidence,
                    status=validate_status(status),
                    source_run=record.source_run,
                    evidence=record.evidence,
                    created_at=record.created_at,
                    updated_at=int(time.time()),
                )
                records[index] = updated
                self._save_records(records)
                return updated
        return None

    @staticmethod
    def _find_existing(records: list[TasteRecord], candidate: TasteRecord) -> Optional[int]:
        for index, record in enumerate(records):
            if record.dedupe_key == candidate.dedupe_key:
                return index
        return None


class TasteProfileManager:
    """Routes user-wide taste and project-local conventions to the right stores."""

    def __init__(
        self,
        user_memory: Optional[Memory],
        project_memory: Optional[Memory] = None,
    ):
        self.user_store = TasteProfileStore(user_memory) if user_memory is not None else None
        self.project_store = (
            TasteProfileStore(project_memory) if project_memory is not None else None
        )

    def render_context(self, max_records: int = 16) -> str:
        records = []
        if self.user_store is not None:
            records.extend(self.user_store.records(scope="user", status=ACTIVE))
        if self.project_store is not None:
            records.extend(self.project_store.records(scope="project", status=ACTIVE))

        records = sorted(records, key=lambda r: (-r.confidence, -r.updated_at))[:max_records]
        if not records:
            return ""

        sections = []
        for scope in ("user", "project"):
            scoped = [record for record in records if record.scope == scope]
            if not scoped:
                continue
            title = "User taste" if scope == "user" else "Project conventions"
            lines = [
                f"- {record.text} ({record.kind}, confidence {record.confidence:.2f})"
                for record in scoped
            ]
            sections.append(f"{title}:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    def learn_from_state(self, state: HarnessState) -> list[TasteRecord]:
        learned = []
        for record in infer_taste_records(state):
            store = self._store_for_scope(record.scope)
            if store is not None:
                learned.append(store.add(record))
        return learned

    def _store_for_scope(self, scope: str) -> Optional[TasteProfileStore]:
        if scope == "project":
            return self.project_store or self.user_store
        return self.user_store or self.project_store


def infer_taste_records(state: HarnessState) -> list[TasteRecord]:
    records: list[TasteRecord] = []
    records.extend(infer_explicit_preferences(state))
    records.extend(infer_verification_commands(state))
    return records


def infer_explicit_preferences(state: HarnessState) -> list[TasteRecord]:
    task = state.task.strip()
    source = {"task": task[:500]}
    records: list[TasteRecord] = []

    for pattern, kind, template in (
        (r"\bI prefer\s+([^.;\n]+)", "preference", "Prefer {value}."),
        (r"\bplease prefer\s+([^.;\n]+)", "preference", "Prefer {value}."),
        (r"\balways\s+([^.;\n]+)", "preference", "Always {value}."),
        (r"\b(?:do not|don't|avoid|never)\s+([^.;\n]+)", "constraint", "Avoid {value}."),
    ):
        for match in re.finditer(pattern, task, flags=re.IGNORECASE):
            value = clean_fragment(match.group(1))
            if value:
                records.append(
                    TasteRecord.create(
                        scope="user",
                        kind=kind,
                        text=template.format(value=value),
                        confidence=0.95,
                        status=ACTIVE,
                        source_run=state.run_id,
                        evidence=source,
                    )
                )

    return records


def infer_verification_commands(state: HarnessState) -> list[TasteRecord]:
    result = state.scratch.get("verification_result")
    if not isinstance(result, dict):
        return []

    records: list[TasteRecord] = []
    for check in result.get("checks", []):
        if not isinstance(check, dict) or not check.get("passed"):
            continue
        command = str(check.get("command") or "").strip()
        check_type = str(check.get("check_type") or "verification").strip()
        if not command:
            continue
        records.append(
            TasteRecord.create(
                scope="project",
                kind="verification_command",
                text=f"Run `{command}` for {check_type} verification.",
                confidence=0.82,
                status=ACTIVE,
                source_run=state.run_id,
                evidence={"check_type": check_type},
            )
        )
    return records


def clean_fragment(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -:,")
    if not value:
        return ""
    return value[0].lower() + value[1:]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def validate_nonempty(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise MemoryValidationError(f"{name} must be non-empty")
    return value


def validate_scope(scope: str) -> str:
    scope = scope.strip().lower()
    if scope not in VALID_SCOPES:
        raise MemoryValidationError(f"scope must be one of: {', '.join(sorted(VALID_SCOPES))}")
    return scope


def validate_status(status: str) -> str:
    status = status.strip().lower()
    if status not in VALID_STATUSES:
        raise MemoryValidationError(
            f"status must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )
    return status
