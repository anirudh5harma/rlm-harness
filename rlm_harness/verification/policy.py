"""Strict verification policy (Phase D).

The pivot plan's Phase D gate:

    "Failing check exits with status=failed, status=unverified if
     checks cannot run, status=verified only if all pass."

This module owns the four statuses and the policy that maps a
`VerificationResult` (or any compatible view of the gate's
output) to one of them. The supervisor (and any other caller)
consults the policy; no caller inlines the mapping.

The four statuses:

* `verified` — every required check ran and passed.
* `failed` — at least one required check failed.
* `unverified` — checks could not run, timed out, or were skipped.
* `not_applicable` — no code or project artifact changed.

`done` requires `verified`. `not_applicable` is a special case
for non-code work (informational tasks, summaries, audits that
didn't touch code); the supervisor maps it to `done` only when
the task is *explicitly* non-code. A code-edit task with no
changed files is suspicious and is treated as `unverified` by
the supervisor, not `not_applicable`.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from rlm_harness.graph.verification import VerificationResult

# Markers the legacy `VerificationGate` writes into a check's
# `output` when it could not run. Used by the classifier to
# distinguish "ran and passed" from "ran and passed trivially
# because nothing was checked".
_UNRUNNABLE_MARKERS = (
    "skipped:",
    "skipped ",
    "could not",
    "not installed",
    "timed out",
    "no .py files changed",  # python_syntax on a non-py project
    "ruff not installed or nothing to check",
    "no tests found",
)


class VerificationStatus(str, Enum):
    """The four strict verification statuses (Phase D)."""

    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


def _check_unrunnable(check) -> bool:
    """A check is *unrunnable* if its output announces a skip, a
    timeout, or a missing tool. The classifier promotes a
    unrunnable check to the `unverified` status when there are
    code changes to verify.
    """
    output = (check.output or "").lower()
    return any(marker in output for marker in _UNRUNNABLE_MARKERS)


def classify_result(result: VerificationResult) -> VerificationStatus:
    """Map a `VerificationResult` to one of the four statuses.

    The mapping is the single source of truth for Phase D.
    Callers (the supervisor, the CLI, the trace reporter) all
    consult this function; no caller inlines the logic.
    """
    if not result.changed_files:
        # No code or project artifact changed. Verification is
        # only meaningful for code edits; for non-code work this
        # is `not_applicable`. The supervisor decides whether a
        # non-edit run may still be `done`.
        return VerificationStatus.NOT_APPLICABLE

    checks = list(result.checks or [])

    # Any failed check dominates everything else.
    if any(not c.passed for c in checks):
        return VerificationStatus.FAILED

    # No failed checks. If any check was unrunnable, the run is
    # `unverified` — we could not actually check what we set out
    # to check. The legacy `passed=True` for skipped checks is
    # exactly the silent-pass bug Phase D fixes.
    if any(_check_unrunnable(c) for c in checks):
        return VerificationStatus.UNVERIFIED

    # All checks passed. We required a check (we have changes);
    # we got checks; they all passed.
    if checks:
        return VerificationStatus.VERIFIED

    # Changes detected but no checks ran at all (e.g., the
    # verification gate short-circuited). Treat as unverified.
    return VerificationStatus.UNVERIFIED


class VerificationPolicy:
    """The strict verification policy.

    The policy is a thin, stateless facade over `classify_result`.
    It exists so callers can take a dependency on a single type
    (and so we have one place to evolve the four-status
    contract).
    """

    def classify(
        self, result: VerificationResult
    ) -> VerificationStatus:
        return classify_result(result)

    @staticmethod
    def done_allowed(status: VerificationStatus) -> bool:
        """Whether the given status allows a `done` exit.

        Only `verified` qualifies. `unverified` is not `done`
        (the run did not actually verify its work). `failed` is
        not `done` (the work is broken). `not_applicable` is not
        `done` for code edits (no checks ran).
        """
        return status == VerificationStatus.VERIFIED

    @staticmethod
    def is_terminal_failure(status: VerificationStatus) -> bool:
        """`failed` is the only status that means the work is broken.
        The supervisor and the CLI both treat it as a hard exit.
        """
        return status == VerificationStatus.FAILED

    @staticmethod
    def phase_d_statuses() -> Iterable[VerificationStatus]:
        return tuple(VerificationStatus)


__all__ = [
    "VerificationPolicy",
    "VerificationStatus",
    "classify_result",
]
