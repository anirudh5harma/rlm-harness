from __future__ import annotations

from enum import Enum
from typing import Optional

from rlm_harness.types import PlanStep


class ErrorCategory(str, Enum):
    PARSE_ERROR = "parse_error"
    SANDBOX_ERROR = "sandbox_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    VERIFICATION_FAILURE = "verification_failure"
    EMPTY_OUTPUT = "empty_output"
    WRONG_OUTPUT = "wrong_output"
    MODEL_REFUSAL = "model_refusal"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    RETRY_WITH_CLARIFICATION = "retry_clarify"
    SIMPLIFY = "simplify"
    ALTERNATIVE_APPROACH = "alternative"
    SKIP = "skip"
    ABORT = "abort"


RECOVERY_MAP: dict[ErrorCategory, list[RecoveryStrategy]] = {
    ErrorCategory.PARSE_ERROR: [
        RecoveryStrategy.RETRY_WITH_CLARIFICATION,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.TIMEOUT: [
        RecoveryStrategy.SIMPLIFY,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.TOOL_ERROR: [
        RecoveryStrategy.ALTERNATIVE_APPROACH,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.SANDBOX_ERROR: [
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.EMPTY_OUTPUT: [
        RecoveryStrategy.RETRY_WITH_CLARIFICATION,
        RecoveryStrategy.SIMPLIFY,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.VERIFICATION_FAILURE: [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.ALTERNATIVE_APPROACH,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.WRONG_OUTPUT: [
        RecoveryStrategy.RETRY_WITH_CLARIFICATION,
        RecoveryStrategy.ALTERNATIVE_APPROACH,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.MODEL_REFUSAL: [
        RecoveryStrategy.ALTERNATIVE_APPROACH,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.UNKNOWN: [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RETRY_WITH_CLARIFICATION,
        RecoveryStrategy.SKIP,
        RecoveryStrategy.ABORT,
    ],
}


class ErrorClassifier:
    @staticmethod
    def classify(
        status: Optional[str],
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> ErrorCategory:
        if timed_out:
            return ErrorCategory.TIMEOUT

        status_lower = (status or "").lower()

        if status_lower == "action_parse_error":
            return ErrorCategory.PARSE_ERROR

        if status_lower == "sandbox_error":
            return ErrorCategory.SANDBOX_ERROR

        if status_lower in {"tool_error", "error"}:
            combined = (stderr + stdout).lower()
            if "modulenotfounderror" in combined or "importerror" in combined:
                return ErrorCategory.TOOL_ERROR
            if "toolerror" in combined or "tool error" in combined:
                return ErrorCategory.TOOL_ERROR
            if not stdout.strip() and not stderr.strip():
                return ErrorCategory.EMPTY_OUTPUT
            return ErrorCategory.TOOL_ERROR

        if status_lower == "ok":
            if not stdout.strip() and not stderr.strip():
                return ErrorCategory.EMPTY_OUTPUT
            return ErrorCategory.UNKNOWN

        return ErrorCategory.UNKNOWN

    @staticmethod
    def classify_for_recovery_failure(reason: str) -> ErrorCategory:
        """For reflection-level failures (wrong output type, no user-facing output, etc.)."""
        reason_lower = reason.lower()
        if "file inventory" in reason_lower or "source dump" in reason_lower:
            return ErrorCategory.WRONG_OUTPUT
        if "no user-facing output" in reason_lower or "did not produce" in reason_lower:
            return ErrorCategory.EMPTY_OUTPUT
        if "did not report" in reason_lower or "verification" in reason_lower:
            return ErrorCategory.VERIFICATION_FAILURE
        return ErrorCategory.WRONG_OUTPUT


class RecoverySelector:
    @staticmethod
    def select(category: ErrorCategory, step: PlanStep) -> RecoveryStrategy:
        strategies = RECOVERY_MAP.get(category, RECOVERY_MAP[ErrorCategory.UNKNOWN])
        idx = min(step.attempts, len(strategies) - 1)
        return strategies[idx]

    @staticmethod
    def generate_hint(strategy: RecoveryStrategy, step: PlanStep, error: str) -> str:
        error_brief = error[:200]
        hints = {
            RecoveryStrategy.RETRY: (
                f"Previous attempt for step {step.id} failed: {error_brief}. "
                "Try again with a different approach. "
                f"(Attempt {step.attempts + 1}/{step.max_attempts})"
            ),
            RecoveryStrategy.RETRY_WITH_CLARIFICATION: (
                f"Previous attempt for step {step.id} failed: {error_brief}. "
                "Re-read the task requirements carefully and return only valid output. "
                f"(Attempt {step.attempts + 1}/{step.max_attempts})"
            ),
            RecoveryStrategy.SIMPLIFY: (
                f"Step {step.id} is too complex. Break it down and handle only one "
                "sub-task now. You can return to the rest later."
            ),
            RecoveryStrategy.ALTERNATIVE_APPROACH: (
                f"Previous approach for step {step.id} failed: {error_brief}. "
                "Consider a completely different strategy — use different tools or "
                "reframe the problem."
            ),
            RecoveryStrategy.SKIP: (
                f"Step {step.id} could not be completed after {step.attempts} attempts. "
                "Skip this step and move to the next one."
            ),
            RecoveryStrategy.ABORT: (
                f"Step {step.id} failed after {step.attempts} attempts and cannot be "
                f"recovered. The task will be aborted. Last error: {error_brief}"
            ),
        }
        return hints.get(strategy, hints[RecoveryStrategy.RETRY])


def is_retryable_decision(strategy: RecoveryStrategy) -> bool:
    """Whether the reflect node should return 'continue' instead of 'error'/'stopped'."""
    return strategy in {
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RETRY_WITH_CLARIFICATION,
        RecoveryStrategy.SIMPLIFY,
        RecoveryStrategy.ALTERNATIVE_APPROACH,
    }
