"""Tests for the strict verification policy (Phase D).

The pivot plan's Phase D gate:

    "Failing check exits with status=failed, status=unverified if
     checks cannot run, status=verified only if all pass."

The four statuses are:

* `verified` — every required check ran and passed.
* `failed` — at least one required check failed.
* `unverified` — checks could not run, timed out, or were skipped.
* `not_applicable` — no code or project artifact changed.

`VerificationPolicy.classify` is the single source of truth for
mapping a `VerificationResult` to one of the four statuses. The
supervisor consumes the policy's output; `done` requires
`verified`.
"""
import unittest

from rlm_harness.graph.verification import (
    VerificationCheck,
    VerificationResult,
)
from rlm_harness.verification.policy import (
    VerificationPolicy,
    VerificationStatus,
    classify_result,
)


def _result(
    *,
    passed: bool = True,
    checks: list | None = None,
    changed_files: list[str] | None = None,
    summary: str = "",
) -> VerificationResult:
    return VerificationResult(
        passed=passed,
        checks=checks or [],
        changed_files=changed_files or [],
        summary=summary,
    )


def _check(
    *,
    check_type: str,
    passed: bool,
    output: str = "",
    command: str = "",
) -> VerificationCheck:
    return VerificationCheck(
        check_type=check_type,
        passed=passed,
        output=output,
        command=command,
    )


class VerificationStatusTests(unittest.TestCase):
    def test_statuses_are_distinct(self):
        statuses = {
            VerificationStatus.VERIFIED,
            VerificationStatus.FAILED,
            VerificationStatus.UNVERIFIED,
            VerificationStatus.NOT_APPLICABLE,
        }
        self.assertEqual(len(statuses), 4)

    def test_statuses_have_expected_string_values(self):
        self.assertEqual(VerificationStatus.VERIFIED.value, "verified")
        self.assertEqual(VerificationStatus.FAILED.value, "failed")
        self.assertEqual(VerificationStatus.UNVERIFIED.value, "unverified")
        self.assertEqual(VerificationStatus.NOT_APPLICABLE.value, "not_applicable")


class ClassifyResultTests(unittest.TestCase):
    def test_no_changes_is_not_applicable(self):
        result = _result(passed=True, checks=[], changed_files=[])
        self.assertEqual(classify_result(result), VerificationStatus.NOT_APPLICABLE)

    def test_changes_all_passed_is_verified(self):
        result = _result(
            passed=True,
            changed_files=["foo.py"],
            checks=[
                _check(check_type="ruff", passed=True),
                _check(check_type="pytest", passed=True),
            ],
        )
        self.assertEqual(classify_result(result), VerificationStatus.VERIFIED)

    def test_changes_with_failure_is_failed(self):
        result = _result(
            passed=False,
            changed_files=["foo.py"],
            checks=[
                _check(check_type="ruff", passed=False, output="syntax error"),
            ],
        )
        self.assertEqual(classify_result(result), VerificationStatus.FAILED)

    def test_changes_with_unrunnable_check_is_unverified(self):
        """If a check could not run (output mentions 'skipped' or
        'could not'), the run is `unverified`, not `passed`.
        """
        result = _result(
            passed=True,  # The legacy `passed` was set True by
            # mistake. The policy must catch this.
            changed_files=["foo.py"],
            checks=[
                _check(
                    check_type="pytest",
                    passed=True,
                    output="pytest skipped: not installed",
                ),
            ],
        )
        self.assertEqual(classify_result(result), VerificationStatus.UNVERIFIED)

    def test_mixed_failure_and_unverified_is_failed(self):
        """A failed check dominates an unverified check."""
        result = _result(
            passed=False,
            changed_files=["foo.py"],
            checks=[
                _check(check_type="ruff", passed=False, output="error"),
                _check(check_type="pytest", passed=True, output="skipped"),
            ],
        )
        self.assertEqual(classify_result(result), VerificationStatus.FAILED)


class VerificationPolicyTests(unittest.TestCase):
    def test_policy_classify_delegates_to_classify_result(self):
        policy = VerificationPolicy()
        result = _result(
            passed=True,
            changed_files=["foo.py"],
            checks=[_check(check_type="ruff", passed=True)],
        )
        self.assertEqual(
            policy.classify(result),
            VerificationStatus.VERIFIED,
        )

    def test_policy_done_requires_verified(self):
        """The supervisor (and any other caller) can ask the policy:
        is this result sufficient to mark the run as `done`?
        Only `verified` qualifies; `unverified` is not `done`."""
        policy = VerificationPolicy()
        for status, is_done in [
            (VerificationStatus.VERIFIED, True),
            (VerificationStatus.NOT_APPLICABLE, False),  # cannot be done
            (VerificationStatus.UNVERIFIED, False),
            (VerificationStatus.FAILED, False),
        ]:
            with self.subTest(status=status):
                self.assertEqual(
                    policy.done_allowed(status), is_done
                )

    def test_policy_exposes_phase_d_constants(self):
        """Phase D is enforced in the policy, not in callers. The
        constants are the public surface."""
        policy = VerificationPolicy()
        self.assertEqual(
            set(policy.phase_d_statuses()),
            {
                VerificationStatus.VERIFIED,
                VerificationStatus.FAILED,
                VerificationStatus.UNVERIFIED,
                VerificationStatus.NOT_APPLICABLE,
            },
        )


if __name__ == "__main__":
    unittest.main()
