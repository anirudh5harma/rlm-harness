from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    status: str
    elapsed_ms: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok" and not self.timed_out
