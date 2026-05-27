from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rlm_harness.memory.store import Memory
from rlm_harness.types import HarnessState, TaskPlan


CHECKPOINT_KIND = "plan_checkpoint"


@dataclass
class PlanCheckpoint:
    step_id: str
    completed_step_ids: list[str] = field(default_factory=list)
    completed_at: float = 0.0
    workspace_git_hash: str = ""
    plan_snapshot: dict = field(default_factory=dict)
    iteration: int = 0


class CheckpointManager:
    def __init__(self, memory: Memory):
        self.memory = memory

    def save(self, state: HarnessState) -> PlanCheckpoint:
        checkpoint = PlanCheckpoint(
            step_id=state.plan.current_step_id or "",
            completed_step_ids=[
                s.id
                for s in state.plan.steps
                if s.status == "completed"
            ],
            completed_at=time.time(),
            workspace_git_hash=_git_hash(state.workspace),
            plan_snapshot=state.plan.model_dump(),
            iteration=int(state.scratch.get("graph_iterations", 0)),
        )
        self.memory.archival_add(
            kind=CHECKPOINT_KIND,
            content=json.dumps(checkpoint.__dict__, sort_keys=True),
            source_thread=state.thread_id,
            metadata={
                "step_id": checkpoint.step_id,
                "completed_step_count": len(checkpoint.completed_step_ids),
            },
        )
        return checkpoint

    def load_latest(self, thread_id: str) -> Optional[PlanCheckpoint]:
        results = self.memory.archival_search(
            query="plan checkpoint resume",
            k=1,
            kind=CHECKPOINT_KIND,
            source_thread=thread_id,
        )
        if not results:
            return None
        try:
            data = json.loads(results[0].memory.content)
        except json.JSONDecodeError:
            return None
        return PlanCheckpoint(**data)

    @staticmethod
    def resume_state(state: HarnessState, checkpoint: PlanCheckpoint) -> HarnessState:
        plan = TaskPlan.model_validate(checkpoint.plan_snapshot)
        for step in plan.steps:
            if step.id in checkpoint.completed_step_ids:
                step.status = "completed"

        from rlm_harness.graph.planning import get_next_pending_step

        next_step = get_next_pending_step(plan)
        plan.current_step_id = next_step.id if next_step else None
        state.plan = plan
        state.scratch["resumed_from_checkpoint"] = True
        state.scratch["resumed_step_id"] = checkpoint.step_id
        return state


def _git_hash(workspace: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""
