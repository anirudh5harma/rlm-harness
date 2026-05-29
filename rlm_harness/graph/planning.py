from __future__ import annotations

import re
from typing import Optional

from rlm_harness.types import PlanStep, PlanStepStatus, TaskPlan

_FULL_STEP_RE = re.compile(r"^(\d+[a-z]?)[.)]\s+(.+)$", re.IGNORECASE)
_SUB_STEP_RE = re.compile(r"^(\d+[a-z]?)[.)]\s+(.+)$", re.IGNORECASE)


def parse_structured_plan(text: str) -> TaskPlan:
    """Parse numbered lines into a structured TaskPlan with optional hierarchy.

    Handles:
        1. Setup environment
        2a. Read config
        2b. Validate schema
        3. Run tests

    as well as dotted variants:
        1. Setup
        2.1 Read config
        2.2 Validate schema

    Returns a TaskPlan with flat steps; dependencies are computed separately.
    """
    steps: list[PlanStep] = []
    seen_ids: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _FULL_STEP_RE.match(line)
        if not match:
            continue

        step_id = match.group(1).strip()
        description = match.group(2).strip()

        if step_id in seen_ids:
            continue
        seen_ids.add(step_id)

        parent_id: Optional[str] = None
        if re.match(r"^\d+[a-z]$", step_id, re.IGNORECASE):
            parent_id = re.sub(r"[a-z]$", "", step_id, flags=re.IGNORECASE)
        elif "." in step_id:
            parent_id = step_id.rsplit(".", 1)[0]

        steps.append(
            PlanStep(
                id=step_id,
                description=description,
                parent_id=parent_id,
            )
        )

    if not steps:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        steps = [
            PlanStep(id=str(i + 1), description=line.lstrip("-* ").strip())
            for i, line in enumerate(lines)
        ]

    detect_dependencies(steps)
    return TaskPlan(steps=steps, current_step_id=steps[0].id if steps else None)


def detect_dependencies(steps: list[PlanStep]) -> None:
    """Derive step dependencies from sequential ordering and parent/child relationships.

    Rules:
    - Sequential steps depend on the preceding step.
    - Sibling steps (children of the same parent) are independent.
    - A parent step depends on all its children being completed.
    """
    if not steps:
        return

    prev: Optional[PlanStep] = None
    for step in steps:
        if prev is not None and step.parent_id is None:
            step.dependencies.append(prev.id)
        elif prev is not None and step.parent_id is not None:
            if prev.parent_id != step.parent_id:
                step.dependencies.append(prev.id)
        prev = step


def format_plan_context(plan: TaskPlan) -> str:
    """Render the plan with status emoji indicators."""
    if not plan.steps:
        return "No plan steps."

    indicators = {
        PlanStepStatus.PENDING: "⬜",
        PlanStepStatus.IN_PROGRESS: "🔄",
        PlanStepStatus.COMPLETED: "✅",
        PlanStepStatus.FAILED: "❌",
        PlanStepStatus.SKIPPED: "⏭️",
    }

    lines: list[str] = []
    for step in plan.steps:
        indicator = indicators.get(step.status, "❓")
        current_marker = " ← current" if step.id == plan.current_step_id else ""
        indent = "  " if step.parent_id else ""
        lines.append(f"{indicator} {indent}Step {step.id}: {step.description}{current_marker}")

    completed = plan.completed_count()
    total = plan.total_count()
    lines.append(f"\nProgress: {completed}/{total} steps completed")
    return "\n".join(lines)


def get_next_pending_step(plan: TaskPlan) -> Optional[PlanStep]:
    """Return the first PENDING step whose dependencies are all COMPLETED."""
    completed_ids = {s.id for s in plan.steps if s.status == PlanStepStatus.COMPLETED}
    pending = [s for s in plan.steps if s.status == PlanStepStatus.PENDING]

    for step in pending:
        if all(dep in completed_ids for dep in step.dependencies):
            return step
    return pending[0] if pending else None


def plan_step_by_id(plan: TaskPlan, step_id: str) -> Optional[PlanStep]:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


def advance_to_next_step(plan: TaskPlan) -> None:
    """Mark current step complete and advance to the next pending step."""
    if plan.current_step_id:
        current = plan_step_by_id(plan, plan.current_step_id)
        if current:
            current.status = PlanStepStatus.COMPLETED

    next_step = get_next_pending_step(plan)
    plan.current_step_id = next_step.id if next_step else None
