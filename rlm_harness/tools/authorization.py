from __future__ import annotations

from dataclasses import dataclass

from rlm_harness.actions import ActionRisk, AnyAction
from rlm_harness.kernel import AutonomyMode
from rlm_harness.tools.registry import SideEffect, ToolDescriptor


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str = ""


def authorize_tool_action(
    action: AnyAction,
    descriptor: ToolDescriptor,
    autonomy: AutonomyMode | str,
) -> AuthorizationDecision:
    mode = normalize_autonomy_mode(autonomy)
    if mode in {AutonomyMode.ASK, AutonomyMode.PLAN}:
        if descriptor.risk == ActionRisk.READ or descriptor.side_effect == SideEffect.COMPLETION:
            return AuthorizationDecision(True)
        return AuthorizationDecision(
            False,
            f"{mode.value} mode is read-only and cannot run {descriptor.name}.",
        )

    if mode == AutonomyMode.PROPOSE:
        if descriptor.risk == ActionRisk.READ or descriptor.side_effect == SideEffect.COMPLETION:
            return AuthorizationDecision(True)
        if action.kind == "propose_file_change":
            return AuthorizationDecision(True)
        return AuthorizationDecision(
            False,
            f"propose mode can queue changes but cannot run {descriptor.name}.",
        )

    if mode == AutonomyMode.SANDBOX and descriptor.risk == ActionRisk.DESTRUCTIVE:
        return AuthorizationDecision(False, f"sandbox mode cannot run {descriptor.name}.")

    return AuthorizationDecision(True)


def normalize_autonomy_mode(value: AutonomyMode | str) -> AutonomyMode:
    if isinstance(value, AutonomyMode):
        return value
    return AutonomyMode(str(value))
