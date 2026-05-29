from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rlm_harness.config import (
    CONFIG_PATH,
    default_api_key,
    default_base_url,
    default_memory_path,
    default_model,
    default_profile_path,
    default_provider,
    default_trace_path,
    masked_secret,
)

READY = "ready"
WARNING = "warning"
BLOCKED = "blocked"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str
    action: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    checks: list[ReadinessCheck]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_readiness_report(workspace: Path, check_docker: bool = True) -> ReadinessReport:
    checks = [
        provider_check(),
        api_key_check(),
        config_path_check(),
        writable_path_check("profile_db", default_profile_path()),
        writable_path_check("memory_db", workspace / default_memory_path()),
        writable_path_check("trace_db", workspace / default_trace_path()),
        module_check("sqlite_vec", required=True),
        module_check("langgraph", required=True),
        module_check("langgraph.checkpoint.sqlite", required=True),
    ]
    if check_docker:
        checks.extend(docker_checks())
    return ReadinessReport(status=overall_status(checks), checks=checks)


def provider_check() -> ReadinessCheck:
    provider = default_provider()
    model = default_model()
    if provider == "stub":
        return ReadinessCheck(
            name="provider",
            status=BLOCKED,
            detail="provider=stub model=stub is for tests, not daily coding work",
            action="Run `harness /provider`, then `harness /model`.",
        )
    return ReadinessCheck(
        name="provider",
        status=READY,
        detail=f"provider={provider} model={model} base_url={default_base_url()}",
    )


def api_key_check() -> ReadinessCheck:
    provider = default_provider()
    if provider == "stub":
        return ReadinessCheck(
            name="api_key",
            status=BLOCKED,
            detail="no real provider is configured",
            action="Run `harness /provider <name> --api-key <key>`.",
        )
    key = default_api_key(provider)
    if not key:
        return ReadinessCheck(
            name="api_key",
            status=BLOCKED,
            detail=f"API key for {provider} is not set",
            action=f"Run `harness /provider {provider} --api-key <key>`.",
        )
    return ReadinessCheck(
        name="api_key",
        status=READY,
        detail=f"{provider} API key {masked_secret(key)}",
    )


def config_path_check() -> ReadinessCheck:
    path = CONFIG_PATH
    if path.exists():
        return ReadinessCheck("config", READY, str(path))
    parent = path.parent
    if writable_directory(parent):
        return ReadinessCheck(
            "config",
            WARNING,
            f"{path} does not exist yet",
            "Run `harness /provider` to save provider configuration.",
        )
    return ReadinessCheck(
        "config",
        BLOCKED,
        f"{parent} is not writable",
        "Choose a writable HARNESS_CONFIG_DIR or fix permissions.",
    )


def writable_path_check(name: str, path: Path) -> ReadinessCheck:
    parent = path.parent
    if writable_directory(parent):
        return ReadinessCheck(name, READY, str(path))
    return ReadinessCheck(
        name=name,
        status=BLOCKED,
        detail=f"{parent} is not writable",
        action="Fix directory permissions or choose a different path with env vars.",
    )


def module_check(module: str, required: bool) -> ReadinessCheck:
    try:
        __import__(module)
    except ImportError:
        return ReadinessCheck(
            name=module,
            status=BLOCKED if required else WARNING,
            detail="missing",
            action=(
                f"Install `{module}` in the Harness environment."
                if required
                else f"Install `{module}` for optional backend support."
            ),
        )
    return ReadinessCheck(name=module, status=READY, detail="installed")


def docker_checks() -> list[ReadinessCheck]:
    if shutil.which("docker") is None:
        return [
            ReadinessCheck(
                name="docker",
                status=WARNING,
                detail="docker CLI not found",
                action="Install Docker for sandboxed runs, or use `--no-sandbox`.",
            )
        ]

    checks = [ReadinessCheck("docker_cli", READY, shutil.which("docker") or "docker")]
    info = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if info.returncode == 0:
        checks.append(ReadinessCheck("docker_daemon", READY, info.stdout.strip()))
    else:
        checks.append(
            ReadinessCheck(
                "docker_daemon",
                WARNING,
                "unavailable",
                "Start Docker for sandboxed runs, or use `--no-sandbox`.",
            )
        )

    image = subprocess.run(
        ["docker", "image", "inspect", "rlm-harness-sandbox:latest"],
        text=True,
        capture_output=True,
        check=False,
    )
    checks.append(
        ReadinessCheck(
            name="sandbox_image",
            status=READY if image.returncode == 0 else WARNING,
            detail="ok" if image.returncode == 0 else "missing",
            action="" if image.returncode == 0 else "Run `harness sandbox build`.",
        )
    )
    return checks


def writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".harness_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def overall_status(checks: list[ReadinessCheck]) -> str:
    if any(check.status == BLOCKED for check in checks):
        return "needs_setup"
    if any(check.status == WARNING for check in checks):
        return "degraded"
    return READY


def render_readiness_report(report: ReadinessReport) -> str:
    lines = [f"readiness\t{report.status}"]
    for check in report.checks:
        line = f"{check.name}\t{check.status}\t{check.detail}"
        if check.action:
            line += f"\t{check.action}"
        lines.append(line)
    if report.status == READY:
        lines.append("next\tRun `harness \"fix a small bug\"` from a project directory.")
    else:
        lines.append("next\tAddress blocked checks, then run `harness readiness` again.")
    return "\n".join(lines)
