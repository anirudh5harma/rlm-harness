from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rlm_harness.sandbox.types import ExecutionResult


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "rlm-harness-sandbox:latest"
    dockerfile: Path = Path("docker/sandbox.Dockerfile")
    workspace: Path = Path(".")
    mount_path: str = "/workspace"
    memory: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 128
    network: str = "none"
    user: str = "sandbox"
    read_only_root: bool = True
    tmpfs: str = "/tmp:rw,noexec,nosuid,size=64m"
    default_timeout_s: float = 60
    start_timeout_s: float = 15
    remove_on_stop: bool = True


class DockerREPL:
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.container_name = f"rlm-harness-sandbox-{uuid.uuid4().hex[:12]}"
        self._process: Optional[subprocess.Popen[str]] = None

    @classmethod
    def build_image(
        cls,
        image: str = "rlm-harness-sandbox:latest",
        dockerfile: Path = Path("docker/sandbox.Dockerfile"),
        context: Path = Path("."),
    ) -> None:
        command = [
            "docker",
            "build",
            "-t",
            image,
            "-f",
            str(dockerfile),
            str(context),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise SandboxError(
                f"docker build failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

    def start(self) -> None:
        if self._process and self._process.poll() is None:
            return

        workspace = self.config.workspace.resolve()
        if not workspace.exists():
            raise SandboxError(f"workspace does not exist: {workspace}")

        command = [
            "docker",
            "run",
            "--rm" if self.config.remove_on_stop else "--init",
            "--name",
            self.container_name,
            "--interactive",
            "--memory",
            self.config.memory,
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            str(self.config.pids_limit),
            "--network",
            self.config.network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            self.config.user,
            "--workdir",
            self.config.mount_path,
            "--mount",
            f"type=bind,src={workspace},dst={self.config.mount_path}",
            self.config.image,
        ]
        if self.config.read_only_root:
            command.insert(-1, "--read-only")
        if self.config.tmpfs:
            command.insert(-1, "--tmpfs")
            command.insert(-1, self.config.tmpfs)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._ensure_started()

    def execute(self, code: str, timeout_s: Optional[float] = None) -> ExecutionResult:
        self.start()
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise SandboxError("sandbox process is not running")
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise SandboxError(f"sandbox container exited early: {stderr}")

        request = {
            "id": uuid.uuid4().hex,
            "code": code,
            "timeout_s": timeout_s or self.config.default_timeout_s,
        }
        self._process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._process.stdin.flush()

        response_line = self._read_response(request["id"], float(request["timeout_s"]) + 5)
        try:
            payload = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise SandboxError(f"sandbox returned invalid JSON: {response_line!r}") from exc

        return ExecutionResult(
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            status=str(payload.get("status", "error")),
            elapsed_ms=int(payload.get("elapsed_ms", 0)),
            timed_out=bool(payload.get("timed_out", False)),
        )

    def stop(self) -> None:
        if not self._process:
            return
        if self._process.poll() is None:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "kill", self.container_name],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self._process.wait(timeout=5)
        if self._process.stdout:
            self._process.stdout.close()
        if self._process.stderr:
            self._process.stderr.close()
        self._process = None

    def __enter__(self) -> DockerREPL:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _ensure_started(self) -> None:
        deadline = time.monotonic() + self.config.start_timeout_s
        while time.monotonic() < deadline:
            if self._process is None:
                raise SandboxError("sandbox process did not start")
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxError(f"sandbox container exited during startup: {stderr}")
            return
        raise SandboxError("sandbox did not start before timeout")

    def _read_response(self, request_id: str, timeout_s: float) -> str:
        if not self._process or not self._process.stdout:
            raise SandboxError("sandbox process has no stdout")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if not line:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read() if self._process.stderr else ""
                    raise SandboxError(f"sandbox exited while waiting for response: {stderr}")
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == request_id:
                return line
        raise SandboxError("timed out waiting for sandbox response")
