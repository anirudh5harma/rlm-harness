from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.sandbox.types import ExecutionResult
from rlm_harness.types import Msg


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


@dataclass(frozen=True)
class RLMSubcallConfig:
    max_depth: int = 3
    max_subcalls: int = 32
    token_budget: int = 200_000
    max_query_chars: int = 8_000
    max_context_chars: int = 200_000
    max_tokens: int = 1024
    temperature: float = 0.1


@dataclass(frozen=True)
class RecursiveCompletionResult:
    content: str
    subcalls: int = 1
    tokens_used: int = 0


class DockerREPL:
    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        completion_client: Optional[LMClient] = None,
        subcall_config: Optional[RLMSubcallConfig] = None,
        recursive_completion: Optional[
            Callable[[str, str, int, Optional[int], Optional[str]], str | RecursiveCompletionResult]
        ] = None,
    ):
        self.config = config or SandboxConfig()
        self.completion_client = completion_client
        self.subcall_config = subcall_config or RLMSubcallConfig()
        self.recursive_completion = recursive_completion
        self.container_name = f"rlm-harness-sandbox-{uuid.uuid4().hex[:12]}"
        self._process: Optional[subprocess.Popen[str]] = None
        self._subcalls = 0
        self._tokens_used = 0

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
        try:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise SandboxError(
                "docker CLI not found; install Docker or run with --no-sandbox"
            ) from exc
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

        command = self._run_command(workspace)
        try:
            self._start_process(command)
        except SandboxError as exc:
            if not self._is_missing_bind_source_error(str(exc)):
                raise
            self._warm_parent_mount(workspace)
            self.container_name = f"rlm-harness-sandbox-{uuid.uuid4().hex[:12]}"
            self._start_process(self._run_command(workspace))

    def _run_command(self, workspace: Path) -> list[str]:
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
        return command

    def _start_process(self, command: list[str]) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                "docker CLI not found; install Docker or run with --no-sandbox"
            ) from exc
        self._ensure_started()

    @staticmethod
    def _is_missing_bind_source_error(message: str) -> bool:
        return (
            'invalid mount config for type "bind"' in message
            and "bind source path does not exist" in message
        )

    def _warm_parent_mount(self, workspace: Path) -> None:
        """Prime Docker Desktop's file-sharing view for freshly-created deep paths."""
        for ancestor in workspace.parents:
            try:
                relative = workspace.relative_to(ancestor)
            except ValueError:
                continue
            script = (
                "from pathlib import Path\n"
                "import sys\n"
                f"target = Path('/workspace') / {str(relative)!r}\n"
                "sys.exit(0 if target.exists() else 1)\n"
            )
            try:
                completed = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--mount",
                        f"type=bind,src={ancestor},dst=/workspace",
                        "--entrypoint",
                        "python",
                        self.config.image,
                        "-c",
                        script,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return
            if completed.returncode == 0:
                return

    def execute(self, code: str, timeout_s: Optional[float] = None) -> ExecutionResult:
        self.start()
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise SandboxError("sandbox process is not running")
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise SandboxError(f"sandbox container exited early: {stderr}")

        request = {
            "id": uuid.uuid4().hex,
            "type": "execute",
            "code": code,
            "timeout_s": timeout_s or self.config.default_timeout_s,
        }
        subcalls_before = self._subcalls
        tokens_before = self._tokens_used
        self._process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._process.stdin.flush()

        response_line = self._read_execute_result(request["id"], float(request["timeout_s"]) + 5)
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
            subcalls=self._subcalls - subcalls_before,
            tokens_used=self._tokens_used - tokens_before,
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
        stable_after = min(deadline, time.monotonic() + 0.2)
        while time.monotonic() < deadline:
            if self._process is None:
                raise SandboxError("sandbox process did not start")
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxError(f"sandbox container exited during startup: {stderr}")
            if time.monotonic() >= stable_after:
                return
            time.sleep(0.02)
        raise SandboxError("sandbox did not start before timeout")

    def _read_execute_result(self, request_id: str, timeout_s: float) -> str:
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
            message_type = payload.get("type")
            if message_type == "rlm_completion_request":
                self._service_rlm_completion(payload)
                continue
            if message_type == "llm_completion_request":
                self._service_llm_completion(payload)
                continue
            if payload.get("id") == request_id and message_type == "execute_result":
                return line
        raise SandboxError("timed out waiting for sandbox response")

    def _service_rlm_completion(self, request: dict) -> None:
        if not self._process or not self._process.stdin:
            raise SandboxError("sandbox process has no stdin")

        request_id = str(request.get("id", ""))
        response = {
            "type": "rlm_completion_response",
            "id": request_id,
            "content": "",
            "error": None,
        }
        try:
            response["content"] = self._complete_for_sandbox(request)
        except Exception as exc:
            response["error"] = str(exc)

        self._process.stdin.write(json.dumps(response, sort_keys=True) + "\n")
        self._process.stdin.flush()

    def _service_llm_completion(self, request: dict) -> None:
        if not self._process or not self._process.stdin:
            raise SandboxError("sandbox process has no stdin")
        request_id = str(request.get("id", ""))
        response = {
            "type": "llm_completion_response",
            "id": request_id,
            "content": "",
            "error": None,
        }
        try:
            response["content"] = self._complete_for_sandbox(request, recursive=False)
        except Exception as exc:
            response["error"] = str(exc)
        self._process.stdin.write(json.dumps(response, sort_keys=True) + "\n")
        self._process.stdin.flush()

    def _complete_for_sandbox(self, request: dict, recursive: bool = True) -> str:
        if self.completion_client is None:
            raise SandboxError("rlm.completion is not enabled for this sandbox")

        depth_hint = int(request.get("depth_hint", -1))
        if depth_hint > self.subcall_config.max_depth:
            raise SandboxError(
                f"rlm.completion depth {depth_hint} exceeds max depth "
                f"{self.subcall_config.max_depth}"
            )
        if self._subcalls >= self.subcall_config.max_subcalls:
            raise SandboxError("rlm.completion subcall limit exceeded")

        query = str(request.get("query", ""))
        context = str(request.get("context", ""))
        if not query.strip():
            raise SandboxError("rlm.completion query must be non-empty")
        if len(query) > self.subcall_config.max_query_chars:
            raise SandboxError("rlm.completion query is too large")
        if len(context) > self.subcall_config.max_context_chars:
            raise SandboxError("rlm.completion context is too large")

        requested_max_tokens = request.get("max_tokens")
        max_tokens = self.subcall_config.max_tokens
        if requested_max_tokens is not None:
            max_tokens = min(max_tokens, int(requested_max_tokens))

        estimated_prompt_tokens = estimate_tokens(query) + estimate_tokens(context)
        projected_tokens = self._tokens_used + estimated_prompt_tokens + max_tokens
        if projected_tokens > self.subcall_config.token_budget:
            raise SandboxError("rlm.completion token budget exceeded")

        if recursive and self.recursive_completion is not None:
            model = request.get("model")
            model_name = str(model) if model else None
            recursive_result = self.recursive_completion(
                query,
                context,
                depth_hint,
                max_tokens,
                model_name,
            )
            if isinstance(recursive_result, RecursiveCompletionResult):
                self._subcalls += max(1, recursive_result.subcalls)
                # Recursive completions account for their model usage in the host runtime.
                # Keeping sandbox-side token accounting to direct LLM calls avoids double counts.
                return recursive_result.content

            self._subcalls += 1
            return str(recursive_result)

        model = request.get("model")
        model_name = str(model) if model else None
        messages = [
            Msg(
                role="system",
                content=(
                    "You are a recursive sub-call inside an RLM sandbox. "
                    "Answer the query using only the provided context."
                ),
            ),
            Msg(role="user", content=f"Query:\n{query}\n\nContext:\n{context}"),
        ]
        old_model = self.completion_client.model
        if model_name:
            self.completion_client.model = model_name
        try:
            completion = self.completion_client.complete(
                messages,
                max_tokens=max_tokens,
                temperature=self.subcall_config.temperature,
            )
        except LMClientError as exc:
            raise SandboxError(f"rlm.completion model request failed: {exc}") from exc
        finally:
            self.completion_client.model = old_model

        self._subcalls += 1
        completion_tokens = completion.completion_tokens or estimate_tokens(completion.content)
        prompt_tokens = completion.prompt_tokens or estimated_prompt_tokens
        self._tokens_used += prompt_tokens + completion_tokens
        return completion.content


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
