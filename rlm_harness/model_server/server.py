from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional


class MLXServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class MLXServerConfig:
    model: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    host: str = "127.0.0.1"
    port: int = 8080
    executable: str = "mlx_lm.server"
    extra_args: Sequence[str] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> list[str]:
        return [
            self.executable,
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]


class MLXServer:
    def __init__(self, config: MLXServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None

    def start(
        self,
        wait: bool = True,
        timeout_s: float = 60,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if self.process and self.process.poll() is None:
            return

        if shutil.which(self.config.executable) is None:
            raise MLXServerError(f"executable not found on PATH: {self.config.executable}")

        self.process = subprocess.Popen(
            self.config.command(),
            env=dict(env) if env is not None else None,
            text=True,
        )

        if wait:
            self.wait_until_ready(timeout_s=timeout_s)

    def is_ready(self, timeout_s: float = 2) -> bool:
        request = urllib.request.Request(self.config.base_url.rstrip("/") + "/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= response.status < 500
        except urllib.error.URLError:
            return False

    def wait_until_ready(self, timeout_s: float = 60, poll_s: float = 0.5) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise MLXServerError(f"mlx server exited with code {self.process.returncode}")
            if self.is_ready(timeout_s=min(poll_s, 2)):
                return
            time.sleep(poll_s)
        raise MLXServerError(f"mlx server did not become ready within {timeout_s}s")

    def wait(self) -> int:
        if not self.process:
            return 0
        return self.process.wait()

    def stop(self, timeout_s: float = 10) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def __enter__(self) -> MLXServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
