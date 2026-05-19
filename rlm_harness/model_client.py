from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from rlm_harness.types import Completion, Msg


class LMClientError(RuntimeError):
    pass


@dataclass
class LMClient:
    provider: str = "stub"
    model: str = "stub"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: Optional[str] = None
    timeout_s: int = 120

    def complete(
        self,
        messages: Iterable[Msg],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Completion:
        started = time.perf_counter()
        messages_list = list(messages)

        if self.provider == "stub":
            return self._stub_complete(messages_list, started)

        if self.provider != "openai-compatible":
            raise LMClientError(f"unknown provider: {self.provider}")

        return self._openai_compatible_complete(
            messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            started=started,
        )

    def _stub_complete(self, messages: list[Msg], started: float) -> Completion:
        user_text = ""
        for message in reversed(messages):
            if message.role == "user":
                user_text = message.content
                break

        lower = user_text.lower()
        if "return a concise numbered plan" in lower:
            content = "1. Inspect the task.\n2. Produce a concise response.\n3. Record the result."
        elif lower.lstrip().startswith("summarize older harness history."):
            content = self._stub_summary(user_text)
        elif "task:" in lower and "return only valid json" in lower:
            content = self._stub_action(user_text)
        elif "decide whether the task is complete" in lower:
            content = "done"
        else:
            content = f"Stub response for task: {user_text.strip()}"

        return Completion(
            content=content,
            model=self.model,
            provider=self.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _stub_action(self, user_text: str) -> str:
        lower = user_text.lower()
        if "list files" in lower or "list the files" in lower:
            code = (
                "from pathlib import Path\n"
                "for path in sorted(Path('/workspace').iterdir(), key=lambda p: p.name):\n"
                "    print(path.name)"
            )
        elif "rlm.completion" in lower or "recursive" in lower or "sub-call" in lower:
            code = (
                "answer = rlm.completion('Summarize the provided context.', "
                "'sandbox recursive subcall context', depth_hint=1)\n"
                "print(answer)"
            )
        else:
            escaped = user_text.replace("\\", "\\\\").replace("'", "\\'")
            code = f"print('Stub action completed for task: {escaped[:200]}')"
        return json.dumps({"type": "python", "code": code}, sort_keys=True)

    @staticmethod
    def _stub_summary(user_text: str) -> str:
        compact = " ".join(user_text.strip().split())
        return f"Archived harness history summary: {compact[:500]}"

    def _openai_compatible_complete(
        self,
        messages: list[Msg],
        max_tokens: int,
        temperature: float,
        started: float,
    ) -> Completion:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LMClientError(f"model request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LMClientError("model response was not valid JSON") from exc

        try:
            content = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LMClientError(
                "model response did not include choices[0].message.content"
            ) from exc

        usage = raw.get("usage") or {}
        return Completion(
            content=content,
            model=raw.get("model", self.model),
            provider=self.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=raw,
        )
