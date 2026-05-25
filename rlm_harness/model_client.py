from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from rlm_harness.observability import maybe_traceable
from rlm_harness.types import Completion, Msg

USER_AGENT = "rlm-harness/0.1 (+https://github.com/anirudh5harma/rlm-harness)"


class LMClientError(RuntimeError):
    pass


@dataclass
class LMClient:
    provider: str = "stub"
    model: str = "stub"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: Optional[str] = None
    timeout_s: int = 120

    @maybe_traceable("LMClient.complete", run_type="llm")
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
        if "the context is available in the repl" in lower:
            content = self._stub_rlm_response(user_text)
        elif "return a concise numbered plan" in lower:
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

    def _stub_rlm_response(self, user_text: str) -> str:
        lower = user_text.lower()
        if is_project_summary_prompt(lower):
            code = (
                "summary = project_summary()\n"
                "answer['content'] = summary\n"
                "answer['ready'] = True\n"
                "print(summary)"
            )
        elif "list files" in lower or "list the files" in lower:
            code = (
                "overview = project_overview()\n"
                "print('\\n'.join(overview['files']))\n"
                "answer['content'] = '\\n'.join(overview['files'])\n"
                "answer['ready'] = True"
            )
        elif "fix failing test" in lower or "fix the failing test" in lower:
            code = (
                "content = read_file('mathlib.py')\n"
                "if 'return a - b' in content:\n"
                "    write_file('mathlib.py', content.replace('return a - b', 'return a + b'))\n"
                "result = run_shell('python -m unittest', timeout=30)\n"
                "output = result['stdout'] + result['stderr']\n"
                "print(output, end='')\n"
                "if result['returncode'] != 0:\n"
                "    raise RuntimeError('tests failed')\n"
                "answer['content'] = output\n"
                "answer['ready'] = True"
            )
        elif "summarize" in lower or "summary" in lower or "explain" in lower:
            code = (
                "overview = project_overview()\n"
                "answer['content'] = str(overview)\n"
                "answer['ready'] = True\n"
                "print(answer['content'])"
            )
        else:
            escaped = user_text.replace("\\", "\\\\").replace("'", "\\'")
            code = (
                f"answer['content'] = 'Stub RLM completed task: {escaped[:160]}'; "
                "answer['ready'] = True; "
                "print(answer['content'])"
            )
        return f"```repl\n{code}\n```"

    def _stub_action(self, user_text: str) -> str:
        lower = user_text.lower()
        if is_project_summary_prompt(lower):
            code = "print(project_summary())"
        elif "list files" in lower or "list the files" in lower:
            code = (
                "result = run_shell('find . -maxdepth 1 -mindepth 1 | sort')\n"
                "print(result['stdout'].replace('./', ''), end='')\n"
                "if result['stderr']:\n"
                "    print(result['stderr'], end='')"
            )
        elif "fix failing test" in lower or "fix the failing test" in lower:
            code = (
                "content = read_file('mathlib.py')\n"
                "if 'return a - b' in content:\n"
                "    write_file('mathlib.py', content.replace('return a - b', 'return a + b'))\n"
                "result = run_shell('python -m unittest', timeout=30)\n"
                "print(result['stdout'], end='')\n"
                "print(result['stderr'], end='')\n"
                "if result['returncode'] != 0:\n"
                "    raise RuntimeError('tests failed')"
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
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
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
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_detail(exc)
            message = f"model request failed: HTTP {exc.code} {exc.reason}"
            if detail:
                message = f"{message}: {detail}"
            raise LMClientError(message) from exc
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


def is_project_summary_prompt(lowered_user_text: str) -> bool:
    return (
        bool(
            re.search(
                r"\b(project|repo|repository|codebase|workspace|application|app)\b",
                lowered_user_text,
            )
        )
        and any(
            intent in lowered_user_text
            for intent in (
                "what is",
                "what's",
                "tell me about",
                "summarize",
                "summary",
                "overview",
                "explain",
                "describe",
            )
        )
    )


def _read_http_error_detail(exc: urllib.error.HTTPError, limit: int = 500) -> str:
    try:
        body = exc.read(limit + 1).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    if len(body) > limit:
        body = body[:limit].rstrip() + "..."
    if not body:
        return ""
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return " ".join(body.split())
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict):
            for key in ("message", "detail", "error"):
                value = error.get(key)
                if value:
                    return str(value)
        elif error:
            return str(error)
        for key in ("message", "detail"):
            value = decoded.get(key)
            if value:
                return str(value)
    return " ".join(body.split())
