from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Optional

from rlm_harness.graph.task_policy import normalize_task_text
from rlm_harness.observability import maybe_traceable
from rlm_harness.types import Completion, Msg, TokenEvent

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
    # Retry the streaming call on transient provider errors
    # (HTTP 408/425/429/500/502/503/504/522/524 or URLError). The
    # default is 3 attempts with exponential backoff
    # (0.5s, 1.0s, 2.0s) so a brief provider blip does not
    # abort a long-running task. Retries are only attempted
    # before any token has been yielded; once the model
    # starts streaming, retrying would duplicate output.
    max_stream_retries: int = 3
    stream_retry_base_delay_s: float = 0.5

    @maybe_traceable("LMClient.complete", run_type="llm")
    def complete(
        self,
        messages: Iterable[Msg],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Completion:
        """Non-streaming completion.

        Kept as a separate implementation from `stream()` because the
        non-streaming path has battle-tested retry logic (the
        `response_format` fallback) that is awkward to express in a
        streaming generator. `stream()` is the new path used by the
        RLM runtime; both share payload construction and SSE
        helpers.
        """
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

    def stream(
        self,
        messages: Iterable[Msg],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[TokenEvent]:
        """Stream a completion as a sequence of `TokenEvent`s.

        Always yields one `start` event first, then 0+ `delta` events,
        then exactly one `finish` (or `error`) event last. The
        non-streaming `complete()` is implemented in terms of this.
        """
        messages_list = list(messages)
        if self.provider == "stub":
            yield from self._stub_stream(messages_list)
            return
        yield from self._openai_compatible_stream(
            messages_list, max_tokens=max_tokens, temperature=temperature
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
        elif "return one typed action json object" in lower:
            content = self._stub_typed_action(user_text)
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
        task_text = extract_labeled_section(user_text, "Query") or user_text
        lower = task_text.lower()
        if "list files" in lower or "list the files" in lower:
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
                "print('Changed files: mathlib.py')\n"
                "print('Verification: python -m unittest')\n"
                "print(output, end='')\n"
                "if result['returncode'] != 0:\n"
                "    raise RuntimeError('tests failed')\n"
                "answer['content'] = 'Changed files: mathlib.py\\nVerification: "
                "python -m unittest\\n' + output\n"
                "answer['ready'] = True"
            )
        elif is_project_audit_prompt(lower):
            code = (
                "audit = project_audit()\n"
                "answer['content'] = audit\n"
                "answer['ready'] = True\n"
                "print(audit)"
            )
        elif is_project_summary_prompt(lower):
            code = (
                "summary = project_summary()\n"
                "answer['content'] = summary\n"
                "answer['ready'] = True\n"
                "print(summary)"
            )
        elif "summarize" in lower or "summary" in lower or "explain" in lower:
            code = (
                "overview = project_overview()\n"
                "answer['content'] = str(overview)\n"
                "answer['ready'] = True\n"
                "print(answer['content'])"
            )
        else:
            content = f"Stub RLM completed task: {user_text[:160]}"
            code = (
                f"answer['content'] = {content!r}; "
                "answer['ready'] = True; "
                "print(answer['content'])"
            )
        return f"```repl\n{code}\n```"

    def _stub_action(self, user_text: str) -> str:
        lower = user_text.lower()
        if is_project_audit_prompt(lower):
            code = (
                "baseline = project_audit()\n"
                "try:\n"
                "    analysis = rlm.completion(\n"
                "        'Find logical and technical gaps in this project. Use the baseline "
                "audit as evidence and return concise findings with recommendations.',\n"
                "        baseline,\n"
                "        depth_hint=1,\n"
                "    )\n"
                "    print(analysis if analysis.strip() else baseline)\n"
                "except Exception:\n"
                "    print(baseline)"
            )
        elif is_project_summary_prompt(lower):
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
            content = f"Stub action completed for task: {user_text[:200]}"
            code = f"print({content!r})"
        return json.dumps({"type": "python", "code": code}, sort_keys=True)

    def _stub_typed_action(self, user_text: str) -> str:
        lower = user_text.lower()
        if is_project_audit_prompt(lower):
            payload = {"kind": "project_audit"}
        elif is_project_summary_prompt(lower):
            payload = {"kind": "project_summary"}
        elif "fix failing test" in lower or "fix the failing test" in lower:
            if "patch applied" in lower or "return a + b" in lower:
                payload = {
                    "kind": "complete_task",
                    "summary": (
                        "Changed files: mathlib.py\n"
                        "Verification: python -m unittest\n"
                        "OK"
                    ),
                    "status": "success",
                    "verification": "python -m unittest",
                }
            else:
                payload = {
                    "kind": "apply_patch",
                    "diff": (
                        "diff --git a/mathlib.py b/mathlib.py\n"
                        "--- a/mathlib.py\n"
                        "+++ b/mathlib.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def add(a, b):\n"
                        "-    return a - b\n"
                        "+    return a + b\n"
                    ),
                    "reason": "Fix the failing addition implementation.",
                }
        elif "list files" in lower or "list the files" in lower:
            payload = {"kind": "list_files", "path": "."}
        else:
            payload = {
                "kind": "complete_task",
                "summary": f"Stub response for task: {user_text[:200]}",
                "status": "success",
            }
        return json.dumps(payload, sort_keys=True)

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
        if should_request_json_response(messages):
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            raw = post_openai_compatible_chat(url, payload, headers, self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_detail(exc)
            if should_retry_without_response_format(exc.code, detail, payload):
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                try:
                    raw = post_openai_compatible_chat(
                        url,
                        fallback_payload,
                        headers,
                        self.timeout_s,
                    )
                except urllib.error.HTTPError as retry_exc:
                    retry_detail = _read_http_error_detail(retry_exc)
                    message = (
                        f"model request failed: HTTP {retry_exc.code} "
                        f"{retry_exc.reason}"
                    )
                    if retry_detail:
                        message = f"{message}: {retry_detail}"
                    raise LMClientError(message) from retry_exc
            else:
                message = f"model request failed: HTTP {exc.code} {exc.reason}"
                if detail:
                    message = f"{message}: {detail}"
                raise LMClientError(message) from exc
        except json.JSONDecodeError as exc:
            raise LMClientError("model response was not valid JSON") from exc
        except urllib.error.URLError as exc:
            raise LMClientError(f"model request failed: {exc}") from exc

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

    def _stub_stream(self, messages: list[Msg]) -> Iterator[TokenEvent]:
        """Stub streaming path. Yields one `delta` containing the full
        stub response. Kept simple on purpose: the stub is a test
        fixture, not a streaming benchmark.
        """
        yield TokenEvent(type="start", model=self.model, provider=self.provider)
        completion = self._stub_complete(messages, time.perf_counter())
        yield TokenEvent(
            type="delta",
            delta=completion.content,
            model=completion.model,
            provider=completion.provider,
        )
        yield TokenEvent(
            type="finish",
            model=completion.model,
            provider=completion.provider,
            usage={
                "prompt_tokens": completion.prompt_tokens or 0,
                "completion_tokens": completion.completion_tokens or 0,
            },
            finish_reason="stop",
        )

    def _openai_compatible_stream(
        self,
        messages: list[Msg],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[TokenEvent]:
        """Streaming path for OpenAI-compatible `/chat/completions`.

        Sends `stream: true`, parses Server-Sent Events of the form
        `data: {json}\n\n`, and yields start / delta / finish events.
        Yields exactly one `error` event on failure (the caller in
        `complete()` translates that to `LMClientError`).
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if should_request_json_response(messages):
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        yield TokenEvent(type="start", model=self.model, provider=self.provider)
        # Retry transient provider errors before any token is
        # yielded. Once `urlopen` returns and the first SSE frame
        # has been read, we commit to the response — duplicating
        # a half-streamed answer would be worse than failing the
        # turn, and the supervisor can drive the next iteration.
        max_attempts = max(1, int(self.max_stream_retries or 1))
        for attempt in range(1, max_attempts + 1):
            try:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    for event in self._iter_sse_events(response):
                        if event is None:
                            # stream terminator (`data: [DONE]`)
                            break
                        kind = event.get("type", "delta")
                        if kind == "error":
                            yield TokenEvent(
                                type="error",
                                error=str(event.get("error") or "stream error"),
                                model=event.get("model", self.model),
                                provider=self.provider,
                            )
                            return
                        if kind == "finish":
                            yield TokenEvent(
                                type="finish",
                                model=event.get("model") or self.model,
                                provider=self.provider,
                                usage=event.get("usage", {}),
                                finish_reason=event.get("finish_reason"),
                            )
                            return
                        if kind == "delta":
                            yield TokenEvent(
                                type="delta",
                                delta=str(event.get("delta") or ""),
                                model=event.get("model", self.model),
                                provider=self.provider,
                            )
                # Provider closed the stream without raising —
                # fall through to the synthetic `finish` event
                # below.
                break
            except urllib.error.HTTPError as exc:
                detail = _read_http_error_detail(exc)
                message = f"model stream failed: HTTP {exc.code} {exc.reason}"
                if detail:
                    message = f"{message}: {detail}"
                if (
                    _is_transient_status(exc.code)
                    and attempt < max_attempts
                ):
                    _sleep_before_retry(
                        attempt, self.stream_retry_base_delay_s
                    )
                    continue
                if _is_transient_status(exc.code):
                    message = (
                        f"{message} (after {attempt} attempts)"
                    )
                yield TokenEvent(
                    type="error", error=message, provider=self.provider
                )
                return
            except urllib.error.URLError as exc:
                if attempt < max_attempts:
                    _sleep_before_retry(
                        attempt, self.stream_retry_base_delay_s
                    )
                    continue
                yield TokenEvent(
                    type="error",
                    error=(
                        f"model stream failed: {exc} "
                        f"(after {attempt} attempts)"
                    ),
                    provider=self.provider,
                )
                return
            except (json.JSONDecodeError, TimeoutError) as exc:
                # Transport-level errors that arrive mid-stream
                # are not retried — we may have already yielded
                # a partial response to the supervisor.
                yield TokenEvent(
                    type="error", error=str(exc), provider=self.provider
                )
                return

        # Provider closed the stream without an explicit finish event.
        # Emit one so the contract (`start` then `finish`) is preserved.
        yield TokenEvent(
            type="finish",
            model=self.model,
            provider=self.provider,
            usage={},
            finish_reason="stop",
        )

    @staticmethod
    def _iter_sse_events(response) -> Iterator[Optional[dict]]:
        """Yield one dict per `data: ...` SSE frame from a streaming
        HTTP response. `None` indicates the `[DONE]` terminator.

        Lines are read in small chunks so that the body of a large
        response does not have to be buffered all at once. Newline
        separators inside a JSON payload are not allowed by SSE, so a
        `\n` boundary is safe.
        """
        buffer = ""
        for raw_chunk in iter(lambda: response.read(4096).decode("utf-8", "replace"), ""):
            buffer += raw_chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for line in frame.splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        yield None
                        return
                    if not payload:
                        continue
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    yield _sse_to_token_event(parsed)


def _sse_to_token_event(payload: dict) -> dict:
    """Translate an OpenAI streaming chunk into our internal shape.

    The OpenAI streaming schema is the de-facto standard for
    OpenAI-compatible providers; we accept the schema as-is and only
    normalise the bits we care about.
    """
    if not isinstance(payload, dict):
        return {"type": "delta", "delta": ""}
    choices = payload.get("choices") or []
    if not choices:
        # Some providers send `usage` in a final chunk with no choices.
        if "usage" in payload:
            return {
                "type": "finish",
                "usage": payload.get("usage", {}),
                "model": payload.get("model"),
            }
        return {"type": "delta", "delta": ""}
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}
    text = ""
    if isinstance(delta, dict):
        text = str(delta.get("content") or "")
    elif isinstance(delta, str):
        text = delta
    finish_reason = choice.get("finish_reason")
    out: dict = {"type": "delta", "delta": text}
    if finish_reason or "usage" in payload:
        out["type"] = "finish"
        out["finish_reason"] = finish_reason
        out["usage"] = payload.get("usage", {})
        out["model"] = payload.get("model")
    return out


def should_request_json_response(messages: list[Msg]) -> bool:
    content = "\n".join(message.content for message in messages).lower()
    return any(
        phrase in content
        for phrase in (
            "return one typed action json object",
            "return only valid json for this action",
            "return exactly one json object",
            "the schema is:",
        )
    )


def should_retry_without_response_format(
    status_code: int,
    detail: str,
    payload: dict,
) -> bool:
    if "response_format" not in payload or status_code not in {400, 422}:
        return False
    lowered = detail.lower()
    return any(
        hint in lowered
        for hint in (
            "response_format",
            "json_object",
            "unsupported parameter",
            "unrecognized request argument",
            "not supported",
        )
    )


# HTTP statuses that are commonly transient — the provider is
# overloaded, rate-limiting, or temporarily unreachable. A short
# backoff almost always succeeds.
TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 522, 524}
)


def _is_transient_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def _sleep_before_retry(attempt: int, base_delay_s: float) -> None:
    """Exponential backoff: ``base * 2 ** (attempt - 1)``.

    Capped at 8 seconds so a long retry chain does not stall the
    CLI for a minute. ``time.sleep`` is a no-op when ``base_delay_s``
    is zero or negative, which lets tests disable the backoff.
    """
    if base_delay_s <= 0:
        return
    delay = min(8.0, base_delay_s * (2 ** (attempt - 1)))
    time.sleep(delay)


def post_openai_compatible_chat(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_s: int,
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = json.loads(response.read().decode("utf-8"))
    if not isinstance(raw, dict):
        raise LMClientError("model response was not a JSON object")
    return raw


def is_project_summary_prompt(lowered_user_text: str) -> bool:
    lowered_user_text = normalize_task_text(lowered_user_text)
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


def is_project_audit_prompt(lowered_user_text: str) -> bool:
    lowered_user_text = normalize_task_text(lowered_user_text)
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
                "gap",
                "gaps",
                "risk",
                "risks",
                "issue",
                "issues",
                "problem",
                "problems",
                "bug",
                "bugs",
                "flaw",
                "flaws",
                "weakness",
                "weaknesses",
                "technical debt",
                "logical",
                "audit",
                "review",
                "critique",
                "evaluate",
                "assess",
                "find any",
                "identify",
                "what must be done",
                "what should be done",
                "what to do next",
                "next step",
                "next steps",
                "done next",
            )
        )
    )


def extract_labeled_section(text: str, label: str) -> str:
    pattern = rf"(?is)\b{re.escape(label)}:\s*(.*?)(?:\n\s*\n|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


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
