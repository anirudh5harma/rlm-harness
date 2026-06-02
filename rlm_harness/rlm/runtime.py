from __future__ import annotations

import contextlib
import io
import json
import re
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import (
    DockerREPL,
    RecursiveCompletionResult,
    RLMSubcallConfig,
    SandboxConfig,
    SandboxError,
)
from rlm_harness.sandbox.types import ExecutionResult
from rlm_harness.types import Msg

REPL_BLOCK_RE = re.compile(
    r"```(?:repl|python)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
MAX_OBSERVATION_CODE_CHARS = 8_000
MAX_OBSERVATION_STREAM_CHARS = 12_000
CONTEXT_PREVIEW_CHARS = 2000


@dataclass
class RLMObservation:
    code: str
    stdout: str = ""
    stderr: str = ""
    status: str = "ok"
    timed_out: bool = False
    elapsed_ms: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass
class RLMResult:
    final_answer: str
    status: str
    iterations: int
    observations: list[RLMObservation] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    subcalls: int = 0
    tokens_used: int = 0


# --- Streaming turn events (Phase A.2) ---------------------------------
# The supervisor consumes these. They are intentionally narrow and
# append-only; the supervisor is the only writer of state, and the
# events are the audit trail it writes from.


@dataclass
class TurnStarted:
    """Emitted once at the start of a turn."""

    query: str
    context_preview: str
    iteration_limit: int


@dataclass
class IterationStarted:
    """Emitted before each model call (within a turn)."""

    iteration: int
    started_at: float


@dataclass
class TokenDelta:
    """One streamed token from the model."""

    delta: str


@dataclass
class IterationFinished:
    """Emitted after a model call. The full response is included so
    the supervisor can checkpoint without re-reading the stream.
    """

    iteration: int
    response: str
    repl_blocks: list[str]
    usage: dict
    latency_ms: int


@dataclass
class ObservationRecorded:
    """Emitted after each REPL block is executed."""

    observation: RLMObservation
    iteration: int


@dataclass
class SubcallStarted:
    """Emitted when a child llm_query / rlm_query starts."""

    kind: str  # "llm" or "rlm"
    prompt_preview: str
    depth: int


@dataclass
class SubcallFinished:
    """Emitted when a child call returns."""

    kind: str
    content: str
    tokens_used: int


@dataclass
class TurnFinished:
    """Emitted once at the end of a turn. Carries the final RLMResult."""

    result: RLMResult


RLMTurnEvent = Union[
    TurnStarted,
    IterationStarted,
    TokenDelta,
    IterationFinished,
    ObservationRecorded,
    SubcallStarted,
    SubcallFinished,
    TurnFinished,
]


class LocalRLMRepl:
    def __init__(self, runtime: RLMRuntime, context: Any):
        self.runtime = runtime
        self.namespace: dict[str, Any] = {
            "__name__": "__rlm_repl__",
            "context": context,
            "answer": {"content": "", "ready": False},
            "SHOW_VARS": self.show_vars,
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
            "rlm_query": self.rlm_query,
            "rlm_query_batched": self.rlm_query_batched,
            "complete_task": self.complete_task,
        }

    def execute(self, code: str) -> RLMObservation:
        stdout = io.StringIO()
        stderr = io.StringIO()
        status = "ok"
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(code, self.namespace, self.namespace)
            except BaseException:
                status = "error"
                traceback.print_exc(file=stderr)
        return RLMObservation(
            code=code,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            status=status,
        )

    @property
    def final_answer(self) -> Optional[str]:
        answer = self.namespace.get("answer")
        if isinstance(answer, dict) and answer.get("ready"):
            return str(answer.get("content", ""))
        return None

    def show_vars(self) -> str:
        ignored = {
            "__name__",
            "context",
            "answer",
            "SHOW_VARS",
            "llm_query",
            "llm_query_batched",
            "rlm_query",
            "rlm_query_batched",
        }
        visible = {
            k: type(v).__name__
            for k, v in self.namespace.items()
            if k not in ignored and not k.startswith("__")
        }
        return json.dumps(visible, sort_keys=True)

    def complete_task(
        self,
        summary: str,
        status: str = "success",
        verification: Optional[str] = None,
    ) -> dict[str, str | bool]:
        if status not in {"success", "partial", "blocked"}:
            raise ValueError("status must be success, partial, or blocked")
        answer = self.namespace["answer"]
        answer["content"] = summary
        answer["ready"] = True
        answer["status"] = status
        answer["verification"] = verification or ""
        return {
            "summary": summary,
            "status": status,
            "verification": verification or "",
            "should_continue": False,
        }

    def llm_query(
        self,
        prompt: str,
        model: Optional[str] = None,
        context: Any = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self.runtime.llm_query(prompt, context=context, model=model, max_tokens=max_tokens)

    def llm_query_batched(
        self,
        prompts: list[str],
        model: Optional[str] = None,
        context: Any = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        return [
            self.llm_query(prompt, model=model, context=context, max_tokens=max_tokens)
            for prompt in prompts
        ]

    def rlm_query(
        self,
        prompt: str,
        context: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self.runtime.rlm_query(prompt, context=context, model=model, max_tokens=max_tokens)

    def rlm_query_batched(
        self,
        prompts: list[str],
        context: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        return [
            self.rlm_query(prompt, context=context, model=model, max_tokens=max_tokens)
            for prompt in prompts
        ]


class RLMRuntime:
    def __init__(
        self,
        client: LMClient,
        workspace: Path,
        max_iterations: int = 8,
        max_depth: int = 3,
        depth: int = 0,
        sandbox_enabled: bool = True,
        sandbox_config: Optional[SandboxConfig] = None,
        subcall_config: Optional[RLMSubcallConfig] = None,
        max_tokens: int = 900,
    ):
        self.client = client
        self.workspace = Path(workspace)
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.depth = depth
        self.sandbox_enabled = sandbox_enabled
        self.sandbox_config = sandbox_config
        self.subcall_config = subcall_config or RLMSubcallConfig(max_depth=max_depth)
        self.max_tokens = max_tokens
        self.subcalls = 0
        self.tokens_used = 0

    def completion(self, query: str, context: Any = "") -> RLMResult:
        if self.sandbox_enabled:
            return self._completion_with_docker(query, context)
        return self._completion_with_local_repl(query, context)

    def stream_turn(
        self, query: str, context: Any = ""
    ) -> Iterator[RLMTurnEvent]:
        """Run one turn of the RLM runtime, yielding events as they happen.

        A *turn* is the unit of work the supervisor runs: start at the
        model, observe repl blocks, sub-call as needed, end at a final
        answer or at the iteration cap. The event stream is the audit
        trail the supervisor writes into the kernel trace and the
        memory store. The non-streaming `completion()` is preserved
        unchanged for callers that want the buffered final result.
        """
        preview = serialize_context(context)[:CONTEXT_PREVIEW_CHARS]
        yield TurnStarted(
            query=query,
            context_preview=preview,
            iteration_limit=self.max_iterations,
        )
        if self.sandbox_enabled:
            yield from self._stream_turn_with_docker(query, context)
        else:
            yield from self._stream_turn_with_local_repl(query, context)

    def _stream_turn_with_local_repl(
        self, query: str, context: Any
    ) -> Iterator[RLMTurnEvent]:
        repl = LocalRLMRepl(self, context)
        messages = self._initial_messages(query, context)
        responses: list[str] = []
        observations: list[RLMObservation] = []
        final: Optional[RLMResult] = None

        for iteration in range(1, self.max_iterations + 1):
            started_at = time.perf_counter()
            yield IterationStarted(iteration=iteration, started_at=started_at)
            response, usage = self._stream_model_call(messages)
            for delta in self._drain_message_chunks(response):
                yield TokenDelta(delta=delta)
            responses.append(response)
            messages.append(Msg(role="assistant", content=response))
            blocks = find_repl_blocks(response)
            yield IterationFinished(
                iteration=iteration,
                response=response,
                repl_blocks=list(blocks),
                usage=usage,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            if not blocks:
                final = RLMResult(
                    final_answer=response.strip(),
                    status="done",
                    iterations=iteration,
                    observations=observations,
                    responses=responses,
                    subcalls=self.subcalls,
                    tokens_used=self.tokens_used,
                )
                break
            for block in blocks:
                observation = repl.execute(block)
                observations.append(observation)
                yield ObservationRecorded(observation=observation, iteration=iteration)
                messages.append(
                    Msg(role="user", content=format_observation(observation))
                )
                if repl.final_answer is not None:
                    final = RLMResult(
                        final_answer=repl.final_answer,
                        status="done",
                        iterations=iteration,
                        observations=observations,
                        responses=responses,
                        subcalls=self.subcalls,
                        tokens_used=self.tokens_used,
                    )
                    break
            if final is not None:
                break

        if final is None:
            final = RLMResult(
                final_answer=stopped_final_answer(responses, observations),
                status="stopped",
                iterations=self.max_iterations,
                observations=observations,
                responses=responses,
                subcalls=self.subcalls,
                tokens_used=self.tokens_used,
            )
        yield TurnFinished(result=final)

    def _stream_turn_with_docker(
        self, query: str, context: Any
    ) -> Iterator[RLMTurnEvent]:
        sandbox_config = self.sandbox_config or SandboxConfig(workspace=self.workspace)
        bootstrap = build_bootstrap_code(context)
        messages = self._initial_messages(query, context)
        responses: list[str] = []
        observations: list[RLMObservation] = []
        final: Optional[RLMResult] = None

        try:
            with DockerREPL(
                sandbox_config,
                completion_client=self.client,
                subcall_config=self.subcall_config,
                recursive_completion=self._recursive_completion_from_sandbox,
            ) as repl:
                repl.execute(bootstrap, timeout_s=sandbox_config.default_timeout_s)
                for iteration in range(1, self.max_iterations + 1):
                    started_at = time.perf_counter()
                    yield IterationStarted(iteration=iteration, started_at=started_at)
                    response, usage = self._stream_model_call(messages)
                    for delta in self._drain_message_chunks(response):
                        yield TokenDelta(delta=delta)
                    responses.append(response)
                    messages.append(Msg(role="assistant", content=response))
                    blocks = find_repl_blocks(response)
                    yield IterationFinished(
                        iteration=iteration,
                        response=response,
                        repl_blocks=list(blocks),
                        usage=usage,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                    )
                    if not blocks:
                        final = RLMResult(
                            response.strip(),
                            "done",
                            iteration,
                            observations,
                            responses,
                            self.subcalls,
                            self.tokens_used,
                        )
                        break
                    for block in blocks:
                        result = repl.execute(
                            block, timeout_s=sandbox_config.default_timeout_s
                        )
                        observation = observation_from_execution(block, result)
                        observations.append(observation)
                        yield ObservationRecorded(
                            observation=observation, iteration=iteration
                        )
                        messages.append(
                            Msg(role="user", content=format_observation(observation))
                        )
                        answer = extract_answer_ready(result.stdout)
                        if answer is not None:
                            final = RLMResult(
                                answer,
                                "done",
                                iteration,
                                observations,
                                responses,
                                self.subcalls + result.subcalls,
                                self.tokens_used + result.tokens_used,
                            )
                            break
                    if final is not None:
                        break
        except SandboxError as exc:
            observations.append(
                RLMObservation(code="", stderr=str(exc), status="sandbox_error")
            )
            final = RLMResult(
                str(exc),
                "error",
                len(responses),
                observations,
                responses,
                self.subcalls,
                self.tokens_used,
            )

        if final is None:
            final = RLMResult(
                stopped_final_answer(responses, observations),
                "stopped",
                self.max_iterations,
                observations,
                responses,
                self.subcalls,
                self.tokens_used,
            )
        yield TurnFinished(result=final)

    def _stream_model_call(self, messages: list[Msg]) -> tuple[str, dict]:
        """Stream one model call. Returns the full response + usage.

        Uses `client.stream` to deliver incremental token events, and
        the `maybe_record_usage` helper to keep the runtime's
        accounting consistent. Sub-calls go through `llm_query` /
        `rlm_query` which already use the non-streaming path; the
        streaming entry point is the top-level model call only.
        """
        chunks: list[str] = []
        usage: dict = {}
        error: Optional[str] = None
        for event in self.client.stream(
            messages, max_tokens=self.max_tokens, temperature=0.1
        ):
            if event.type == "delta":
                chunks.append(event.delta)
            elif event.type == "finish":
                usage = event.usage or usage
            elif event.type == "error":
                error = event.error or "stream error"
                break
        if error is not None:
            # Surface as a sandbox-style error so the supervisor can
            # classify and recover; do not raise (callers are the
            # non-streaming `completion()` and the streaming
            # `stream_turn()`, neither of which is allowed to raise).
            return f"__rlm_stream_error__:{error}", usage
        response = "".join(chunks)
        self._add_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return response, usage

    @staticmethod
    def _drain_message_chunks(response: str) -> list[str]:
        """Split a buffered response into per-token deltas.

        A real model emits a stream of small deltas; a stub or
        non-streaming call returns the whole message at once. To keep
        the event shape uniform (one `TokenDelta` per chunk), the
        helper tokenises the message on whitespace boundaries so the
        caller can rebuild the response by concatenation.
        """
        if not response:
            return []
        # Treat each whitespace-separated token as one delta. This is
        # a coarse approximation: it is not byte-faithful, but it is
        # good enough for the supervisor to estimate latency and to
        # checkpoint at a per-iteration boundary.
        return [chunk for chunk in re.split(r"(\s+)", response) if chunk]

    def _completion_with_local_repl(self, query: str, context: Any) -> RLMResult:
        repl = LocalRLMRepl(self, context)
        messages = self._initial_messages(query, context)
        responses: list[str] = []
        observations: list[RLMObservation] = []

        for iteration in range(1, self.max_iterations + 1):
            completion = self.client.complete(messages, max_tokens=self.max_tokens, temperature=0.1)
            self._add_usage(completion.prompt_tokens, completion.completion_tokens)
            response = completion.content
            responses.append(response)
            messages.append(Msg(role="assistant", content=response))
            blocks = find_repl_blocks(response)
            if not blocks:
                return RLMResult(
                    final_answer=response.strip(),
                    status="done",
                    iterations=iteration,
                    observations=observations,
                    responses=responses,
                    subcalls=self.subcalls,
                    tokens_used=self.tokens_used,
                )
            for block in blocks:
                observation = repl.execute(block)
                observations.append(observation)
                messages.append(Msg(role="user", content=format_observation(observation)))
                final_answer = repl.final_answer
                if final_answer is not None:
                    return RLMResult(
                        final_answer=final_answer,
                        status="done",
                        iterations=iteration,
                        observations=observations,
                        responses=responses,
                        subcalls=self.subcalls,
                        tokens_used=self.tokens_used,
                    )
        return RLMResult(
            final_answer=stopped_final_answer(responses, observations),
            status="stopped",
            iterations=self.max_iterations,
            observations=observations,
            responses=responses,
            subcalls=self.subcalls,
            tokens_used=self.tokens_used,
        )

    def _completion_with_docker(self, query: str, context: Any) -> RLMResult:
        sandbox_config = self.sandbox_config or SandboxConfig(workspace=self.workspace)
        bootstrap = build_bootstrap_code(context)
        messages = self._initial_messages(query, context)
        responses: list[str] = []
        observations: list[RLMObservation] = []
        try:
            with DockerREPL(
                sandbox_config,
                completion_client=self.client,
                subcall_config=self.subcall_config,
                recursive_completion=self._recursive_completion_from_sandbox,
            ) as repl:
                repl.execute(bootstrap, timeout_s=sandbox_config.default_timeout_s)
                for iteration in range(1, self.max_iterations + 1):
                    completion = self.client.complete(
                        messages, max_tokens=self.max_tokens, temperature=0.1
                    )
                    self._add_usage(completion.prompt_tokens, completion.completion_tokens)
                    response = completion.content
                    responses.append(response)
                    messages.append(Msg(role="assistant", content=response))
                    blocks = find_repl_blocks(response)
                    if not blocks:
                        return RLMResult(
                            response.strip(),
                            "done",
                            iteration,
                            observations,
                            responses,
                            self.subcalls,
                            self.tokens_used,
                        )
                    for block in blocks:
                        result = repl.execute(block, timeout_s=sandbox_config.default_timeout_s)
                        observation = observation_from_execution(block, result)
                        observations.append(observation)
                        messages.append(Msg(role="user", content=format_observation(observation)))
                        answer = extract_answer_ready(result.stdout)
                        if answer is not None:
                            return RLMResult(
                                answer,
                                "done",
                                iteration,
                                observations,
                                responses,
                                self.subcalls + result.subcalls,
                                self.tokens_used + result.tokens_used,
                            )
        except SandboxError as exc:
            observations.append(RLMObservation(code="", stderr=str(exc), status="sandbox_error"))
            return RLMResult(
                str(exc),
                "error",
                len(responses),
                observations,
                responses,
                self.subcalls,
                self.tokens_used,
            )
        return RLMResult(
            stopped_final_answer(responses, observations),
            "stopped",
            self.max_iterations,
            observations,
            responses,
            self.subcalls,
            self.tokens_used,
        )

    def _recursive_completion_from_sandbox(
        self,
        query: str,
        context: str,
        depth_hint: int,
        max_tokens: Optional[int],
        model: Optional[str] = None,
    ) -> RecursiveCompletionResult:
        subcalls_before = self.subcalls
        tokens_before = self.tokens_used
        if depth_hint >= 0 and depth_hint >= self.max_depth:
            content = self.llm_query(query, context=context, model=model, max_tokens=max_tokens)
        else:
            content = self.rlm_query(query, context=context, model=model, max_tokens=max_tokens)
        return RecursiveCompletionResult(
            content=content,
            subcalls=max(1, self.subcalls - subcalls_before),
            tokens_used=max(0, self.tokens_used - tokens_before),
        )

    def llm_query(
        self,
        prompt: str,
        context: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.subcalls >= self.subcall_config.max_subcalls:
            raise RuntimeError("subcall limit exceeded")
        self.subcalls += 1
        content = f"Query:\n{prompt}"
        if context is not None:
            content += f"\n\nContext:\n{serialize_context(context)}"
        messages = [
            Msg(
                role="system",
                content=(
                    "Answer the query using only the provided context "
                    "when context is provided."
                ),
            ),
            Msg(role="user", content=content),
        ]
        old_model = self.client.model
        if model:
            self.client.model = model
        try:
            completion = self.client.complete(
                messages,
                max_tokens=max_tokens or self.subcall_config.max_tokens,
                temperature=self.subcall_config.temperature,
            )
        finally:
            self.client.model = old_model
        self._add_usage(completion.prompt_tokens, completion.completion_tokens)
        return completion.content

    def rlm_query(
        self,
        prompt: str,
        context: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.depth + 1 > self.max_depth:
            return self.llm_query(prompt, context=context, model=model, max_tokens=max_tokens)
        if self.subcalls >= self.subcall_config.max_subcalls:
            raise RuntimeError("subcall limit exceeded")
        self.subcalls += 1
        child = RLMRuntime(
            self.client,
            workspace=self.workspace,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            sandbox_enabled=self.sandbox_enabled,
            sandbox_config=self.sandbox_config,
            subcall_config=self.subcall_config,
            max_tokens=max_tokens or self.max_tokens,
        )
        result = child.completion(prompt, context=context or "")
        self.subcalls += result.subcalls
        self.tokens_used += result.tokens_used
        return result.final_answer

    def _initial_messages(self, query: str, context: Any) -> list[Msg]:
        manifest = manifest_for_context(context)
        # The manifest is the long-context surface; it is bounded
        # by the budget in `context.manifest.build_manifest_for_doc`
        # (default 20k tokens). We do *not* truncate it here: a
        # half-manifest is worse than no manifest, and the
        # supervisor can always build a tighter one.
        manifest_text = serialize_context(manifest)
        return [
            Msg(role="system", content=RLM_SYSTEM_PROMPT),
            Msg(
                role="user",
                content=(
                    f"Query:\n{query}\n\n"
                    "The context is available in the REPL as variable `context`. "
                    f"Context manifest:\n{manifest_text}"
                ),
            ),
        ]

    def _add_usage(self, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
        self.tokens_used += int(prompt_tokens or 0) + int(completion_tokens or 0)


RLM_SYSTEM_PROMPT = """
You are an RLM runtime inside a coding harness. Solve the query by writing Python
in ```repl blocks.

The REPL contains:
- context
- answer
- llm_query / llm_query_batched
- rlm_query / rlm_query_batched
- complete_task(summary, status='success', verification=None)
- SHOW_VARS
- When running in the sandbox, workspace tools such as project_summary, project_audit,
  project_overview, list_files, read_file, read_file_slice, chunk_file, search_code,
  propose_file_change, list_pending_changes, apply_pending_change,
  clear_pending_changes, run_shell, and git_status

Inspect context programmatically. Set answer['content'] and answer['ready'] = True
when done, or call complete_task(summary, status, verification). If you have
enough information to answer after seeing observations, reply with the final
user-facing answer in plain text and do not include another ```repl block.
Write that final answer in friendly, ordinary English. Hide internal counts,
raw JSON, exhaustive file lists, and git noise unless the user asks for them.

In Docker, the workspace is mounted at /workspace. Do not use host absolute paths
such as /Users/... inside REPL code. Prefer the workspace tools with relative
paths like read_file('package.json') or search_code('pattern', 'src').

Use the REPL as your workspace, not as a dumping ground. Prefer targeted reads,
searches, summaries, and recursive calls over printing huge files. When the task
requires understanding many files or long text, split the material into chunks
with read_file_slice/chunk_file and ask llm_query/rlm_query focused sub-questions,
then synthesize and verify.
For code-editing tasks, infer the user's intent, inspect the relevant files before
editing, make the smallest correct change, run focused verification when possible,
and report the changed files plus verification result in a concise friendly note.
For risky edits, dependency changes, prompt/policy changes, or changes outside
the user's apparent request, use propose_file_change() and show the pending diff
instead of applying silently. Destructive shell commands are blocked unless the
user has explicitly approved them.

For project identity or overview questions such as "what is this project", call
project_summary() when it is available and return that summary. Do not answer by
printing raw source code or a raw file list.

For project review, audit, risk, issue, or gap-analysis questions, use
project_audit() as a baseline and inspect relevant source/config files before
returning evidence-backed findings. Do not answer these questions with only a
file inventory.
"""


def find_repl_blocks(text: str) -> list[str]:
    return [match.strip() for match in REPL_BLOCK_RE.findall(text)]


def format_observation(observation: RLMObservation) -> str:
    code, code_truncated = truncate_text(observation.code, MAX_OBSERVATION_CODE_CHARS)
    parts = [f"Code executed:\n```python\n{code}\n```", f"Status: {observation.status}"]
    if code_truncated:
        parts.append("Code note: truncated before returning to the model.")
    if observation.stdout:
        stdout, truncated = truncate_text(observation.stdout, MAX_OBSERVATION_STREAM_CHARS)
        parts.append("STDOUT:\n" + stdout)
        if truncated or observation.stdout_truncated:
            parts.append("STDOUT note: truncated before returning to the model.")
    if observation.stderr:
        stderr, truncated = truncate_text(observation.stderr, MAX_OBSERVATION_STREAM_CHARS)
        parts.append("STDERR:\n" + stderr)
        if truncated or observation.stderr_truncated:
            parts.append("STDERR note: truncated before returning to the model.")
    return "\n\n".join(parts)


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    marker = (
        f"\n\n... [truncated {omitted} chars; "
        "narrow the query or inspect a smaller slice] ...\n\n"
    )
    return (text[:head] + marker + text[-tail:], True)


def serialize_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True)
    except TypeError:
        return repr(context)


def manifest_for_context(context: Any) -> dict:
    """Return a small manifest describing `context` for the prompt.

    The manifest is what the prompt carries when the model is
    given a long-context working set. Today this is a thin
    passthrough that:

    * returns `context` if it is already a dict (legacy callers);
    * returns the manifest from a `ContextVar`-like object (has a
      `map()` method) — Phase B's long-context path;
    * returns `{"_raw": serialize_context(context)}` as a last
      resort for unknown objects.

    The supervisor (Phase B.4) builds the per-turn manifest from a
    `ContextVar` and stores it in `state.scratch["context_manifest"]`;
    the runtime consults that key when it is present.
    """
    if context is None:
        return {"_raw": ""}
    if isinstance(context, dict):
        return context
    # `ContextVar` (Phase B) exposes `map()` returning a manifest.
    map_method = getattr(context, "map", None)
    if callable(map_method):
        try:
            return dict(map_method())
        except Exception:
            pass
    return {"_raw": serialize_context(context)}


def build_bootstrap_code(context: Any) -> str:
    encoded = json.dumps(context if is_json_serializable(context) else serialize_context(context))
    return (
        "import json, os, subprocess\n"
        "from pathlib import Path\n"
        f"context = json.loads({json.dumps(encoded)})\n"
        "ctx = context\n"
        "answer = {'content': '', 'ready': False}\n"
        "def _emit_answer_if_ready():\n"
        "    if isinstance(answer, dict) and answer.get('ready'):\n"
        "        print('__RLM_FINAL_ANSWER__' + json.dumps(str(answer.get('content', ''))))\n"
        "def complete_task(summary, status='success', verification=None):\n"
        "    if status not in {'success', 'partial', 'blocked'}:\n"
        "        raise ValueError('status must be success, partial, or blocked')\n"
        "    answer['content'] = str(summary)\n"
        "    answer['ready'] = True\n"
        "    answer['status'] = status\n"
        "    answer['verification'] = verification or ''\n"
        "    return {'summary': str(summary), 'status': status, "
        "'verification': verification or '', 'should_continue': False}\n"
    )


def is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def observation_from_execution(code: str, result: ExecutionResult) -> RLMObservation:
    return RLMObservation(
        code=code,
        stdout=result.stdout,
        stderr=result.stderr,
        status=result.status,
        timed_out=result.timed_out,
        elapsed_ms=result.elapsed_ms,
    )


def extract_answer_ready(stdout: str) -> Optional[str]:
    marker = "__RLM_FINAL_ANSWER__"
    for line in stdout.splitlines():
        if line.startswith(marker):
            try:
                return str(json.loads(line[len(marker) :]))
            except json.JSONDecodeError:
                return line[len(marker) :]
    return None


def stopped_final_answer(responses: list[str], observations: list[RLMObservation]) -> str:
    for response in reversed(responses):
        text = remove_repl_blocks(response).strip()
        if text:
            return text

    for observation in reversed(observations):
        stdout = strip_answer_markers(observation.stdout).strip()
        if stdout:
            return stdout
        stderr = observation.stderr.strip()
        if stderr:
            return (
                "The RLM runtime stopped before producing a final answer. "
                f"Last error:\n{stderr}"
            )
    return "The RLM runtime stopped before producing a final answer."


def remove_repl_blocks(text: str) -> str:
    return REPL_BLOCK_RE.sub("", text)


def strip_answer_markers(stdout: str) -> str:
    lines = []
    marker = "__RLM_FINAL_ANSWER__"
    for line in stdout.splitlines():
        if not line.startswith(marker):
            lines.append(line)
    return "\n".join(lines)
