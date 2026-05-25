from __future__ import annotations

import contextlib
import io
import json
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.sandbox.types import ExecutionResult
from rlm_harness.types import Msg

REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


@dataclass
class RLMObservation:
    code: str
    stdout: str = ""
    stderr: str = ""
    status: str = "ok"
    timed_out: bool = False
    elapsed_ms: int = 0


@dataclass
class RLMResult:
    final_answer: str
    status: str
    iterations: int
    observations: list[RLMObservation] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    subcalls: int = 0
    tokens_used: int = 0


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
    ) -> str:
        if depth_hint >= 0 and depth_hint > self.max_depth:
            return self.llm_query(query, context=context, max_tokens=max_tokens)
        return self.rlm_query(query, context=context, max_tokens=max_tokens)

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
        return [
            Msg(role="system", content=RLM_SYSTEM_PROMPT),
            Msg(
                role="user",
                content=(
                    f"Query:\n{query}\n\n"
                    "The context is available in the REPL as variable `context`. "
                    f"Context preview:\n{serialize_context(context)[:2000]}"
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
- SHOW_VARS
- When running in the sandbox, workspace tools such as project_summary, project_audit,
  project_overview, list_files, read_file, search_code, run_shell, and git_status

Inspect context programmatically. Set answer['content'] and answer['ready'] = True
when done. If you have enough information to answer after seeing observations,
reply with the final user-facing answer in plain text and do not include another
```repl block.

For project identity or overview questions such as "what is this project", call
project_summary() when it is available and return that summary. Do not answer by
printing raw source code.

For project review, audit, risk, issue, or gap-analysis questions, use
project_audit() as a baseline and inspect relevant source/config files before
returning evidence-backed findings. Do not answer these questions with only a
file inventory.
"""


def find_repl_blocks(text: str) -> list[str]:
    return [match.strip() for match in REPL_BLOCK_RE.findall(text)]


def format_observation(observation: RLMObservation) -> str:
    parts = [f"Code executed:\n```python\n{observation.code}\n```", f"Status: {observation.status}"]
    if observation.stdout:
        parts.append("STDOUT:\n" + observation.stdout)
    if observation.stderr:
        parts.append("STDERR:\n" + observation.stderr)
    return "\n\n".join(parts)


def serialize_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True)
    except TypeError:
        return repr(context)


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
