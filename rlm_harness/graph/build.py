from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, Optional

from rlm_harness.graph.nodes import Nodes
from rlm_harness.graph.task_policy import is_code_editing_task
from rlm_harness.types import HarnessState


class HarnessGraph:
    def __init__(self, nodes: Nodes, max_loops: Optional[int] = None):
        self.nodes = nodes
        self.max_loops = max_loops or nodes.runtime.max_iterations

    def invoke(self, state: HarnessState) -> HarnessState:
        state = self.nodes.memory_read(state)
        state = self.nodes.plan(state)
        state = self.nodes.memory_write(state)
        loops = 0
        while loops < self.max_loops:
            loops += 1
            state.budget.iterations_used = loops
            if state.budget.is_exhausted and state.status != "done":
                state = self.nodes.finalize_partial(state)
                return state
            state = self.nodes.act(state)
            state = self.nodes.memory_write(state)
            if state.status == "done":
                return self.nodes.learn(self.nodes.done(state))
            state = self.nodes.execute_action(state)
            state = self.nodes.memory_write(state)
            state = self.nodes.verify(state)
            state = self.nodes.memory_write(state)
            state = self.nodes.observe(state)
            state = self.nodes.memory_write(state)
            state = self.nodes.reflect(state)
            state = self.nodes.memory_write(state)
            if state.status == "done":
                return self.nodes.learn(self.nodes.done(state))
            if state.status in {"error", "stopped"}:
                return self.nodes.learn(state)
        state.status = "stopped"
        if state.final_answer is None:
            state.final_answer = "Stopped before the task reached a completed state."
        return self.nodes.learn(state)


class LangGraphHarnessGraph:
    def __init__(self, graph, checkpointer_connection: Optional[sqlite3.Connection] = None):
        self.graph = graph
        self.checkpointer_connection = checkpointer_connection

    def invoke(self, state: HarnessState) -> HarnessState:
        result = self.graph.invoke(state, config=graph_config(state))
        if isinstance(result, HarnessState):
            return result
        return HarnessState.model_validate(result)

    def stream(self, state: HarnessState):
        yield from self.graph.stream(state, config=graph_config(state), stream_mode="updates")

    def close(self) -> None:
        if self.checkpointer_connection is not None:
            self.checkpointer_connection.close()


def _build_langgraph(nodes: Nodes, checkpoint_path: Optional[Path] = None):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph backend requested but langgraph is not installed") from exc

    checkpointer = None
    connection = None
    if checkpoint_path is not None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph SQLite checkpointing requested but "
                "langgraph-checkpoint-sqlite is not installed"
            ) from exc
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        checkpointer = SqliteSaver(connection)

    graph = StateGraph(HarnessState)
    graph.add_node("memory_read", nodes.memory_read)
    graph.add_node("plan", nodes.plan)
    graph.add_node("memory_after_plan", nodes.memory_write)
    graph.add_node("memory_after_act", nodes.memory_write)
    graph.add_node("execute_action", nodes.execute_action)
    graph.add_node("memory_after_execute", nodes.memory_write)
    graph.add_node("verify", nodes.verify)
    graph.add_node("memory_after_verify", nodes.memory_write)
    graph.add_node("memory_after_observe", nodes.memory_write)
    graph.add_node("memory_after_reflect", nodes.memory_write)
    graph.add_node("act", nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("done", nodes.done)
    graph.add_node("learn", nodes.learn)

    graph.set_entry_point("memory_read")
    graph.add_edge("memory_read", "plan")
    graph.add_edge("plan", "memory_after_plan")
    graph.add_edge("memory_after_plan", "act")
    graph.add_edge("act", "memory_after_act")
    graph.add_conditional_edges(
        "memory_after_act",
        lambda state: "done" if state.status == "done" else "execute_action",
        {"done": "done", "execute_action": "execute_action"},
    )
    graph.add_edge("execute_action", "memory_after_execute")
    graph.add_conditional_edges(
        "memory_after_execute",
        lambda state: "verify" if is_code_editing_task(state.task) else "observe",
        {"verify": "verify", "observe": "observe"},
    )
    graph.add_edge("verify", "memory_after_verify")
    graph.add_edge("memory_after_verify", "observe")
    graph.add_edge("observe", "memory_after_observe")
    graph.add_edge("memory_after_observe", "reflect")
    graph.add_edge("reflect", "memory_after_reflect")
    graph.add_conditional_edges(
        "memory_after_reflect",
        lambda state: state.status if state.status in {"done", "error", "stopped"} else "act",
        {"done": "done", "error": "learn", "stopped": "learn", "act": "act"},
    )
    graph.add_edge("done", "learn")
    graph.add_edge("learn", END)
    return LangGraphHarnessGraph(graph.compile(checkpointer=checkpointer), connection)


def build_graph(
    nodes: Nodes,
    backend: Literal["auto", "simple", "langgraph"] = "auto",
    checkpoint_path: Optional[Path] = None,
):
    if backend == "simple":
        return HarnessGraph(nodes)
    if backend == "langgraph":
        return _build_langgraph(nodes, checkpoint_path=checkpoint_path)
    try:
        return _build_langgraph(nodes, checkpoint_path=checkpoint_path)
    except RuntimeError:
        return HarnessGraph(nodes)


def graph_config(state: HarnessState) -> dict:
    return {"configurable": {"thread_id": state.thread_id}}
