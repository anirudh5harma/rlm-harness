from __future__ import annotations

from typing import Literal

from rlm_harness.graph.nodes import Nodes
from rlm_harness.types import HarnessState


class HarnessGraph:
    def __init__(self, nodes: Nodes, max_loops: int = 3):
        self.nodes = nodes
        self.max_loops = max_loops

    def invoke(self, state: HarnessState) -> HarnessState:
        state = self.nodes.plan(state)
        loops = 0
        while loops < self.max_loops:
            loops += 1
            state = self.nodes.act(state)
            state = self.nodes.observe(state)
            state = self.nodes.reflect(state)
            if state.status == "done":
                return self.nodes.done(state)
            if state.status == "error":
                return state
        state.status = "stopped"
        return state


class LangGraphHarnessGraph:
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, state: HarnessState) -> HarnessState:
        result = self.graph.invoke(state)
        if isinstance(result, HarnessState):
            return result
        return HarnessState.model_validate(result)


def _build_langgraph(nodes: Nodes):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph backend requested but langgraph is not installed") from exc

    graph = StateGraph(HarnessState)
    graph.add_node("plan", nodes.plan)
    graph.add_node("act", nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("done", nodes.done)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: state.status if state.status in {"done", "error"} else "act",
        {"done": "done", "error": END, "act": "act"},
    )
    graph.add_edge("done", END)
    return LangGraphHarnessGraph(graph.compile())


def build_graph(nodes: Nodes, backend: Literal["auto", "simple", "langgraph"] = "auto"):
    if backend == "simple":
        return HarnessGraph(nodes)
    if backend == "langgraph":
        return _build_langgraph(nodes)
    try:
        return _build_langgraph(nodes)
    except RuntimeError:
        return HarnessGraph(nodes)
