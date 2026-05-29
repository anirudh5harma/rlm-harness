from __future__ import annotations

from rlm_harness.kernel.state import RunState


def append_event(state: RunState, event) -> RunState:
    state.events.append(event)
    state.event_cursor = max(state.event_cursor, event.sequence)
    return state
