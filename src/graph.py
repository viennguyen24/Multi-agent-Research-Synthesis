from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.agents import (
    researcher_node,
    planner_node,
    writer_node,
    critic_node,
    supervisor_node,
)


def build_graph():
    """Construct the 5-node research synthesis graph.

    Topology:
        [ENTRY] → researcher → planner → writer → critic → supervisor → [END]
                      ↑                     ↑                   │
                      │                     └── REVISE ─────────┤
                      └──────────── REPLAN ─────────────────────┘

    Planner and Supervisor use Command API for conditional routing.
    Writer and Critic use fixed edges.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("supervisor", supervisor_node)

    graph.set_entry_point("researcher")

    graph.add_edge("researcher", "planner")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "supervisor")

    return graph.compile()
