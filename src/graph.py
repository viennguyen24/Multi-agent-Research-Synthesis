from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.agents import lead_researcher_node, editor_node, critic_node


def _route_after_lead(state: ResearchState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("lead_researcher", lead_researcher_node)
    graph.add_node("editor", editor_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("lead_researcher")

    graph.add_conditional_edges(
        "lead_researcher",
        _route_after_lead,
        {"continue": "editor", "done": END},
    )
    graph.add_edge("editor", "critic")
    graph.add_edge("critic", "lead_researcher")

    return graph.compile()
