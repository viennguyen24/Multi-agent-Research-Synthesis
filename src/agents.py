from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState
from src.llm import get_llm
from src.util import AGENT_ROLES, MAX_ITERATIONS


def _call_llm(role: str, user_prompt: str) -> str:
    """Invoke the LLM with a role system prompt and return the response text."""
    print(f"  -> requesting {role}...", flush=True)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=AGENT_ROLES[role]),
        HumanMessage(content=user_prompt),
    ])
    print(f"  OK {role} responded ({len(response.content)} chars)", flush=True)
    return response.content


def lead_researcher_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0)

    # why: first pass creates the plan, subsequent passes evaluate draft+critique
    if iteration == 0:
        prompt = f"Research query:\n{state['query']}\n\nProduce a structured research plan."
        plan = _call_llm("lead_researcher", prompt)
        return {
            "plan": plan,
            "next": "continue",
            "iteration": 1,
            "messages": [f"[lead_researcher] Created research plan (iteration 1)"],
        }

    if iteration >= MAX_ITERATIONS:
        return {
            "next": "done",
            "iteration": iteration,
            "messages": [f"[lead_researcher] Max iterations ({MAX_ITERATIONS}) reached — finalizing"],
        }

    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"Current draft:\n{state.get('draft', '')}\n\n"
        f"Critic feedback:\n{state.get('critique', '')}\n\n"
        "Evaluate the draft against the critique. "
        "If the draft is publication-ready, respond with exactly: DONE\n"
        "Otherwise respond with CONTINUE followed by specific guidance for the editor."
    )
    evaluation = _call_llm("lead_researcher", prompt)

    if evaluation.strip().upper().startswith("DONE"):
        return {
            "next": "done",
            "iteration": iteration,
            "messages": [f"[lead_researcher] Draft approved at iteration {iteration}"],
        }

    return {
        "plan": evaluation,
        "next": "continue",
        "iteration": iteration + 1,
        "messages": [f"[lead_researcher] Requesting revision (iteration {iteration + 1})"],
    }


def editor_node(state: ResearchState) -> dict:
    prompt = f"Research query:\n{state['query']}\n\nResearch plan / guidance:\n{state.get('plan', '')}\n\n"

    critique = state.get("critique", "")
    if critique:
        prompt += f"Previous critique to address:\n{critique}\n\n"

    draft = state.get("draft", "")
    if draft:
        prompt += f"Previous draft to revise:\n{draft}\n\n"
        prompt += "Revise the draft to resolve every issue from the critique."
    else:
        prompt += "Write the initial research draft."

    new_draft = _call_llm("editor", prompt)
    return {
        "draft": new_draft,
        "messages": [f"[editor] {'Revised' if draft else 'Created'} draft"],
    }


def critic_node(state: ResearchState) -> dict:
    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"Draft to review:\n{state.get('draft', '')}\n\n"
        "List specific, actionable problems. Do not rewrite — only identify issues."
    )
    critique = _call_llm("critic", prompt)
    return {
        "critique": critique,
        "messages": ["[critic] Reviewed draft"],
    }
