from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from langgraph.types import Command
from langgraph.graph import END

from src.state import (
    ResearchState,
    ResearchContextOutput,
    ResearchPlanOutput,
    CritiqueOutput,
    SupervisorOutput,
    ResearchContext,
    ResearchPlan,
    Draft,
    DraftMetadata,
    Critique,
    Revision,
)
from src.llm import get_llm
from src.util import AGENT_ROLES, MAX_REVISIONS


def _call_llm(role: str, user_prompt: str, schema: type | None = None):
    llm = get_llm()
    messages = [
        {"role": "system", "content": AGENT_ROLES[role]},
        {"role": "user", "content": user_prompt},
    ]
    tag = "structured" if schema else "text"
    print(f"  -> requesting {role} ({tag})...", flush=True)
    result = llm.complete(messages, schema=schema)
    label = "structured" if schema else f"{len(result)} chars"
    print(f"  OK {role} responded ({label})", flush=True)
    return result


def researcher_node(state: ResearchState) -> Command[Literal["planner"]]:
    """Domain knowledge gathering node.

    Surveys the research landscape for the user query: identifies related works,
    key concepts, debates, and open questions. On replan, incorporates the
    Supervisor's replan_direction as supplementary context.
    Always routes to the Planner.
    """
    query = state["query"]
    revisions = state.get("revisions", [])

    # why: on replan the Supervisor provides direction for re-research
    replan_context = ""
    if revisions and revisions[-1]["decision"] == "replan":
        last_revision = revisions[-1]
        replan_context = (
            f"\n\nPrevious attempt failed. Supervisor guidance:\n"
            f"Failures: {last_revision['failures']}\n"
            f"Learnings: {last_revision['learnings']}\n"
            f"New direction: {last_revision['replan_direction']}"
        )

    prompt = f"Research query:\n{query}{replan_context}"
    result = _call_llm("researcher", prompt, schema=ResearchContextOutput)

    now = datetime.now().isoformat()
    context_entry: ResearchContext = {
        "timestamp": now,
        "id": str(uuid4()),
        "query": query,
        "key_works": [kw.model_dump() for kw in result.key_works],
        "key_concepts": result.key_concepts,
        "research_landscape": result.research_landscape,
        "uncertain_areas": result.uncertain_areas,
        "additional_notes": result.additional_notes,
    }

    return Command(
        update={
            "research_context": [context_entry],
            "messages": [f"[researcher] Gathered research context ({len(result.key_works)} key works)"],
        },
        goto="planner",
    )


def planner_node(state: ResearchState) -> Command[Literal["writer", "researcher"]]:
    """Research plan design node.

    Receives the user query and latest research context. Designs a structured
    plan with sections, research questions, writing guidelines and success
    criteria. Routes to Writer if plan is complete, or back to Researcher
    if knowledge gaps are identified.
    """
    query = state["query"]
    research_contexts = state.get("research_context", [])
    latest_context = research_contexts[-1] if research_contexts else {}

    prompt = (
        f"Research query:\n{query}\n\n"
        f"Research context:\n{latest_context}\n\n"
        "Design a comprehensive research plan. If the research context has gaps "
        "that would prevent writing a quality synthesis, set needs_more_research=true "
        "and provide specific research_queries."
    )
    result = _call_llm("planner", prompt, schema=ResearchPlanOutput)

    plan_dict: ResearchPlan = {
        "title": result.title,
        "research_context": latest_context,
        "target_audience": result.target_audience,
        "guidelines": result.guidelines,
        "success_criteria": result.success_criteria,
        "table_of_contents": result.table_of_contents,
        "introduction": result.introduction,
        "sections": [s.model_dump() for s in result.sections],
        "conclusion": result.conclusion,
    }

    # why: route back to researcher if planner identified knowledge gaps
    if result.needs_more_research and result.research_queries:
        return Command(
            update={
                "plan": plan_dict,
                "messages": [
                    f"[planner] Identified {len(result.research_queries)} knowledge gaps, routing back to researcher"
                ],
            },
            goto="researcher",
        )

    return Command(
        update={
            "plan": plan_dict,
            "messages": [f"[planner] Created research plan: '{result.title}'"],
        },
        goto="writer",
    )


def writer_node(state: ResearchState) -> dict:
    """Research document writing node.

    On first pass: produces a complete initial draft from the research plan.
    On revision: improves the latest draft using Supervisor feedback and
    Critic issues. Always outputs a full new document version.
    Fixed edge to Critic.
    """
    plan = state.get("plan", {})
    drafts = state.get("drafts", [])
    revisions = state.get("revisions", [])
    critiques = state.get("critiques", [])
    session_id = state.get("session_id", "")

    is_revision = len(drafts) > 0 and len(revisions) > 0

    if is_revision:
        previous_draft = drafts[-1]
        last_revision = revisions[-1]
        last_critique = critiques[-1] if critiques else {}

        prompt = (
            f"Research plan:\n{plan}\n\n"
            f"Current draft (version {previous_draft['version']}):\n{previous_draft['document']}\n\n"
            f"Supervisor feedback:\n{last_revision['feedback']}\n\n"
            f"Priority issues to address: {last_revision['issues']}\n\n"
            f"Full critique details:\n{last_critique}\n\n"
            "Revise the draft to address all identified issues. Produce a complete "
            "updated document — do not leave placeholders or omit sections."
        )
        action = "revision"
        based_on = previous_draft["version"]
        issues_addressed = last_revision.get("issues", [])
        feedback_refs = [last_revision.get("feedback", "")]
    else:
        prompt = (
            f"Research plan:\n{plan}\n\n"
            "Write the initial research synthesis document following the plan exactly. "
            "Answer every research question in each section. Produce a complete, "
            "well-structured document."
        )
        action = "initial"
        based_on = 0
        issues_addressed = []
        feedback_refs = []

    document = _call_llm("writer", prompt)
    now = datetime.now().isoformat()
    version = len(drafts) + 1

    draft_entry: Draft = {
        "metadata": DraftMetadata(
            session_id=session_id,
            created_at=now,
            last_modified=now,
            agent="writer",
        ),
        "version": version,
        "timestamp": now,
        "document": document,
        "word_count": len(document.split()),
        "action": action,
        "based_on_version": based_on,
        "issues_addressed": issues_addressed,
        "feedback": feedback_refs,
    }

    return {
        "drafts": [draft_entry],
        "messages": [
            f"[writer] {'Revised' if is_revision else 'Created'} draft v{version} "
            f"({draft_entry['word_count']} words)"
        ],
    }


def critic_node(state: ResearchState) -> dict:
    """Draft review and issue identification node.

    Reviews the latest draft against the research plan. Produces a structured
    issues list with severity ratings and locations. Does not rewrite —
    only identifies problems for the Supervisor to act on.
    Fixed edge to Supervisor.
    """
    drafts = state.get("drafts", [])
    plan = state.get("plan", {})
    latest_draft = drafts[-1]
    critiques = state.get("critiques", [])

    prompt = (
        f"Research plan with success criteria:\n{plan}\n\n"
        f"Draft to review (version {latest_draft['version']}):\n{latest_draft['document']}\n\n"
        "Review this draft thoroughly. For each issue found, specify the exact "
        "location, issue type, severity, and a clear description. "
        "Do not rewrite — only identify problems."
    )
    result = _call_llm("critic", prompt, schema=CritiqueOutput)

    critique_version = len(critiques) + 1
    now = datetime.now().isoformat()

    critique_entry: Critique = {
        "version": critique_version,
        "draft_version": latest_draft["version"],
        "timestamp": now,
        "summary": result.summary,
        "issues": [issue.model_dump() for issue in result.issues],
        "issue_counts": {
            "critical": result.critical_count,
            "major": result.major_count,
            "minor": result.minor_count,
            "total": len(result.issues),
        },
    }

    return {
        "critiques": [critique_entry],
        "messages": [
            f"[critic] Reviewed draft v{latest_draft['version']}: "
            f"{result.critical_count} critical, {result.major_count} major, "
            f"{result.minor_count} minor issues"
        ],
    }


def supervisor_node(
    state: ResearchState,
) -> Command[Literal["writer", "researcher", "__end__"]]:
    """Holistic evaluation and routing decision node.

    Evaluates the draft against success criteria and the original query.
    Decides: accept (end), revise (back to Writer with targeted feedback),
    or replan (back to Researcher with new direction).
    Enforces MAX_REVISIONS to prevent infinite loops.
    """
    drafts = state.get("drafts", [])
    critiques = state.get("critiques", [])
    plan = state.get("plan", {})
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    revisions = state.get("revisions", [])

    latest_draft = drafts[-1]
    latest_critique = critiques[-1]

    # why: hard stop to prevent infinite revision loops
    if iteration_count >= MAX_REVISIONS:
        revision_entry: Revision = {
            "version": len(revisions) + 1,
            "draft_version": latest_draft["version"],
            "decision": "accept",
            "reasoning": f"Max revisions ({MAX_REVISIONS}) reached — accepting current draft",
            "issues": [],
            "feedback": "",
            "expected_outcome": "",
            "failures": "",
            "learnings": "",
            "replan_direction": "",
        }
        return Command(
            update={
                "revisions": [revision_entry],
                "iteration_count": iteration_count,
                "messages": [
                    f"[supervisor] Max revisions ({MAX_REVISIONS}) reached — finalizing draft v{latest_draft['version']}"
                ],
            },
            goto=END,
        )

    prompt = (
        f"Original research query:\n{query}\n\n"
        f"Research plan and success criteria:\n{plan}\n\n"
        f"Current draft (version {latest_draft['version']}):\n{latest_draft['document']}\n\n"
        f"Critique summary: {latest_critique['summary']}\n"
        f"Issue counts: {latest_critique['issue_counts']}\n"
        f"Issues:\n{latest_critique['issues']}\n\n"
        f"Iteration: {iteration_count + 1} of {MAX_REVISIONS}\n\n"
        "Evaluate holistically. Decide: 'accept' if publication-ready, "
        "'revise' if specific sections need work, or 'replan' if the "
        "approach is fundamentally flawed."
    )
    result = _call_llm("supervisor", prompt, schema=SupervisorOutput)

    revision_entry: Revision = {
        "version": len(revisions) + 1,
        "draft_version": latest_draft["version"],
        "decision": result.decision,
        "reasoning": result.reasoning,
        "issues": result.priority_issues,
        "feedback": result.feedback,
        "expected_outcome": result.expected_outcome,
        "failures": result.failures,
        "learnings": result.learnings,
        "replan_direction": result.replan_direction,
    }

    base_update = {
        "revisions": [revision_entry],
        "iteration_count": iteration_count + 1,
    }

    if result.decision == "accept":
        return Command(
            update={
                **base_update,
                "messages": [
                    f"[supervisor] Accepted draft v{latest_draft['version']} — done"
                ],
            },
            goto=END,
        )

    if result.decision == "replan":
        return Command(
            update={
                **base_update,
                "messages": [
                    f"[supervisor] Draft fundamentally off-track — replanning"
                ],
            },
            goto="researcher",
        )

    # why: default to revise — send targeted feedback to Writer
    return Command(
        update={
            **base_update,
            "messages": [
                f"[supervisor] Requesting revision of draft v{latest_draft['version']} "
                f"({len(result.priority_issues)} priority issues)"
            ],
        },
        goto="writer",
    )
