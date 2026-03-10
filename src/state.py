from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field



class KeyWork(BaseModel):
    """A single related work entry in the research context."""
    title: str = Field(description="Title of the related work")
    authors: list[str] = Field(description="Author names")
    findings: str = Field(description="High-level findings summary")


class ResearchContextOutput(BaseModel):
    """LLM output schema for the Researcher node.

    Captures domain knowledge: related works, key concepts, field landscape,
    and open questions that inform downstream planning.
    """
    key_works: list[KeyWork] = Field(
        description="Related works regarding this topic"
    )
    key_concepts: list[str] = Field(
        description="Concepts fundamental to the user question"
    )
    research_landscape: str = Field(
        description="Overview of field: debates, consensus, trends"
    )
    uncertain_areas: list[str] = Field(
        description="What is not known or needs further research"
    )
    additional_notes: list[str] = Field(
        default_factory=list,
        description="Observations worth noting that don't fit elsewhere",
    )


class SectionDef(BaseModel):
    """Recursive section definition for a research plan outline."""
    id: str = Field(description="Section number e.g. '1.0', '2.3'")
    title: str = Field(description="Section title")
    research_questions: list[str] = Field(
        description="Questions this section must answer"
    )
    requirements: list[str] = Field(
        description="What must be covered in this section"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of prerequisite sections this builds on",
    )
    target_length: str = Field(description="Target length e.g. '500 words'")
    purpose: str = Field(description="Why this section exists in the plan")
    sub_sections: list[SectionDef] = Field(
        default_factory=list, description="Nested sub-sections"
    )


class ResearchPlanOutput(BaseModel):
    """LLM output schema for the Planner node.

    Produces a structured research plan with sections, writing guidelines,
    and routing decision (proceed to write or request more research).
    """
    title: str = Field(description="Title of the research synthesis")
    target_audience: str = Field(
        description="Intended audience e.g. 'academic', 'general', 'practitioner'"
    )
    guidelines: dict = Field(
        description="Writing guidelines: tone, formality, use of examples, etc."
    )
    success_criteria: list[str] = Field(
        description="Measurable criteria for a successful synthesis"
    )
    table_of_contents: list[str] = Field(
        description="High-level summary of article structure"
    )
    introduction: str = Field(
        description="Background context and motivation for the draft"
    )
    sections: list[SectionDef] = Field(
        description="Ordered section definitions with nested structure"
    )
    conclusion: str = Field(
        description="Summary of the research plan with respect to the user query"
    )
    needs_more_research: bool = Field(
        description="True if knowledge gaps require routing back to Researcher"
    )
    research_queries: list[str] = Field(
        default_factory=list,
        description="Specific queries for the Researcher if gaps were found",
    )


class IssueItem(BaseModel):
    """A single issue identified during critique."""
    id: str = Field(description="Issue identifier e.g. 'ISS_001'")
    location: str = Field(
        description="Section or paragraph where the issue occurs"
    )
    type: str = Field(
        description=(
            "Issue category: logical_gap, unsupported_claim, missing_angle, "
            "structural, clarity, evidence_quality, contradiction"
        )
    )
    severity: str = Field(description="Severity level: critical, major, or minor")
    description: str = Field(description="What is wrong and why it matters")
    related_issues: list[str] = Field(
        default_factory=list, description="IDs of related issues"
    )


class CritiqueOutput(BaseModel):
    """LLM output schema for the Critic node.

    Reviews the draft against the plan and produces a structured issues list.
    Does not rewrite — only identifies problems.
    """
    summary: str = Field(description="Overview of critique findings")
    issues: list[IssueItem] = Field(description="Specific issues found in the draft")
    critical_count: int = Field(description="Number of critical-severity issues")
    major_count: int = Field(description="Number of major-severity issues")
    minor_count: int = Field(description="Number of minor-severity issues")


class SupervisorOutput(BaseModel):
    """LLM output schema for the Supervisor node.

    Holistic evaluation against success criteria. Decides whether to accept,
    revise (send back to Writer), or replan (send back to Researcher).
    """
    decision: str = Field(
        description="Routing decision: 'accept', 'revise', or 'replan'"
    )
    reasoning: str = Field(description="Justification for the decision")
    priority_issues: list[str] = Field(
        default_factory=list,
        description="Issue IDs the Writer should focus on if revising",
    )
    feedback: str = Field(
        default="",
        description="Specific feedback for the Writer on what to improve",
    )
    expected_outcome: str = Field(
        default="",
        description="What a successful next revision looks like",
    )
    failures: str = Field(
        default="",
        description="What went wrong with the current approach (used on replan)",
    )
    learnings: str = Field(
        default="",
        description="Key takeaways from this iteration",
    )
    replan_direction: str = Field(
        default="",
        description="Guidance for the Researcher/Planner if replanning",
    )


# ── LangGraph State TypedDicts ──────────────────────────────────────────────


class ResearchContext(TypedDict):
    """Snapshot of domain knowledge produced by one Researcher pass."""
    timestamp: str
    id: str
    query: str
    key_works: list[dict]
    key_concepts: list[str]
    research_landscape: str
    uncertain_areas: list[str]
    additional_notes: list[str]


class ResearchPlan(TypedDict):
    """Structured research plan produced by the Planner."""
    title: str
    research_context: dict
    target_audience: str
    guidelines: dict
    success_criteria: list[str]
    table_of_contents: list[str]
    introduction: str
    sections: list[dict]
    conclusion: str


class DraftMetadata(TypedDict):
    """Provenance metadata for a single draft version."""
    session_id: str
    created_at: str
    last_modified: str
    agent: str


class Draft(TypedDict):
    """A complete draft version produced by the Writer."""
    metadata: DraftMetadata
    version: int
    timestamp: str
    document: str
    word_count: int
    action: str               # "initial" | "revision"
    based_on_version: int
    issues_addressed: list[str]
    feedback: list[str]


class Critique(TypedDict):
    """Structured critique of a draft produced by the Critic."""
    version: int
    draft_version: int
    timestamp: str
    summary: str
    issues: list[dict]
    issue_counts: dict        # {critical, major, minor, total}


class Revision(TypedDict):
    """Supervisor judgment on a draft+critique pair."""
    version: int
    draft_version: int
    decision: str             # "accept" | "revise" | "replan"
    reasoning: str
    issues: list[str]
    feedback: str
    expected_outcome: str
    failures: str
    learnings: str
    replan_direction: str


class ErrorRecord(TypedDict):
    """An error encountered during graph execution."""
    agent: str
    timestamp: str
    description: str


class ResearchState(TypedDict):
    """Minimal shared state that persists across the entire graph.

    Append-only lists use ``operator.add`` as reducer so each node
    appends without overwriting previous entries. Access the latest
    entry via list[-1].
    """
    # immutable core — set once at graph invocation
    query: str
    session_id: str
    created_at: str

    # coordination counter for the revision loop
    iteration_count: int

    # append-only artifact histories
    research_context: Annotated[list[ResearchContext], operator.add]
    plan: ResearchPlan
    drafts: Annotated[list[Draft], operator.add]
    critiques: Annotated[list[Critique], operator.add]
    revisions: Annotated[list[Revision], operator.add]
    errors: Annotated[list[ErrorRecord], operator.add]
    messages: Annotated[list[str], operator.add]

