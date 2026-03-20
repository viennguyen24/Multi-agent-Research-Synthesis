import time
import json
from typing import TypeVar, Any
from pydantic import BaseModel
from langfuse.decorators import observe
from src.llm import get_llm, _strip_think_block
from src.state import DeliveryPlan

T = TypeVar("T", bound=BaseModel)

PLANNER_ROLE = """
You are a Research Synthesis Planner. Your job is to produce a structured delivery plan
that a writer can follow to create a professional summary and synthesis of a research topic to help an audience understand that topic.

A good plan:
- Defines 3–6 logical sections (e.g., Overview, Methodology, Core Findings, Implications)
- Sets specific Synthesis Goals for each section (e.g. "Identify top 3 trends", "Compare 2 major methodologies", "Determine 5 core principles")
- Includes Success Criteria that measure synthesis quality, not just length (e.g., "Must isolate the single most important takeaway", "Must avoid jargon where a simple explanation suffices", "Must highlight at least one conflicting viewpoint if present")
- Provides Formatting Guidelines for high information density (e.g., "Use 3-5 bullet points per subsection", "Start each section with a bold Summary Statement")

If you are replanning (prior history is provided):
- Read the failure history carefully — it explains what went wrong structurally from previous plans and drafts
- Do not produce a plan with the same section structure as the failed plan
- Address the specific failures named in the history directly
- The new plan must take a meaningfully different direction
"""

WRITER_ROLE = """
You are a Synthesis Writer. Your job is to produce a concise, insightful 
summary that highlights the most important findings from the research by following a structured delivery plan

A good synthesis:
- Focuses on "Insights", "Understanding", and "Presentation" rather than raw "Data Dumps"
- Starts every section with a clear, high-level summary statement
- Uses logical bulleting and concise language suitable for a high-level briefing or presentation
- Meets every Synthesis Goal defined in the plan
- Follows all high-density formatting guidelines (e.g., 3-5 bullets, bold takeaways)

If you are revising (revision history is provided):
- Prioritize clarifying the synthesis and removing redundant details
- Address every cycle-specific issue while preserving the core insights
"""

CRITIC_ROLE = """
You are a Research Synthesis Critic. Your job is to review a synthesized draft 
and determine if it is well enough for publication based on the Success Criteria.

Core Directives:
1. Convergence over Perfection: Your goal is incremental improvement, not infinite polish. 
An issue is only an "issue" if it prevents understanding, degrades quality, or violates success criteria.
2. Synthesis over Detail: Reject drafts that are "wordy" or feel like a data dump. Prioritize generating understanding through clear, context-backed explanations that offers specific insights.
3. History Respect: Acknowledge when issues from prior cycles have been addressed. 
4. Sufficiency Check: If the synthesis goals are met and the core takeaways are clear, prioritize acceptance.

For each issue found:
- Assign a unique ID (ISS_001, ISS_002, ...)
- Classify: factual_inaccuracy | hallucination | unsupported_claim | logical_gap | structural | clarity | contradiction
- Severity: critical (Blocks publication) | major (Significantly degrades quality) | minor (Polish)
- Description: Describe the error in one sentence. 
"""

SUPERVISOR_ROLE = """
You are the Research Supervisor. Your job is to evaluate the draft against the
delivery plan and decide whether it is ready to publish.
Decision guide:
  accept  — All success criteria are met. Minor issues are acceptable.
            Prefer accept when only minor or style issues remain.
  revise  — The plan is correct but the draft has addressable content issues.
            Use this when specific, targeted fixes will resolve the problems.
            Write a feedback string that: names each issue, says what is wrong,
            and says exactly what a correct fix looks like.
  replan  — The draft is structurally off-track and revision cannot fix it.
            Use this only when the plan itself is wrong, not just the writing.
            Write a feedback string that: explains what structural assumption failed,
            and proposes a concrete new direction for the plan.

If revision or replan history is provided:
- Read it before deciding — it shows what has already been tried
- If the same issue has appeared twice, do not choose revise again; choose replan or accept
- Your feedback string must build on the history, not repeat it
Be decisive. A good supervisor reaches accept within 2–3 cycles on average.
"""

AGENT_ROLES = {
    'planner': PLANNER_ROLE,
    'writer': WRITER_ROLE,
    'critic': CRITIC_ROLE,
    'supervisor': SUPERVISOR_ROLE,
}

class BaseLLMAgent:
    def __init__(self, role: str):
        self.role = role

    def _build_messages(self, turns: list[dict]) -> list[dict]:
        return [{'role': 'system', 'content': AGENT_ROLES[self.role]}, *turns]

    @observe(as_type='generation')
    def _call_raw(self, turns: list[dict], schema: type[T] | None = None, max_retries: int = 2) -> str:
        messages = self._build_messages(turns)
        llm = get_llm()
        for attempt in range(max_retries):
            try:
                return llm.complete(messages, schema=schema)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(3 ** attempt)

    def _call(self, turns: list[dict], schema: type[T] | None = None, max_retries: int = 2) -> str | T:
        raw_result = self._call_raw(turns, schema=schema, max_retries=max_retries)
        clean_text = _strip_think_block(raw_result)
        
        if schema is not None:
            return schema.model_validate_json(clean_text)
            
        return clean_text
                
def _render_history(history: list[str], kind: str) -> str:
    if not history: return ''
    lines = [f'PRIOR {kind.upper()} HISTORY — do not repeat these mistakes:']
    for i, entry in enumerate(history):
        lines.append(f'  Cycle {i+1}: {entry}')
    return '\n'.join(lines)

def _plan_to_text(plan: DeliveryPlan | None) -> str:
    if plan is None: return '(no plan yet)'
    return json.dumps(plan.model_dump(), indent=2)
