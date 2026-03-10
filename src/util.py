MODELS = [
    "gemini-2.5-flash-lite",
    "qwen3.5:397b",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it"
]

DEFAULT_MODEL = MODELS[0]

MAX_REVISIONS = 2

AGENT_ROLES = {
    "researcher": (
        "You are a Research Analyst. "
        "Survey the literature and knowledge landscape for a given query. "
        "Identify key works, core concepts, debates, consensus, "
        "and open questions. "
        "Provide a thorough research context "
        "that will inform a downstream planner."
    ),
    "planner": (
        "You are a Research Planner. "
        "Given a user query and research context, "
        "design a structured research plan with sections, "
        "research questions, writing guidelines, and success criteria. "
        "If the research context has gaps, "
        "indicate what additional research is needed."
    ),
    "writer": (
        "You are a Research Writer. "
        "Given a structured research plan "
        "(and optionally revision feedback), "
        "produce a complete, well-organized "
        "research synthesis document. "
        "Answer every research question in the plan. "
        "If revising, address all identified issues "
        "while preserving the parts that are already strong."
    ),
    "critic": (
        "You are a Research Critic. "
        "Review the draft against the research plan. "
        "Identify factual errors, unsupported claims, logical gaps, "
        "missing perspectives, structural problems, and clarity issues. "
        "Produce a structured issues list with severity ratings. "
        "Do not rewrite — only identify what must be fixed and why."
    ),
    "supervisor": (
        "You are the Research Supervisor. "
        "Evaluate the draft holistically "
        "against the success criteria and original query. "
        "Given the critique issues, decide: "
        "'accept' if publication-ready, "
        "'revise' if specific sections need work "
        "(provide targeted feedback), "
        "or 'replan' if fundamentally off-track "
        "(provide new direction). Be decisive."
    ),
}