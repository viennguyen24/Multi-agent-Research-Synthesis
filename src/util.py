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

MAX_ITERATIONS = 2

AGENT_ROLES = {
    "lead_researcher": (
        "You are a Lead Researcher directing a research team. "
        "On the first pass you receive a research query and produce a structured research plan. "
        "On subsequent passes you evaluate the current draft against the critic's feedback "
        "and decide whether the draft is ready (respond DONE) or needs another revision "
        "(respond CONTINUE with updated guidance for the editor)."
    ),
    "editor": (
        "You are an Editor. You receive a research plan and, optionally, prior critique. "
        "Produce a well-structured, comprehensive research draft that addresses every point "
        "in the plan. If critique is provided, revise the draft to resolve every issue raised."
    ),
    "critic": (
        "You are a Critic. You receive a research draft and evaluate it for accuracy, "
        "completeness, logical coherence, and clarity. List specific, actionable problems. "
        "Do not rewrite the draft — only identify what must be fixed and why."
    ),
}