# Python Multi-Agent System Style Guide

## 1. Project Overview

This is a Python multi-agent system. Code reviews should be aware of the following architectural concepts:
- **Nodes** are individual agent functions that receive a state dict and return a state patch dict
- **State** is a typed dict that flows between nodes — it is never mutated in-place
- **LLM access** is centralised through `get_llm()` which accepts an `LLMConfig` dataclass
- **Prompts** are separated from node logic into dedicated prompt modules

### A. Role Specialization (SRP)
- **Constraint:** Every agent MUST have a single, well-defined responsibility (e.g., Planner, Researcher, Executor, Critic).
- **Check:** Flag "God Agents" that handle too many unrelated tasks or agents whose name does not reflect their actual responsibility (e.g. a `critic_node` that also rewrites content).

### B. State Management & Isolation
- **Constraint:** Use explicit state objects (e.g., Pydantic models or Dataclasses or TypedDicts) to pass context between agents.
- **Immutability:** Prefer returning new state objects rather than mutating existing ones to prevent side effects in complex agent loops.
- **Check:** Flag any node that reads a state key it does not own or that writes to a key outside its defined responsibility.

### C. Tool Definitions
- **Standard:** Every tool provided to an LLM MUST have a clear docstring explicitly defining parameters, functionality, and return types.
- **Check:** Tools without type hints or clear "purpose" descriptions should be flagged, as they lead to model hallucinations.
- **Check:** Flag tools with vague names like `process()`, `handle()`, or `run()` — names must describe the action and domain, or tools that perform more than one logical action — each tool should do exactly one thing.

### D. Agent System Prompts
- **No hardcoding:** System role descriptions (agent personas) must be defined as constants in a dedicated prompts module, never as inline strings inside node functions.
- **Action-oriented verbs:** Use "Analyze," "Extract," "Transform," or "Validate" — not "Try to find" or "Think about."
- **No politeness noise:** Remove "Please," "I would like you to," and "Thank you" — they waste tokens and reduce instruction following.
- **Output format:** Every system prompt that expects structured output must explicitly define the expected format (e.g. JSON schema, exact keywords like `DONE` / `CONTINUE`).
- **Check:** Flag any system prompt that does not specify what a valid response looks like.

## 2. Python Version & General Style

- Python 3.11+
- Use `snake_case` for functions, variables, and modules
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for module-level constants
- Use f-strings for string formatting, not `.format()` or `%`

---

## 3. Type Hints

- All function signatures **must** have type hints on parameters and return values
- Avoid `Any` unless genuinely unavoidable; flag every occurrence

**Bad:**
```python
def editor_node(state):
    ...

def get_llm(provider, overrides):
    ...
```

**Good:**
```python
def editor_node(state: ResearchState) -> dict:
    ...

def get_llm(config: LLMConfig) -> BaseLLM:
    ...
```
---