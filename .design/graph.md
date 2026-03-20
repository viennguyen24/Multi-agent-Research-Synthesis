# Multi-Agent Research Synthesizer — Design Document

## Overview

A multi-agent research synthesis pipeline built on LangGraph. Given a user query and one or more source documents, the system produces a structured, insight-driven synthesis through iterative refinement. Four specialized agents — Planner, Writer, Critic, and Supervisor — collaborate in a directed graph with explicit feedback loops governed by a central decision-maker.

The key design philosophy: **each agent is stateless per call, but the session is stateful**. History, artifacts, and routing decisions are carried explicitly through LangGraph state, not implicitly through a shared context window.

## Graph Architecture

```
[ENTRY] → Planner → Writer → Critic → Supervisor ─┬→ END (accept)
              ↑        ↑                          |
              |        └──────────────────────────┘ (revise)
              |                                   |
              └───────────────────────────────────┘(replan)
```
Circuit breakers are implemented as guard wrappers on the Planner and Writer nodes to minimize re-planning and revising.

## Agent Roles and Context Design

A deliberate principle throughout: **each agent receives only the context it needs to do its specific job**. This is enforced at the prompt construction level, not at the state level — all artifacts live in state, but each agent's selectively choose its context.

### Planner

Responsible for decomposing the research query into a structured output plan that contains all key questions to answered as well as synthesis goals, formatting guidelines, and measurable success criteria.

On replan, the Planner receives its prior plan as an assistant turn and the replan failure history as the final user turn. This multi-turn message structure mirrors the natural editing workflow — the model sees what it previously produced before being told why it failed — which tends to produce more targeted structural changes than a flat single-turn prompt.


### Writer

Responsible for producing the output draft by following the delivery plan.

On revision, the Writer uses the same multi-turn structure: the initial write prompt followed by the prior draft as an assistant turn, with the revision history appended as the final user message. This means the Writer literally "sees its own prior output" before receiving the critique, grounding the revision in context rather than requiring it to reconstruct what it wrote.


### Critic

Stateless by design. The Critic evaluates the current draft against the delivery plan and success criteria, producing a structured `CritiqueOutput` with a summary and a typed, severity-ranked issues list.

The Critic receives the revision and replan histories to enable convergence-aware evaluation — it can acknowledge when prior issues have been addressed and avoid re-raising resolved problems as new findings. Without this, repeated revision cycles tend to produce issue list churn rather than genuine quality improvement.

### Supervisor

The session's decision-maker and history owner. It evaluates the holistic quality of the draft against the success criteria and decides the routing: accept, revise, or replan.

The Supervisor receives the full revision and replan histories, making it the only agent with complete session visibility. Its role is to detect patterns the other agents cannot see in isolation — specifically, whether the same issues are recurring across cycles.

On revise, the Supervisor writes a feedback that names each issue, states what is wrong, and describes exactly what a correct fix looks like and will be appended to the log history for replan or revision accordingly.

---

### Feedback Loop and History Design

The core challenge in a multi-node revision loop is that **each agent call is a fresh LLM invocation with no implicit memory**. The Writer on revision cycle 3 has no knowledge that cycles 1 and 2 happened unless that information is explicitly injected into its prompt.

For now, we solve this by storing previous feedback loop into their own list. Each entry is a single Supervisor-written feedback string from one cycle and we append these context into the corresponding agent that uses it later.

```
PRIOR REVISION HISTORY — do not repeat these mistakes:
  Cycle 1: <supervisor feedback from cycle 1>
  Cycle 2: <supervisor feedback from cycle 2>
```

This gives every agent a compact, ordered record of what has been tried and what the outcome was. The Writer can avoid re-introducing issues it was told to fix. The Planner can take a meaningfully different structural direction on replan.