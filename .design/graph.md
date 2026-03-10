# Multi-Agent Research Synthesizer
## Graph
See `design_graph.md`

```
[ENTRY] → RESEARCHER → PLANNER → WRITER → CRITIC → SUPERVISOR → [END]
                        ↑            ↑               |
                        |            |               |
                        |            └────────── REVISE
                        |                            |
                        └──────────────────────── REPLAN
```

**Normal path:** Researcher → Planner → Writer → Critic → Supervisor → END

**Revision loop:** Supervisor decides the draft is overall on the right track but has fixable issues, so routes back to Writer with targeted feedback, Writer revises, Critic re-evaluates, Supervisor decides again

**Replan loop:** Supervisor decides the entire research direction is wrong and must be aborted, so routes to Planner re-outlines, and full pipeline resumes

---

## Agent Roles

### Researcher

In real research, you don't plan before you understand the landscape. A planner operating on a raw user query alone risks producing a shallow or misdirected outline. The Researcher's job is to establish grounded context before any planning happens. It focuses on breadth of context, gathering as much high level information as possible to create ground on the research direction.

**What it does:**
- Identifies the relevant knowledge domains and key concepts in the query
- Maps the research landscape: what is established, what is debated, what is uncertain
- Surfaces key prior work and relevant perspectives in the area
- Recommends a preliminary research direction based on this survey
---

### Planner

This agent defines the questions the final output must answer, decide how to structure the output, and set the guidlines for what a good answer looks like.

**What it does:**
- Receives the user query and the Researcher's background context
- Decomposes the topic into a nested section structure
- Defines specific research questions each section must answer
- Establishes writing guidelines: target audience, tone, style, depth
- Sets explicit success criteria the final output will be evaluated against
- Routes back to Researcher if it identifies a genuine knowledge gap or an angle worth exploring.
---

### Writer

The Writer here is responsible for turning a structured plan into a complete, coherent document, and for improving it when feedback arrives. It answers every single question listed in the structured reasearch plan, and focuses on getting the in-depth explanation for each questions. Outputs a complete draft document, versioned which will be handed to Critic.

**What it does:**

*First draft:* Follows the research plan directly. Answers every question defined per section. Structures and formats everything into one complete document.

*Revision:* Receives the most recent draft plus structured feedback from Supervisor (high-level direction) and Critic (specific issues). Addresses each issue and rewrites affected sections.

---

### Critic

This agent purely finds problems and outlines them for another agent who will plan how to fix them.

**What it does:**
- Checks every section against the plan's defined research questions — does this section actually answer what it was supposed to?
- Evaluates factual soundness: are claims well-supported, clearly explained, and appropriately evidenced?
- Identifies logical gaps, missing angles, weak transitions, and unsupported assertions
- Flags contradictions between sections
- Produces a structured issues list with: issue description, affected section, severity (blocking / major / minor), and a concrete suggestion

---

### Supervisor

The Supervisor is the only agent that sees the whole picture: the original query, the research plan, all draft versions, and the Critic's issues. Its job is to decide whether the current state is good enough and what the right next step is.

**What it does:**
- Evaluates the current draft holistically against the success criteria defined in the plan
- Assesses whether the Critic's issues are blocking or acceptable
- Decides one of three outcomes:
  - **DONE** — draft meets criteria, or maximum revision iterations reached → END
  - **REVISE** — specific sections need work → routes back to Writer with targeted feedback
  - **REPLAN** — the overall research direction or structure is fundamentally wrong → routes back to Researcher
---