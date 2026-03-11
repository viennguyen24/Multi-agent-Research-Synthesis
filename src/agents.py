import time
import json

from src.state import ResearchState
from src.llm import get_llm
from src.util import AGENT_ROLES, MAX_ITERATIONS


def _call_llm(role: str, user_prompt: str, max_retries: int = 2) -> str:
    print(f"  -> requesting {role}...", flush=True)
    messages = [
        {"role": "system", "content": AGENT_ROLES[role]},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(max_retries):
        try:
            llm = get_llm()
            response_text = llm.complete(messages)
            print(f"  OK {role} responded ({len(response_text)} chars)", flush=True)
            return response_text
        except Exception as e:
            print(f"  [!] LLM connection error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts.") from e
            # Sleep before retrying to give the network/API time to recover
            time.sleep(2 ** attempt)


def _manifest_summary(state: ResearchState) -> str:
    manifest = state.get("manifest_json", {})
    summary = {
        "doc_id": manifest.get("doc_id"),
        "markdown_path": manifest.get("markdown_path"),
        "image_count": len(manifest.get("images", [])),
        "table_count": len(manifest.get("tables", [])),
        "equation_count": len(manifest.get("equations", [])),
        "references_preview": manifest.get("references", [])[:10],
    }
    return json.dumps(summary, indent=2)


def _build_chunk_directory(state: ResearchState) -> str:
    """Numbered table of contents for the lead researcher — headings only, no body text."""
    chunks = state.get("source_chunks", [])
    if not chunks:
        return "(no chunks available)"
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        headings = chunk.get("headings", [])
        label = " > ".join(headings) if headings else "(no heading)"
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def _select_chunk_indices(state: ResearchState, char_budget: int = 10000) -> list[int]:
    """Score chunks by heading/query keyword overlap and return the top indices in document order."""
    chunks = state.get("source_chunks", [])

    # For now, get query words from which we determine the top chunks to select, from the state
    # Technically, best practice to have unified query processing across all agents, so it is done here with query words in the state
    query_words = set(state.get("query", "").lower().split())

    scored: list[tuple[int, int]] = []
    for i, chunk in enumerate(chunks):
        heading_words = set(" ".join(chunk.get("headings", [])).lower().split())
        score = len(query_words & heading_words)
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[int] = []
    used = 0
    for _score, i in scored:
        text = chunks[i].get("contextualized_text", "")
        if used + len(text) > char_budget:
            continue
        selected.append(i)
        used += len(text)

    return sorted(selected)


def _get_selected_chunk_texts(state: ResearchState) -> str:
    """Return the contextualized text for each selected chunk index, joined with a separator."""
    chunks = state.get("source_chunks", [])
    indices = state.get("selected_chunk_indices", [])
    if not indices:
        return "(no document chunks selected)"
    separator = "\n\n---\n\n"
    parts = [chunks[i].get("contextualized_text", "") for i in indices if i < len(chunks)]
    return separator.join(parts)


def lead_researcher_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0)

    selected_indices = _select_chunk_indices(state)
    chunk_directory = _build_chunk_directory(state)
    manifest = _manifest_summary(state)

    document_context = (
        "Document chunk directory (index: heading breadcrumb):\n"
        f"{chunk_directory}\n\n"
        "Artifact manifest summary:\n"
        f"{manifest}"
    )

    if iteration == 0:
        prompt = (
            f"Research query:\n{state['query']}\n\n"
            f"{document_context}\n\n"
            "Produce a structured research plan that references the relevant document sections above."
        )
        plan = _call_llm("lead_researcher", prompt)
        return {
            "plan": plan,
            "selected_chunk_indices": selected_indices,
            "next": "continue",
            "iteration": 1,
            "messages": [
                f"[lead_researcher] Created research plan (iteration 1, {len(selected_indices)} chunks selected)"
            ],
        }

    if iteration >= MAX_ITERATIONS:
        return {
            "next": "done",
            "iteration": iteration,
            "messages": [f"[lead_researcher] Max iterations ({MAX_ITERATIONS}) reached — finalizing"],
        }

    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"{document_context}\n\n"
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
        "selected_chunk_indices": selected_indices,
        "next": "continue",
        "iteration": iteration + 1,
        "messages": [f"[lead_researcher] Requesting revision (iteration {iteration + 1})"],
    }


def editor_node(state: ResearchState) -> dict:
    chunk_texts = _get_selected_chunk_texts(state)
    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"Relevant document excerpts:\n{chunk_texts}\n\n"
        f"Research plan / guidance:\n{state.get('plan', '')}\n\n"
    )

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
    chunk_texts = _get_selected_chunk_texts(state)
    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"Relevant document excerpts:\n{chunk_texts}\n\n"
        f"Draft to review:\n{state.get('draft', '')}\n\n"
        "List specific, actionable problems. Do not rewrite — only identify issues."
    )
    critique = _call_llm("critic", prompt)
    return {
        "critique": critique,
        "messages": ["[critic] Reviewed draft"],
    }
