import time
import json

from src.state import ResearchState
from src.llm import get_llm
from src.util import AGENT_ROLES, MAX_ITERATIONS
from src.ingestion import extract_multimodal_pdf_artifacts


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


def _build_document_context(state: ResearchState, markdown_limit: int = 12000) -> str:
    markdown = state.get("source_markdown", "")
    markdown_excerpt = markdown[:markdown_limit]

    manifest = state.get("manifest_json", {}) or {}
    manifest_summary = {
        "doc_id": manifest.get("doc_id"),
        "markdown_path": manifest.get("markdown_path"),
        "image_count": len(manifest.get("images", [])),
        "table_count": len(manifest.get("tables", [])),
        "equation_count": len(manifest.get("equations", [])),
        "references_preview": manifest.get("references", [])[:10],
    }

    return (
        "Source document context (Markdown excerpt):\n"
        f"{markdown_excerpt}\n\n"
        "Artifact manifest summary:\n"
        f"{json.dumps(manifest_summary, indent=2)}"
    )


def ingest_pdf_node(state: ResearchState) -> dict:
    artifacts = extract_multimodal_pdf_artifacts(state["source_pdf_path"])
    message = (
        "[ingest_pdf] Extracted multimodal artifacts "
        f"(images={artifacts['image_count']}, tables={artifacts['table_count']}, equations={artifacts['equation_count']})"
    )
    return {
        "artifact_root": artifacts["artifact_root"],
        "manifest_path": artifacts["manifest_path"],
        "manifest_json": artifacts["manifest_json"],
        "source_markdown": artifacts["source_markdown"],
        "messages": [message],
    }


def lead_researcher_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0)
    document_context = _build_document_context(state)

    if iteration == 0:
        prompt = (
            f"Research query:\n{state['query']}\n\n"
            f"{document_context}\n\n"
            "Produce a structured research plan that uses the markdown and referenced artifacts."
        )
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
        "next": "continue",
        "iteration": iteration + 1,
        "messages": [f"[lead_researcher] Requesting revision (iteration {iteration + 1})"],
    }


def editor_node(state: ResearchState) -> dict:
    document_context = _build_document_context(state)
    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"{document_context}\n\n"
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
    document_context = _build_document_context(state)
    prompt = (
        f"Research query:\n{state['query']}\n\n"
        f"{document_context}\n\n"
        f"Draft to review:\n{state.get('draft', '')}\n\n"
        "List specific, actionable problems. Do not rewrite — only identify issues."
    )
    critique = _call_llm("critic", prompt)
    return {
        "critique": critique,
        "messages": ["[critic] Reviewed draft"],
    }
