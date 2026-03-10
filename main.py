import argparse
from datetime import datetime
from uuid import uuid4

from src.graph import build_graph
from src.llm import GLOBAL_CONFIG, Provider

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"

_PROVIDER_FLAGS = {
    "ollama":     Provider.OLLAMA,
    "openrouter": Provider.OPENROUTER,
    "gemini":     Provider.GOOGLE_AI_STUDIO,
}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent research synthesis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ollama",     action="store_true", help="Use Ollama provider")
    group.add_argument("--openrouter", action="store_true", help="Use OpenRouter provider")
    group.add_argument("--gemini",     action="store_true", help="Use Google AI Studio (Gemini) provider")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Research query")
    return parser.parse_args()


def main():
    args = _parse_args()

    # resolve which provider flag was set (fallback Google AI studio)
    selected = next(
        (prov for flag, prov in _PROVIDER_FLAGS.items() if getattr(args, flag)),
        Provider.GOOGLE_AI_STUDIO,
    )
    GLOBAL_CONFIG.provider = selected

    graph = build_graph()

    result = graph.invoke({
        "query": args.query,
        "session_id": str(uuid4()),
        "created_at": datetime.now().isoformat(),
        "iteration_count": 0,
        "research_context": [],
        "plan": {},
        "drafts": [],
        "critiques": [],
        "revisions": [],
        "errors": [],
        "messages": [],
    })

    print("\n--- Agent Log ---")
    for msg in result.get("messages", []):
        print(msg)

    drafts = result.get("drafts", [])
    if drafts:
        final_draft = drafts[-1]
        print(f"\n--- Final Draft (v{final_draft['version']}, {final_draft['word_count']} words) ---")
        print(final_draft["document"])
    else:
        print("\n--- No draft produced ---")


if __name__ == "__main__":
    main()

