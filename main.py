from datetime import datetime
from uuid import uuid4

from src.graph import build_graph

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"
DEFAULT_SOURCE_PDF = "Transformers.pdf"


def main():
    graph = build_graph()

    result = graph.invoke({
        "query": DEFAULT_QUERY,
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
