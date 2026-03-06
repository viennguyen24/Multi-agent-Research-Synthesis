import argparse
from src.graph import build_graph
import src.llm

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"


def main():
    parser = argparse.ArgumentParser(description="Run the research synthesis agent.")
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument("--open-router", action="store_true", help="Use OpenRouter provider")
    provider_group.add_argument("--ollama", action="store_true", help="Use Ollama Cloud provider (default)")
    parser.add_argument("--query", type=str, default=None, help="Research query")
    args = parser.parse_args()

    if args.open_router:
        src.llm.GLOBAL_CONFIG.provider = "openrouter"
    else:
        src.llm.GLOBAL_CONFIG.provider = "ollama"

    graph = build_graph()

    result = graph.invoke({
        "query": args.query or DEFAULT_QUERY,
        "plan": "",
        "draft": "",
        "critique": "",
        "iteration": 0,
        "next": "continue",
        "messages": [],
    })

    print("\n--- Agent Log ---")
    for msg in result.get("messages", []):
        print(msg)

    print("\n--- Final Draft ---")
    print(result.get("draft", "(no draft produced)"))


if __name__ == "__main__":
    main()
