import argparse
from src.graph import build_graph
import src.llm

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"


def main():
    parser = argparse.ArgumentParser(description="Run the research synthesis agent.")
    parser.add_argument("--open-router", action="store_true", help="Use OpenRouter provider")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama Cloud provider (default)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Research query")
    args = parser.parse_args()

    if args.open_router:
        src.llm.GLOBAL_CONFIG.provider = "openrouter"
    else:
        src.llm.GLOBAL_CONFIG.provider = "ollama"

    graph = build_graph()

    result = graph.invoke({
        "query": args.query,
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
