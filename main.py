from src.graph import build_graph

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"


def main():
    graph = build_graph()

    result = graph.invoke({
        "query": DEFAULT_QUERY,
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
