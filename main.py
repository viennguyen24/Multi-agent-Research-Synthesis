import argparse
import sys
from pathlib import Path
from src.graph import build_graph
import src.llm
from src.ingestion import extract_multimodal_pdf_artifacts

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"
DEFAULT_SOURCE_PDF = "Transformers.pdf"


def main():
    parser = argparse.ArgumentParser(description="Run the research synthesis agent.")
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument("--open-router", action="store_true", help="Use OpenRouter provider")
    provider_group.add_argument("--ollama", action="store_true", help="Use Ollama Cloud provider (default)")
    parser.add_argument("--query", type=str, default=None, help="Research query")
    parser.add_argument(
        "--pdf",
        type=str,
        metavar="PATH",
        default=DEFAULT_SOURCE_PDF,
        help="Path to the PDF to analyse (default: %(default)s)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")

    if args.open_router:
        src.llm.GLOBAL_CONFIG.provider = "openrouter"
    else:
        src.llm.GLOBAL_CONFIG.provider = "ollama"

    artifacts = extract_multimodal_pdf_artifacts(str(pdf_path))
    preprocessing_message = (
        "[preprocessing] Extracted multimodal artifacts "
        f"(images={artifacts['image_count']}, tables={artifacts['table_count']}, equations={artifacts['equation_count']})"
    )

    graph = build_graph()

    result = graph.invoke({
        "query": args.query or DEFAULT_QUERY,
        "source_markdown": artifacts["source_markdown"],
        "manifest_json": artifacts["manifest_json"],
        "plan": "",
        "draft": "",
        "critique": "",
        "iteration": 0,
        "next": "continue",
        "messages": [preprocessing_message],
    })

    print("\n--- Agent Log ---")
    for msg in result.get("messages", []):
        print(msg)

    print("\n--- Final Draft ---")
    print(result.get("draft", "(no draft produced)"))


if __name__ == "__main__":
    main()
