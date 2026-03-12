import argparse
import sys
import time
from pathlib import Path
from src.graph import build_graph
import src.llm
from src.processing.document import DocProcessor

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
        help="Path to the PDF to analyse (default: %(default)s)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Pause after document extraction and require user confirmation to continue",
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
    
    _t0 = time.perf_counter()
    processor = DocProcessor()
    artifacts = processor.process_document(str(pdf_path))
    _pdf_elapsed = time.perf_counter() - _t0
    
    if artifacts.chunk_count > 0:
        print(f"[preprocessing] PDF extraction completed in {_pdf_elapsed:.2f}s", flush=True)
        
    if args.interactive:
        try:
            response = input("Press Enter to continue, or type 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAborted.")
        if response == "q":
            sys.exit("Execution stopped by user.")
    
    status = "Extracted" if artifacts.chunk_count > 0 else "FAILED TO EXTRACT (running without documents)"
    preprocessing_message = (
        f"[preprocessing] {status} multimodal artifacts "
        f"(images={artifacts.image_count}, tables={artifacts.table_count}, "
        f"equations={artifacts.equation_count}, chunks={artifacts.chunk_count})"
    )

    graph = build_graph()

    result = graph.invoke({
        "query": args.query or DEFAULT_QUERY,
        "source_chunks": artifacts.source_chunks,
        "selected_chunk_indices": [],
        "manifest_json": artifacts.manifest_json,
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
