import argparse
import sys
import time
from pathlib import Path
from src.database import get_database_provider
from src.graph import build_graph
import src.llm
from src.processing.document import DocProcessor
from src.processing.document._common import _slugify
from src.llm import GLOBAL_CONFIG, Provider
from src.logging.logger import AgentLogger

DEFAULT_QUERY = "Explain the CAP theorem in distributed systems"
DEFAULT_SOURCE_PDF = "Transformers.pdf"

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
    group.add_argument("--gemini",     action="store_true", help="Use Google AI Studio (Gemini) provider (default)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Research query")
    parser.add_argument(
        "--pdf",
        type=str,
        metavar="PATH",
        default=DEFAULT_SOURCE_PDF,
        help="Path to the PDF to analyse (default: %(default)s)"
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Skip document processing and attempt to load artifacts from the existing SQLite database",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Pause after document extraction and require user confirmation to continue",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # resolve which provider flag was set (fallback Google AI studio)
    selected = next(
        (prov for flag, prov in _PROVIDER_FLAGS.items() if getattr(args, flag)),
        Provider.GOOGLE_AI_STUDIO,
    )
    GLOBAL_CONFIG.provider = selected

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")
    
    _t0 = time.perf_counter()
    doc_id = _slugify(pdf_path.stem)
    db = get_database_provider()
    
    if args.use_db:
        db.setup()
        artifacts = db.load_result(doc_id)
        if not artifacts:
            sys.exit(f"error: No cached database entry found for doc_id '{doc_id}'. The existing processor.db does not match the requested PDF. Please run without --use-db to re-process the document.")
    else:
        db.reset()
        processor = DocProcessor()
        artifacts = processor.process_document(str(pdf_path))
        db.save_result(artifacts)
        
    _pdf_elapsed = time.perf_counter() - _t0
    
    if artifacts.chunk_count > 0:
        source_str = "DATABASE" if args.use_db else "Docling"
        print(f"[preprocessing] PDF extraction from {source_str} completed in {_pdf_elapsed:.2f}s", flush=True)
        
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

    logger = AgentLogger()
    langfuse_handler = logger.get_langgraph_handler()

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
    }, config={"callbacks": [langfuse_handler]})

    print("\n--- Agent Log ---")
    for msg in result.get("messages", []):
        print(msg)

    print("\n--- Final Draft ---")
    print(result.get("draft", "(no draft produced)"))
    
    logger.flush()


if __name__ == "__main__":
    main()

