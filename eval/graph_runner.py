from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.llm.llm import current_session_id, init_from_config
from src.logging.logger import AgentLogger, VALIDATION_ERRORS_DIR
from src.memory.objectstore import (
    DEFAULT_OBJECT_STORE_CONFIG,
    LocalObjectStore,
    R2ObjectStore,
)
from src.memory.research.config import StorageConfig
from src.memory.research.database import ResearchDatabase
from src.processing.chunker import get_text_chunker
from src.processing.context.contextualizer import ContextConfig, Contextualizer
from src.processing.context.document import DocumentContextConfig, DocumentContextualizer
from src.processing.document import DocProcessor
from src.processing.embedder.provider import get_text_embedder
from src.processing.export.pandoc_builder import PandocBuilder
from src.retriever import Retriever
from src.state import MAX_CYCLES, make_initial_review_state
from src.tools.registry import build_tool_registry


DEFAULT_QUERY = "Explain this paper to an audience of laypeople"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
_PROCESSOR_BACKEND_ALIASES = {
    "llama": "llama_parse",
    "llama_parse": "llama_parse",
}
_TEXT_SPLITTER_ALIASES = {
    "none": None,
    "semantic": "semantic",
}


@dataclass(slots=True)
class DocumentProcessResult:
    doc_ids: list[str]
    paper_titles: list[str]
    preprocessing_messages: list[str]


@dataclass(slots=True)
class GraphRunResult:
    session_id: str
    status: str
    final_state: dict[str, Any]
    node_events: list[Any]
    pptx_path: str | None
    final_warnings: list[str]
    error_text: str | None = None


def _configure_llm(llm_config_path: str | None) -> None:
    init_from_config(config_path=llm_config_path)


def _get_callbacks(logging_enabled: bool, logger: AgentLogger, session_id: str):
    callbacks = []
    if not logging_enabled:
        os.environ["LANGFUSE_ENABLED"] = "false"
        from langfuse.decorators import langfuse_context

        langfuse_context.configure(enabled=False)
    else:
        current_session_id.set(session_id)
        callbacks.append(logger.get_langgraph_handler(session_id=session_id))
    return callbacks, logger


def _make_object_store(object_store: str | None, logger: AgentLogger) -> Any:
    if object_store == "local":
        return LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)
    if object_store == "r2":
        return R2ObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)
    try:
        return R2ObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)
    except Exception:
        logger.log("Falling back to local object store", level="warning")
        return LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)


def _sanitize_filename(name: str) -> str:
    if not name:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in name)
    safe = re.sub(r"[ _]+", "_", safe).strip("_")
    return safe[:64]


def _partial_deck_warnings(messages: list[str]) -> list[str]:
    return [
        msg for msg in messages if "RETRIES EXHAUSTED" in msg or "PARTIAL DECK" in msg
    ]


def _build_initial_state(
    query: str,
    preprocessing_messages: list[str],
    doc_ids: list[str],
    paper_titles: list[str],
    session_id: str,
    max_slides: int,
    skip_supervisor: bool,
    force_replan: bool,
    max_cycles: int,
) -> dict[str, Any]:
    return {
        "query": query or DEFAULT_QUERY,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_ids": doc_ids,
        "paper_titles": paper_titles,
        "max_slides": max_slides,
        "skip_supervisor": skip_supervisor,
        "plan_number": 1,
        "force_replan_at_max_cycles": force_replan,
        "slide_numbers": [],
        "presentation_plan": None,
        "review": make_initial_review_state(max_cycles=max_cycles),
        "retrieval_queries": [],
        "tool_calls": [],
        "tool_results": [],
        "slides_written": [],
        "critic_results": [],
        "review_summaries": [],
        "messages": preprocessing_messages,
        "errors": [],
    }


def _open_database(database_path: str | None) -> ResearchDatabase:
    if database_path is None:
        return ResearchDatabase()
    return ResearchDatabase(StorageConfig(db_path=Path(database_path)))


def process_documents(
    pdf_paths: list[str],
    llm_config_path: str | None,
    database_path: str | None,
    processor: str = "llama_parse",
    text_splitter: str = "semantic",
    interactive: bool = False,
    object_store: str | None = None,
    logging_enabled: bool = False,
    no_cache_control: bool = False,
    no_context_batching: bool = False,
) -> DocumentProcessResult:
    logger = AgentLogger()
    session_id = str(uuid.uuid4())
    _configure_llm(llm_config_path)
    _get_callbacks(logging_enabled, logger, session_id)
    store = _make_object_store(object_store, logger)
    with _open_database(database_path) as db:
        embedder = get_text_embedder()
        doc_ids: list[str] = []
        paper_titles: list[str] = []
        preprocessing_messages: list[str] = []
        seen_doc_ids: set[str] = set()

        for pdf_path_str in pdf_paths:
            artifacts, message = _process_document(
                pdf_path_str=pdf_path_str,
                processor=processor,
                text_splitter=text_splitter,
                interactive=interactive,
                logger=logger,
                db=db,
                object_store=store,
                embedder=embedder,
                no_cache_control=no_cache_control,
                no_context_batching=no_context_batching,
            )
            preprocessing_messages.append(message)
            if not artifacts:
                continue
            if artifacts.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(artifacts.doc_id)
            doc_ids.append(artifacts.doc_id)
            paper_titles.append(Path(artifacts.source_path).stem)
    return DocumentProcessResult(
        doc_ids=doc_ids,
        paper_titles=paper_titles,
        preprocessing_messages=preprocessing_messages,
    )


def _process_document(
    pdf_path_str: str,
    processor: str,
    text_splitter: str,
    interactive: bool,
    logger: AgentLogger,
    db: Any,
    object_store: Any,
    embedder: Any,
    no_cache_control: bool,
    no_context_batching: bool,
) -> tuple[Any, str]:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")
    start = time.perf_counter()
    backend_name = _PROCESSOR_BACKEND_ALIASES[processor]
    chunker_name = _TEXT_SPLITTER_ALIASES[text_splitter]
    text_chunker = get_text_chunker(chunker_name) if chunker_name else None
    contextualizer = Contextualizer(
        config=ContextConfig(
            model="context",
            cache_control=not no_cache_control,
            use_batch=not no_context_batching,
        ),
        object_store=object_store,
        logger=logger,
    )
    document_contextualizer = DocumentContextualizer(
        config=DocumentContextConfig(model="context"),
        logger=logger,
    )
    processor_instance = DocProcessor(
        backend=backend_name,
        text_chunker=text_chunker,
        db=db,
        contextualizer=contextualizer,
        document_contextualizer=document_contextualizer,
        embedder=embedder,
        logger=logger,
        object_store=object_store,
    )
    artifacts = asyncio.run(processor_instance.process_document(str(pdf_path)))
    elapsed = time.perf_counter() - start
    if artifacts and artifacts.chunk_count > 0:
        print(f"[preprocessing] {pdf_path.name} completed in {elapsed:.2f}s", flush=True)
    if interactive:
        response = input(
            f"Finished processing {pdf_path.name}. Press Enter to continue, or 'q' to quit: "
        ).strip().lower()
        if response == "q":
            sys.exit("Execution stopped by user.")
    status = "Processed" if artifacts and artifacts.chunk_count > 0 else "FAILED TO PROCESS"
    if artifacts:
        message = (
            f"[preprocessing] {pdf_path.name}: {status} "
            f"(images={artifacts.image_count}, tables={artifacts.table_count}, "
            f"equations={artifacts.equation_count}, chunks={artifacts.chunk_count})"
        )
    else:
        message = f"[preprocessing] {pdf_path.name}: {status}"
    return artifacts, message


def run_existing_documents(
    query: str,
    doc_ids: list[str],
    paper_titles: list[str],
    llm_config_path: str | None,
    output_dir: str | None,
    database_path: str | None,
    existing_docs_only: bool,
    clear_run_artifacts: bool,
    max_slides: int = 15,
    max_cycles: int = MAX_CYCLES,
    skip_supervisor: bool = False,
    force_replan: bool = False,
    object_store: str | None = None,
    logging_enabled: bool = False,
    reference_doc: str | None = None,
) -> GraphRunResult:
    logger = AgentLogger()
    session_id = str(uuid.uuid4())
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    if VALIDATION_ERRORS_DIR.exists():
        shutil.rmtree(VALIDATION_ERRORS_DIR)
    VALIDATION_ERRORS_DIR.mkdir(exist_ok=True)
    _configure_llm(llm_config_path)
    callbacks, logger = _get_callbacks(logging_enabled, logger, session_id)
    store = _make_object_store(object_store, logger)

    with _open_database(database_path) as db:
        if existing_docs_only:
            missing = [doc_id for doc_id in doc_ids if db.load_document(doc_id) is None]
            if missing:
                raise FileNotFoundError(
                    f"Missing processed documents for existing-docs-only run: {', '.join(missing)}"
                )
        if clear_run_artifacts:
            db.clear_proto_slides()
            db.clear_slide_review_events()

        retriever = Retriever(db, get_text_embedder())
        tool_registry = build_tool_registry(retriever=retriever, research_db=db)
        agent_tool_allowlist = {
            "slide_writer": ["retrieve_artifacts"],
            "planner": [],
            "critic": [],
            "supervisor": [],
            "parse_supervisor": [],
            "research_to_slide": [],
        }
        initial_state = _build_initial_state(
            query=query,
            preprocessing_messages=[],
            doc_ids=doc_ids,
            paper_titles=paper_titles,
            session_id=session_id,
            max_slides=max_slides,
            skip_supervisor=skip_supervisor,
            force_replan=force_replan,
            max_cycles=max_cycles,
        )
        graph = build_graph(
            tool_registry=tool_registry,
            agent_tool_allowlist=agent_tool_allowlist,
        )
        final_state = initial_state
        node_events: list[Any] = []
        error_text = None
        try:
            for index, event in enumerate(
                graph.stream(
                    initial_state,
                    config={"callbacks": callbacks, "recursion_limit": 100},
                    stream_mode="values",
                )
            ):
                final_state = event
                node_events.append(
                    {
                        "transcript_id": "",
                        "sequence_index": index,
                        "event_type": "state",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "node_name": event.get("last_node") if isinstance(event, dict) else None,
                        "payload": {
                            "message_count": len(event.get("messages", [])),
                            "error_count": len(event.get("errors", [])),
                        },
                    }
                )
        except Exception as exc:
            error_text = str(exc)

        plan_title = ""
        plan_subtitle = ""
        raw_name = paper_titles[0] if paper_titles else session_id
        presentation_plan = final_state.get("presentation_plan")
        if presentation_plan and getattr(presentation_plan, "title", None):
            raw_name = presentation_plan.title
            plan_title = presentation_plan.title
            plan_subtitle = getattr(presentation_plan, "subtitle", "") or ""
        safe_name = _sanitize_filename(raw_name) or session_id
        output_path.mkdir(parents=True, exist_ok=True)
        pptx_count = len(list(output_path.glob("*.pptx")))
        pptx_path = output_path / f"{pptx_count + 1} - {safe_name}.pptx"

        exported_path = None
        status = "failed" if error_text else "success"
        if final_state.get("review", {}).get("export_ready"):
            try:
                reference_doc_path = Path(reference_doc).expanduser().resolve() if reference_doc else None
                exported_path = str(
                    PandocBuilder(
                        output_path=pptx_path,
                        db=db,
                        title=plan_title,
                        subtitle=plan_subtitle,
                        object_store=store,
                        reference_doc=reference_doc_path,
                    ).build()
                )
            except ValueError as exc:
                error_text = str(exc)
                status = "partial"
        elif not error_text:
            status = "partial"

        return GraphRunResult(
            session_id=session_id,
            status=status,
            final_state=final_state,
            node_events=node_events,
            pptx_path=exported_path,
            final_warnings=_partial_deck_warnings(final_state.get("messages", [])),
            error_text=error_text,
        )
