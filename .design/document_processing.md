# Document Processing Design

Detailed technical implementation of document ingestion, chunking, and artifact extraction.

## Multimodal Artifacts

Each run ingests the PDF and writes artifacts to `artifacts/<doc>`, which includes:

- `document.md`
- `chunks.jsonl`
- `images/`
- `tables/`
- `equations.jsonl`
- `manifest.json`

### Artifact Content

- `document.md`: Human-readable audit artifact. Not loaded into agent state. Any agent that needs the full annotated document can retrieve its path from the manifest and read it from disk.
- `chunks.jsonl`: One JSON object per line, produced by Docling's `HybridChunker`. Each record contains raw text, contextualized text with heading breadcrumbs, ancestor headings, and associated captions.
- `equations.jsonl`: Separate equation records for deterministic downstream processing.
- `manifest.json`: Index of all extracted assets (images, tables, equations) and references mapping markdown tokens to concrete artifacts.

## Document Chunking

Documents are split into semantically coherent, token-aware chunks using Docling's `HybridChunker`, which operates directly on the `DoclingDocument` object before any Markdown export. It respects document structure — chunks never break mid-sentence or mid-paragraph, and each chunk carries the full heading breadcrumb of the section it belongs to.

> [!NOTE]
> "Token indices sequence length is longer than the specified maximum sequence length for this model" is a documented false alarm in Docling's `HybridChunker`.

## Agent Integration

### How agents use chunks

The lead researcher receives a **chunk directory** (heading breadcrumbs only) to understand document structure. It selects relevant chunk indices based on keyword overlap with the query. The editor and critic nodes then receive only the `contextualized_text` of those selected chunks.

### Extending to full RAG

The keyword scorer can be replaced with an embedding similarity search. By embedding the `contextualized_text` at ingestion time, retrieval can switch to vector similarity without altering the downstream agent logic.
