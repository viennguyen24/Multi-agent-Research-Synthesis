# Document Processing Design

Detailed technical implementation of document ingestion, chunking, and artifact extraction.

## OCR Backend Architecture

The pipeline uses the **Strategy pattern** to decouple document extraction from any specific OCR library. All backends implement a single interface:

```python
class OCRBackend(ABC):
    @abstractmethod
    def extract(self, source_pdf_path: str) -> ExtractionResult: ...
```

### Backend Registry

`processor.py` maintains a `BACKEND_REGISTRY` mapping string keys to concrete backend classes:

| Key | Class | Status |
|------|------|--------|
| `"lighton"` | `LightOnOCRBackend` | ✅ Active (Too slow CPU inference) |
| `"docling"` | `DoclingBackend` | ✅ Active (has issues with subscripts in text)|
| `"chandra"` | `ChandraOCRBackend` | 🚧 Not written yet |
| `"glm"` | `GLMOCRBackend` | 🚧 Not written yet |

`DocProcessor` accepts an optional `backend` parameter (string key or `OCRBackend` instance). It defaults to `"lighton"`.

### Adding a new backend

1. Create a new module in `src/processing/document/backends/`.
2. Subclass `OCRBackend` and implement `extract()` to return an `ExtractionResult`.
3. Register the class in `processor.BACKEND_REGISTRY`.
4. (Optional) Add a CLI flag in `main.py` (e.g. `--ocr-backend`) and pass the key to `DocProcessor(backend=...)`.

### File layout

```
src/processing/document/
├── _common.py               # Schema-level helpers (no OCR dependency)
├── backend_base.py          # OCRBackend ABC
├── backends/
│   ├── __init__.py
│   ├── docling_backend.py   # Full Docling pipeline (parse → chunk → extract → manifest)
│   └── lighton_backend.py   # Stub for LightOnOCR-2-1B
├── processor.py             # DocProcessor + backend factory/registry
└── schema.py                # Shared data models
```

## Multimodal Artifacts

Each run ingests the PDF and writes artifacts to `artifacts/<doc>`, which includes:

- `research.db`: SQLite database (located in `data/`) containing:
    - `images`: Binary BLOBs of extracted pictures.
    - `tables`: HTML representations of tables.
    - `equations`: LaTeX or raw text formulas.
    - `text_chunks`: Semantic markdown chunks with metadata.
    - `manifest`: Index of document-artifact associations.

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
