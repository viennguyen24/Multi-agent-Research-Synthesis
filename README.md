# Multi-Agent Research Synthesis

LangGraph-coordinated research workflow:
Document Ingestion -> Lead Researcher -> Editor -> Critic loop.

## Setup

### 1. Python 3.11+

```bash
python --version
```

### 2. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Note: installing `docling` can take longer than the previous setup because it brings document-processing dependencies.
Also Docling requires 1.5-2GB for its custom model.
### 3. API keys

```bash
copy .env.sample .env
```

Edit `.env` — replace the placeholder values with your OpenRouter API key (from https://openrouter.ai/keys) and Ollama API key (from https://ollama.com/settings/keys).

### 4. (Optional) Change model

Edit `DEFAULT_MODEL` in `src/util.py`. Use any model string from https://openrouter.ai/models or Ollama.

## Run

```bash
# To use Ollama Cloud (default):
python main.py --ollama

# To use OpenRouter:
python main.py --open-router
```

The PDF input defaults to `Transformers.pdf` in the project root directory. You can change this by adding `--pdf "Path to your PDF file here"` or editing `DEFAULT_SOURCE_PDF` in `main.py`.
You change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.

## Multimodal Artifacts

Each run ingests the PDF and writes artifacts to:

- `artifacts/<doc>/document.md`
- `artifacts/<doc>/chunks.jsonl`
- `artifacts/<doc>/images/`
- `artifacts/<doc>/tables/`
- `artifacts/<doc>/equations.jsonl`
- `artifacts/<doc>/manifest.json`

### Artifact content

- `document.md`: human-readable audit artifact. Not loaded into agent state. Any agent that needs the full annotated document can reach it via `state["manifest_json"]["markdown_path"]` and read it from disk.
- `chunks.jsonl`: one JSON object per line, produced by Docling's `HybridChunker`. Each record contains:
  - `text`: raw chunk content with no heading context
  - `contextualized_text`: heading breadcrumb prepended to the text (e.g. `"3 Model Architecture\n3.2 Attention\n<body text>"`) — use this field when passing content to an LLM or embedding model
  - `headings`: list of ancestor heading strings from document root to the nearest section, e.g. `["Attention is All You Need", "3 Model Architecture", "3.2 Attention"]`
  - `captions`: list of any figure/table captions associated with the chunk
- `equations.jsonl`: separate equation records for deterministic downstream processing.
- `manifest.json`: index of all extracted assets and references. Includes:
  - `images[]` with IDs and file paths
  - `tables[]` with IDs and file paths
  - `equations[]` with IDs, equation text, and markdown anchors
  - `references[]` tokens mapping markdown references to concrete artifacts

## Document Chunking

Documents are split into semantically coherent, token-aware chunks using Docling's `HybridChunker`, which operates directly on the `DoclingDocument` object before any Markdown export. It respects document structure — chunks never break mid-sentence or mid-paragraph, and each chunk carries the full heading breadcrumb of the section it belongs to.
"Token indices sequence length is longer than the specified maximum sequence length for this model (970 > 512). Running this sequence through the model will result in indexing errors" - This error is a documented false alarm (https://docling-project.github.io/docling/faq/#hybridchunker-triggers-warning-token-indices-sequence-length-is-longer-than-the-specified-maximum-sequence-length-for-this-model).

### How agents use chunks

The lead researcher receives a **chunk directory** — a numbered list of heading breadcrumbs with no body text — and uses it to understand the document's structure. Before calling the LLM, the lead researcher node selects relevant chunk indices by scoring each chunk's heading words against the research query (keyword overlap). The selected indices are stored in `ResearchState["selected_chunk_indices"]`.

The editor and critic receive only the `contextualized_text` of those selected chunks — no other document content is passed to them.

### Extending to full RAG

The keyword scorer in `_select_chunk_indices` (`src/agents.py`) can be replaced with an embedding similarity search without changing anything else. Embed each `chunk["contextualized_text"]` at ingestion time and store the vectors alongside `chunks.jsonl`; at query time, retrieve the top-k by cosine similarity instead of word overlap and return the same list of indices.

## Graph Flow

```
START → lead_researcher (selects chunk indices)
                  │
                  ├─ next=="continue" → editor → critic ─┐
                  │         (uses selected chunks)        │
                  └◄─────────────────────────────────────┘
                  │
                  └─ next=="done" → END
```