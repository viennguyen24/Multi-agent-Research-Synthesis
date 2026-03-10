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
Also Docling requires 1.5-2GB for it's custom model.
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

The PDF input is currently hardcoded to `Transformers.pdf` in `main.py`.
You can still change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.

## Multimodal Artifacts

Each run ingests `Transformers.pdf` and writes artifacts to:

- `artifacts/transformers/document.md`
- `artifacts/transformers/images/`
- `artifacts/transformers/tables/`
- `artifacts/transformers/equations.jsonl`
- `artifacts/transformers/manifest.json`

### Artifact contract

- `document.md`: primary text context for agents; equations are kept inline in markdown text.
- `equations.jsonl`: separate equation records for deterministic downstream processing.
- `manifest.json`: index of all extracted assets and references. Includes:
  - `images[]` with IDs and file paths
  - `tables[]` with IDs and file paths
  - `equations[]` with IDs, equation text, and markdown anchors
  - `references[]` tokens mapping markdown references to concrete artifacts

Agents consume markdown content plus manifest metadata so they can reference tables, images, and equations reliably.

## Graph Flow

```
START → lead_researcher
                  │
                  ├─ next=="continue" → editor → critic ─┐
                  │                                      │
                  └◄─────────────────────────────────────┘
                  │
                  └─ next=="done" → END
```