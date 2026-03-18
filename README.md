# Multi-Agent Research Synthesis

LangGraph-coordinated research workflow:
Document Ingestion -> Lead Researcher -> Editor -> Critic loop.

## Setup

### 1. Python 3.11+

```bash

```

### 2. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/Scripts/activate
pip install -r requirements.txt
```

Note: installing `docling` can take longer than the previous setup because it brings document-processing dependencies.
Also Docling requires 1.5-2GB for its custom model.
### 3. API keys

```bash
copy .env.sample .env
```

Edit `.env` — replace the placeholder values with your LLM Provider's API key. We have options for
- [OpenRouter API](https://openrouter.ai/keys) 
- [Ollama Cloud API](https://ollama.com/settings/keys)
- [Google AI Studio](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)

### 4. Langfuse Logging Setup

To enable observability, ensure the following API keys are set in your `.env` file (you can get these from your Langfuse project settings):
```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```
The codebase uses `langfuse` which will automatically pick up these environment variables to trace agent runs.

### 5. (Optional) Change model

Edit `DEFAULT_MODEL` in `src/util.py`.

## Run

```bash
# To use Ollama Cloud (default):
python main.py --ollama

# To use OpenRouter:
python main.py --open-router

# To use Google Gemini
python main.py --gemini
```

### Optional Commandline Arguments

The PDF input defaults to `Transformers.pdf` in the project root directory. You can change this by adding `--pdf "Path to your PDF file here"` or editing `DEFAULT_SOURCE_PDF` in `main.py`.

You change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.
Adding the argument `-i` or `--interactive` adds a prompt for whether the user wants to continue, which pops up if a document is extracted and after the document extraction process is complete.

Adding `--use-db` (or `--skip-processing`) skips the Docling document extraction process and instead attempts to load the parsed PDF chunks and metadata directly from the `processor.db` SQLite database if it exists, saving valuable API and compute time during iterative runs pipeline tuning.

## Graph Flow

```
START → lead_researcher (selects chunk indices)
                  │
                  ├─ next=="continue" → editor → critic ─┐
                  │         (uses selected chunks)       │
                  └◄─────────────────────────────────────┘
                  │
                  └- next=="done" → END
```

## Telemetry & Logging

The system implements a dual-layer observability strategy to maintain clean agent logic while ensuring comprehensive tracing:
- **Workflow Tracing**: Captures the high-level orchestration, state transitions, and routing overhead as the research document flows between the specialized agents.
- **Cognitive Tracing**: Instruments the underlying LLM calls to capture precise generation metrics (latency, token usage) and raw prompt details completely independently of the graph execution.