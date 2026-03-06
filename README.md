# Multi-Agent Research Synthesis

LangGraph-coordinated research workflow: Lead Researcher → Editor → Critic loop via OpenRouter.

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

You can change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.

## Graph Flow

```
START → lead_researcher
             │
             ├─ next=="continue" → editor → critic ─┐
             │                                       │
             └◄──────────────────────────────────────┘
             │
             └─ next=="done" → END
```