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

Edit `.env` — replace the placeholder values with your LLM Provider's API key. We have options for
- [OpenRouter API](https://openrouter.ai/keys) 
- [Ollama Cloud API](https://ollama.com/settings/keys)
- [Google AI Studio](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)

### 4. (Optional) Change model

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

You can change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.

## Graph Flow

```
[ENTRY] → researcher → planner → writer → critic → supervisor → [END]
                ↑                     ↑                   │
                │                     └── REVISE ─────────┤
                └──────────── REPLAN ─────────────────────┘
```