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

### 3. API key

```bash
copy .env.sample .env
```

Edit `.env` — replace `your_key_here` with your OpenRouter API key from https://openrouter.ai/keys

### 4. (Optional) Change model

Edit `DEFAULT_MODEL` in `src/util.py`. Use any model string from https://openrouter.ai/models

## Run

```bash
python main.py
```

Change the research query by editing `DEFAULT_QUERY` in `main.py`.

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