import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.util import DEFAULT_MODEL, OPENROUTER_BASE_URL

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))


def get_llm(model_name: str | None = None) -> BaseChatModel:
    """Return a ChatOpenAI instance pointed at OpenRouter.

    Swap internals here to switch providers — callers only see BaseChatModel.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in environment or .env")

    resolved_model = model_name or DEFAULT_MODEL
    if not resolved_model.endswith(":free"):
        resolved_model += ":free"

    return ChatOpenAI(
        model=resolved_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )
