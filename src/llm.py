from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
from google import genai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
import re

from src.util import DEFAULT_MODEL

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

def _strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3).
    Also strips any leading/trailing whitespace so the remainder is clean JSON.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class Provider(str, Enum):
    OPENROUTER        = "openrouter"
    OLLAMA            = "ollama"
    GOOGLE_AI_STUDIO  = "google_ai_studio"

# For now we will use provider, model, base url and api key fields. Other fields will be updated once a need is found
@dataclass
class LLMConfig:
    provider: Provider | str = Provider.OLLAMA
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool | None = None
    base_url: str | None = None
    api_key: str | None = None

    def resolved_api_key(self, env_var: str) -> str:
        if self.api_key:
            return self.api_key
        key = os.environ.get(env_var)
        if not key:
            raise EnvironmentError(f"Missing API key: set LLMConfig.api_key or {env_var}")
        return key



class OpenRouterLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.resolved_api_key("OPENROUTER_API_KEY"),
            base_url=config.base_url,
        )

    def complete(self, messages: list[dict], schema: type[T] | None = None, strip_think: bool = True, **kwargs) -> str | T:
        req_params = {
            "model": self.config.model,
            "messages": messages,
        }
        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            req_params["temperature"] = temp
            
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            req_params["max_tokens"] = mt

        if schema is not None:
            raise NotImplementedError("Structured output not yet supported for OpenRouter")

        resp = self._client.chat.completions.create(**req_params)
        raw = resp.choices[0].message.content
        return _strip_think_block(raw) if strip_think else raw

class OllamaLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.resolved_api_key("OLLAMA_API_KEY")
        self._client = Client(
            host=config.base_url,
            headers={'Authorization': f'Bearer {api_key}'}
        )

    def complete(self, messages: list[dict], schema: type[T] | None = None, strip_think: bool = True, **kwargs) -> str | T:
        """Invoke Ollama chat. If schema is provided, constrain output to JSON schema."""
        options = {}
        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            options["temperature"] = temp

        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            options["num_predict"] = mt

        req_params = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        if options:
            req_params["options"] = options

        if schema is not None:
            req_params["format"] = schema.model_json_schema()

        resp = self._client.chat(**req_params)
        raw_content = _strip_think_block(resp.message.content) if strip_think else resp.message.content
        print(raw_content)
        if schema is not None:
            return schema.model_validate_json(raw_content)
        return raw_content

class GeminiLLM:
    """Google AI Studio provider via the google-genai SDK."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = genai.Client(
            api_key=config.resolved_api_key("GOOGLE_AI_STUDIO_API_KEY"),
        )

    def complete(self, messages: list[dict], schema: type[T] | None = None, **kwargs) -> str | T:
        # flatten chat messages into a single prompt string for generate_content
        contents = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

        gen_config: dict = {}
        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            gen_config["temperature"] = temp

        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            gen_config["max_output_tokens"] = mt

        if schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_json_schema"] = schema.model_json_schema()

        response = self._client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=gen_config if gen_config else None,
        )

        raw_content = response.text
        if schema is not None:
            return schema.model_validate_json(raw_content)
        return raw_content

_PROVIDERS: dict[Provider, dict] = {
    Provider.OPENROUTER: {
        "cls": OpenRouterLLM,
        "defaults": {
            "model": f"{DEFAULT_MODEL}:free",
            "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        }
    },
    Provider.OLLAMA: {
        "cls": OllamaLLM,
        "defaults": {
            "model": f"{DEFAULT_MODEL}-cloud",
            "base_url": os.environ.get("OLLAMA_BASE_URL", "https://ollama.com"),
        }
    },
    Provider.GOOGLE_AI_STUDIO: {
        "cls": GeminiLLM,
        "defaults": {
            "model": "gemini-2.5-flash-lite",
        }
    },
}

GLOBAL_CONFIG = LLMConfig()

def get_llm(config: LLMConfig | None = None) -> OpenRouterLLM | OllamaLLM | GeminiLLM:
    if config is None:
        config = GLOBAL_CONFIG

    provider = Provider(config.provider)

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Valid: {list(_PROVIDERS)}")

    for field_name, default_val in _PROVIDERS[provider]["defaults"].items():
        if getattr(config, field_name) is None:
            setattr(config, field_name, default_val)

    return _PROVIDERS[provider]["cls"](config)
