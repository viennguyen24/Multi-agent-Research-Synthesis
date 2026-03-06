from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client

from src.util import DEFAULT_MODEL

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA     = "ollama"

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

    def complete(self, messages: list[dict], **kwargs) -> str:
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

        resp = self._client.chat.completions.create(**req_params)
        return resp.choices[0].message.content

class OllamaLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.resolved_api_key("OLLAMA_API_KEY")
        self._client = Client(
            host=config.base_url,
            headers={'Authorization': f'Bearer {api_key}'}
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
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

        resp = self._client.chat(**req_params)
        return resp["message"]["content"]

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
}

GLOBAL_CONFIG = LLMConfig()

def get_llm(config: LLMConfig | None = None) -> OpenRouterLLM | OllamaLLM:
    if config is None:
        config = GLOBAL_CONFIG

    provider = Provider(config.provider)

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Valid: {list(_PROVIDERS)}")

    for field_name, default_val in _PROVIDERS[provider]["defaults"].items():
        if getattr(config, field_name) is None:
            setattr(config, field_name, default_val)

    return _PROVIDERS[provider]["cls"](config)
