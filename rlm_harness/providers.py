from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    label: str
    base_url: str
    api_key_env: tuple[str, ...]
    models: tuple[str, ...]
    description: str


PROVIDERS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        name="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=("OPENROUTER_API_KEY",),
        models=(
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
            "qwen/qwen-2.5-coder-32b-instruct",
            "deepseek/deepseek-chat",
        ),
        description="Broad model router; recommended default.",
    ),
    "openai": ProviderPreset(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env=("OPENAI_API_KEY",),
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"),
        description="OpenAI models and OpenAI API keys.",
    ),
    "groq": ProviderPreset(
        name="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env=("GROQ_API_KEY",),
        models=(
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct",
        ),
        description="Fast hosted inference.",
    ),
    "together": ProviderPreset(
        name="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env=("TOGETHER_API_KEY",),
        models=(
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ),
        description="Open-source models via Together.",
    ),
    "fireworks": ProviderPreset(
        name="fireworks",
        label="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env=("FIREWORKS_API_KEY",),
        models=(
            "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
            "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "accounts/fireworks/models/deepseek-v3",
        ),
        description="Fireworks hosted models.",
    ),
    "deepinfra": ProviderPreset(
        name="deepinfra",
        label="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env=("DEEPINFRA_API_KEY",),
        models=(
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-V3",
        ),
        description="DeepInfra hosted models.",
    ),
    "opencode-go": ProviderPreset(
        name="opencode-go",
        label="OpenCode Go",
        base_url="https://api.opencode.ai/v1",
        api_key_env=("OPENCODE_API_KEY",),
        models=("opencode/sonic", "opencode/coder", "opencode/agent"),
        description="OpenCode Go provider if you have an OpenCode API key.",
    ),
    "custom": ProviderPreset(
        name="custom",
        label="Custom chat-completions endpoint",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=("HARNESS_API_KEY",),
        models=("model/name",),
        description="Any chat-completions endpoint that accepts OpenAI-style requests.",
    ),
    "stub": ProviderPreset(
        name="stub",
        label="Stub / offline test",
        base_url="",
        api_key_env=(),
        models=("stub",),
        description="Offline deterministic test provider.",
    ),
}

ALIASES = {
    "openai-compatible": "openrouter",
    "openrouter.ai": "openrouter",
    "open-code": "opencode-go",
    "opencode": "opencode-go",
    "opencodego": "opencode-go",
    "local": "custom",
}


def provider_names() -> list[str]:
    return list(PROVIDERS)


def normalize_provider(name: Optional[str]) -> str:
    if not name:
        return "openrouter"
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    return ALIASES.get(normalized, normalized)


def provider_preset(name: Optional[str]) -> ProviderPreset:
    normalized = normalize_provider(name)
    return PROVIDERS.get(normalized, PROVIDERS["custom"])


def provider_env_names(name: Optional[str]) -> tuple[str, ...]:
    return provider_preset(name).api_key_env


def provider_base_url(name: Optional[str]) -> str:
    return provider_preset(name).base_url


def static_models(name: Optional[str]) -> list[str]:
    return list(provider_preset(name).models)


def fetch_provider_models(
    provider: str,
    base_url: str,
    api_key: Optional[str] = None,
    timeout_s: int = 20,
) -> list[str]:
    if provider == "stub" or not base_url:
        return static_models(provider)
    url = base_url.rstrip("/") + "/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return static_models(provider)
    data = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        return static_models(provider)
    models: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
        else:
            model_id = str(item)
        if model_id:
            models.append(str(model_id))
    return sorted(dict.fromkeys(models)) or static_models(provider)
