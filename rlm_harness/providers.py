from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

USER_AGENT = "rlm-harness/0.1 (+https://github.com/anirudh5harma/rlm-harness)"


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    label: str
    base_url: str
    api_key_env: tuple[str, ...]
    models: tuple[str, ...]
    description: str
    models_url: str | None = None


PROVIDERS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        name="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=("OPENROUTER_API_KEY",),
        models=(
            "qwen/qwen3.7-max",
            "x-ai/grok-build-0.1",
            "google/gemini-3.5-flash",
            "openai/gpt-chat-latest",
            "openrouter/owl-alpha",
        ),
        description="Broad model router; recommended default.",
    ),
    "openai": ProviderPreset(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env=("OPENAI_API_KEY",),
        models=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1"),
        description="OpenAI models and OpenAI API keys.",
    ),
    "groq": ProviderPreset(
        name="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env=("GROQ_API_KEY",),
        models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "groq/compound",
            "groq/compound-mini",
        ),
        description="Fast hosted inference.",
    ),
    "together": ProviderPreset(
        name="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env=("TOGETHER_API_KEY",),
        models=(
            "zai-org/GLM-5.1",
            "moonshotai/Kimi-K2.5",
            "openai/gpt-oss-120b",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3.1",
        ),
        description="Open-source models via Together.",
    ),
    "fireworks": ProviderPreset(
        name="fireworks",
        label="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env=("FIREWORKS_API_KEY",),
        models=(
            "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
            "accounts/fireworks/models/kimi-k2p5",
            "accounts/fireworks/models/deepseek-v3p2",
            "accounts/fireworks/models/gpt-oss-120b",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        ),
        description="Fireworks hosted models.",
        models_url=(
            "https://api.fireworks.ai/v1/accounts/fireworks/models"
            "?filter=supports_serverless%3Dtrue"
        ),
    ),
    "deepinfra": ProviderPreset(
        name="deepinfra",
        label="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env=("DEEPINFRA_API_KEY",),
        models=(
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "openai/gpt-oss-120b",
        ),
        description="DeepInfra hosted models.",
        models_url="https://api.deepinfra.com/v1/models",
    ),
    "opencode-go": ProviderPreset(
        name="opencode-go",
        label="OpenCode Go",
        base_url="https://opencode.ai/zen/go/v1",
        api_key_env=("OPENCODE_API_KEY",),
        models=(
            "glm-5.1",
            "glm-5",
            "kimi-k2.6",
            "kimi-k2.5",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "mimo-v2.5-pro",
            "mimo-v2.5",
        ),
        description="OpenCode Go provider if you have an OpenCode API key.",
    ),
    "custom": ProviderPreset(
        name="custom",
        label="Custom chat-completions endpoint",
        base_url="http://127.0.0.1:8080/v1",
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
    "openai-compatible": "custom",
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
    preset = provider_preset(provider)
    url = preset.models_url or base_url.rstrip("/") + "/models"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return static_models(provider)
    if isinstance(raw, dict):
        data = raw.get("data") or raw.get("models")
    else:
        data = raw
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
    if provider == "opencode-go":
        supported = set(static_models(provider))
        models = [model for model in models if model in supported]
    return sorted(dict.fromkeys(models)) or static_models(provider)
