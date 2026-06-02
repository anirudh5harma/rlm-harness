from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from rlm_harness.providers import (
    normalize_provider,
    provider_base_url,
    provider_env_names,
    static_models,
)

DEFAULT_MODEL = "stub"
DEFAULT_PROVIDER = "stub"
CONFIG_DIR = Path(os.environ.get("HARNESS_CONFIG_DIR", Path.home() / ".harness"))
CONFIG_PATH = Path(os.environ.get("HARNESS_CONFIG", CONFIG_DIR / "config.json"))


def env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def load_user_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_user_config(updates: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    config = load_user_config(config_path)
    config.update({key: value for key, value in updates.items() if value is not None})
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return config


def default_provider() -> str:
    configured = load_user_config().get("provider")
    value = (
        env_first(
            "HARNESS_PROVIDER",
            "RLM_HARNESS_PROVIDER",
            default=configured or DEFAULT_PROVIDER,
        )
        or DEFAULT_PROVIDER
    )
    return normalize_provider(value)


def default_model() -> str:
    configured_env = env_first("HARNESS_MODEL", "RLM_HARNESS_MODEL")
    if configured_env:
        return configured_env

    provider = default_provider()
    config = load_user_config()
    configured_provider = config.get("provider")
    configured_model = config.get("model")
    if configured_model and (
        not configured_provider or normalize_provider(str(configured_provider)) == provider
    ):
        return str(configured_model)
    return static_models(provider)[0] if static_models(provider) else DEFAULT_MODEL


def default_base_url() -> str:
    provider = default_provider()
    config = load_user_config()
    configured_provider = config.get("provider")
    configured = (
        config.get("base_url")
        if not configured_provider or normalize_provider(str(configured_provider)) == provider
        else None
    )
    provider_default = provider_base_url(provider)
    return env_first(
        "HARNESS_BASE_URL",
        "RLM_HARNESS_BASE_URL",
        "OPENAI_BASE_URL",
        default=configured or provider_default,
    ) or provider_default


def default_api_key(provider: Optional[str] = None) -> Optional[str]:
    config = load_user_config()
    active_provider = normalize_provider(provider or default_provider())
    configured_provider = config.get("provider")
    configured = (
        config.get("api_key")
        if (
            not configured_provider
            or normalize_provider(str(configured_provider)) == active_provider
        )
        else None
    )
    return env_first(
        "HARNESS_API_KEY",
        "RLM_HARNESS_API_KEY",
        *provider_env_names(active_provider),
        default=configured,
    )


def default_trace_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_TRACE_DB", ".rlm_harness/traces.db"))


def default_memory_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_MEMORY_DB", ".rlm_harness/memory.db"))


def default_extension_root() -> Path:
    """Where installed extensions live. Mirrors the pi-mono
    `~/.pi/agent/` layout: project-local first, then user-global.
    """
    return Path(
        os.environ.get(
            "RLM_HARNESS_EXTENSION_ROOT",
            str(Path.home() / ".harness" / "extensions"),
        )
    )


def default_profile_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_PROFILE_DB", CONFIG_DIR / "profile.db"))


def masked_secret(value: Optional[str]) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"
