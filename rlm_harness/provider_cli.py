from __future__ import annotations

import argparse
import getpass
import json
import sys
from typing import Optional

from rlm_harness.config import (
    default_api_key,
    default_base_url,
    default_model,
    default_provider,
    masked_secret,
    save_user_config,
)
from rlm_harness.providers import (
    PROVIDERS,
    fetch_provider_models,
    normalize_provider,
    provider_names,
    provider_preset,
    static_models,
)


def prompt_numbered_choice(options: list[str], prompt: str) -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    try:
        raw = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index]
    return raw


def print_provider_options() -> None:
    print("Available providers:")
    for index, name in enumerate(provider_names(), start=1):
        preset = PROVIDERS[name]
        print(f"  {index}. {name}\t{preset.label}\t{preset.description}")


def cmd_model(args: argparse.Namespace) -> int:
    provider = normalize_provider(args.provider or default_provider())
    base_url = args.base_url
    if not base_url:
        base_url = (
            default_base_url()
            if provider == default_provider()
            else provider_preset(provider).base_url
        )
    if args.model:
        save_user_config({"model": args.model})
        print(f"model set to {args.model}")
        return 0

    models = (
        static_models(provider)
        if args.offline
        else fetch_provider_models(provider, base_url, default_api_key(provider))
    )
    if args.json_output:
        print(json.dumps({"provider": provider, "models": models}, sort_keys=True))
        return 0

    print(f"Available models for {provider}:")
    for index, model_name in enumerate(models, start=1):
        current = " *" if model_name == default_model() else ""
        print(f"  {index}. {model_name}{current}")
    selected = prompt_numbered_choice(
        models,
        "Select model number/name, or press Enter to keep current: ",
    )
    if selected:
        save_user_config({"model": selected})
        print(f"model set to {selected}")
    return 0


def cmd_provider(args: argparse.Namespace) -> int:
    provider_arg = " ".join(args.provider) if isinstance(args.provider, list) else args.provider
    if not provider_arg:
        print_provider_options()
        selected = prompt_numbered_choice(
            provider_names(),
            "Select provider number/name, or press Enter to keep current: ",
        )
        if not selected:
            print(f"current provider\t{default_provider()}")
            print(f"base_url\t{default_base_url()}")
            print(f"api_key\t{masked_secret(default_api_key())}")
            return 0
        provider_arg = selected

    provider = normalize_provider(provider_arg)
    if provider not in PROVIDERS:
        print(f"Unknown provider: {provider_arg}", file=sys.stderr)
        print_provider_options()
        return 1

    preset = provider_preset(provider)
    updates = {
        "provider": provider,
        "base_url": args.base_url or preset.base_url,
    }
    if provider == "stub":
        updates["model"] = "stub"
    elif args.set_default_model:
        updates["model"] = static_models(provider)[0]

    if args.api_key:
        updates["api_key"] = args.api_key
    elif provider != "stub" and args.prompt_key and sys.stdin.isatty():
        entered = getpass.getpass(f"{preset.label} API key: ").strip()
        if entered:
            updates["api_key"] = entered

    save_user_config(updates)
    print(f"provider set to {provider}")
    print(f"base_url set to {updates['base_url']}")
    if updates.get("model"):
        print(f"model set to {updates['model']}")
    if updates.get("api_key"):
        print("api_key saved")
    elif provider != "stub" and not default_api_key(provider):
        env_hint = preset.api_key_env[0] if preset.api_key_env else "HARNESS_API_KEY"
        print(f"api_key not set; run: harness /provider {provider} --api-key <key>")
        print(f"or export {env_hint}=<key>")
    return 0


def add_provider_commands(subparsers) -> None:
    model = subparsers.add_parser("model", help="Choose or list models.")
    model.add_argument("model", nargs="?")
    model.add_argument("--provider", default=None)
    model.add_argument("--base-url", default=None, help=argparse.SUPPRESS)
    model.add_argument("--json", dest="json_output", action="store_true")
    model.add_argument("--offline", action="store_true", help="Use bundled model suggestions only.")
    model.set_defaults(func=cmd_model)

    provider = subparsers.add_parser("provider", help="Choose provider and save API key.")
    provider.add_argument(
        "provider",
        nargs="*",
        help="Provider to use. Omit to choose from a list.",
    )
    provider.add_argument(
        "--api-key",
        default=None,
        help="API key to save in ~/.harness/config.json.",
    )
    provider.add_argument(
        "--base-url",
        default=None,
        help="Override the provider base URL.",
    )
    provider.add_argument(
        "--keep-model",
        dest="set_default_model",
        action="store_false",
        default=True,
        help="Keep the current model when switching providers.",
    )
    provider.add_argument(
        "--no-prompt",
        dest="prompt_key",
        action="store_false",
        default=True,
        help="Do not prompt for an API key.",
    )
    provider.set_defaults(func=cmd_provider)
