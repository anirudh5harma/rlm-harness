"""Extension install / update / list (Phase F).

The pivot plan's Phase F adds the `harness install <source>` family
of commands for installing, updating, and listing extensions. This
is the pi-mono "pi packages" pattern: extensions live under
`~/.harness/extensions/` (or `.harness/extensions/` for project-local)
and are loaded at startup.

Phase F ships a no-op stub for the install surface. The full
extension model (npm-style discovery, sandbox scoping, hook
lifecycle) is queued for a follow-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm_harness.config import default_extension_root


def cmd_install(args: argparse.Namespace) -> int:
    """Install an extension from a source URL or local path.

    Phase F stub: prints the resolved path under the extension
    root and exits 0. Real install logic (download, verify,
    sandbox scoping) is queued.

    `--refresh` (no source) refreshes all installed extensions.
    """
    root = Path(args.extension_root)
    if args.refresh:
        if args.json_output:
            print(
                json.dumps(
                    {"status": "stub", "action": "refresh"}, sort_keys=True
                )
            )
        else:
            print("[stub] would refresh installed extensions.")
        return 0
    target = root / args.source.replace(":", "_").replace("/", "_")
    if args.json_output:
        print(
            json.dumps(
                {
                    "source": args.source,
                    "extension_root": str(root),
                    "target": str(target),
                    "status": "stub",
                },
                sort_keys=True,
            )
        )
    else:
        print(
            f"[stub] would install {args.source} into {target}\n"
            "Extension support is not yet implemented (Phase F stub)."
        )
    return 0


def cmd_extensions_list(args: argparse.Namespace) -> int:
    """List installed extensions. Phase F stub returns an empty list."""
    if args.json_output:
        print(json.dumps({"extensions": []}, sort_keys=True))
    else:
        print("No extensions installed.")
    return 0


def add_install_command(subparsers) -> None:
    install = subparsers.add_parser(
        "install", help="Install, refresh, or list harness extensions."
    )
    install.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Extension source (npm:foo, git:..., path:...)",
    )
    install.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh all installed extensions instead of adding a new one.",
    )
    install.add_argument(
        "--list",
        dest="list_installed",
        action="store_true",
        help="List installed extensions instead of adding a new one.",
    )
    install.add_argument(
        "--extension-root",
        type=Path,
        default=default_extension_root(),
        help=argparse.SUPPRESS,
    )
    install.add_argument("--json", dest="json_output", action="store_true")
    install.set_defaults(func=_dispatch_install)


def _dispatch_install(args: argparse.Namespace) -> int:
    if args.list_installed:
        return cmd_extensions_list(args)
    return cmd_install(args)
