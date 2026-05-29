from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rlm_harness.config import default_profile_path


def cmd_doctor(args: argparse.Namespace) -> int:
    profile_path = default_profile_path()
    checks = {
        "python": sys.version.split()[0],
        "harness_cli": shutil.which("harness") or "not installed as command yet",
        "docker_cli": "ok" if shutil.which("docker") else "missing",
        "langgraph": module_status("langgraph"),
        "langgraph_checkpoint_sqlite": module_status("langgraph.checkpoint.sqlite"),
        "sqlite_vec": module_status("sqlite_vec"),
        "profile_db": str(profile_path),
    }
    if shutil.which("docker"):
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        checks["docker_daemon"] = (
            completed.stdout.strip() if completed.returncode == 0 else "unavailable"
        )
        image_check = subprocess.run(
            ["docker", "image", "inspect", "rlm-harness-sandbox:latest"],
            text=True,
            capture_output=True,
            check=False,
        )
        checks["sandbox_image"] = "ok" if image_check.returncode == 0 else "missing"
    if args.json_output:
        print(json.dumps(checks, sort_keys=True))
    else:
        for name, value in checks.items():
            print(f"{name}\t{value}")
    failing = {"missing", "unavailable"}
    required_checks = (
        "docker_cli",
        "langgraph",
        "langgraph_checkpoint_sqlite",
        "sqlite_vec",
        "docker_daemon",
        "sandbox_image",
    )
    return 0 if all(checks.get(name) not in failing for name in required_checks) else 1


def module_status(module: str) -> str:
    try:
        __import__(module)
    except ImportError:
        return "missing"
    return "ok"


def cmd_update(args: argparse.Namespace) -> int:
    app_dir = Path(os.environ.get("HARNESS_APP_DIR", Path.home() / ".local/share/harness"))
    src_dir = app_dir / "src"
    repo_url = os.environ.get(
        "HARNESS_REPO_URL",
        "https://github.com/anirudh5harma/rlm-harness.git",
    )
    ref = os.environ.get("HARNESS_REF", "main")
    venv_dir = app_dir / "venv"
    pip_bin = venv_dir / "bin" / "pip"

    if args.in_place:
        return _update_in_place(args)

    if not (src_dir / ".git").is_dir():
        print(
            f"Update requires an install-managed copy at {src_dir}. "
            f"Re-run the install script or use --in-place for a local checkout.",
            file=sys.stderr,
        )
        return 1

    print(f"Fetching {repo_url} ({ref})...")
    result = subprocess.run(
        ["git", "-C", str(src_dir), "fetch", "--tags", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git fetch failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    subprocess.run(
        ["git", "-C", str(src_dir), "checkout", ref],
        text=True,
        capture_output=True,
        check=False,
    )

    branch_check = subprocess.run(
        ["git", "-C", str(src_dir), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if branch_check.returncode == 0:
        pull_result = subprocess.run(
            ["git", "-C", str(src_dir), "pull", "--ff-only", "origin", ref],
            text=True,
            capture_output=True,
            check=False,
        )
        if pull_result.returncode != 0:
            print(f"git pull failed: {pull_result.stderr.strip()}", file=sys.stderr)
            return 1

    if not pip_bin.exists():
        print(f"pip not found at {pip_bin}. Re-run the install script.", file=sys.stderr)
        return 1

    print("Upgrading package...")
    pip_result = subprocess.run(
        [str(pip_bin), "install", "--upgrade", str(src_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if pip_result.returncode != 0:
        print(f"pip install failed: {pip_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("harness updated to latest.")
    if not args.no_sandbox_rebuild and shutil.which("docker"):
        print("Rebuilding sandbox image...")
        subprocess.run(
            [
                str(venv_dir / "bin" / "harness"),
                "sandbox",
                "build",
                "--dockerfile",
                str(src_dir / "docker/sandbox.Dockerfile"),
                "--context",
                str(src_dir),
            ],
            check=False,
        )
    return 0


def _update_in_place(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").is_dir():
        print(f"No .git directory found at {repo_root}", file=sys.stderr)
        return 1

    print("Fetching origin...")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--tags", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git fetch failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    branch_result = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if branch_result.returncode != 0:
        print("HEAD is detached; cannot pull.", file=sys.stderr)
        return 1

    branch = branch_result.stdout.strip().removeprefix("refs/heads/")
    pull_result = subprocess.run(
        ["git", "-C", str(repo_root), "pull", "--ff-only", "origin", branch],
        text=True,
        capture_output=True,
        check=False,
    )
    if pull_result.returncode != 0:
        print(f"git pull failed: {pull_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("Reinstalling package...")
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{repo_root}[dev]"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pip_result.returncode != 0:
        print(f"pip install failed: {pip_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("harness updated to latest (in-place).")
    return 0


def add_maintenance_commands(subparsers) -> None:
    doctor = subparsers.add_parser("doctor", help="Check local setup.")
    doctor.add_argument("--json", dest="json_output", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    update = subparsers.add_parser("update", help="Fetch latest harness from GitHub and upgrade.")
    update.add_argument(
        "--in-place",
        action="store_true",
        help="Update from the local dev checkout instead of the managed install.",
    )
    update.add_argument(
        "--no-sandbox-rebuild",
        action="store_true",
        help="Skip sandbox image rebuild after update.",
    )
    update.set_defaults(func=cmd_update)
