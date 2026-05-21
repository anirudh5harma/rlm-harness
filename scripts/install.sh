#!/usr/bin/env sh
set -eu

REPO_URL="${HARNESS_REPO_URL:-https://github.com/anirudh5harma/rlm-harness.git}"
REF="${HARNESS_REF:-main}"
PREFIX="${HARNESS_PREFIX:-$HOME/.local}"
APP_DIR="${HARNESS_APP_DIR:-$HOME/.local/share/harness}"
BIN_DIR="$PREFIX/bin"
VENV_DIR="$APP_DIR/venv"
SRC_DIR="$APP_DIR/src"
PYTHON_BIN="${PYTHON:-}"
NO_PATH_UPDATE="${HARNESS_NO_PATH_UPDATE:-0}"

say() { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required but was not found"; }

need_python() {
  candidate="$1"
  command -v "$candidate" >/dev/null 2>&1 || return 1
  "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

pick_python() {
  if [ -n "$PYTHON_BIN" ]; then
    need_python "$PYTHON_BIN" || fail "$PYTHON_BIN must be Python 3.10 or newer"
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  for candidate in python3.12 python3.11 python3.10 python3; do
    if need_python "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  fail "Python 3.10 or newer is required. Set PYTHON=/path/to/python3.12 and retry."
}

path_contains_bin() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return 0 ;;
    *) return 1 ;;
  esac
}

choose_profile() {
  if [ -n "${PROFILE:-}" ]; then
    printf '%s\n' "$PROFILE"
  elif [ "${SHELL##*/}" = "zsh" ]; then
    printf '%s\n' "$HOME/.zshrc"
  elif [ "${SHELL##*/}" = "bash" ]; then
    printf '%s\n' "$HOME/.bashrc"
  else
    printf '%s\n' "$HOME/.profile"
  fi
}

ensure_path_hint() {
  if path_contains_bin; then
    return 0
  fi
  if [ "$NO_PATH_UPDATE" = "1" ]; then
    return 0
  fi
  profile="$(choose_profile)"
  mkdir -p "$(dirname "$profile")"
  touch "$profile"
  if ! grep -F "$BIN_DIR" "$profile" >/dev/null 2>&1; then
    {
      printf '\n# Harness installer\n'
      printf 'export PATH="%s:$PATH"\n' "$BIN_DIR"
    } >> "$profile"
    say "Added $BIN_DIR to PATH in $profile"
  fi
}

PYTHON_BIN="$(pick_python)"
need "$PYTHON_BIN"
need git

say "Installing Harness"
say "  repo: $REPO_URL"
say "  ref:  $REF"
say "  app:  $APP_DIR"
say "  python: $PYTHON_BIN ($($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

mkdir -p "$APP_DIR" "$BIN_DIR"

if [ -d "$SRC_DIR/.git" ]; then
  say "Updating existing source checkout..."
  git -C "$SRC_DIR" fetch --quiet --tags origin
else
  rm -rf "$SRC_DIR"
  git clone --quiet "$REPO_URL" "$SRC_DIR"
fi

git -C "$SRC_DIR" checkout --quiet "$REF"
if git -C "$SRC_DIR" symbolic-ref -q HEAD >/dev/null 2>&1; then
  git -C "$SRC_DIR" pull --ff-only --quiet origin "$REF" || true
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade "$SRC_DIR[graph]"

cat > "$BIN_DIR/harness" <<EOF
#!/usr/bin/env sh
exec "$VENV_DIR/bin/harness" "\$@"
EOF
chmod +x "$BIN_DIR/harness"
ensure_path_hint

say ""
say "Harness installed."
say ""
say "Next steps:"
say "  1. Ensure $BIN_DIR is on your PATH."
say "     export PATH=\"$BIN_DIR:\$PATH\""
say "  2. Choose a provider, save your API key, then select a model:"
say "     harness /provider"
say "     harness /model"
say "  3. Start in any directory:"
say "     harness"
say ""
if path_contains_bin; then
  "$BIN_DIR/harness" doctor || true
else
  say "Open a new shell or run the export above, then run: harness doctor"
fi
