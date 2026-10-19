#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TURBOVLA_ROOT="$REPO_ROOT/third_party/TurboVLA"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

# This creates third_party/TurboVLA/.venv.  It is intentionally separate from
# the RoboSynChallenge / EmbodiChain simulator environment.
uv sync \
  --project "$TURBOVLA_ROOT" \
  --python 3.11 \
  --frozen \
  --extra robotwin \
  --no-dev

TURBOVLA_PYTHON="$TURBOVLA_ROOT/.venv/bin/python"
"$TURBOVLA_PYTHON" -c \
  'import starVLA, turbovla; print("TurboVLA import OK:", starVLA.__file__, turbovla.__file__)'

printf '\nSet this before evaluation:\nexport TURBOVLA_PYTHON=%q\n' "$TURBOVLA_PYTHON"
