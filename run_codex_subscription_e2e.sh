#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_GOAL="Analyze Tokyo's next 48 hours temperature trend and highlight the biggest rise and drop windows."
DEFAULT_SOURCE_URL="https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2"

GOAL="${1:-$DEFAULT_GOAL}"
SOURCE_URL="${2:-$DEFAULT_SOURCE_URL}"

echo "[1/7] Checking required commands (python3, npx)..."
for cmd in python3 npx; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Missing required command '$cmd'."
    exit 1
  fi
done

echo "[2/7] Preparing latest Codex wrapper..."
WRAPPER_PATH="$ROOT_DIR/codex-latest.sh"
cat > "$WRAPPER_PATH" <<'SH'
#!/usr/bin/env bash
exec npx -y @openai/codex "$@"
SH
chmod +x "$WRAPPER_PATH"

echo "[3/7] Preparing Python virtual environment..."
if [ ! -d "$ROOT_DIR/.venv" ]; then
  python3 -m venv "$ROOT_DIR/.venv"
fi

echo "[4/7] Installing Python dependencies..."
"$ROOT_DIR/.venv/bin/pip" install -e "$ROOT_DIR"

echo "[5/7] Ensuring Codex subscription login..."
if ! "$WRAPPER_PATH" login status >/dev/null 2>&1; then
  echo "Codex is not logged in. Starting device-auth login flow..."
  "$WRAPPER_PATH" login --device-auth
fi

echo "[6/7] Writing .env for codex subscription mode..."
export ROOT_DIR
"$ROOT_DIR/.venv/bin/python" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])

content = f"""MODEL_PROVIDER=codex_subscription
MODEL_NAME=gpt-5.3-codex
CODEX_SUBSCRIPTION_BIN={root / 'codex-latest.sh'}

TIDB_ZERO_TAG=agent-demo
TIDB_DB_NAME=agent_sandbox
RUNS_DIR=.runs
MAX_TOOL_ITERATIONS=12
SQL_ROW_LIMIT=200
HTTP_TIMEOUT_SEC=20
MODEL_TIMEOUT_SEC=180
MODEL_MAX_RETRIES=3
FETCH_MAX_CHARS=60000
"""

(root / ".env").write_text(content, encoding="utf-8")
print("Wrote", root / ".env")
PY

echo "[7/7] Running end-to-end demo..."
RUN_LOG="$(mktemp -t zero-agent-demo-run.XXXXXX.log)"

set +e
"$ROOT_DIR/.venv/bin/zero-agent-demo" run "$GOAL" --source-url "$SOURCE_URL" | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "$RUN_STATUS" -ne 0 ]; then
  echo "ERROR: Demo run failed. Full log: $RUN_LOG"
  exit "$RUN_STATUS"
fi

RUN_ID="$("$ROOT_DIR/.venv/bin/python" - "$RUN_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"run_id=([0-9a-fA-F-]{36})", text)
print(match.group(1) if match else "")
PY
)"

if [ -z "$RUN_ID" ]; then
  echo "WARN: Run finished, but could not parse run_id. Check log: $RUN_LOG"
  exit 0
fi

echo "[post] Running audit and connection check..."
"$ROOT_DIR/.venv/bin/zero-agent-demo" audit "$RUN_ID" || true
"$ROOT_DIR/.venv/bin/zero-agent-demo" conn "$RUN_ID" --redact-password || true

echo
echo "DONE: Codex subscription E2E test completed."
echo "run_id=$RUN_ID"
echo "run_log=$RUN_LOG"
echo "To replay full timeline:"
echo "  $ROOT_DIR/.venv/bin/zero-agent-demo replay $RUN_ID"
