#!/usr/bin/env bash
# MarginGuard - one-command native orchestrator (no containers).
#
#   ./run_all.sh                  full pipeline + train + start all services
#   ./run_all.sh --skip-pipeline  reuse existing data/model, just start services
#   ./run_all.sh --pipeline-only  build lakehouse + train, then exit
#   PYTHON=python3.12 ./run_all.sh   force an interpreter
#   API_PORT=8100 DASH_PORT=3100 MLFLOW_PORT=5100 ./run_all.sh   if the defaults are taken
#     (macOS AirPlay Receiver squats on :5000 - change MLFLOW_PORT or disable AirPlay Receiver)
set -euo pipefail
cd "$(dirname "$0")"

SKIP_PIPELINE=0; PIPELINE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-pipeline) SKIP_PIPELINE=1 ;;
    --pipeline-only) PIPELINE_ONLY=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

API_PORT="${API_PORT:-8000}"; DASH_PORT="${DASH_PORT:-3000}"; MLFLOW_PORT="${MLFLOW_PORT:-5000}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore PYTHONPATH="$PWD"

banner() { printf '\n==========================================================\n%s\n==========================================================\n' "$1"; }

# ---- Python environment (3.10-3.13; the ML wheels do not support newer yet) --------------
if [ ! -x venv/bin/python ]; then
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    for c in python3.12 python3.11 python3.13 python3.10; do
      if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
  fi
  echo "Creating virtual environment..."
  if [ -n "$PY" ]; then
    "$PY" -m venv venv
    venv/bin/python -m pip install --quiet --upgrade pip
    venv/bin/python -m pip install --quiet -r requirements.txt
  elif command -v uv >/dev/null 2>&1; then
    uv venv venv --python 3.12
    uv pip install --python venv/bin/python -r requirements.txt
  else
    echo "Need Python 3.10-3.13 (or uv). Install one, or set PYTHON=/path/to/python." >&2; exit 1
  fi
fi
PY=venv/bin/python

if [ "$SKIP_PIPELINE" -eq 0 ]; then
  banner "Phase 1: Downloading market data (yfinance; synthetic fallback)"
  $PY -m pipeline.fetch_data
  banner "Phase 2: Bronze - validated, partitioned Parquet"
  $PY -m pipeline.ingest_bronze
  banner "Phase 3: Silver - SQL + technical indicators + target"
  $PY -m pipeline.build_silver
  banner "Phase 4: Gold - purged temporal splits"
  $PY -m pipeline.build_gold
  banner "Phase 5: Train LightGBM, log to MLflow, write drift report"
  $PY -m ml.train
  $PY -m ml.drift_engine >/dev/null
fi
[ "$PIPELINE_ONLY" -eq 1 ] && { echo "Pipeline complete."; exit 0; }

[ -f data/gold/model.txt ] || { echo "No trained model found; run without --skip-pipeline first." >&2; exit 1; }

banner "Phase 6: Starting services"

port_busy() { lsof -iTCP:"$1" -sTCP:LISTEN -P -n >/dev/null 2>&1; }
for pair in "API_PORT:$API_PORT" "DASH_PORT:$DASH_PORT" "MLFLOW_PORT:$MLFLOW_PORT"; do
  if port_busy "${pair#*:}"; then
    echo "Port ${pair#*:} (${pair%%:*}) is already in use. Free it or override, e.g. ${pair%%:*}=$(( ${pair#*:} + 100 )) ./run_all.sh" >&2
    exit 1
  fi
done

PIDS=()
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill "$1" 2>/dev/null || true; }
cleanup() { trap - INT TERM EXIT; echo; echo "Stopping services..."; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill_tree "$p"; done; }
trap cleanup INT TERM EXIT

export MG_CORS_ORIGINS="http://localhost:$DASH_PORT,http://127.0.0.1:$DASH_PORT"
export MG_MLFLOW_URL="http://127.0.0.1:$MLFLOW_PORT"

venv/bin/mlflow server --backend-store-uri "sqlite:///$PWD/mlruns.db" \
  --default-artifact-root "$PWD/mlruns" --host 127.0.0.1 --port "$MLFLOW_PORT" >logs_mlflow.txt 2>&1 &
PIDS+=($!)
venv/bin/uvicorn serving.main:app --host 127.0.0.1 --port "$API_PORT" >logs_api.txt 2>&1 &
PIDS+=($!)

if [ ! -d dashboard/node_modules ]; then (cd dashboard && npm install --silent); fi
(cd dashboard && NEXT_PUBLIC_API_URL="http://127.0.0.1:$API_PORT" npm run dev -- --port "$DASH_PORT" >../logs_dashboard.txt 2>&1) &
PIDS+=($!)

echo "--------------------------------------------------------"
echo "All local services starting (logs: logs_*.txt):"
echo " - FastAPI Backend:   http://127.0.0.1:$API_PORT/docs"
echo " - MLflow Tracking:   http://127.0.0.1:$MLFLOW_PORT"
echo " - Risk Cockpit:      http://localhost:$DASH_PORT"
echo "--------------------------------------------------------"
echo "Press [CTRL+C] to stop all services."
wait
