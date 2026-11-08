<div align="center">

# 🛡️ MarginGuard

### Market-risk lakehouse & tail-risk prediction engine

Predict volatility breakouts before they become margin calls — from raw market data to a live, monitored model, running entirely on your own machine.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-Parquet-FFF000?logo=duckdb&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-calibrated-2E8B57)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently-drift%20monitoring-E4572E)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-shadcn%2Fui-000000?logo=nextdotjs&logoColor=white)
![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)

</div>

---

## Overview

MarginGuard is a complete, container-free **data lakehouse and MLOps platform** for market risk. It ingests daily bars for
NIFTY 50 and large-cap Indian equities, refines them through a **Bronze → Silver → Gold medallion** built on DuckDB and
Parquet, and trains a **calibrated LightGBM classifier** that answers one question:

> *Will this stock make a move larger than 2.5σ within the next 3 trading sessions?*

That is the kind of volatility breakout that triggers margin calls. The model returns a real probability, compares it with
the historical base rate, assigns a risk tier (`LOW` / `ELEVATED` / `CRITICAL`) and recommends a brokerage action — through a
REST API and an operational console.

```
 yfinance ─► BRONZE ─► SILVER ─► GOLD ─► LightGBM + calibration ─► MLflow
 (synthetic   raw,      16 features   purged         │
  fallback)  validated  + target      splits        ├─► Evidently drift (KS + PSI)
                                                    └─► FastAPI :8000 ─► Next.js console :3000
```

## Highlights

| | |
|---|---|
| 🏗️ **Medallion lakehouse** | Schema-enforced Bronze, SQL-built Silver, leakage-safe Gold. Hive-partitioned Snappy Parquet, row-level quality rules, deduplication and a lineage ledger for every stage. |
| 🧮 **Quant-grade features** | 16 stationary features: multi-horizon returns, EMA/MACD, Wilder RSI & ATR, Parkinson volatility, volume z-score, volatility-regime ratios and a market-wide vol factor. |
| 🔒 **No look-ahead** | Date-based splits with a purge window at every boundary, independently re-computed targets and regression tests that fail if the purge is removed. |
| 🎯 **Calibrated probabilities** | Class-imbalance handled with `scale_pos_weight`, then Platt-calibrated so a "17% risk" means 17%. Thresholds are fixed on validation before the test set is touched. |
| 📈 **Full MLOps loop** | MLflow experiment tracking (SQLite), SHAP explainability, and Evidently + PSI feature-drift monitoring against the training distribution. |
| ⚡ **Fast serving** | FastAPI inference in **~1–4 ms** (budget 50 ms), validated inputs, restricted CORS, live drift telemetry. |
| 🖥️ **Operational console** | Dark-mode Next.js + shadcn/ui cockpit: KPIs, interactive risk simulator, drift monitor and feature importance. |
| 🛠️ **Robust data handling** | Auto-adjusts for corporate actions (e.g. the Tata Motors demerger), resolves renamed tickers, and falls back to a synthetic GBM + jump generator when offline. |
| ✅ **Tested & reproducible** | 16 offline tests on a synthetic lakehouse, pinned dependencies, one-command orchestration. |

## Results

Evaluated **out-of-time** (Sep 2025 – Nov 2026), on data the model never saw, with thresholds chosen beforehand.

| Metric | Value | Context |
|---|---|---|
| **PR-AUC** | **0.166** | **1.60× lift** over the 10.4% base rate |
| ROC-AUC | 0.642 | 0.5 = random |
| F1 | 0.243 | at the validation-selected threshold |
| Brier score | 0.091 | on calibrated probabilities |
| Inference latency | ~1–4 ms | p50, single request |

A volatility-breakout signal from daily bars is inherently low-signal; MarginGuard delivers a consistent, honestly
measured edge and is designed as a risk **screening and prioritisation** layer. Methodology and caveats: [docs/decisions.md](docs/decisions.md).

---

## Quick start

**Requirements:** Python 3.10–3.13 (3.12 recommended) or [`uv`](https://docs.astral.sh/uv/) · Node.js 18+ · internet on first run.

```bash
./run_all.sh
```

One command builds the virtual environment, downloads data, runs the full pipeline, trains and logs the model, and starts
all three services. Press **Ctrl+C** to stop them.

| Service | URL |
|---|---|
| 🖥️ Risk console | http://localhost:3000 |
| 📘 API docs (Swagger) | http://127.0.0.1:8000/docs |
| 🧪 MLflow tracking UI | http://127.0.0.1:5000 |

```bash
./run_all.sh --skip-pipeline    # reuse existing data + model, just start the services
./run_all.sh --pipeline-only    # rebuild the lakehouse and model, no servers
```

**Ports busy?** `run_all.sh` pre-checks them and never attaches to another app. Override as needed:

```bash
API_PORT=8100 DASH_PORT=3100 MLFLOW_PORT=5100 ./run_all.sh
```
> On macOS, AirPlay Receiver uses port 5000 — set `MLFLOW_PORT=5100` (or turn AirPlay Receiver off).

<details>
<summary><b>Run each stage manually</b></summary>

```bash
python3.12 -m venv venv && source venv/bin/activate        # or: uv venv venv --python 3.12
pip install -r requirements.txt

python -m pipeline.fetch_data        # raw bars          → data/raw_cache/
python -m pipeline.ingest_bronze     # validated Parquet → data/bronze/year=/month=/symbol=/
python -m pipeline.build_silver      # features + target → data/silver/features.parquet
python -m pipeline.build_gold        # purged splits     → data/gold/
python -m ml.train                   # model + MLflow run
python -m ml.drift_engine            # drift report      → data/reports/

uvicorn serving.main:app --port 8000
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --port 5000
cd dashboard && npm install && npm run dev
```
Run Python modules from the repository root with `python -m …`.
</details>

---

## The console

1. **KPI banner** — active model and run, out-of-time PR-AUC with lift, live p50/p95 latency, drift status.
2. **Risk simulator** — pick a symbol to load its latest engineered features, adjust any of the 16 inputs and score it in
   real time: calibrated probability, multiple of the base rate, risk tier and recommended action.
3. **Drift monitor** — per-feature KS p-value, PSI and status; compare the latest market window *or* live API traffic
   against the training set.
4. **Feature importance** — mean |SHAP| showing what drives the score.

**Reading the drift badge.** It turns red when at least half of the features fail the drift test. Right now the latest
market (a calm uptrend) looks very different from the 2019–2024 training period, which includes the COVID crash — so
volatility, trend and momentum features register as shifted. PSI (`<0.10` stable · `0.10–0.25` moderate · `>0.25` major)
is the most informative guide, and a red badge is the cue to retrain on fresher data.

---

## API

`http://127.0.0.1:8000` — interactive documentation at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service, model and data availability |
| `POST /api/v1/predict-risk` | 16-feature vector → calibrated probability, tier, action, latency |
| `GET /api/v1/model-info` | run id, thresholds, validation/test metrics, SHAP, p50/p95 latency |
| `GET /api/v1/drift-status?source=market\|live` | per-feature KS p-value and PSI |
| `GET /api/v1/symbols` · `GET /api/v1/latest-features?symbol=…` | discover symbols and pre-fill inputs |

```bash
curl -s "http://127.0.0.1:8000/api/v1/latest-features?symbol=RELIANCE.NS" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(json.dumps({'symbol':d['symbol'],**d['features']}))" > req.json
curl -s -X POST http://127.0.0.1:8000/api/v1/predict-risk -H 'Content-Type: application/json' -d @req.json
```
```json
{"symbol":"RELIANCE.NS","risk_score":0.0176,"base_rate":0.1069,"risk_level":"LOW",
 "action_trigger":"Maintain Normal Margin Limits","latency_ms":1.3,"model_run_id":"…"}
```

Errors are explicit: `422` invalid or out-of-range input · `503` model/data not ready · `409` not enough live traffic
for live drift · `404` unknown symbol. Risk tiers use thresholds fitted on validation and stored with the model.

---

## How it works

**Universe** — `^NSEI`, `RELIANCE.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `TMPV.NS`, `TMCV.NS` (Tata Motors'
post-demerger successors), 2019 → today.

| Layer | Location | What happens |
|---|---|---|
| **Bronze** | `data/bronze/year=/month=/symbol=/` | Explicit schema casts, row-level validation (positive prices, OHLC consistency, non-negative volume), de-duplication, Snappy Parquet |
| **Silver** | `data/silver/features.parquet` | Corporate-action guard, trading calendar with capped forward-fill, 16 features, forward-looking target — DuckDB SQL plus pandas for recursive indicators |
| **Gold** | `data/gold/` | 70/15/15 chronological train/val/test with purge windows; training set doubles as the drift reference |
| **Lineage** | `data/lineage.json` | Inputs, outputs, row counts and rejects for every stage |

**Target.** `1` when `max |r(t+k)|` for `k = 1..3` exceeds `2.5 × σ₂₀(t)`, using the daily volatility known at time *t*
(≈10% positive rate). The last three sessions per symbol stay unlabeled rather than guessed.

**Model.** LightGBM with `scale_pos_weight`, early stopping on validation PR-AUC, Platt calibration, and operating points
(`elevated` = F1-optimal, `critical` = ≥50% validation precision) fixed before the test set is scored. Every run is logged to
MLflow with parameters, metrics, SHAP importance and the model artifact.

Deep dives: **[Architecture](docs/architecture.md)** · **[Decision log](docs/decisions.md)** · **[Original plan](docs/plan.md)**

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_PORT` / `DASH_PORT` / `MLFLOW_PORT` | 8000 / 3000 / 5000 | service ports for `run_all.sh` |
| `PYTHON` | auto-detected | interpreter for the virtual environment |
| `MG_DATA_DIR` | `./data` | lakehouse root |
| `MG_MLFLOW_URI` | `sqlite:///./mlruns.db` | MLflow tracking store |
| `MG_ARTIFACT_DIR` | `./mlruns` | MLflow artifacts |
| `MG_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | allowed browser origins |
| `MG_MLFLOW_URL` | `http://127.0.0.1:5000` | MLflow link in the console |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | API address for the dashboard |

`run_all.sh` derives the CORS and URL settings from whatever ports you choose.

## Quality

```bash
venv/bin/python -m pytest                      # 16 tests, ~11 s, fully offline
cd dashboard && npm run lint && npm run build
```

The suite builds a synthetic lakehouse and exercises the real pipeline: bronze validation and partitioning, target
re-computation, indicator look-ahead safety, corporate-action adjustment, purged chronological splits, model artifacts,
PSI/KS/calibration, and the full API contract.

## Project structure

```
pipeline/    config · lineage · fetch_data · ingest_bronze · build_silver · build_gold
ml/          train · evaluate · calibration · drift_engine
serving/     main.py (FastAPI) · schemas.py (Pydantic v2)
scripts/     generate_synthetic_trades.py  (GBM + Poisson-jump trade simulator)
dashboard/   Next.js · Tailwind v4 · shadcn/ui
tests/       pytest suite
docs/        plan.md · architecture.md · decisions.md
run_all.sh   one-command orchestrator
```

Need a large offline dataset? Generate 1M+ simulated trades:

```bash
python -m scripts.generate_synthetic_trades --trades 1000000 --out data/synthetic_trades.parquet
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Port … is already in use` | override ports as shown in Quick start |
| Console says "Backend unavailable" | start the API; make sure `NEXT_PUBLIC_API_URL` and CORS ports match (`logs_api.txt`) |
| API returns `503` | build the model: `./run_all.sh --pipeline-only` |
| Evidently `infer_dtype_from_object` error | use the pinned `pandas==2.3.3` |
| Dependency install fails | use Python 3.12 (`PYTHON=python3.12 ./run_all.sh`) |
| Live drift says "not enough traffic" | score at least 30 requests first |

## Do I need API keys or a `.env` file?

No. Everything runs locally: the model is trained and served on your machine, market data comes from yfinance's public
endpoints, and MLflow uses a local SQLite file. No external AI/LLM service, credentials or secrets are involved. The
`MG_*` / port variables above are optional overrides with working defaults.

## Roadmap

- Automatic retraining when drift crosses a PSI threshold
- Intraday and order-book features
- Per-symbol drift testing and persistent drift history
- API authentication for multi-user deployments

---

## License

[MIT](LICENSE)

<div align="center">

**MarginGuard** — from raw ticks to a monitored, explainable risk signal.

</div>
