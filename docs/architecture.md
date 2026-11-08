# MarginGuard — Architecture

MarginGuard is a native (no containers) market-risk lakehouse and MLOps stack. It ingests daily
bars, refines them through a Bronze → Silver → Gold medallion, trains a LightGBM classifier that
predicts **tail-volatility events** (a >2.5σ move within the next 3 sessions — the kind of move that
triggers margin calls), tracks experiments in MLflow, monitors feature drift, and serves everything
through FastAPI and a Next.js console.

> Original specification: [plan.md](plan.md). Where the implementation deviates from it, see
> [decisions.md](decisions.md).

## 1. System view

```
  yfinance ──(retries, successor resolution)──┐        ┌── synthetic GBM+jump generator (fallback)
                                              ▼        ▼
                                   data/raw_cache/*.parquet            pipeline/fetch_data.py
                                              │
                          validate · dedup · cast · partition          pipeline/ingest_bronze.py
                                              ▼
   BRONZE  data/bronze/year=YYYY/month=MM/symbol=*/…parquet  (Snappy, Hive-partitioned)
                                              │
        DuckDB SQL: corp-action guard → calendar → capped ffill →     pipeline/build_silver.py
        returns / rolling vol / Parkinson / volume z / target
        pandas: EMA · MACD · Wilder RSI · Wilder ATR · ratios
                                              ▼
   SILVER  data/silver/features.parquet   (16 features + target + audit cols; unlabeled tail kept)
                                              │
        date-based 70/15/15 split, purge H sessions at each cut        pipeline/build_gold.py
                                              ▼
   GOLD    data/gold/{train,val,test}.parquet · baseline_reference.parquet (= train features)
                                              │
        LightGBM (scale_pos_weight, early stop on PR-AUC)              ml/train.py
        Platt calibration + thresholds fitted on VAL only              ml/calibration.py, ml/evaluate.py
                      │                                   │
                      ▼                                   ▼
          MLflow (SQLite mlruns.db,            data/gold/model.txt + model_meta.json
          mlruns/ artifacts)                              │
                                                          ▼
   ml/drift_engine.py (Evidently KS + PSI) ◄──────  serving/main.py  (FastAPI :8000)
                                                          │  JSON/HTTP, CORS-restricted
                                                          ▼
                                              dashboard/  (Next.js + shadcn/ui :3000)
```

Every stage appends to `data/lineage.json` (inputs, outputs, row counts, rejects, positive rate…),
giving a queryable audit trail of how each layer was produced.

## 2. Layers and contracts

### Bronze — raw, validated, immutable-per-run
| Aspect | Contract |
|---|---|
| Schema | `timestamp TIMESTAMP, symbol VARCHAR, open/high/low/close DOUBLE, volume BIGINT, ingested_at TIMESTAMP` (cast explicitly; mismatches fail the build) |
| Row rules | all prices > 0; `high ≥ max(open,low,close)`; `low ≤ min(open,high,close)`; `volume ≥ 0`; non-null keys. Failing rows are dropped and **counted** in lineage |
| Dedup | one row per `(symbol, calendar day)`, latest `ingested_at` wins |
| Layout | Snappy Parquet, `PARTITION_BY (year, month, symbol)`; rebuilt from scratch each run (idempotent) |

### Silver — clean feature store
Stage A is a single DuckDB SQL statement; stage B is pandas for recursive indicators.

1. **Corporate-action guard** – overnight gaps beyond ±35 % are back-adjusted into earlier prices.
2. **Calendar** – a session is a date where ≥ 50 % of symbols alive on it traded. Each symbol is
   gridded onto the sessions inside its own listing span; missing bars are forward-filled for at most
   3 consecutive sessions and flagged `is_filled`.
3. **Features** (all computable at the close of day *t*):

| Feature | Definition |
|---|---|
| `ret_1d/5d/20d` | log returns |
| `ema_ratio`, `ema_trend` | EMA9/EMA21, EMA21/EMA50 |
| `macd_norm`, `macd_hist_norm` | MACD(12,26) and histogram(signal 9), divided by close |
| `rsi_14` | Wilder RSI (α = 1/14) |
| `volatility_20d` | stdev₂₀(log ret) · √252 |
| `parkinson_vol_20d` | √(mean₂₀(ln(H/L)²)/(4 ln 2)) · √252 |
| `atr_ratio` | Wilder ATR(14) / close |
| `volume_zscore_20d` | (volume − mean₂₀)/stdev₂₀, 0 if undefined |
| `abs_ret_z` | \|r_t\| / σ₂₀ |
| `vol_ratio_5_20`, `vol_ratio_20_60` | σ₅/σ₂₀, σ₂₀/σ₆₀ |
| `mkt_volatility_20d` | `volatility_20d` of the benchmark `^NSEI` on the same date |

4. **Target** `high_volatility_event`: `1` iff `max_{k=1..3} |r_{t+k}| > 2.5 · σ₂₀(t)` (daily σ).
   The last 3 sessions per symbol have no complete forward window → target is `NULL`, never guessed.

### Gold — model-consumption mart
Splits are cut on **dates**, 70/15/15, and the last `H = 3` dates of train and val are purged because
their labels reach into the next split. `baseline_reference.parquet` holds the **train** features —
the distribution the model learned from, and therefore the drift reference.

## 3. Model & MLOps

| Concern | Implementation |
|---|---|
| Learner | `LGBMClassifier`, depth 4 / 15 leaves, `scale_pos_weight = neg/pos`, early stopping (patience 150) on validation **PR-AUC** |
| Calibration | Platt scaling (`a·logit(p)+b`) fitted on validation → `risk_score` is a real probability |
| Operating points | `elevated` = F1-optimal on validation; `critical` = lowest score with ≥ 50 % validation precision (fallback: 95th percentile). Fixed **before** the test set is scored |
| Tracking | MLflow, `sqlite:///mlruns.db`, artifacts in `mlruns/`; logs params, val/test metrics, SHAP importance, model, `model_meta.json` |
| Explainability | mean \|SHAP\| (TreeExplainer) on the test set, stored in `model_meta.json` and shown in the UI |
| Drift | Evidently `DataDriftPreset(method="ks")` + own PSI; reference = train; current = latest 60 sessions (`market`) or recent API inputs (`live`) |

Current model (see [decisions.md D-10](decisions.md) for how to read these): out-of-time
PR-AUC 0.166 vs 0.104 base rate (**1.60×**), ROC-AUC 0.642, 9 trees.

## 4. Serving API (`serving/`)

| Method & path | Purpose |
|---|---|
| `GET /health` | model + silver availability |
| `POST /api/v1/predict-risk` | validated feature vector → calibrated probability, tier, action, latency |
| `GET /api/v1/model-info` | run id, thresholds, metrics, SHAP, rolling p50/p95 latency, request count |
| `GET /api/v1/drift-status?source=market\|live` | per-feature KS p-value + PSI; cached 60 s; `live` returns 409 until ≥ 30 requests |
| `GET /api/v1/symbols`, `GET /api/v1/latest-features?symbol=` | pre-fill the simulator from silver |

The model and metadata load in the FastAPI lifespan hook. If they are missing the process stays up,
`/health` reports `degraded`, and inference returns 503. Request fields are
validated by Pydantic v2 ranges; the field list is asserted equal to `FEATURE_COLS` in tests so
training and serving cannot silently diverge. `model_meta.json` also records the feature list and the API refuses
to load a model trained on a different one.

## 5. Dashboard (`dashboard/`)
Next.js (App Router, client components) + shadcn/ui (base-nova style, Tailwind v4), dark theme.
`lib/api.ts` is the typed client. Components: `kpi-banner` (model, PR-AUC lift, latency, drift),
`risk-simulator` (symbol → pre-filled 16 features → score), `drift-monitor` (KS/PSI table, market vs
live toggle), `feature-importance` (SHAP bars).

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MG_DATA_DIR` | `<repo>/data` | lakehouse root |
| `MG_MLFLOW_URI` | `sqlite:///<repo>/mlruns.db` | MLflow tracking store |
| `MG_ARTIFACT_DIR` | `<repo>/mlruns` | MLflow artifact root |
| `MG_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | allowed browser origins |
| `MG_MLFLOW_URL` | `http://127.0.0.1:5000` | link shown in the UI |
| `API_PORT` / `DASH_PORT` / `MLFLOW_PORT` | 8000 / 3000 / 5000 | `run_all.sh` ports |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | API base URL for the dashboard |

All paths are resolved lazily (`pipeline.config.get_paths()`), which is what lets the test-suite run
the whole pipeline in a temp directory.

## 7. Repository map

```
pipeline/   config, lineage, fetch_data, ingest_bronze, build_silver, build_gold
ml/         train, evaluate, calibration, drift_engine
serving/    main (FastAPI), schemas (Pydantic v2)
scripts/    generate_synthetic_trades (GBM + Poisson jumps, 1M+ trades → daily bars)
dashboard/  Next.js console
tests/      pipeline, drift/calibration, API (offline, synthetic lakehouse fixture)
docs/       plan.md (spec), architecture.md, decisions.md
run_all.sh  end-to-end orchestrator
```

## 8. Testing
`pytest` (≈ 11 s, offline) builds a synthetic lakehouse in a temp dir and runs the real pipeline,
then asserts: bronze rejects/dedups bad rows and is partitioned; target equals an independent
recomputation; recursive indicators have no look-ahead; corporate actions are back-adjusted; gold splits
are chronological, disjoint and purged (verified to fail when the purge is removed); model artifacts and
thresholds are sane; PSI/KS/Platt behave; API contract, validation (422), tiers, drift and 409/404
paths.
