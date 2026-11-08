# Project 2: Medallion Architecture Data Lake & Market Risk Predictive Engine (Native Local Execution)

## 1. Project Overview & JD Alignment

* **Role Alignment:** Data Scientist / ML Engineer / Data Lakehouse Track (Mirae Asset Sharekhan AI CoE).
* **Objective:** Build an end-to-end quantitative financial data lakehouse and machine learning operations (MLOps) platform running entirely natively on a local development workstation without containerization. The system ingests raw market time-series data into a 3-tier Medallion architecture (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) using DuckDB and Apache Parquet, engineers technical indicators, trains an imbalanced LightGBM classifier to predict high-volatility tail-risk events, tracks runs via a local SQLite-backed MLflow instance, monitors feature drift with Evidently AI, and exposes predictions through a FastAPI microservice and a Next.js (shadcn/ui) operational console.
* **Key JD Matches:**
  * Data Lake / Lakehouse layers, partitioning, schema enforcement, and data lineage.
  * SQL transformations, DuckDB data processing, and Snappy-compressed Parquet pipelines.
  * Supervised machine learning, quantitative time-series feature engineering, and handling class imbalance.
  * MLOps: Experiment tracking with MLflow, live feature drift monitoring with Evidently AI, Python virtual environment management, and interactive dashboarding.

---

## 2. Dataset Specification & Ingestion Blueprint

To ensure an autonomous coding agent can execute this project end-to-end without manual downloads or paid API keys, the data pipeline uses automated fetch scripts with `yfinance` to simulate multi-ticker exchange trade bars across large liquid Indian and global equity indices.

### A. Raw Data Sources
* **Tickers (Benchmark Portfolio):**
  * Index: `^NSEI` (NIFTY 50)
  * High-Beta Equities: `TATAMOTORS.NS`, `RELIANCE.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`
* **Granularity & Horizon:**
  * 5 years of daily bars (`1d`) from `2019-01-01` to current date for robust multi-cycle training (market crashes, rate hike cycles, rallies).
* **Fallback Generator:** An integrated synthetic trade log generator (`scripts/generate_synthetic_trades.py`) using Geometric Brownian Motion with Poisson jumps to generate 1,000,000+ realistic transaction rows if offline.

### B. Raw Ingestion Schema (Bronze)
| Column Name | Data Type | Description / Constraints |
|---|---|---|
| `timestamp` | `TIMESTAMP_NTZ` / `DATETIME` | Bar start timestamp (UTC) |
| `symbol` | `VARCHAR` | Stock symbol (e.g., `TATAMOTORS.NS`) |
| `open` | `DOUBLE` | Opening price ($\gt 0$) |
| `high` | `DOUBLE` | Highest price ($\ge \text{open}, \text{low}, \text{close}$) |
| `low` | `DOUBLE` | Lowest price ($\le \text{open}, \text{high}, \text{close}$) |
| `close` | `DOUBLE` | Closing price ($\gt 0$) |
| `volume` | `BIGINT` | Total trade volume ($\ge 0$) |
| `ingested_at` | `TIMESTAMP` | Pipeline ingestion audit timestamp |

---

## 3. Detailed Data Pipeline Architecture (Medallion Standard)

```
             [ Data Fetcher: yfinance API / Synthetic Generator ]
                                      │
                                      ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  BRONZE LAYER: Raw Data Lake                         │
   │  Path: data/bronze/year=YYYY/month=MM/symbol=*.parquet               │
   │  - Storage: Snappy Parquet partitioned by year, month, and symbol    │
   │  - Validation: Null checks, zero-price filtering, timestamp ordering │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼ DuckDB Transformation & SQL
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  SILVER LAYER: Clean Feature Store                   │
   │  Path: data/silver/features.parquet                                  │
   │  - Deduplication and Forward-fill for holiday gaps                   │
   │  - Engineered Technical Features:                                    │
   │    • Returns: 1d, 5d, 20d Log Returns                                │
   │    • Moving Averages: EMA(9), EMA(21), EMA(50), MACD(12, 26, 9)     │
   │    • Volatility: 20-period Rolling StDev, Parkinson Volatility       │
   │    • Momentum & Range: RSI(14), Average True Range (ATR-14)          │
   │    • Volume Dynamics: Rolling Volume Z-Score                         │
   │  - Supervised Target Definition: High-Volatility Event (0 or 1)      │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼ Temporal Partitioning (No Lookahead)
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  GOLD LAYER: Model Consumption Mart                  │
   │  Paths: data/gold/train.parquet, val.parquet, test.parquet           │
   │  - Time-based splits: 70% Train, 15% Validation, 15% Out-of-Time Test │
   │  - Standardized feature matrices ($X$) and ground truth targets ($y$)│
   │  - Baseline reference dataset exported for Evidently AI drift checks │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │              Machine Learning & MLOps Infrastructure                 │
   │  - LightGBM Classifier with scale_pos_weight for imbalance           │
   │  - Local MLflow Tracking: PR-AUC, ROC-AUC, F1, SQLite backend        │
   │  - Evidently AI: Population Stability Index & Kolmogorov-Smirnov test│
   │  - FastAPI Microservice: Sub-50ms inference & live drift telemetry  │
   │  - Next.js (App Router) + shadcn/ui: Real-time MLOps cockpit         │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Engineering & Target Mathematical Formulations

### 1. Log Returns
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### 2. Exponential Moving Average (EMA)
$$\text{EMA}_t(k) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}(k), \quad \text{where } \alpha = \frac{2}{k + 1}$$

### 3. Relative Strength Index (RSI - 14 Period)
$$U_t = \max(P_t - P_{t-1}, 0), \quad D_t = \max(P_{t-1} - P_t, 0)$$
$$\text{RS} = \frac{\text{EMA}_{14}(U)}{\text{EMA}_{14}(D)}, \quad \text{RSI} = 100 - \left(\frac{100}{1 + \text{RS}}\right)$$

### 4. Normalized Rolling Volatility (20-day annualized)
$$\sigma_{20, t} = \sqrt{\frac{252}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r})^2}$$

### 5. Ground Truth Target Label (`high_volatility_event`)
A binary event indicating an extreme volatility breakout over the next forward horizon ($H = 3$ days):
$$Y_t = \begin{cases} 1, & \text{if } \max_{k \in [1, H]} |r_{t+k}| > 2.5 \times \sigma_{20, t} \\ 0, & \text{otherwise} \end{cases}$$

*(Yields an imbalanced target distribution of $\sim 5\% - 8\%$ positive cases, mirroring real-world brokerage margin call events).*

---

## 5. Native Workspace Directory Structure

```text
market-risk-lakehouse/
├── venv/                          # Local Python virtual environment
├── requirements.txt               # Backend Python dependencies
├── run_all.sh                     # Native orchestrator running all services
├── data/
│   ├── raw_cache/                 # Cached raw ticker downloads
│   ├── bronze/                    # Partitioned raw Parquet: year=YYYY/month=MM/
│   ├── silver/                    # Cleaned feature tables (features.parquet)
│   └── gold/                      # Final splits (train.parquet, val.parquet, test.parquet)
├── mlruns.db                      # Local SQLite database for MLflow tracking
├── mlruns/                        # MLflow local artifact store
├── pipeline/
│   ├── __init__.py
│   ├── fetch_data.py              # Automated yfinance downloader with retries
│   ├── ingest_bronze.py           # DuckDB partitioned Parquet writer
│   ├── build_silver.py            # Vectorized technical indicator calculation
│   └── build_gold.py              # Temporal train/val/test splitter
├── ml/
│   ├── __init__.py
│   ├── train.py                   # LightGBM training with local MLflow logging
│   ├── evaluate.py                # PR-AUC, ROC-AUC, Brier score, SHAP importance
│   └── drift_engine.py            # Evidently AI DataDriftPreset report generator
├── serving/
│   ├── __init__.py
│   ├── main.py                    # FastAPI service exposing /predict-risk & /drift-status
│   └── schemas.py                 # Pydantic v2 validation models
└── dashboard/                     # Next.js 14 App Router + shadcn/ui frontend
    ├── package.json
    ├── components.json            # shadcn/ui configuration
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── src/
        ├── app/
        │   ├── layout.tsx         # Dark-mode dashboard layout
        │   └── page.tsx           # Operational MLOps & Risk Management cockpit
        ├── components/
        │   ├── ui/                # shadcn primitives: card, badge, table, input, button
        │   ├── kpi-banner.tsx     # Active Model, PR-AUC, Latency, Drift Status
        │   ├── drift-monitor.tsx  # Drift table rendering p-values and alert badges
        │   └── risk-simulator.tsx # Interactive form for instant real-time risk scoring
        └── lib/
            ├── api.ts             # Typed Fetch client communicating with FastAPI
            └── utils.ts           # Class merging utilities
```

---

## 6. Implementation Code Specifications

### Step 1: Data Downloader (`pipeline/fetch_data.py`)

```python
import datetime
import os
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["TATAMOTORS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"]
CACHE_DIR = Path("data/raw_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_ticker_history(ticker: str, start: str = "2019-01-01", end: str = None) -> pd.DataFrame:
    if end is None:
        end = datetime.datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
        
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "date"
    df = df.rename(columns={date_col: "timestamp"})
    df["symbol"] = ticker
    df["ingested_at"] = pd.Timestamp.utcnow()
    clean_df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "ingested_at"]]
    
    clean_symbol = ticker.replace("^", "").replace(".NS", "")
    out_path = CACHE_DIR / f"{clean_symbol}.parquet"
    clean_df.to_parquet(out_path, index=False)
    print(f"Cached {len(clean_df)} records for {ticker} -> {out_path}")
    return clean_df

if __name__ == "__main__":
    for sym in TICKERS:
        download_ticker_history(sym)
```

### Step 2: Bronze Ingestion via DuckDB (`pipeline/ingest_bronze.py`)

```python
import duckdb
from pathlib import Path

BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

def write_bronze_layer():
    con = duckdb.connect(database=":memory:")
    query = f"""
    COPY (
        SELECT 
            CAST(timestamp AS TIMESTAMP) AS timestamp,
            CAST(symbol AS VARCHAR) AS symbol,
            CAST(open AS DOUBLE) AS open,
            CAST(high AS DOUBLE) AS high,
            CAST(low AS DOUBLE) AS low,
            CAST(close AS DOUBLE) AS close,
            CAST(volume AS BIGINT) AS volume,
            CAST(ingested_at AS TIMESTAMP) AS ingested_at,
            YEAR(timestamp) AS year,
            LPAD(CAST(MONTH(timestamp) AS VARCHAR), 2, '0') AS month
        FROM read_parquet('data/raw_cache/*.parquet')
        WHERE open > 0 AND high >= low AND volume >= 0
    ) 
    TO '{BRONZE_DIR}' (FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE 1);
    """
    con.execute(query)
    con.close()
    print("Bronze Layer materialized with Year/Month partitioning.")

if __name__ == "__main__":
    write_bronze_layer()
```

### Step 3: Silver Feature Computation (`pipeline/build_silver.py`)

```python
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

def build_silver():
    con = duckdb.connect()
    raw_df = con.execute("SELECT * FROM read_parquet('data/bronze/*/*/*.parquet') ORDER BY timestamp ASC").df()
    
    dfs = []
    for symbol, group in raw_df.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        group["log_ret"] = np.log(group["close"] / group["close"].shift(1))
        
        # EMAs
        group["ema_9"] = group["close"].ewm(span=9, adjust=False).mean()
        group["ema_21"] = group["close"].ewm(span=21, adjust=False).mean()
        group["ema_50"] = group["close"].ewm(span=50, adjust=False).mean()
        group["ema_ratio"] = group["ema_9"] / (group["ema_21"] + 1e-8)
        
        # RSI 14
        delta = group["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        group["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Rolling Volatility (20-day annualized)
        group["volatility_20d"] = group["log_ret"].rolling(window=20).std() * np.sqrt(252)
        
        # Average True Range (ATR 14)
        high_low = group["high"] - group["low"]
        high_close = (group["high"] - group["close"].shift()).abs()
        low_close = (group["low"] - group["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group["atr_14"] = tr.rolling(window=14).mean()
        group["atr_ratio"] = group["atr_14"] / (group["close"] + 1e-8)
        
        # Binary target: 3-period forward volatility spike
        forward_max_ret = group["log_ret"].shift(-1).abs().rolling(window=3).max()
        group["high_volatility_event"] = (forward_max_ret > 2.5 * group["volatility_20d"]).astype(int)
        
        dfs.append(group.dropna())
        
    final_df = pd.concat(dfs, ignore_index=True)
    out_file = SILVER_DIR / "features.parquet"
    final_df.to_parquet(out_file, index=False)
    print(f"Silver Feature Table created: {len(final_df)} records -> {out_file}")

if __name__ == "__main__":
    build_silver()
```

### Step 4: Gold Splitter (`pipeline/build_gold.py`)

```python
import os
import duckdb
from pathlib import Path

GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

def build_gold():
    con = duckdb.connect(database=":memory:")
    df = con.execute("SELECT * FROM read_parquet('data/silver/features.parquet') ORDER BY timestamp ASC").df()
    
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_df.to_parquet(GOLD_DIR / "train.parquet", index=False)
    val_df.to_parquet(GOLD_DIR / "val.parquet", index=False)
    test_df.to_parquet(GOLD_DIR / "test.parquet", index=False)
    test_df.to_parquet(GOLD_DIR / "baseline_reference.parquet", index=False)
    print("Gold Layer created with leak-free temporal partitions.")

if __name__ == "__main__":
    build_gold()
```

### Step 5: LightGBM Training & Local MLflow (`ml/train.py`)

```python
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

FEATURE_COLS = ["ema_ratio", "rsi_14", "volatility_20d", "atr_ratio", "volume"]

def train_model():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Sharekhan_Market_Risk_Engine")
    
    train = pd.read_parquet("data/gold/train.parquet")
    val = pd.read_parquet("data/gold/val.parquet")
    test = pd.read_parquet("data/gold/test.parquet")
    
    X_train, y_train = train[FEATURE_COLS], train["high_volatility_event"]
    X_val, y_val = val[FEATURE_COLS], val["high_volatility_event"]
    X_test, y_test = test[FEATURE_COLS], test["high_volatility_event"]
    
    scale_weight = float((len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5))
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 250,
        "max_depth": 5,
        "scale_pos_weight": scale_weight,
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="lgbm-volatility-native"):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds_prob = model.predict_proba(X_test)[:, 1]
        preds_bin = (preds_prob > 0.5).astype(int)
        
        pr_auc = average_precision_score(y_test, preds_prob)
        roc_auc = roc_auc_score(y_test, preds_prob)
        f1 = f1_score(y_test, preds_bin, zero_division=0)
        
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.lightgbm.log_model(model, "model")
        
        model.booster_.save_model("data/gold/model.txt")
        print(f"Training Complete. PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    train_model()
```

### Step 6: Evidently AI Drift Detection (`ml/drift_engine.py`)

```python
import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

FEATURE_COLS = ["ema_ratio", "rsi_14", "volatility_20d", "atr_ratio", "volume"]

def run_drift_check(current_df: pd.DataFrame, baseline_path="data/gold/baseline_reference.parquet") -> dict:
    reference_df = pd.read_parquet(baseline_path)[FEATURE_COLS]
    current_df = current_df[FEATURE_COLS]
    
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    summary = drift_report.as_dict()
    metrics = summary["metrics"][0]["result"]
    
    return {
        "dataset_drift_detected": metrics["dataset_drift"],
        "share_of_drifted_columns": metrics["share_of_drifted_columns"],
        "drifted_features": [
            col for col, data in metrics["drift_by_columns"].items() if data["drift_detected"]
        ],
        "feature_details": {
            col: {
                "p_value": round(data["drift_score"], 4),
                "drift_detected": data["drift_detected"]
            } for col, data in metrics["drift_by_columns"].items()
        }
    }
```

### Step 7: FastAPI Microservice (`serving/main.py`)

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ml.drift_engine import run_drift_check

app = FastAPI(title="Sharekhan Market Risk Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = ["ema_ratio", "rsi_14", "volatility_20d", "atr_ratio", "volume"]
booster = None

@app.on_event("startup")
def load_model():
    global booster
    try:
        booster = lgb.Booster(model_file="data/gold/model.txt")
    except Exception as e:
        print(f"Warning: Model file not loaded on startup: {e}")

class RiskPredictionRequest(BaseModel):
    symbol: str = Field(..., example="TATAMOTORS.NS")
    ema_ratio: float = Field(..., example=1.04)
    rsi_14: float = Field(..., example=68.5)
    volatility_20d: float = Field(..., example=0.28)
    atr_ratio: float = Field(..., example=0.035)
    volume: float = Field(..., example=1500000)

class RiskPredictionResponse(BaseModel):
    symbol: str
    risk_score: float
    risk_level: str
    action_trigger: str

@app.post("/api/v1/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest):
    if booster is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    features = np.array([[payload.ema_ratio, payload.rsi_14, payload.volatility_20d, payload.atr_ratio, payload.volume]])
    prob = float(booster.predict(features)[0])
    
    if prob >= 0.70:
        level, action = "CRITICAL", "Reduce Margin Leverage Immediately"
    elif prob >= 0.40:
        level, action = "ELEVATED", "Issue Risk Warning to Account"
    else:
        level, action = "LOW", "Maintain Normal Margin Limits"
        
    return RiskPredictionResponse(
        symbol=payload.symbol,
        risk_score=round(prob, 4),
        risk_level=level,
        action_trigger=action
    )

@app.get("/api/v1/drift-status")
def get_drift_status():
    test_df = pd.read_parquet("data/gold/test.parquet")
    return run_drift_check(test_df.tail(100))
```

---

## 7. Next.js + shadcn/ui Dashboard Architecture

Scaffold the operational dashboard using Next.js 14 and `shadcn/ui` in the `dashboard/` directory:

```bash
npx create-next-app@latest dashboard --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm
cd dashboard
npx shadcn@latest init -d
npx shadcn@latest add card badge table button input alert tabs progress
npm install lucide-react clsx tailwind-merge
```

### Dashboard View (`dashboard/src/app/page.tsx`)

```tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export default function RiskConsole() {
  const [drift, setDrift] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [formData, setFormData] = useState({
    symbol: "TATAMOTORS.NS",
    ema_ratio: 1.04,
    rsi_14: 68.5,
    volatility_20d: 0.28,
    atr_ratio: 0.035,
    volume: 1500000
  });

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/drift-status")
      .then(res => res.json())
      .then(data => setDrift(data))
      .catch(console.error);
  }, []);

  const handlePredict = async () => {
    const res = await fetch("http://127.0.0.1:8000/api/v1/predict-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });
    const result = await res.json();
    setPrediction(result);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 p-8 space-y-8 font-sans">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Market Risk & Lakehouse Cockpit</h1>
          <p className="text-sm text-slate-400">Sharekhan AI CoE • Quantitative Model Ops (Native Execution)</p>
        </div>
        <div className="flex gap-3">
          <Badge variant="outline" className="text-emerald-400 border-emerald-500/30">
            Pipeline: DuckDB Lakehouse Active
          </Badge>
          <Badge variant="outline" className="text-blue-400 border-blue-500/30">
            Model: LightGBM Production
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-slate-900 border-slate-800 text-slate-100">
          <CardHeader><CardTitle className="text-sm font-medium">Data Drift Status</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {drift ? (drift.dataset_drift_detected ? "Drift Detected" : "Stable") : "Checking..."}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Drifted Feature Share: {drift ? `${(drift.share_of_drifted_columns * 100).toFixed(0)}%` : "..."}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800 text-slate-100">
          <CardHeader><CardTitle className="text-sm font-medium">Evaluation Metric (Holdout)</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-400">PR-AUC Optimized</div>
            <p className="text-xs text-slate-400 mt-1">Trained on imbalanced risk breakouts</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800 text-slate-100">
          <CardHeader><CardTitle className="text-sm font-medium">Local Tracking Store</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-400">MLflow Native</div>
            <p className="text-xs text-slate-400 mt-1">sqlite:///mlruns.db on port 5000</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card className="bg-slate-900 border-slate-800 text-slate-100">
          <CardHeader><CardTitle>Live Quantitative Risk Tester</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-xs text-slate-400">Ticker Symbol</label>
              <Input 
                value={formData.symbol} 
                onChange={e => setFormData({...formData, symbol: e.target.value})}
                className="bg-slate-950 border-slate-700" 
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-slate-400">14-Day RSI</label>
                <Input 
                  type="number" 
                  value={formData.rsi_14} 
                  onChange={e => setFormData({...formData, rsi_14: parseFloat(e.target.value)})}
                  className="bg-slate-950 border-slate-700" 
                />
              </div>
              <div>
                <label className="text-xs text-slate-400">Rolling Volatility (20d)</label>
                <Input 
                  type="number" 
                  value={formData.volatility_20d} 
                  onChange={e => setFormData({...formData, volatility_20d: parseFloat(e.target.value)})}
                  className="bg-slate-950 border-slate-700" 
                />
              </div>
            </div>
            <Button onClick={handlePredict} className="w-full bg-blue-600 hover:bg-blue-500">
              Run Real-Time Inference
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800 text-slate-100">
          <CardHeader><CardTitle>Inference Diagnostics</CardTitle></CardHeader>
          <CardContent>
            {prediction ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-950 rounded border border-slate-800">
                  <span className="text-sm">Risk Probability Score</span>
                  <span className="font-mono text-lg font-bold">{(prediction.risk_score * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-950 rounded border border-slate-800">
                  <span className="text-sm">Risk Assessment Tier</span>
                  <Badge variant={prediction.risk_level === "CRITICAL" ? "destructive" : "default"}>
                    {prediction.risk_level}
                  </Badge>
                </div>
                <div className="p-3 bg-slate-950 rounded border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Recommended Brokerage Action</span>
                  <span className="text-sm font-medium">{prediction.action_trigger}</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Submit feature inputs to view model probability output.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

---

## 8. Single-Command Native Orchestration Script (`run_all.sh`)

Create `run_all.sh` to run the end-to-end data pipeline, model training, and all services concurrently on localhost:

```bash
#!/usr/bin/env bash
set -e

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "=========================================================="
echo "Phase 1: Downloading Market Data (NSE / Global Tickers)"
echo "=========================================================="
python pipeline/fetch_data.py

echo "=========================================================="
echo "Phase 2: Ingesting into Bronze Medallion Lakehouse"
echo "=========================================================="
python pipeline/ingest_bronze.py

echo "=========================================================="
echo "Phase 3: Calculating Silver Technical Indicators"
echo "=========================================================="
python pipeline/build_silver.py

echo "=========================================================="
echo "Phase 4: Materializing Gold Modeling Partitions"
echo "=========================================================="
python pipeline/build_gold.py

echo "=========================================================="
echo "Phase 5: Training LightGBM Model & Logging to MLflow"
echo "=========================================================="
python ml/train.py

echo "=========================================================="
echo "Phase 6: Starting Background Services"
echo "=========================================================="

# Start local MLflow server on port 5000
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --port 5000 &
MLFLOW_PID=$!

# Start FastAPI backend on port 8000
uvicorn serving.main:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# Start Next.js frontend on port 3000
cd dashboard && npm run dev &
NEXT_PID=$!
cd ..

echo "--------------------------------------------------------"
echo "All local services active:"
echo " • FastAPI Backend:    http://127.0.0.1:8000/docs"
echo " • MLflow Tracking:    http://127.0.0.1:5000"
echo " • shadcn/ui Cockpit:  http://localhost:3000"
echo "--------------------------------------------------------"
echo "Press [CTRL+C] to stop all background services."

trap "kill $MLFLOW_PID $FASTAPI_PID $NEXT_PID" EXIT
wait
```

---

## 9. Verification & Execution Runbook

To execute the entire project from scratch:

```bash
# Make script executable and run
chmod +x run_all.sh
./run_all.sh
```

### Verification Checklist:
- [ ] Parquet files generated in `data/bronze/`, `data/silver/`, and `data/gold/`.
- [ ] MLflow UI at `http://127.0.0.1:5000` displays runs with logged PR-AUC and parameters.
- [ ] FastAPI Swagger docs at `http://127.0.0.1:8000/docs` accept `POST /api/v1/predict-risk` calls.
- [ ] Dashboard at `http://localhost:3000` renders drift diagnostics and returns risk predictions.