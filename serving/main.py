"""MarginGuard risk API: real-time inference, model metadata and live drift telemetry."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ml.calibration import apply_platt
from ml.drift_engine import latest_market_window, run_drift_check
from pipeline.config import FEATURE_COLS, get_paths
from serving.schemas import (DriftStatus, HealthResponse, LatestFeatures, ModelInfo, RiskPredictionRequest,
                             RiskPredictionResponse)

log = logging.getLogger("margin_guard.api")

ACTIONS = {
    "CRITICAL": "Reduce Margin Leverage Immediately",
    "ELEVATED": "Issue Risk Warning to Account",
    "LOW": "Maintain Normal Margin Limits",
}
DRIFT_TTL_S = 60          # Evidently is ~seconds-slow; cache results
MIN_LIVE_SAMPLES = 30     # below this a live-traffic KS test is meaningless
MLFLOW_URL = os.environ.get("MG_MLFLOW_URL", "http://127.0.0.1:5000")


class State:
    booster: lgb.Booster | None = None
    meta: dict = {}
    latencies: deque = deque(maxlen=1000)
    live_inputs: deque = deque(maxlen=500)
    requests: int = 0
    drift_cache: dict = {}
    lock = threading.Lock()


state = State()


def load_artifacts() -> None:
    paths = get_paths()
    try:
        state.booster = lgb.Booster(model_file=str(paths.model_file))
        state.meta = json.loads(paths.model_meta.read_text())
        if state.meta["features"] != FEATURE_COLS:
            raise ValueError("model_meta features differ from pipeline.config.FEATURE_COLS - retrain")
    except Exception as exc:  # keep the API up so /health can report the problem
        state.booster, state.meta = None, {}
        log.warning("Model not loaded: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="MarginGuard Market Risk Engine API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model() -> lgb.Booster:
    if state.booster is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run the training pipeline first.")
    return state.booster


def _tier(prob: float) -> str:
    t = state.meta.get("thresholds", {"elevated": 0.4, "critical": 0.7})
    return "CRITICAL" if prob >= t["critical"] else "ELEVATED" if prob >= t["elevated"] else "LOW"


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    silver = get_paths().silver_file.exists()
    loaded = state.booster is not None
    return HealthResponse(status="ok" if loaded and silver else "degraded", model_loaded=loaded,
                          silver_available=silver)


@app.post("/api/v1/predict-risk", response_model=RiskPredictionResponse, tags=["inference"])
def predict_risk(payload: RiskPredictionRequest):
    booster = _require_model()
    t0 = time.perf_counter()
    row = {c: getattr(payload, c) for c in FEATURE_COLS}
    raw = booster.predict(np.array([[row[c] for c in FEATURE_COLS]], dtype=float))
    prob = float(apply_platt(raw, state.meta["platt"])[0])
    level = _tier(prob)
    latency = (time.perf_counter() - t0) * 1000
    with state.lock:
        state.latencies.append(latency)
        state.live_inputs.append(row)
        state.requests += 1
    return RiskPredictionResponse(
        symbol=payload.symbol, risk_score=round(prob, 4),
        base_rate=round(state.meta["metrics"]["validation"]["prevalence"], 4),
        risk_level=level, action_trigger=ACTIONS[level], latency_ms=round(latency, 3),
        model_run_id=state.meta["run_id"],
    )


@app.get("/api/v1/model-info", response_model=ModelInfo, tags=["ops"])
def model_info():
    _require_model()
    m = state.meta
    with state.lock:
        lat = list(state.latencies)
        served = state.requests
    as_of = None
    if get_paths().silver_file.exists():
        as_of = str(pd.read_parquet(get_paths().silver_file, columns=["timestamp"])["timestamp"].max().date())
    return ModelInfo(
        model_name=m["model_name"], run_id=m["run_id"], trained_at=m["trained_at"], n_features=len(m["features"]),
        best_iteration=m["best_iteration"], thresholds=m["thresholds"], metrics=m["metrics"],
        shap_importance=m["shap_importance"],
        latency_ms_p50=round(float(np.percentile(lat, 50)), 3) if lat else None,
        latency_ms_p95=round(float(np.percentile(lat, 95)), 3) if lat else None,
        requests_served=served, mlflow_url=MLFLOW_URL, data_as_of=as_of,
    )


@app.get("/api/v1/drift-status", response_model=DriftStatus, tags=["monitoring"])
def drift_status(source: str = Query("market", pattern="^(market|live)$")):
    """`market`: latest 60 sessions of silver vs. the training reference.
    `live`: inputs recently sent to /predict-risk (needs >= 30 samples)."""
    cached = state.drift_cache.get(source)
    if cached and time.time() - cached[0] < DRIFT_TTL_S:
        return cached[1]
    if source == "live":
        with state.lock:
            current = pd.DataFrame(list(state.live_inputs))
        if len(current) < MIN_LIVE_SAMPLES:
            raise HTTPException(status_code=409, detail=f"Need >= {MIN_LIVE_SAMPLES} scored requests for live drift "
                                                         f"(have {len(current)}).")
    else:
        if not get_paths().silver_file.exists():
            raise HTTPException(status_code=503, detail="Silver layer not built.")
        current = latest_market_window()
    result = DriftStatus(source=source, **run_drift_check(current))
    state.drift_cache[source] = (time.time(), result)
    return result


@app.get("/api/v1/symbols", response_model=list[str], tags=["reference"])
def symbols():
    f = get_paths().silver_file
    if not f.exists():
        raise HTTPException(status_code=503, detail="Silver layer not built.")
    return sorted(pd.read_parquet(f, columns=["symbol"])["symbol"].unique().tolist())


@app.get("/api/v1/latest-features", response_model=LatestFeatures, tags=["reference"])
def latest_features(symbol: str):
    """Most recent engineered feature vector for a symbol (to pre-fill the simulator)."""
    f = get_paths().silver_file
    if not f.exists():
        raise HTTPException(status_code=503, detail="Silver layer not built.")
    df = pd.read_parquet(f, columns=["timestamp", "symbol", *FEATURE_COLS])
    df = df[df["symbol"] == symbol]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol!r}")
    row = df.sort_values("timestamp").iloc[-1]
    return LatestFeatures(symbol=symbol, as_of=str(row["timestamp"].date()),
                          features={c: float(row[c]) for c in FEATURE_COLS})
