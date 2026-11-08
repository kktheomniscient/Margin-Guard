"""Pydantic v2 request/response contracts. Field names MUST equal pipeline.config.FEATURE_COLS
(enforced by tests/test_api.py)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["LOW", "ELEVATED", "CRITICAL"]


class RiskPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "symbol": "RELIANCE.NS", "ret_1d": -0.012, "ret_5d": -0.03, "ret_20d": 0.01,
        "ema_ratio": 0.99, "ema_trend": 1.0, "macd_norm": -0.004, "macd_hist_norm": -0.002,
        "rsi_14": 41.0, "volatility_20d": 0.24, "parkinson_vol_20d": 0.22, "atr_ratio": 0.014,
        "volume_zscore_20d": 0.8, "abs_ret_z": 0.8, "vol_ratio_5_20": 1.3, "vol_ratio_20_60": 1.1,
        "mkt_volatility_20d": 0.16}})

    symbol: str = Field(min_length=1, max_length=20)
    ret_1d: float = Field(ge=-1.0, le=1.0, description="1-day log return")
    ret_5d: float = Field(ge=-2.0, le=2.0, description="5-day log return")
    ret_20d: float = Field(ge=-3.0, le=3.0, description="20-day log return")
    ema_ratio: float = Field(gt=0, le=5, description="EMA(9)/EMA(21)")
    ema_trend: float = Field(gt=0, le=5, description="EMA(21)/EMA(50)")
    macd_norm: float = Field(ge=-1, le=1, description="MACD(12,26)/close")
    macd_hist_norm: float = Field(ge=-1, le=1, description="MACD histogram/close")
    rsi_14: float = Field(ge=0, le=100, description="Wilder RSI(14)")
    volatility_20d: float = Field(ge=0, le=10, description="Annualised 20d stdev of log returns")
    parkinson_vol_20d: float = Field(ge=0, le=10, description="Annualised 20d Parkinson volatility")
    atr_ratio: float = Field(ge=0, le=1, description="Wilder ATR(14)/close")
    volume_zscore_20d: float = Field(ge=-20, le=20, description="Volume z-score vs 20d window")
    abs_ret_z: float = Field(ge=0, le=50, description="|1d return| in 20d-sigma units")
    vol_ratio_5_20: float = Field(ge=0, le=20, description="5d vol / 20d vol")
    vol_ratio_20_60: float = Field(ge=0, le=20, description="20d vol / 60d vol")
    mkt_volatility_20d: float = Field(ge=0, le=10, description="Benchmark index annualised 20d vol")


class RiskPredictionResponse(BaseModel):
    symbol: str
    risk_score: float = Field(description="Calibrated probability of a >2.5-sigma move in the next 3 sessions")
    base_rate: float = Field(description="Training-set event frequency, for context")
    risk_level: RiskLevel
    action_trigger: str
    latency_ms: float
    model_run_id: str


class ModelInfo(BaseModel):
    model_name: str
    run_id: str
    trained_at: str
    n_features: int
    best_iteration: int
    thresholds: dict[str, float]
    metrics: dict[str, dict[str, float]]
    shap_importance: dict[str, float]
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    requests_served: int
    mlflow_url: str
    data_as_of: str | None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    silver_available: bool


class FeatureDrift(BaseModel):
    p_value: float
    psi: float
    drift_detected: bool
    psi_status: Literal["stable", "warn", "alert"]


class DriftStatus(BaseModel):
    source: Literal["market", "live"]
    dataset_drift_detected: bool
    share_of_drifted_columns: float
    drifted_features: list[str]
    feature_details: dict[str, FeatureDrift]
    n_reference: int
    n_current: int
    stattest: str


class LatestFeatures(BaseModel):
    symbol: str
    as_of: str
    features: dict[str, float]
