"""Single source of truth for paths, symbols, feature lists and target definition.

Paths are resolved lazily from environment variables so tests (and alternate
deployments) can redirect the whole lakehouse without touching code:

    MG_DATA_DIR      root of the lakehouse (default: <repo>/data)
    MG_MLFLOW_URI    MLflow tracking URI    (default: sqlite:///<repo>/mlruns.db)
    MG_ARTIFACT_DIR  MLflow artifact root   (default: <repo>/mlruns)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Universe -------------------------------------------------------------
# TATAMOTORS.NS was demerged (late 2025) into TMPV.NS (passenger vehicles,
# continuous history) and TMCV.NS (commercial vehicles, listed Nov 2025).
# Each requested symbol maps to an ordered list of candidates; the first that
# returns data wins, otherwise *all* successors are used (see fetch_data.py).
TICKERS: list[str] = [
    "^NSEI",
    "TATAMOTORS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]
SUCCESSORS: dict[str, list[str]] = {"TATAMOTORS.NS": ["TMPV.NS", "TMCV.NS"]}
START_DATE = "2019-01-01"

# --- Features / target ----------------------------------------------------
FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ema_ratio",        # EMA9 / EMA21
    "ema_trend",        # EMA21 / EMA50
    "macd_norm",        # MACD(12,26) / close
    "macd_hist_norm",   # (MACD - signal(9)) / close
    "rsi_14",
    "volatility_20d",   # annualised rolling stdev of log returns
    "parkinson_vol_20d",
    "atr_ratio",        # ATR14 / close
    "volume_zscore_20d",
    "abs_ret_z",        # |r_t| / sigma_20d : size of the latest shock in sigma units
    "vol_ratio_5_20",   # short-term vs medium-term realised vol (vol clustering / acceleration)
    "vol_ratio_20_60",  # medium-term vs long-term vol (mean-reversion of vol)
    "mkt_volatility_20d",  # market-wide vol regime (benchmark index)
]
MARKET_SYMBOL = "^NSEI"
TARGET_COL = "high_volatility_event"
HORIZON_DAYS = 3          # H: forward window for the event
SIGMA_MULT = 2.5          # event threshold in daily-sigma units
TRADING_DAYS = 252
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # remainder (0.15) is out-of-time test

BRONZE_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "ingested_at"]


@dataclass(frozen=True)
class Paths:
    data: Path
    raw_cache: Path
    bronze: Path
    silver: Path
    gold: Path
    reports: Path
    lineage: Path
    mlflow_uri: str
    artifact_root: str

    @property
    def silver_file(self) -> Path:
        return self.silver / "features.parquet"

    @property
    def model_file(self) -> Path:
        return self.gold / "model.txt"

    @property
    def model_meta(self) -> Path:
        return self.gold / "model_meta.json"


def get_paths() -> Paths:
    data = Path(os.environ.get("MG_DATA_DIR", REPO_ROOT / "data")).resolve()
    return Paths(
        data=data,
        raw_cache=data / "raw_cache",
        bronze=data / "bronze",
        silver=data / "silver",
        gold=data / "gold",
        reports=data / "reports",
        lineage=data / "lineage.json",
        mlflow_uri=os.environ.get("MG_MLFLOW_URI", f"sqlite:///{REPO_ROOT / 'mlruns.db'}"),
        artifact_root=os.environ.get("MG_ARTIFACT_DIR", str(REPO_ROOT / "mlruns")),
    )


def clean_symbol(symbol: str) -> str:
    """Filesystem/partition-safe symbol name (``^NSEI`` -> ``NSEI``)."""
    return symbol.replace("^", "").replace(".NS", "")
