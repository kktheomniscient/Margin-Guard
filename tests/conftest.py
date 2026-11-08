"""Test fixtures: a tiny synthetic lakehouse in a temp dir (offline, deterministic)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pipeline.config import BRONZE_COLS, MARKET_SYMBOL, clean_symbol
from scripts.generate_synthetic_trades import synthetic_daily_bars

SYMBOLS = [MARKET_SYMBOL, "AAA.NS", "BBB.NS", "CCC.NS"]


def make_raw_cache(root, start="2019-01-01", end="2022-12-31", trades=40_000, mutate=None):
    cache = root / "data" / "raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for sym in SYMBOLS:
        df = synthetic_daily_bars(sym, start, end, trades)[BRONZE_COLS]
        if mutate:
            df = mutate(sym, df)
        df.to_parquet(cache / f"{clean_symbol(sym)}.parquet", index=False)


def point_env(monkeypatch, root):
    monkeypatch.setenv("MG_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("MG_MLFLOW_URI", f"sqlite:///{root / 'mlruns.db'}")
    monkeypatch.setenv("MG_ARTIFACT_DIR", str(root / "mlruns"))


@pytest.fixture(scope="session")
def lakehouse(tmp_path_factory):
    """Full pipeline (bronze -> silver -> gold -> trained model) on synthetic data."""
    root = tmp_path_factory.mktemp("lakehouse")
    old = {k: os.environ.get(k) for k in ("MG_DATA_DIR", "MG_MLFLOW_URI", "MG_ARTIFACT_DIR")}
    os.environ["MG_DATA_DIR"] = str(root / "data")
    os.environ["MG_MLFLOW_URI"] = f"sqlite:///{root / 'mlruns.db'}"
    os.environ["MG_ARTIFACT_DIR"] = str(root / "mlruns")
    make_raw_cache(root)
    from ml.train import train_model
    from pipeline.build_gold import build_gold
    from pipeline.build_silver import build_silver
    from pipeline.ingest_bronze import write_bronze_layer

    write_bronze_layer()
    build_silver()
    build_gold()
    meta = train_model()
    yield {"root": root, "meta": meta}
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
