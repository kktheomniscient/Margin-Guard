from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import (FEATURE_COLS, HORIZON_DAYS, SIGMA_MULT, TARGET_COL, get_paths)
from tests.conftest import make_raw_cache, point_env


# ---------------------------------------------------------------- bronze
def test_bronze_rejects_bad_rows_and_dedups(tmp_path, monkeypatch):
    point_env(monkeypatch, tmp_path)

    def corrupt(sym, df):
        if sym != "AAA.NS":
            return df
        df = df.copy()
        df.loc[3, "high"] = df.loc[3, "low"] - 1          # high < low
        df.loc[4, "close"] = -5                            # non-positive price
        df.loc[5, "volume"] = -1                           # negative volume
        return pd.concat([df, df.iloc[[10]]], ignore_index=True)   # exact duplicate

    make_raw_cache(tmp_path, end="2019-06-30", trades=8_000, mutate=corrupt)
    from pipeline.ingest_bronze import write_bronze_layer

    stats = write_bronze_layer()
    assert stats["rows_rejected"] == 4  # 3 invalid rows + 1 duplicate
    files = list(get_paths().bronze.rglob("*.parquet"))
    assert files and all("year=" in str(f) and "month=" in str(f) and "symbol=" in str(f) for f in files)


# ---------------------------------------------------------------- silver
@pytest.fixture(scope="module")
def silver(lakehouse):
    return pd.read_parquet(get_paths().silver_file)


def test_silver_schema_and_no_bad_values(silver):
    assert set(FEATURE_COLS + [TARGET_COL, "symbol", "timestamp"]) <= set(silver.columns)
    assert not np.isinf(silver[FEATURE_COLS].to_numpy()).any()
    assert silver[FEATURE_COLS].notna().all().all()
    assert silver["rsi_14"].between(0, 100).all()
    assert not silver.duplicated(["symbol", "timestamp"]).any()


def test_target_matches_definition(silver):
    """Y_t = 1 iff max_{k=1..H}|r_{t+k}| > 2.5 * sigma_20d(t), recomputed independently."""
    checked = 0
    for _, g in silver.groupby("symbol"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        r = g["ret_1d"]
        sigma = r.rolling(20).std()
        fwd = pd.concat([r.shift(-k).abs() for k in range(1, HORIZON_DAYS + 1)], axis=1).max(axis=1, skipna=False)
        ok = fwd.notna() & sigma.notna() & g[TARGET_COL].notna()
        expected = (fwd[ok] > SIGMA_MULT * sigma[ok]).astype(int)
        assert (g.loc[ok, TARGET_COL].astype(int) == expected).all()
        checked += int(ok.sum())
        # the final H sessions have no complete forward window -> unlabeled, never guessed
        assert g[TARGET_COL].tail(HORIZON_DAYS).isna().all()
    assert checked > 1000


def test_features_have_no_lookahead(lakehouse):
    """Recursive indicators computed on data truncated at t equal those computed on the full series."""
    from pipeline.build_silver import _recursive_features

    s = pd.read_parquet(get_paths().silver_file)
    g = s[s["symbol"] == "AAA.NS"].sort_values("timestamp").reset_index(drop=True)
    cols = ["timestamp", "close", "ret_1d", "sigma_20d", "true_range"]
    base = g.assign(true_range=(g["high"] - g["low"]), sigma_5d=g["sigma_20d"], sigma_60d=g["sigma_20d"])
    full = _recursive_features(base)
    cut = _recursive_features(base.iloc[:300])
    for c in ["ema_ratio", "ema_trend", "macd_norm", "macd_hist_norm", "rsi_14", "atr_ratio"]:
        np.testing.assert_allclose(full[c].iloc[:300].to_numpy(), cut[c].to_numpy(), equal_nan=True)


def test_corporate_action_is_back_adjusted(tmp_path, monkeypatch):
    point_env(monkeypatch, tmp_path)

    def demerge(sym, df):
        if sym != "AAA.NS":
            return df
        df = df.copy()
        cut = len(df) // 2
        for c in ("open", "high", "low", "close"):
            df.loc[df.index[cut:], c] *= 0.5          # unadjusted -50% step, like a demerger
        return df

    make_raw_cache(tmp_path, end="2020-12-31", trades=12_000, mutate=demerge)
    from pipeline.build_silver import build_silver
    from pipeline.ingest_bronze import write_bronze_layer

    write_bronze_layer()
    df = build_silver()
    assert df.loc[df["symbol"] == "AAA.NS", "ret_1d"].abs().max() < 0.35


# ---------------------------------------------------------------- gold
def test_gold_splits_are_chronological_disjoint_and_purged(lakehouse):
    p = get_paths()
    tr, va, te = (pd.read_parquet(p.gold / f"{n}.parquet") for n in ("train", "val", "test"))
    assert tr["timestamp"].max() < va["timestamp"].min() <= va["timestamp"].max() < te["timestamp"].min()
    silver_dates = sorted(pd.read_parquet(p.silver_file, columns=["timestamp"])["timestamp"].unique())
    for left, right in ((tr, va), (va, te)):
        gap = sum(left["timestamp"].max() < d < right["timestamp"].min() for d in silver_dates)
        assert gap >= HORIZON_DAYS - 1, "labels of the earlier split would leak into the next one"
    n = len(tr) + len(va) + len(te)
    assert 0.6 < len(tr) / n < 0.75
    assert te[TARGET_COL].isin([0, 1]).all()


def test_baseline_reference_is_train(lakehouse):
    p = get_paths()
    ref = pd.read_parquet(p.gold / "baseline_reference.parquet")
    assert len(ref) == len(pd.read_parquet(p.gold / "train.parquet"))


# ---------------------------------------------------------------- model
def test_model_artifacts_and_calibration(lakehouse):
    m = lakehouse["meta"]
    assert m["features"] == FEATURE_COLS
    t = m["thresholds"]
    assert 0 < t["elevated"] <= t["critical"] < 1
    test = m["metrics"]["test"]
    assert 0 <= test["pr_auc"] <= 1 and 0 <= test["roc_auc"] <= 1
    assert get_paths().model_file.exists()
