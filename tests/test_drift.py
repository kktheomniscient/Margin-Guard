from __future__ import annotations

import numpy as np
import pandas as pd

from ml.calibration import apply_platt, fit_platt
from ml.drift_engine import population_stability_index, run_drift_check
from pipeline.config import FEATURE_COLS, get_paths


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert population_stability_index(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert population_stability_index(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_run_drift_check_flags_shifted_features(lakehouse):
    ref = pd.read_parquet(get_paths().gold / "baseline_reference.parquet")[FEATURE_COLS]
    same = run_drift_check(ref.sample(300, random_state=1))
    assert same["share_of_drifted_columns"] <= 0.2 and not same["dataset_drift_detected"]

    shifted = ref.sample(300, random_state=1).copy()
    shifted["rsi_14"] += 30
    shifted["volatility_20d"] *= 4
    res = run_drift_check(shifted)
    assert {"rsi_14", "volatility_20d"} <= set(res["drifted_features"])
    assert res["feature_details"]["rsi_14"]["psi_status"] == "alert"


def test_platt_calibration_recovers_base_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(int)
    raw = np.clip(0.5 + 0.3 * (y - 0.1) + rng.normal(0, 0.1, 4000), 0.01, 0.99)  # inflated, informative
    cal = apply_platt(raw, fit_platt(raw, y))
    assert abs(cal.mean() - y.mean()) < 0.01
    assert np.all(np.diff(cal[np.argsort(raw)]) >= -1e-12)  # monotonic
