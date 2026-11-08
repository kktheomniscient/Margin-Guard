"""Feature-drift detection: Evidently (Kolmogorov-Smirnov) + Population Stability Index.

Reference  = TRAIN features (what the model learned from).
Current    = either the most recent market window from silver ("market") or recently
             scored requests captured by the API ("live").
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from pipeline.config import FEATURE_COLS, get_paths

KS_ALPHA = 0.05          # KS p-value below this => feature drifted
PSI_WARN, PSI_ALERT = 0.10, 0.25   # conventional PSI bands: <0.1 stable, 0.1-0.25 moderate, >0.25 major
DATASET_DRIFT_SHARE = 0.5          # dataset-level drift if >= 50% of features drifted (Evidently default)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI with reference-decile bins (open-ended tails); epsilon-smoothed to avoid log(0)."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference.dropna(), edges)[0] / max(reference.notna().sum(), 1)
    cur_p = np.histogram(current.dropna(), edges)[0] / max(current.notna().sum(), 1)
    ref_p, cur_p = np.clip(ref_p, 1e-4, None), np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def latest_market_window(sessions: int = 60) -> pd.DataFrame:
    """All symbols over the most recent `sessions` trading days (includes the unlabeled tail)."""
    df = pd.read_parquet(get_paths().silver_file, columns=["timestamp", "symbol", *FEATURE_COLS])
    cutoff = sorted(df["timestamp"].unique())[-sessions]
    return df[df["timestamp"] >= cutoff]


def run_drift_check(current_df: pd.DataFrame, baseline_path: str | None = None,
                    html_path: str | None = None) -> dict:
    baseline_path = baseline_path or str(get_paths().gold / "baseline_reference.parquet")
    reference = pd.read_parquet(baseline_path)[FEATURE_COLS].astype(float)
    current = current_df[FEATURE_COLS].astype(float)

    definition = DataDefinition(numerical_columns=FEATURE_COLS)
    snapshot = Report([DataDriftPreset(method="ks", threshold=KS_ALPHA)]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    if html_path:
        snapshot.save_html(html_path)

    p_values: dict[str, float] = {}
    share = None
    for m in snapshot.dict()["metrics"]:
        name = m["metric_name"]
        if name.startswith("ValueDrift"):
            col = m["config"]["column"]
            p_values[col] = float(m["value"])
        elif name.startswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])

    details = {}
    for col in FEATURE_COLS:
        psi = population_stability_index(reference[col], current[col])
        details[col] = {
            "p_value": round(p_values[col], 4),
            "psi": round(psi, 4),
            "drift_detected": bool(p_values[col] < KS_ALPHA),
            "psi_status": "alert" if psi >= PSI_ALERT else "warn" if psi >= PSI_WARN else "stable",
        }
    drifted = [c for c, d in details.items() if d["drift_detected"]]
    share = share if share is not None else len(drifted) / len(FEATURE_COLS)
    return {
        "dataset_drift_detected": share >= DATASET_DRIFT_SHARE,
        "share_of_drifted_columns": round(share, 4),
        "drifted_features": drifted,
        "feature_details": details,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "stattest": f"ks (alpha={KS_ALPHA}) + psi",
    }


if __name__ == "__main__":
    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)
    result = run_drift_check(latest_market_window(), html_path=str(paths.reports / "drift_report.html"))
    (paths.reports / "drift.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "feature_details"}, indent=2))
