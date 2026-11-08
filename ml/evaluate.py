"""Evaluation helpers: ranking metrics, operating thresholds, SHAP importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             roc_auc_score)


def compute_metrics(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    prevalence = float(np.mean(y_true))
    pr_auc = float(average_precision_score(y_true, prob))
    return {
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence if prevalence else float("nan"),  # vs. random = prevalence
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),  # computed on calibrated probabilities
        "prevalence": prevalence,
        "alert_rate": float(pred.mean()),
    }


def pick_thresholds(y_val: pd.Series, prob_val: np.ndarray, min_critical_precision: float = 0.5) -> dict[str, float]:
    """Operating points chosen on VALIDATION only.

    elevated = score maximising F1; critical = lowest score whose precision >= 50%
    (falls back to the 95th percentile of scores if that precision is unreachable).
    """
    prec, rec, thr = precision_recall_curve(y_val, prob_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    elevated = float(thr[int(np.argmax(f1))])
    ok = np.where(prec[:-1] >= min_critical_precision)[0]
    critical = float(thr[ok[0]]) if len(ok) else float(np.quantile(prob_val, 0.95))
    return {"elevated": elevated, "critical": max(critical, elevated)}


def shap_importance(booster, X: pd.DataFrame, max_rows: int = 1000) -> dict[str, float]:
    """Mean |SHAP| per feature (TreeExplainer, last-iteration booster)."""
    import shap

    sample = X.tail(max_rows)
    values = shap.TreeExplainer(booster).shap_values(sample)
    if isinstance(values, list):  # older shap returns [neg, pos]
        values = values[-1]
    imp = np.abs(np.asarray(values)).mean(axis=0)
    return dict(sorted(zip(X.columns, map(float, imp)), key=lambda kv: -kv[1]))
