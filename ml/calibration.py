"""Platt scaling: maps the (scale_pos_weight-inflated) raw LightGBM score to a calibrated probability.

Fitted on the VALIDATION split only. Monotonic, so ranking metrics (PR/ROC-AUC) are unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(raw_prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    lr = LogisticRegression(C=1e6).fit(_logit(raw_prob).reshape(-1, 1), y)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def apply_platt(raw_prob, platt: dict[str, float]) -> np.ndarray:
    z = platt["a"] * _logit(raw_prob) + platt["b"]
    return 1.0 / (1.0 + np.exp(-z))
