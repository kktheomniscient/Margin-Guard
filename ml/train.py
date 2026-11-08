"""LightGBM tail-risk classifier with SQLite-backed MLflow tracking."""
from __future__ import annotations

import datetime as dt
import json

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from ml.calibration import apply_platt, fit_platt
from ml.evaluate import compute_metrics, pick_thresholds, shap_importance
from pipeline.config import FEATURE_COLS, HORIZON_DAYS, SIGMA_MULT, TARGET_COL, get_paths

EXPERIMENT = "MarginGuard_Market_Risk_Engine"


def train_model() -> dict:
    paths = get_paths()
    mlflow.set_tracking_uri(paths.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=f"file://{paths.artifact_root}")
    mlflow.set_experiment(EXPERIMENT)

    train, val, test = (pd.read_parquet(paths.gold / f"{n}.parquet") for n in ("train", "val", "test"))
    X_tr, y_tr = train[FEATURE_COLS], train[TARGET_COL]
    X_va, y_va = val[FEATURE_COLS], val[TARGET_COL]
    X_te, y_te = test[FEATURE_COLS], test[TARGET_COL]

    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    params = {
        "objective": "binary",
        "metric": ["average_precision", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 600,          # upper bound; early stopping picks the real count
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "scale_pos_weight": neg / max(pos, 1),
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lgbm-volatility-native") as run:
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
        raw_val = model.predict_proba(X_va)[:, 1]
        platt = fit_platt(raw_val, y_va.to_numpy())          # calibrate on validation only
        p_val = apply_platt(raw_val, platt)
        p_te = apply_platt(model.predict_proba(X_te)[:, 1], platt)
        thresholds = pick_thresholds(y_va, p_val)
        val_m = compute_metrics(y_va, p_val, thresholds["elevated"])
        test_m = compute_metrics(y_te, p_te, thresholds["elevated"])  # threshold fixed BEFORE seeing test
        shap_imp = shap_importance(model.booster_, X_te)

        mlflow.log_params({**{k: v for k, v in params.items() if k != "metric"},
                           "features": ",".join(FEATURE_COLS), "horizon_days": HORIZON_DAYS,
                           "sigma_mult": SIGMA_MULT, "best_iteration": model.best_iteration_})
        mlflow.log_metrics({**{f"val_{k}": v for k, v in val_m.items()},
                            **{f"test_{k}": v for k, v in test_m.items()},
                            # headline names kept as in the plan:
                            "pr_auc": test_m["pr_auc"], "roc_auc": test_m["roc_auc"], "f1_score": test_m["f1"]})
        mlflow.log_params({"platt_a": platt["a"], "platt_b": platt["b"]})
        mlflow.log_dict(shap_imp, "shap_importance.json")
        mlflow.lightgbm.log_model(model, name="model")

        paths.gold.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(paths.model_file))
        meta = {
            "run_id": run.info.run_id,
            "experiment": EXPERIMENT,
            "model_name": "LightGBM (binary, scale_pos_weight)",
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "features": FEATURE_COLS,
            "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            "thresholds": thresholds,   # in calibrated-probability space
            "platt": platt,
            "metrics": {"validation": val_m, "test": test_m},
            "shap_importance": shap_imp,
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
        }
        paths.model_meta.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(paths.model_meta))

    print(f"Training complete (run {run.info.run_id[:8]}, {meta['best_iteration']} trees)")
    print(f"  thresholds: elevated={thresholds['elevated']:.3f} critical={thresholds['critical']:.3f}")
    print(f"  TEST  PR-AUC {test_m['pr_auc']:.4f} (prevalence {test_m['prevalence']:.3f}, "
          f"lift {test_m['pr_auc_lift']:.2f}x) | ROC-AUC {test_m['roc_auc']:.4f} | F1 {test_m['f1']:.4f}")
    return meta


if __name__ == "__main__":
    train_model()
