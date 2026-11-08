"""Gold layer: leakage-safe temporal train / validation / out-of-time test splits.

* Splits are cut on *calendar dates* (not row positions) so one date never straddles two splits.
* Labels look H sessions into the future, so the last H dates of train and val are
  purged (embargo). Without this, train labels would be computed from validation-period prices.
* Only labeled rows are modelled; the unlabeled tail stays in silver (used for live drift checks).
* baseline_reference.parquet = TRAIN features: the distribution the model learned from, which is
  what production data must be compared against.
"""
from __future__ import annotations

import pandas as pd

from pipeline import lineage
from pipeline.config import (FEATURE_COLS, HORIZON_DAYS, TARGET_COL, TRAIN_FRAC, VAL_FRAC, get_paths)


def split_by_date(df: pd.DataFrame, purge: int = HORIZON_DAYS) -> dict[str, pd.DataFrame]:
    dates = pd.Series(sorted(df["timestamp"].unique()))
    n = len(dates)
    t_end, v_end = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    train_dates = dates[: max(t_end - purge, 0)]
    val_dates = dates[t_end : max(v_end - purge, t_end)]
    test_dates = dates[v_end:]
    return {
        "train": df[df["timestamp"].isin(train_dates)],
        "val": df[df["timestamp"].isin(val_dates)],
        "test": df[df["timestamp"].isin(test_dates)],
    }


def build_gold() -> dict[str, int]:
    paths = get_paths()
    paths.gold.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(paths.silver_file)
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    splits = split_by_date(df)
    sizes, outputs = {}, []
    for name, part in splits.items():
        out = paths.gold / f"{name}.parquet"
        part.to_parquet(out, index=False, compression="snappy")
        sizes[name] = len(part)
        outputs.append(str(out))
        span = f"{part['timestamp'].min():%Y-%m-%d} -> {part['timestamp'].max():%Y-%m-%d}"
        print(f"  {name:<5} {len(part):>6} rows  pos={part[TARGET_COL].mean():.2%}  {span}")
    ref = paths.gold / "baseline_reference.parquet"
    splits["train"][["symbol", "timestamp", *FEATURE_COLS]].to_parquet(ref, index=False, compression="snappy")
    outputs.append(str(ref))
    lineage.record("gold", [str(paths.silver_file)], outputs, sum(sizes.values()), splits=sizes,
                   purge_days=HORIZON_DAYS)
    print("Gold layer created with purged, leak-free temporal partitions.")
    return sizes


if __name__ == "__main__":
    build_gold()
