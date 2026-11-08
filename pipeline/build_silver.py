"""Silver layer: clean, gap-filled, feature-engineered table + supervised target.

Stage A (DuckDB SQL): dedup read from bronze, trading-calendar alignment with
    bounded forward-fill, log returns, rolling vol / Parkinson / volume z-score,
    forward-looking target.
Stage B (pandas):     recursive indicators that have no clean SQL form
    (EMA, MACD, Wilder RSI, Wilder ATR).

Target (see docs/decisions.md D-03):
    Y_t = 1  iff  max_{k=1..H} |r_{t+k}|  >  SIGMA_MULT * sigma_{t}
where sigma_t is the *daily* (non-annualised) 20d stdev of log returns known at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

from pipeline import lineage
from pipeline.config import (FEATURE_COLS, MARKET_SYMBOL, HORIZON_DAYS, SIGMA_MULT, TARGET_COL, TRADING_DAYS, get_paths)

CA_GAP_THRESHOLD = 0.35  # |overnight gap| treated as a corporate action
MAX_FFILL_DAYS = 3  # never invent more than 3 consecutive sessions for one symbol

STAGE_A_SQL = f"""
WITH raw AS (
    SELECT symbol, CAST(timestamp AS DATE) AS d, open, high, low, close, volume,
           ln(open / lag(close) OVER (PARTITION BY symbol ORDER BY timestamp)) AS gap
    FROM read_parquet('{{bronze}}/*/*/*/*.parquet', hive_partitioning = true)
),
-- Corporate-action guard: an overnight gap beyond +-{CA_GAP_THRESHOLD:.0%} (demerger / split not
-- back-adjusted by the vendor) is treated as a discontinuity; all EARLIER prices are
-- rescaled by the gap so returns stay continuous. Volume is left untouched.
adj AS (
    SELECT *, coalesce(sum(CASE WHEN abs(gap) > ln(1 + {CA_GAP_THRESHOLD}) THEN gap END)
                 OVER (PARTITION BY symbol ORDER BY d ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING), 0) AS ca_ln
    FROM raw
),
bars AS (
    SELECT symbol, d, open * exp(ca_ln) AS open, high * exp(ca_ln) AS high,
           low * exp(ca_ln) AS low, close * exp(ca_ln) AS close, volume, ca_ln
    FROM adj
),
span AS (SELECT symbol, min(d) AS d0, max(d) AS d1 FROM bars GROUP BY symbol),
-- A date is a trading session if at least half of the symbols alive on it traded.
cal AS (
    SELECT d FROM (
        SELECT b.d,
               count(*) AS traded,
               (SELECT count(*) FROM span s WHERE b.d BETWEEN s.d0 AND s.d1) AS alive
        FROM bars b GROUP BY b.d
    ) WHERE traded >= 0.5 * alive
),
grid AS (  -- every (symbol, session) inside the symbol's own listing span
    SELECT s.symbol, c.d FROM span s JOIN cal c ON c.d BETWEEN s.d0 AND s.d1
),
joined AS (
    SELECT g.symbol, g.d, b.open, b.high, b.low, b.close, b.volume, (b.close IS NULL) AS is_filled
    FROM grid g LEFT JOIN bars b USING (symbol, d)
),
filled AS (
    SELECT symbol, d, is_filled,
           last_value(open  IGNORE NULLS) OVER w AS open,
           last_value(high  IGNORE NULLS) OVER w AS high,
           last_value(low   IGNORE NULLS) OVER w AS low,
           last_value(close IGNORE NULLS) OVER w AS close,
           coalesce(volume, 0)                  AS volume,
           -- consecutive filled sessions so far (to cap the fill length)
           sum(CASE WHEN is_filled THEN 0 ELSE 1 END) OVER w AS grp
    FROM joined
    WINDOW w AS (PARTITION BY symbol ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),
capped AS (
    SELECT *, count(*) OVER (PARTITION BY symbol, grp ORDER BY d) - 1 AS fill_run FROM filled
),
base AS (
    SELECT *, ln(close / lag(close) OVER s) AS log_ret,
           lag(close) OVER s AS prev_close
    FROM capped WHERE fill_run <= {MAX_FFILL_DAYS}
    WINDOW s AS (PARTITION BY symbol ORDER BY d)
),
feat AS (
    SELECT *,
        ln(close / lag(close, 5)  OVER s) AS ret_5d,
        ln(close / lag(close, 20) OVER s) AS ret_20d,
        CASE WHEN count(log_ret) OVER w20 = 20 THEN stddev_samp(log_ret) OVER w20 END AS sigma_20d,
        CASE WHEN count(*) OVER w20 = 20
             THEN sqrt(avg(power(ln(high / low), 2)) OVER w20 / (4 * ln(2))) * sqrt({TRADING_DAYS}) END AS parkinson_vol_20d,
        CASE WHEN count(log_ret) OVER w5  = 5  THEN stddev_samp(log_ret) OVER w5  END AS sigma_5d,
        CASE WHEN count(log_ret) OVER w60 = 60 THEN stddev_samp(log_ret) OVER w60 END AS sigma_60d,
        greatest(high - low, abs(high - prev_close), abs(low - prev_close)) AS true_range,
        CASE WHEN count(*) OVER w20 = 20
             THEN (volume - avg(volume) OVER w20) / nullif(stddev_samp(volume) OVER w20, 0) END AS volume_zscore_20d,
        -- forward window: exactly H later sessions must exist, else target is unknown (NULL)
        CASE WHEN count(log_ret) OVER wf = {HORIZON_DAYS} THEN max(abs(log_ret)) OVER wf END AS fwd_max_abs_ret
    FROM base
    WINDOW s   AS (PARTITION BY symbol ORDER BY d),
           w20 AS (PARTITION BY symbol ORDER BY d ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w5  AS (PARTITION BY symbol ORDER BY d ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
           w60 AS (PARTITION BY symbol ORDER BY d ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
           wf  AS (PARTITION BY symbol ORDER BY d ROWS BETWEEN 1 FOLLOWING AND {HORIZON_DAYS} FOLLOWING)
)
SELECT symbol, CAST(d AS TIMESTAMP) AS timestamp, open, high, low, close, volume, is_filled,
       log_ret AS ret_1d, ret_5d, ret_20d, sigma_20d, sigma_5d, sigma_60d,
       sigma_20d * sqrt({TRADING_DAYS}) AS volatility_20d,
       parkinson_vol_20d, true_range, coalesce(volume_zscore_20d, 0) AS volume_zscore_20d,
       fwd_max_abs_ret
FROM feat ORDER BY symbol, d
"""


def _recursive_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()
    c = g["close"]
    ema9 = c.ewm(span=9, adjust=False, min_periods=9).mean()
    ema21 = c.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    g["ema_ratio"] = ema9 / ema21
    g["ema_trend"] = ema21 / ema50
    g["macd_norm"] = macd / c
    g["macd_hist_norm"] = (macd - signal) / c
    # Wilder smoothing (alpha = 1/14), the textbook RSI/ATR definition
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    denom = avg_gain + avg_loss
    g["rsi_14"] = np.where(denom > 0, 100 * avg_gain / denom.where(denom > 0), 50.0)
    g.loc[avg_gain.isna(), "rsi_14"] = np.nan
    g["abs_ret_z"] = g["ret_1d"].abs() / g["sigma_20d"]
    g["vol_ratio_5_20"] = g["sigma_5d"] / g["sigma_20d"]
    g["vol_ratio_20_60"] = g["sigma_20d"] / g["sigma_60d"]
    g["atr_ratio"] = g["true_range"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    return g


def build_silver() -> pd.DataFrame:
    paths = get_paths()
    paths.silver.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    base = con.execute(STAGE_A_SQL.replace("{bronze}", str(paths.bronze))).df()
    n_ca = con.execute(
        f"SELECT count(*) FROM (SELECT symbol, ln(open/lag(close) OVER (PARTITION BY symbol ORDER BY timestamp)) g "
        f"FROM read_parquet('{paths.bronze}/*/*/*/*.parquet', hive_partitioning=true)) "
        f"WHERE abs(g) > ln(1 + {CA_GAP_THRESHOLD})"
    ).fetchall()[0][0]

    df = pd.concat([_recursive_features(g) for _, g in base.groupby("symbol", sort=True)], ignore_index=True)
    mkt = df.loc[df["symbol"] == MARKET_SYMBOL].set_index("timestamp")["volatility_20d"]
    df["mkt_volatility_20d"] = df["timestamp"].map(mkt)  # NaN if the benchmark did not trade that day
    df[TARGET_COL] = (df["fwd_max_abs_ret"] > SIGMA_MULT * df["sigma_20d"]).astype("Int64")
    df.loc[df["fwd_max_abs_ret"].isna(), TARGET_COL] = pd.NA  # unlabeled tail (last H sessions)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS).drop(columns=["true_range", "sigma_5d", "sigma_60d"])
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    out = paths.silver_file
    con.register("silver_df", df)
    con.execute(f"COPY silver_df TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
    con.close()

    labeled = df[TARGET_COL].dropna()
    rate = float(labeled.astype(int).mean())
    lineage.record("silver", [str(paths.bronze)], [str(out)], len(df), labeled_rows=int(len(labeled)),
                   positive_rate=round(rate, 4), corporate_actions_adjusted=int(n_ca), features=FEATURE_COLS)
    print(f"Corporate-action discontinuities back-adjusted: {n_ca}")
    print(f"Silver: {len(df)} rows ({len(labeled)} labeled), positive rate {rate:.2%} -> {out}")
    return df


if __name__ == "__main__":
    build_silver()
