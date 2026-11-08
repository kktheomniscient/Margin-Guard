"""Synthetic trade-log generator (offline fallback).

Price model per symbol: Geometric Brownian Motion with
  * stochastic, mean-reverting log-volatility (volatility clustering, so the
    tail-risk target is genuinely learnable),
  * Poisson-arriving log-normal jumps (crash / gap events),
  * an overnight gap + an intraday random-walk path sampled at trade times.

Produces a tick-level trade log (timestamp, symbol, price, qty) which is then
aggregated to daily OHLCV bars with DuckDB - the same shape yfinance returns.

    python -m scripts.generate_synthetic_trades --trades 1000000 --out data/synthetic_trades.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def generate_trades(
    symbols: list[str],
    start: str = "2019-01-01",
    end: str | None = None,
    total_trades: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    days = pd.bdate_range(start, end)
    n_days = len(days)
    per_day = max(5, total_trades // (len(symbols) * n_days))
    rng = np.random.default_rng(seed)
    frames = []
    for s_idx, sym in enumerate(symbols):
        mu = 0.12 / 252
        # AR(1) log-vol around ~1.4% daily
        log_vol = np.empty(n_days)
        log_vol[0] = np.log(0.014)
        shocks = rng.normal(0, 0.12, n_days)
        for i in range(1, n_days):
            log_vol[i] = log_vol[i - 1] + 0.08 * (np.log(0.014) - log_vol[i - 1]) + shocks[i]
        sigma = np.exp(log_vol)
        jumps = rng.poisson(0.015, n_days) * rng.normal(-0.01, 0.045, n_days)
        gap_frac = 0.3  # share of daily variance realised overnight
        gap = rng.normal(0, sigma * np.sqrt(gap_frac)) + jumps
        intraday_sd = sigma * np.sqrt(1 - gap_frac) / np.sqrt(per_day)
        steps = rng.normal(0, 1, (n_days, per_day)) * intraday_sd[:, None]
        steps[:, 0] += mu
        day_open_logp = np.log(rng.uniform(300, 3000)) + np.cumsum(
            np.concatenate([[0.0], (steps.sum(axis=1) + gap)[:-1]])
        ) + gap
        intraday_logp = day_open_logp[:, None] + np.cumsum(steps, axis=1)
        price = np.exp(intraday_logp)
        seconds = np.sort(rng.integers(0, 6 * 3600 + 1800, (n_days, per_day)), axis=1)
        ts = (
            days.values.astype("datetime64[s]")[:, None]
            + np.timedelta64(3 * 3600 + 45 * 60, "s")  # 09:15 IST in UTC
            + seconds.astype("timedelta64[s]")
        )
        qty = np.maximum(1, rng.lognormal(4.0, 1.0, (n_days, per_day)).astype(np.int64))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts.ravel(),
                    "symbol": sym,
                    "price": np.round(price.ravel(), 2),
                    "qty": qty.ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def trades_to_daily_bars(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a trade log into daily OHLCV bars (DuckDB SQL)."""
    con = duckdb.connect()
    con.register("trades", trades)
    df = con.execute(
        """
        SELECT date_trunc('day', timestamp)                    AS timestamp,
               symbol,
               arg_min(price, timestamp)                       AS open,
               max(price)                                      AS high,
               min(price)                                      AS low,
               arg_max(price, timestamp)                       AS close,
               sum(qty)::BIGINT                                AS volume
        FROM trades GROUP BY 1, 2 ORDER BY symbol, timestamp
        """
    ).df()
    con.close()
    df["ingested_at"] = pd.Timestamp.now("UTC").tz_localize(None)
    return df


def synthetic_daily_bars(symbol: str, start: str, end: str | None = None, trades: int = 60_000) -> pd.DataFrame:
    seed = sum(map(ord, symbol))  # deterministic across runs (unlike hash())
    return trades_to_daily_bars(generate_trades([symbol], start, end, trades, seed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/synthetic_trades.parquet")
    ap.add_argument("--symbols", nargs="+", default=["SYN1", "SYN2", "SYN3", "SYN4", "SYN5"])
    a = ap.parse_args()
    t = generate_trades(a.symbols, total_trades=a.trades)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(a.out, index=False, compression="snappy")
    print(f"Wrote {len(t):,} synthetic trades -> {a.out}")
