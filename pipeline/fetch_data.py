"""Download daily bars with yfinance (retries + successor resolution + synthetic fallback)."""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
import yfinance as yf

from pipeline import lineage
from pipeline.config import BRONZE_COLS, START_DATE, SUCCESSORS, TICKERS, clean_symbol, get_paths

log = logging.getLogger("fetch")


def _download(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            return df  # empty = unknown/delisted symbol; retrying will not help
        except Exception as exc:  # network hiccup / rate limit
            log.warning("%s attempt %d/%d failed: %s", ticker, attempt, retries, exc)
            time.sleep(2**attempt)
    return pd.DataFrame()


def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    df = df.rename(columns={"Date": "timestamp", "date": "timestamp", "index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["symbol"] = ticker
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["ingested_at"] = pd.Timestamp.now("UTC").tz_localize(None)
    return df[BRONZE_COLS]


def fetch_symbol(requested: str, start: str, end: str) -> tuple[list[pd.DataFrame], str]:
    """Return (frames, source). Tries the symbol, then its successors, then synthetic."""
    df = _download(requested, start, end)
    if not df.empty:
        return [_normalise(df, requested)], "yfinance"
    frames = []
    for succ in SUCCESSORS.get(requested, []):
        d = _download(succ, start, end)
        if d.empty:
            log.warning("successor %s of %s returned no data", succ, requested)
            continue
        log.warning("%s has no data; using successor %s (%d rows)", requested, succ, len(d))
        frames.append(_normalise(d, succ))
    if frames:
        return frames, "yfinance-successor"
    from scripts.generate_synthetic_trades import synthetic_daily_bars

    log.error("%s: no yfinance data and no successors - FALLING BACK TO SYNTHETIC DATA", requested)
    return [synthetic_daily_bars(requested, start, end)[BRONZE_COLS]], "synthetic"


def fetch_all(tickers: list[str] | None = None, start: str = START_DATE, end: str | None = None) -> list[str]:
    paths = get_paths()
    paths.raw_cache.mkdir(parents=True, exist_ok=True)
    for old in paths.raw_cache.glob("*.parquet"):  # raw cache mirrors the current universe exactly
        old.unlink()
    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    written, total, sources = [], 0, {}
    for req in tickers or TICKERS:
        frames, source = fetch_symbol(req, start, end)
        for f in frames:
            sym = f["symbol"].iloc[0]
            out = paths.raw_cache / f"{clean_symbol(sym)}.parquet"
            f.to_parquet(out, index=False, compression="snappy")
            written.append(str(out))
            total += len(f)
            sources[sym] = {"source": source, "rows": len(f)}
            print(f"Cached {len(f):>5} rows for {sym:<14} [{source}] -> {out.name}")
    lineage.reset()  # a fresh fetch starts a new lineage chain
    lineage.record("fetch", ["yfinance"], written, total, sources=sources)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_all()
