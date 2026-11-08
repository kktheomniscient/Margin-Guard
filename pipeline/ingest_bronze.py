"""Bronze layer: schema-enforced, validated, Snappy-Parquet partitioned by year/month/symbol."""
from __future__ import annotations

import shutil

import duckdb

from pipeline import lineage
from pipeline.config import get_paths

# Row-level quality rules (bronze contract). A row failing ANY rule is quarantined.
VALID_ROW = """
    timestamp IS NOT NULL AND symbol IS NOT NULL
    AND open > 0 AND high > 0 AND low > 0 AND close > 0
    AND high >= greatest(open, low, close)
    AND low  <= least(open, high, close)
    AND volume >= 0
"""


def write_bronze_layer() -> dict:
    paths = get_paths()
    if not any(paths.raw_cache.glob("*.parquet")):
        raise FileNotFoundError(f"No raw files in {paths.raw_cache}; run pipeline.fetch_data first")
    if paths.bronze.exists():  # idempotent: rebuild the layer from scratch
        shutil.rmtree(paths.bronze)
    paths.bronze.mkdir(parents=True)
    src = f"read_parquet('{paths.raw_cache}/*.parquet')"

    con = duckdb.connect(":memory:")
    # Deduplicate on (symbol, day): keep the most recently ingested record.
    con.execute(
        f"""
        CREATE TEMP VIEW typed AS
        SELECT CAST(timestamp AS TIMESTAMP)  AS timestamp,
               CAST(symbol AS VARCHAR)       AS symbol,
               CAST(open AS DOUBLE)          AS open,
               CAST(high AS DOUBLE)          AS high,
               CAST(low AS DOUBLE)           AS low,
               CAST(close AS DOUBLE)         AS close,
               CAST(volume AS BIGINT)        AS volume,
               CAST(ingested_at AS TIMESTAMP) AS ingested_at
        FROM {src}
        """
    )
    total = con.execute("SELECT count(*) FROM typed").fetchone()[0]
    con.execute(
        f"""
        CREATE TEMP TABLE clean AS
        SELECT * EXCLUDE rn FROM (
            SELECT *, row_number() OVER (PARTITION BY symbol, CAST(timestamp AS DATE)
                                         ORDER BY ingested_at DESC) AS rn
            FROM typed WHERE {VALID_ROW}
        ) WHERE rn = 1
        """
    )
    kept = con.execute("SELECT count(*) FROM clean").fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT *, YEAR(timestamp) AS year,
                   LPAD(CAST(MONTH(timestamp) AS VARCHAR), 2, '0') AS month
            FROM clean ORDER BY symbol, timestamp
        ) TO '{paths.bronze}' (FORMAT PARQUET, COMPRESSION SNAPPY,
                               PARTITION_BY (year, month, symbol), OVERWRITE_OR_IGNORE 1)
        """
    )
    n_files = len(list(paths.bronze.rglob("*.parquet")))
    con.close()
    stats = {"rows_in": total, "rows_out": kept, "rows_rejected": total - kept, "files": n_files}
    lineage.record("bronze", [str(paths.raw_cache)], [str(paths.bronze)], kept,
                   rows_rejected=total - kept, partitions="year/month/symbol", compression="snappy")
    print(f"Bronze: {kept}/{total} rows kept ({total - kept} rejected) in {n_files} Parquet files "
          f"-> {paths.bronze}")
    return stats


if __name__ == "__main__":
    write_bronze_layer()
