"""Tiny data-lineage ledger: every pipeline stage appends what it read/wrote."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from pipeline.config import get_paths


def record(stage: str, inputs: list[str], outputs: list[str], rows: int, **extra: Any) -> None:
    p = get_paths()
    p.data.mkdir(parents=True, exist_ok=True)
    ledger: list[dict] = []
    if p.lineage.exists():
        try:
            ledger = json.loads(p.lineage.read_text())
        except json.JSONDecodeError:
            ledger = []
    ledger.append(
        {
            "stage": stage,
            "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "inputs": inputs,
            "outputs": outputs,
            "rows": rows,
            **extra,
        }
    )
    p.lineage.write_text(json.dumps(ledger[-200:], indent=2, default=str))


def reset() -> None:
    p = get_paths()
    if p.lineage.exists():
        p.lineage.unlink()
