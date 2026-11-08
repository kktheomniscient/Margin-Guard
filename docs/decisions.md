# MarginGuard — Decision Log

Format: **Context → Decision → Consequence.** Newest decisions are appended at the bottom.
The spec being implemented is [plan.md](plan.md).

---

## D-01 Follow the plan's intent, correct its flaws
**Context.** The plan's code has several correctness problems (listed below as D-03…D-09, D-12).
**Decision.** Keep the architecture, stack, endpoints and directory layout; fix the flaws and record each here.
**Consequence.** The code intentionally differs from the plan's snippets in the places called out below.

## D-02 Ticker universe: `TATAMOTORS.NS` was demerged
**Context.** Yahoo returns 404 for `TATAMOTORS.NS`. The company demerged (Oct 2025) into `TMPV.NS`
(passenger vehicles, full history) and `TMCV.NS` (commercial vehicles, listed 2025-11-12, ~246 bars).
**Decision.** Request order is *symbol → its successors (`config.SUCCESSORS`) → synthetic generator*. Both successors are used.
Synthetic fallback is loud (ERROR log) and recorded in lineage.
**Consequence.** 7 symbols. `TMCV.NS` has only ~197 modelled rows — usable, but thin. Horizon is 2019-01-01 → today (~7.8 years, not the "5 years" in the plan text).
Yahoo `auto_adjust=True` prices are used.

## D-03 Target uses *daily* sigma (plan compared against an annualised one)
**Context.** The plan's formula `|r| > 2.5·σ₂₀` is defined with the annualised σ (×√252 ≈ 15.9) in code, which would make the event nearly impossible.
**Decision.** Compare against the daily 20-day stdev known at *t*; `volatility_20d` (annualised) remains a feature.
The last H sessions are `NULL` (not 0). Forward window requires exactly H later sessions.
**Consequence.** Observed positive rate ≈ **10 %**, above the plan's 5–8 % estimate (equity returns are fat-tailed). The estimate was a guess, not a requirement; class weighting handles it.

## D-04 Wider, stationary feature set (plan: 5 features incl. raw volume)
**Context.** Raw `volume` is non-stationary and scale-dependent per symbol. The target is *relative* (a future move vs *current* σ), so absolute-level features carry little signal: the first 12-feature model scored PR-AUC lift of only ~1.2× on test.
**Decision.** 16 features (see architecture §2). Replace raw volume with a 20d z-score; add regime-relative features (`abs_ret_z`, `vol_ratio_5_20`, `vol_ratio_20_60`) and a market-vol feature from `^NSEI`.
The additions were judged on **validation** PR-AUC, not test. Plan-listed indicators (MACD, Parkinson, volume z-score, multi-horizon returns) are all included.
**Consequence.** Lift rose to ~1.5× (val) / 1.6× (test). The API request has 16 fields instead of 5; the simulator pre-fills them from silver.

## D-05 Wilder smoothing for RSI and ATR
**Context.** The plan used simple rolling means; its own formula section specifies an EMA, and RSI/ATR are conventionally Wilder-smoothed.
**Decision.** `ewm(alpha=1/14, adjust=False, min_periods=14)` for RSI and ATR. RSI = 100·avgGain/(avgGain+avgLoss) (algebraically equal, avoids ÷0; flat = 50).
**Consequence.** Matches textbook values. Warm-up rows (< 60 sessions per symbol) are dropped.

## D-06 Silver is SQL-first, pandas only for recursive indicators
**Context.** The plan advertises DuckDB SQL transformations but implements everything in pandas.
**Decision.** DuckDB window functions do the calendar, ffill, returns, rolling vol/Parkinson/z-score and target; EMA/MACD/RSI/ATR (recursive, no clean SQL form) use pandas `ewm`.
Forward-fill is bounded (≤ 3 sessions) and flagged `is_filled`; calendar = dates where ≥ 50 % of live symbols traded, so one symbol's local holiday doesn't create fake sessions for all.
**Consequence.** 3 filled rows today (index vs equity holiday mismatches). Filled rows keep `ret_1d = 0`; negligible but visible via the flag.

## D-07 Corporate-action guard
**Context.** Yahoo did not back-adjust the TMPV demerger: a −51 % one-day "return" on 2025-10-14 would poison σ₂₀ for 20 sessions and create a fake event. No other |return| in the data exceeds 20 %.
**Decision.** Overnight gap beyond ±35 % ⇒ treat as a discontinuity and rescale all *earlier* OHLC by the gap ratio (volume untouched). Count is written to lineage (`corporate_actions_adjusted`).
**Consequence.** Max |return| after the fix is 19.7 % (COVID crash). A genuine >35 % overnight crash in a large cap would be wrongly smoothed — acceptable for this universe; make the threshold configurable if the universe widens.

## D-08 Gold: date-based split with purge; drift reference = train
**Context.** The plan splits by row position and sets the baseline reference to the *test* set, then compares `test.tail(100)` against it — a subset of itself, which can never show real drift. Also, labels look 3 sessions ahead, so train labels near the cut are computed from validation-period prices.
**Decision.** Cut on unique dates (a date never straddles splits); purge the last H dates of train and val; `baseline_reference.parquet` = train features. Only labeled rows enter gold.
**Consequence.** ~6 sessions lost around cuts. A regression test fails if the purge is removed. Drift is measured against what the model actually learned from.

## D-09 Calibrated probabilities and validation-fitted thresholds
**Context.** `scale_pos_weight` inflates scores (a "62 % risk" would be ~6× the base rate), so the plan's fixed 0.4/0.7 tiers and "probability" label are misleading; the plan also scores F1 at an arbitrary 0.5 cut.
**Decision.** Platt scaling fitted on validation; `elevated` = F1-optimal on validation, `critical` = lowest score with ≥ 50 % validation precision, else 95th percentile. Thresholds are chosen before the test set is scored and stored in `model_meta.json`; the API reads them.
**Consequence.** Scores are interpretable (compare with the 10.7 % base rate; UI shows "× normal"). Tier cut-offs are low in absolute terms (12.5 % / 16.6 %) because discrimination is modest — that is honest, not a bug.

## D-10 Reading the model metrics honestly
**Result.** Out-of-time test: PR-AUC 0.166 vs 0.104 base rate (1.60×), ROC-AUC 0.642, F1 0.243, Brier 0.091 (calibrated).
**Decision.** Report lift over base rate next to PR-AUC (KPI card and logs); do not tune on test. Early-stopping on validation PR-AUC lands on 9 trees at lr 0.03; raising patience 50→150 did not change that, i.e. the signal is small and the model overfits quickly.
**Consequence.** This is a modest but real edge, typical for predicting volatility breakouts from daily bars. The plan's implicit "tail-risk engine" should be treated as a screening/priority signal, not a standalone margin trigger. Test set is ~1.9k rows / ~195 events, so metrics carry sampling noise.

## D-11 Dependency & environment choices
- **Python 3.12 venv** (via `python3.12` or `uv`): the system `python3` is 3.14, for which parts of the ML stack lack support. `run_all.sh` searches 3.10–3.13.
- **`pandas<3`**: Evidently 0.7 imports a pandas internal removed in 3.x.
- **Evidently 0.7 new API** (`Report([...]).run(current, reference)`), not the plan's legacy `evidently.report`/`metric_preset` modules. Evidently provides the KS test; **PSI is implemented in-house** (reference-decile bins) because the plan asks for both and Evidently's preset exposes one test per column.
- **MLflow 3.x**: `log_model(name=...)`; `MLFLOW_DISABLE_AGENT_HINT=1` silences a log hint.
- **LightGBM sklearn API** with `first_metric_only` early stopping on `average_precision`.

## D-12 Serving changes vs the plan
- `@app.on_event("startup")` (deprecated) → lifespan handler.
- `allow_origins=["*"]` + `allow_credentials=True` is both insecure and invalid per the CORS spec → explicit origin list (`MG_CORS_ORIGINS`), GET/POST only.
- Pydantic `Field(example=…)` (v1 style) → `json_schema_extra`; request fields carry range constraints (422 on junk).
- `/drift-status` ran Evidently on every request (≈1 s) → 60 s cache. Added `source=live` (ring buffer of the last 500 scored inputs; 409 until ≥ 30) for the plan's "live drift telemetry".
- Added `/health`, `/model-info` (feeds the KPI banner incl. real p50/p95 latency), `/symbols`, `/latest-features`.
- The API refuses a model whose recorded features differ from `FEATURE_COLS`.
- Measured inference latency ≈ 1–4 ms (budget 50 ms).

## D-13 Dashboard stack
- `create-next-app@latest` currently yields **Next.js 16 / React 19 / Tailwind v4**, and `shadcn init` the **base-nova** style on `@base-ui/react`; the plan's Next 14 / Radix pins were not forced. The folder layout from the plan is kept.
- shadcn's CLI wrote `lib/utils.ts` as `export { cn } from "cn"` using the npm package `cn` (published by shadcn from `shadcn-ui/cn`, a drop-in for clsx + tailwind-merge). It was checked and kept instead of hand-writing the helper; the plan's separate `clsx`/`tailwind-merge` installs are therefore not needed.
- Uses theme tokens (`bg-card`, `text-muted-foreground`) instead of the plan's hard-coded `slate-*` classes, so the UI follows the shadcn theme. `dark` class set on `<html>`.
- Fixed shadcn's generated self-referencing `--font-sans` (it fell back to a serif font).
- Plan hard-coded the "PR-AUC Optimized" / "MLflow Native" tiles; they now show real values from `/model-info`.
- Nested `.git` created by `create-next-app` was removed (repo isn't under git).

## D-14 Bronze partitioning produces many small files
`year/month/symbol` (as specified) over 7 symbols × ~90 months ≈ 580 tiny Parquet files. Fine at this scale and good for demonstrating partition pruning, but wrong for production volumes — partition by `year` (or `symbol`) and compact if data grows.

## D-15 Orchestrator hardening (`run_all.sh`)
- Plan's `trap "kill $PIDS" EXIT` was installed after services started and `kill 0` is unsafe when the script isn't a process-group leader. Now PIDs are tracked and each process **tree** is killed.
- Ports are configurable and **pre-checked** (this machine already ran other apps on :3000/:8000; macOS AirPlay occupies :5000). The script refuses to start rather than silently talking to another service.
- Flags `--skip-pipeline` / `--pipeline-only`; CORS origin and API URL are derived from the chosen ports.
- Entry points are run as modules (`python -m pipeline.fetch_data`) so package imports work from the repo root.

## D-16 Synthetic generator semantics
The plan wants 1M+ synthetic transactions. `scripts/generate_synthetic_trades.py` produces a tick-level trade log (GBM + AR(1) log-vol clustering + Poisson jumps, overnight gap + intraday path) and aggregates it to daily bars with DuckDB. As a fetch fallback it generates ~60k trades per symbol; as a CLI it writes the full 1M+ log. Seeds are deterministic (`sum(ord)`; Python's `hash()` is salted per process).
**Consequence.** Synthetic data can never silently masquerade as real: lineage records `source` per symbol.

## D-17 Everything path-related is lazy and env-overridable
`get_paths()` reads `MG_*` env vars at call time. This lets the test-suite build and serve a complete lakehouse in a temp dir without monkey-patching module globals.

---

## Known limitations / follow-ups
- Modest predictive edge (D-10); only daily bars, no intraday or order-book data.
- Drift KS test is hypersensitive because the 7 symbols are cross-correlated (effective sample ≪ 420 rows); PSI is shown alongside. The current "94 % drifted" reflects a genuine regime change (calm, trending 2026 market vs 2019–24 training), but a production system should add a retraining trigger and a per-symbol or block-bootstrapped test.
- `TMCV.NS` has < 1 year of history.
- Live drift is in-memory (lost on API restart) and the latency window is per-process.
- No auth on the API (local-only by design; CORS restricted).
- Dashboard has no automated tests; it was verified by build, lint and a browser run.
