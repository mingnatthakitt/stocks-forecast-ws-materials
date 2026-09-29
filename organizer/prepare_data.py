#!/usr/bin/env python3
"""
prepare_data.py — Build the workshop datasets from real market data.

Instruments: SPY (S&P 500 ETF) plus four popular mega-caps — NVDA, AAPL, MSFT, TSLA —
so teams model a small panel rather than a single series.

Run this ONCE before the workshop (organisers only — participants never see
the private files):

    python3 prepare_data.py
    python3 prepare_data.py --start 2014-01-01 --end 2026-12-31   # custom window

Outputs (long format: one row per Date × Ticker, sorted by Date then Ticker)
-------
participant/data/workshop_participant.csv     2014-01-01 .. 2024-12-31 (distributed)
organizer/private/workshop_2025_full.csv      2025 incl. private labels (never distributed)
organizer/private/workshop_2026_holdout.csv   2026 YTD holdout (hidden until the end)

Columns: Date, Ticker, Open, High, Low, Close, Volume (+ next_day_return in the
private files, NaN-free — trailing rows without a next day are dropped).

IMPORTANT — adjusted prices: downloads use auto_adjust=True. NVDA (10:1 in 2024),
AAPL (4:1 in 2020) and TSLA (5:1 and 3:1) all split during the window; raw prices
would inject fake ±40–90% "returns" at split dates and poison every model. Adjusted
OHLCV removes splits and dividends, which is the standard input for return modelling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTICIPANT_DIR = PROJECT_ROOT / "participant" / "data"
PRIVATE_DIR = Path(__file__).resolve().parent / "private"

TICKERS = ["SPY", "NVDA", "AAPL", "MSFT", "TSLA"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

TRAIN_START = "2014-01-01"
TRAIN_VAL_END = "2024-12-31"   # participant window: 2014-2024 (train 2014-22, val 23-24)
COMP_END = "2025-12-31"        # private competition year: 2025
HOLDOUT_END = "2026-12-31"     # optional hidden holdout: 2026


def download_panel(start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for every ticker; return a clean long-format frame."""
    print(f"Downloading {', '.join(TICKERS)} {start} .. {end} from Yahoo Finance ...")
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        sys.exit("ERROR: yfinance returned no data. Check your internet connection.")

    # Fail loudly (with a readable message) if any ticker is missing entirely.
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0)) | set(raw.columns.get_level_values(1))
    else:
        available = set(TICKERS[:1])
    missing = [t for t in TICKERS if t not in available]
    if missing:
        sys.exit(f"ERROR: Yahoo returned no data for {missing}. Re-run, or remove the "
                 f"ticker(s) from TICKERS — the workshop expects all {len(TICKERS)} instruments.")

    frames = []
    for ticker in TICKERS:
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance flips column-level order across versions — detect it.
            if set(TICKERS) <= set(raw.columns.get_level_values(0)):
                df = raw[ticker].copy()
            else:
                df = raw.xs(ticker, axis=1, level=1).copy()
        else:
            df = raw.copy()
        df = df[OHLCV].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["Ticker"] = ticker
        frames.append(df)
    panel = pd.concat(frames).sort_index(kind="stable")            # by Date, then Ticker
    panel.index.name = "Date"

    # Data hygiene: force sane types, round, and run sanity checks per ticker.
    panel["Volume"] = panel["Volume"].astype("int64")
    for col in ["Open", "High", "Low", "Close"]:
        panel[col] = panel[col].round(2)

    assert (panel["Low"] <= panel["High"]).all(), "Corrupt data: Low > High somewhere."
    assert (panel[["Open", "High", "Low", "Close"]] > 0).all().all(), "Corrupt data: non-positive prices."
    assert not panel.reset_index().duplicated(["Date", "Ticker"]).any(), \
        "Duplicate (Date, Ticker) rows found."
    for ticker, g in panel.groupby("Ticker"):
        assert g.index.is_monotonic_increasing, f"{ticker}: dates not sorted."
        ret = g["Close"].pct_change().abs().max()
        assert ret < 0.5, f"{ticker}: max daily |return| {ret:.1%} — split not adjusted?!"
    return panel


def add_next_day_return(panel: pd.DataFrame) -> pd.DataFrame:
    """next_day_return computed from Close *within each ticker* (never across)."""
    out = panel.copy()
    out["next_day_return"] = (
        out.groupby("Ticker")["Close"].transform(lambda s: s.shift(-1) / s - 1.0)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build workshop datasets (multi-ticker panel).")
    ap.add_argument("--start", default=TRAIN_START, help="global start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=HOLDOUT_END, help="global end date (YYYY-MM-DD)")
    ap.add_argument("--skip-holdout", action="store_true",
                    help="do not build the 2026 holdout (e.g. if running before 2026)")
    args = ap.parse_args()

    PARTICIPANT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_panel(args.start, args.end)
    full = add_next_day_return(raw)
    print(f"Downloaded {len(raw)} rows across {raw['Ticker'].nunique()} tickers: "
          f"{raw.index[0].date()} .. {raw.index[-1].date()}")

    # ---- 1. Participant file: 2014-2024, raw OHLCV only (no target column) ----
    participant = raw.loc[TRAIN_START:TRAIN_VAL_END].copy()
    p_path = PARTICIPANT_DIR / "workshop_participant.csv"
    participant.to_csv(p_path, index_label="Date")
    print(f"\n[participant] {p_path}  ({len(participant)} rows, "
          f"{participant.index[0].date()} .. {participant.index[-1].date()}, OHLCV only)")

    # ---- 2. Private 2025 competition file: features + hidden labels ----
    comp = full.loc["2025-01-01":COMP_END].copy()
    n_before = len(comp)
    comp = comp[comp["next_day_return"].notna()]
    c_path = PRIVATE_DIR / "workshop_2025_full.csv"
    comp.to_csv(c_path, index_label="Date")
    print(f"[private   ] {c_path}  ({len(comp)} rows, {n_before - len(comp)} trailing dropped, "
          f"includes labels)")

    # ---- 3. Optional hidden 2026 holdout ----
    hold = full.loc["2026-01-01":args.end].copy()
    hold = hold[hold["next_day_return"].notna()]
    if args.skip_holdout or hold.empty:
        print("[holdout   ] skipped (no 2026 data yet or --skip-holdout)")
    else:
        h_path = PRIVATE_DIR / "workshop_2026_holdout.csv"
        hold.to_csv(h_path, index_label="Date")
        print(f"[private   ] {h_path}  ({len(hold)} rows, "
              f"{hold.index[0].date()} .. {hold.index[-1].date()})")

    # ---- Per-ticker stats the organisers can quote in the debrief ----
    print("\nPer-ticker sanity summary (all downloaded data):")
    print(f"{'ticker':<7}{'days':>6}{'mean ret':>10}{'std':>9}{'up-share':>10}{'zero-MAE':>10}")
    for ticker, g in full.groupby("Ticker"):
        rets = g["next_day_return"].dropna()
        print(f"{ticker:<7}{len(rets):>6}{rets.mean():>+10.5f}{rets.std():>9.5f}"
              f"{(rets > 0).mean():>10.3f}{rets.abs().mean():>10.5f}")
    pooled = full["next_day_return"].dropna()
    print(f"{'POOLED':<7}{len(pooled):>6}{pooled.mean():>+10.5f}{pooled.std():>9.5f}"
          f"{(pooled > 0).mean():>10.3f}{pooled.abs().mean():>10.5f}")
    print("\nDone. Remember: files in organizer/private/ must NEVER be shared with participants.")


if __name__ == "__main__":
    main()
