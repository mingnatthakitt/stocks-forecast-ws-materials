#!/usr/bin/env python3
"""
paper_trading.py — ORGANISER TOOL. Historical paper-trading simulation.

    python3 paper_trading.py --results results/round1.json --out results/round1_paper.json
    options: --capital 10000  --threshold 0.0  --cost-bps 0

Strategy (intentionally simple — the teaching point is accuracy != profit):
    predicted return > threshold  ->  LONG  (earn that day's actual SPY return)
    otherwise                     ->  CASH (earn 0)
Position changes long<->cash can be charged a transaction cost (--cost-bps 5
means 0.05% of the portfolio each switch). A buy-and-hold benchmark is included.

Reads the predictions stored in the round JSON written by evaluate.py; writes a
small JSON that leaderboard.py renders. Entirely historical, entirely virtual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ORGANIZER_DIR = Path(__file__).resolve().parent
PRIVATE_DIR = ORGANIZER_DIR / "private"
RESULTS_DIR = ORGANIZER_DIR / "results"


def _resolve(value: str) -> Path:
    """Resolve a results path, trying the places the user plausibly meant.

    A relative path is first taken as-is against the current working directory
    (so the documented `--results results/round1.json` works from the repo root
    exactly as written), then re-anchored on the script's own directory, then on
    organizer/results/ for a bare filename. Without this the documented command
    raised FileNotFoundError from the repo root, and the bare auto-detect below
    reported "no results JSON" even though organizer/results/round1.json exists.
    """
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    for cand in (p, ORGANIZER_DIR / p, RESULTS_DIR / p):
        if cand.exists():
            return cand
    # Nothing exists yet. Outputs land next to their inputs; bare names mean
    # organizer/results/, which is where every documented artefact lives.
    if len(p.parts) == 1:
        return RESULTS_DIR / p
    return p


def load_labels(year: str) -> pd.DataFrame:
    path = PRIVATE_DIR / ("workshop_2026_holdout.csv" if year == "2026"
                          else "workshop_2025_full.csv")
    private = pd.read_csv(path, index_col="Date", parse_dates=True)
    return private[["Ticker", "next_day_return"]].dropna()


def simulate(preds: dict[str, float], labels: pd.DataFrame, capital: float,
             threshold: float, cost_rate: float) -> dict:
    """Panel simulation: each ticker trades its own sub-portfolio with an equal
    share of the capital; the reported value is the SUM of the sub-portfolios."""
    tickers = sorted(labels["Ticker"].unique())
    share = capital / len(tickers)
    total = 0.0
    n_trades = 0
    up_days = down_days = 0
    curve = []
    for ticker in tickers:
        sub = labels[labels["Ticker"] == ticker]
        equity = share
        position = 0.0
        for date, row in sub.iterrows():
            target = 1.0 if preds.get(f"{date.date()}|{ticker}", 0.0) > threshold else 0.0
            if target != position:
                equity *= 1.0 - cost_rate * abs(target - position)
                n_trades += 1
                position = target
            ret = float(row["next_day_return"])
            equity *= 1.0 + position * ret
            curve.append((f"{date.date()}|{ticker}", round(equity, 2)))
            up_days += position > 0 and ret > 0
            down_days += position > 0 and ret < 0
        total += equity
    return {
        "final_value": round(total, 2),
        "return_pct": round((total / capital - 1) * 100, 2),
        "n_trades": n_trades,
        "days_in_market_up": up_days,
        "days_in_market_down": down_days,
        "equity_curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-trade stored predictions (organiser only).")
    ap.add_argument("--results", default=None, nargs="?", const="round1.json",
                    help="results/roundN.json from evaluate.py (default: "
                         "round1.json, else round2.json)")
    ap.add_argument("--out", default=None, help="output json (default: <results>_paper.json)")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="go LONG only when predicted return exceeds this")
    ap.add_argument("--cost-bps", type=float, default=0.0,
                    help="transaction cost per position switch, basis points (e.g. 5 = 0.05%%)")
    args = ap.parse_args()
    if not args.results:
        for cand in ("round1.json", "round2.json"):
            if (RESULTS_DIR / cand).exists():
                args.results = cand
                break
        else:
            sys.exit(f"No results JSON found in {RESULTS_DIR} — run evaluate.py "
                     f"first, or pass --results.")

    results_path = _resolve(args.results)
    if not results_path.exists():
        sys.exit(f"No results JSON at {results_path} — run evaluate.py first, "
                 f"or pass --results.")
    results = json.loads(results_path.read_text())
    labels = load_labels(results["eval_year"])
    cost_rate = args.cost_bps / 10_000.0

    out = {"eval_year": results["eval_year"], "capital": args.capital,
           "threshold": args.threshold, "cost_bps": args.cost_bps, "teams": {}}

    ok_teams = [t for t in results["teams"] if t.get("status") == "ok"]
    preds_by_team = {t["team"]: t["predictions"] for t in ok_teams}  # {"date|ticker": pred}

    # Buy-and-hold benchmark: all-in every day, on every ticker.
    bh_preds = {f"{d.date()}|{r['Ticker']}": 1.0 for d, r in labels.iterrows()}
    bh = simulate(bh_preds, labels, args.capital, threshold=-1.0, cost_rate=cost_rate)
    out["teams"]["BUY & HOLD"] = bh
    print(f"BUY & HOLD: ${args.capital:,.0f} -> ${bh['final_value']:,.0f} "
          f"({bh['return_pct']:+.2f}%)")

    for team, preds in preds_by_team.items():
        sim = simulate(preds, labels, args.capital, args.threshold, cost_rate)
        out["teams"][team] = sim
        print(f"{team:<28} ${args.capital:,.0f} -> ${sim['final_value']:,.0f} "
              f"({sim['return_pct']:+.2f}%)  trades={sim['n_trades']}")

    # A non-zero cost run gets its own filename. Writing the 10 bps demo over
    # round<N>_paper.json used to silently replace the zero-cost headline result
    # the leaderboard renders — the number you quote on the projector.
    if args.out:
        out_path = _resolve(args.out)
    else:
        suffix = f"_paper_cost{args.cost_bps:g}" if args.cost_bps else "_paper"
        out_path = results_path.with_name(results_path.stem + suffix + ".json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
