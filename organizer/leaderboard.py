#!/usr/bin/env python3
"""
leaderboard.py — ORGANISER TOOL. Renders the projector leaderboard.

    python3 leaderboard.py --round1 results/round1.json
    python3 leaderboard.py --round1 results/round1.json --round2 results/round2.json \
        --paper1 results/round1_paper.json --paper2 results/round2_paper.json \
        --holdout results/holdout.json --out results/leaderboard.html

Produces a single self-contained leaderboard.html (dark trading-terminal look).
Open it in any browser on the projector and press F5 / Cmd-R after each
evaluation — no server, no internet needed.

Tables rendered (whichever inputs were provided):
  * Round 1  — Human-only models ranked by MAE
  * Round 2  — Human + AI models ranked by MAE, with the Human-vs-AI Δ column
  * Paper trading — virtual $10,000 portfolios for both rounds + buy & hold
  * 2026 holdout — the surprise generalisation test (2025 vs 2026 MAE per round)
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ORGANIZER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ORGANIZER_DIR / "results"


def _resolve(value: str) -> Path:
    """Resolve a results path, trying the places the user plausibly meant.

    A relative path is first taken as-is against the current working directory
    (so `--results results/round1.json` works from the repo root exactly as
    documented), then re-anchored on the script's own directory, then on
    organizer/results/ for a bare filename. Without this the defaults and the
    documented commands only worked when you happened to be sitting in
    organizer/, and `python organizer/leaderboard.py` from the repo root
    rendered an empty board.
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


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = _resolve(path)
    return json.loads(p.read_text()) if p.exists() else None


# Every MAE in this workshop sits at 0.015x: the whole competition is decided in
# the fourth decimal, so 4 dp rounds genuinely different models to the same
# number on the projector. 5 dp is the finest that still reads from the back row.
MAE_DP = 5


def fmt(x: float | None, nd: int = MAE_DP, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%" if pct else f"{x:.{nd}f}"


def delta_cell(mae1: float | None, mae2: float | None) -> tuple[str, str]:
    """Return (text, css_class) for the Human -> AI change column."""
    if not mae1 or not mae2:
        return "—", ""
    d = (mae2 - mae1) / mae1 * 100
    cls = "good" if d < -1e-9 else ("bad" if d > 1e-9 else "flat")
    arrow = "▼" if d < 0 else ("▲" if d > 0 else "■")
    return f"{arrow} {d:+.1f}%", cls


def team_rows(results: dict, key: str = "mae") -> list[dict]:
    ok = [t for t in results["teams"] if t.get("status") == "ok"]
    fail = [t for t in results["teams"] if t.get("status") != "ok"]
    ok.sort(key=lambda t: t[key])
    return ok + fail


def build_table_round(results: dict, title: str, baseline: float | None,
                      compare: dict | None = None) -> str:
    rows_html = []
    for i, t in enumerate(team_rows(results), 1):
        if t.get("status") == "ok":
            mae = fmt(t["mae"])
            rmse = fmt(t["rmse"])
            diracc = fmt(t["dir_acc"], 3, pct=True)
            if compare:
                other = compare.get(t["team"])
                d_txt, d_cls = delta_cell(t["mae"], (other or {}).get("mae"))
            else:
                d_txt, d_cls = "", ""
        else:
            mae = rmse = diracc = "DNF"
            d_txt, d_cls = "—", ""
        rows_html.append(
            f"<tr><td class='rank'>{i if t.get('status') == 'ok' else '—'}</td>"
            f"<td class='team'>{html.escape(t['team'])}</td>"
            f"<td class='arch'>{html.escape(str(t.get('arch', '?')))}</td>"
            f"<td class='num'>{mae}</td><td class='num'>{rmse}</td>"
            f"<td class='num'>{diracc}</td>"
            + (f"<td class='num {d_cls}'>{d_txt}</td>" if compare else "")
            + "</tr>")
    header = ("<tr><th>#</th><th>Team</th><th>Model</th><th>MAE ↓</th><th>RMSE</th>"
              "<th>Dir. Acc ↑</th>" + ("<th>vs Human</th>" if compare else "") + "</tr>")
    baseline_html = (f"<div class='baseline'>baseline to beat — always predict 0.0: "
                     f"<b>MAE {fmt(baseline)}</b></div>") if baseline else ""
    return (f"<section><h2>{title}</h2>{baseline_html}"
            f"<table>{header}{''.join(rows_html)}</table></section>")


def build_table_paper(paper: dict, title: str) -> str:
    capital = paper.get("capital", 10_000)
    order = sorted(paper["teams"].items(), key=lambda kv: kv[1]["final_value"], reverse=True)
    rows = []
    for team, s in order:
        ret = s["return_pct"]
        cls = "good" if ret > 0 else ("bad" if ret < 0 else "flat")
        is_bh = team == "BUY & HOLD"
        rows.append(
            f"<tr{' class=benchmark' if is_bh else ''}><td class='team'>"
            f"{html.escape(team)}</td><td class='num'>${capital:,.0f}</td>"
            f"<td class='num'>${s['final_value']:,.0f}</td>"
            f"<td class='num {cls}'>{ret:+.2f}%</td>"
            f"<td class='num'>{s['n_trades']}</td></tr>")
    header = ("<tr><th>Team</th><th>Start</th><th>Final</th><th>Return</th>"
              "<th>Trades</th></tr>")
    return (f"<section><h2>{title}</h2>"
            f"<div class='baseline'>virtual money only — long when predicted return "
            f"&gt; threshold, else cash</div>"
            f"<table>{header}{''.join(rows)}</table></section>")


def build_table_holdout(holdout: dict, r1: dict | None, r2: dict | None) -> str:
    def mae_of(results: dict | None, team: str) -> float | None:
        if not results:
            return None
        for t in results["teams"]:
            if t["team"] == team and t.get("status") == "ok":
                return t["mae"]
        return None

    def hold_mae(team: str, rnd: int) -> float | None:
        """2026 MAE for one team's model from one round. Holdout entries carry a
        `round` tag; a flat legacy results file has none, so fall back to any."""
        best = None
        for t in holdout["teams"]:
            if t["team"] != team or t.get("status") != "ok":
                continue
            if t.get("round") == rnd:
                return t["mae"]
            if t.get("round") is None and best is None:
                best = t["mae"]
        return best

    order, seen = [], set()
    for t in holdout["teams"]:          # preserve the evaluator's ranking order
        if t["team"] not in seen:
            seen.add(t["team"])
            order.append(t["team"])

    rows = []
    for team in order:
        cells = []
        for results, rnd in ((r1, 1), (r2, 2)):
            cells.append(f"<td class='num'>{fmt(mae_of(results, team))}</td>"
                         f"<td class='num'>{fmt(hold_mae(team, rnd))}</td>")
        rows.append(f"<tr><td class='team'>{html.escape(team)}</td>{''.join(cells)}</tr>")
    header = ("<tr><th rowspan='1'>Team</th><th colspan='2'>Human-only</th>"
              "<th colspan='2'>Human + AI</th></tr>"
              "<tr><th></th><th>2025 MAE</th><th>2026 MAE</th>"
              "<th>2025 MAE</th><th>2026 MAE</th></tr>")
    base = holdout.get("baseline_zero_mae")
    return (f"<section><h2>🔮 Surprise holdout — unseen {html.escape(str(holdout['eval_year']))} "
            f"data</h2><div class='baseline'>did the Round-2 improvement transfer to data "
            f"nobody optimised against? (predict-0 baseline: {fmt(base)})</div>"
            f"<table>{header}{''.join(rows)}</table></section>")


CSS = """
:root { --bg:#0b0f14; --panel:#11161d; --line:#1e2630; --text:#d7e2ee;
        --dim:#7d8b9b; --green:#2dd4a7; --red:#ff5d73; --gold:#f5c451; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text);
       font:15px/1.5 'SF Mono',ui-monospace,Menlo,Consolas,monospace; padding:34px; }
.wrap { max-width:1060px; margin:0 auto; }
h1 { font-size:24px; letter-spacing:2px; color:var(--gold); text-transform:uppercase; }
h1 .q { color:var(--text); }
.sub { color:var(--dim); margin:6px 0 26px; font-size:13px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:18px 22px; margin-bottom:22px; }
h2 { font-size:15px; letter-spacing:1.5px; text-transform:uppercase; color:var(--green);
     margin-bottom:10px; }
table { width:100%; border-collapse:collapse; }
th { text-align:left; color:var(--dim); font-weight:600; font-size:12px;
     text-transform:uppercase; letter-spacing:1px; padding:8px 10px;
     border-bottom:1px solid var(--line); }
td { padding:9px 10px; border-bottom:1px solid var(--line); }
tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.rank { color:var(--dim); width:34px; }
.team { font-weight:600; }
.arch { color:var(--dim); }
.good { color:var(--green); } .bad { color:var(--red); } .flat { color:var(--dim); }
.baseline { color:var(--dim); font-size:12.5px; margin:4px 0 12px; }
.baseline b { color:var(--gold); }
tr.benchmark td { border-top:2px solid var(--gold); color:var(--gold); }
tr.benchmark .team::after { content:' (benchmark)'; color:var(--dim); font-weight:400; }
footer { color:var(--dim); font-size:12px; margin-top:8px; }
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the projector leaderboard.")
    ap.add_argument("--round1", nargs="?", const="round1.json", default=None,
                    metavar="JSON", help="default: round1.json in organizer/results/")
    ap.add_argument("--round2", nargs="?", const="round2.json", default=None, metavar="JSON")
    ap.add_argument("--paper1", nargs="?", const="round1_paper.json", default=None, metavar="JSON")
    ap.add_argument("--paper2", nargs="?", const="round2_paper.json", default=None, metavar="JSON")
    ap.add_argument("--holdout", nargs="?", const="holdout.json", default=None, metavar="JSON")
    ap.add_argument("--out", default="leaderboard.html")
    ap.add_argument("--title", default="Can AI Predict the Market?")
    ap.add_argument("--live", nargs="?", const=10, type=int, default=0, metavar="SECONDS",
                    help="make the rendered page reload itself every N seconds "
                         "(default: off — it is a frozen snapshot until re-rendered)")
    args = ap.parse_args()

    # Bare `leaderboard.py` should just work on whatever has been scored — the
    # flags exist for the rare case of scoring a different round by hand.
    r1, r2 = load(args.round1 or "round1.json"), load(args.round2 or "round2.json")
    p1, p2 = load(args.paper1 or "round1_paper.json"), load(args.paper2 or "round2_paper.json")
    hold = load(args.holdout or "holdout.json")
    sections: list[str] = []
    if r1:
        sections.append(build_table_round(r1, "Round 1 — Human-only optimisation",
                                          r1.get("baseline_zero_mae")))
    if r2:
        sections.append(build_table_round(r2, "Round 2 — Human + AI optimisation",
                                          r2.get("baseline_zero_mae"), compare={
                                              t["team"]: t for t in (r1 or {}).get("teams", [])}))
    if p1:
        sections.append(build_table_paper(p1, "📈 Paper trading — Round 1 models"))
    if p2:
        sections.append(build_table_paper(p2, "📈 Paper trading — Round 2 models"))
    if hold:
        sections.append(build_table_holdout(hold, r1, r2))
    if not sections:
        raise SystemExit("Nothing to render — pass at least one results JSON.")

    generated = Path(args.round1 or args.round2 or "round1.json").name
    # --live adds a meta-refresh so the projector tab picks up a re-render on its
    # own. Without it the file is a frozen snapshot and needs a manual F5 — which
    # is what you want for the holdout reveal, where you control the moment.
    refresh = (f"<meta http-equiv='refresh' content='{args.live}'>" if args.live else "")
    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>{refresh}
<title>{html.escape(args.title)} — Leaderboard</title><style>{CSS}</style></head><body>
<div class='wrap'>
<h1><span class='q'>{html.escape(args.title)}</span> · Leaderboard</h1>
<div class='sub'>private test set · SPY + NVDA + AAPL + MSFT + TSLA pooled ·
MAE = mean absolute error of next-day-return predictions ·
lower is better · updated from {html.escape(generated)}</div>
{''.join(sections)}
<footer>Educational exercise — historical data, virtual portfolios, not investment advice.</footer>
</div></body></html>"""
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    if args.live:
        print(f"Wrote {out.resolve()}  — refreshing itself every {args.live}s; "
              f"re-run this command to fold in new results.")
    else:
        print(f"Wrote {out.resolve()}  — open it in the projector browser and "
              f"refresh after each round (or use --live).")


if __name__ == "__main__":
    main()
