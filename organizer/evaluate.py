#!/usr/bin/env python3
"""
evaluate.py — ORGANISER TOOL. Scores team submissions against the private labels.

    python3 evaluate.py --round 1
    python3 evaluate.py --round 2
    python3 evaluate.py --holdout          # the surprise 2026 generalisation test

Expects a `submissions/` directory next to this script where each team is either
a folder or a zip (submission_<TEAM>_round<N>.zip), as produced by the notebook's
export cell:

    organizer/
    ├── evaluate.py
    ├── submissions/
    │   ├── TeamAlpha/            # unzipped folder ...
    │   └── submission_TeamBeta_round2.zip   # ... or a zip, both work
    └── private/                  # never shared with participants

Every team's predict(df) receives the SAME input: the full participant OHLCV panel
(SPY, NVDA, AAPL, MSFT, TSLA · 2014–2024) plus the evaluation year's OHLCV — warm-up
included, labels absent. Predictions are scored only on (Date, Ticker) rows that
have private labels.

Metrics (primary metric is MAE, pooled across all tickers)
-------
MAE      pooled mean absolute error of next-day-return predictions (lower is better)
RMSE     pooled diagnostic
DirAcc   pooled fraction of rows where sign(pred) == sign(actual), ties -> down
by_ticker  per-ticker MAE breakdown (SPY is the easiest, TSLA the hardest)

Writes results/round<N>.json or results/holdout.json (predictions included so that
paper_trading.py / leaderboard.py can reuse them; labels are never written).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as _queue_mod
import re
import shutil
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ORGANIZER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ORGANIZER_DIR.parent
PARTICIPANT_CSV = PROJECT_ROOT / "participant" / "data" / "workshop_participant.csv"
PRIVATE_DIR = ORGANIZER_DIR / "private"
RESULTS_DIR = ORGANIZER_DIR / "results"
SUBMISSIONS_DIR = ORGANIZER_DIR / "submissions"
SANDBOX_DIR = RESULTS_DIR / "_sandbox"
MAX_UNPACKED_MB = 200.0   # a model + features file unpacks to a few MB, not hundreds

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
YEARS = {
    "2025": {"file": "workshop_2025_full.csv", "start": "2025-01-01", "end": "2025-12-31"},
    "2026": {"file": "workshop_2026_holdout.csv", "start": "2026-01-01", "end": "2026-12-31"},
}


def row_keys(df: pd.DataFrame) -> np.ndarray:
    """Unique key per row: 'YYYY-MM-DD|TICKER' (matches the private files)."""
    return np.array([f"{d.date()}|{t}" for d, t in zip(df.index, df["Ticker"])])


def build_eval_input(year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ohlcv_input incl. warm-up, labels DataFrame with Ticker + target)."""
    spec = YEARS[year]
    private = pd.read_csv(PRIVATE_DIR / spec["file"], index_col="Date", parse_dates=True)
    history = pd.read_csv(PARTICIPANT_CSV, index_col="Date", parse_dates=True)

    year_ohlcv = private[["Ticker"] + OHLCV]
    labels = private[["Ticker", "next_day_return"]].dropna()
    input_df = pd.concat([history[["Ticker"] + OHLCV], year_ohlcv])
    input_df = input_df.sort_index(kind="stable")
    return input_df, labels


# --------------------------------------------------------------------------
# Per-team worker (runs in a spawned process so a hung/failing team cannot
# stall the evaluation; the parent enforces a timeout).
# --------------------------------------------------------------------------
def _worker(team_dir: str, payload: dict, queue: mp.Queue) -> None:  # noqa: ANN001
    try:
        df = pd.DataFrame(
            payload["data"],
            index=pd.DatetimeIndex(payload["index"]),
            columns=payload["columns"],
        )
        sys.path.insert(0, team_dir)
        # Fresh import per team; drop cached modules from previously scored teams.
        for mod in list(sys.modules):
            if mod in ("predict", "team_features", "team_models"):
                del sys.modules[mod]
        import predict as team_predict  # noqa: PLC0415 (deliberate late import)

        preds = team_predict.predict(df)
        preds = pd.Series(np.asarray(preds, dtype=float), index=df.index)
        if preds.isna().any():
            bad = preds[preds.isna()].index[:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"predict() returned NaN for {bad}... — rows without enough "
                             "history must be predicted as 0.0, never NaN.")
        queue.put({"ok": True, "values": preds.tolist()})
    except Exception:  # noqa: BLE001 — report anything back to the parent
        queue.put({"ok": False, "error": traceback.format_exc(limit=8)})


def evaluate_team(team_dir: Path, input_df: pd.DataFrame, timeout: float) -> dict:
    payload = {
        "data": input_df.values.tolist(),
        "index": [str(i) for i in input_df.index],
        "columns": list(input_df.columns),
    }
    queue: mp.Queue = mp.Queue()
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_worker, args=(str(team_dir), payload, queue))
    proc.start()

    import time  # noqa: PLC0415 (local to keep the module import block tidy)

    deadline = time.monotonic() + timeout
    result = None
    while True:
        try:
            result = queue.get(timeout=0.5)
            break
        except _queue_mod.Empty:
            if not proc.is_alive():
                # The child died (os._exit, native crash). Drain once in case the
                # result landed between the aliveness checks, then fail fast
                # instead of burning the whole timeout on a dead process.
                try:
                    result = queue.get(timeout=1.0)
                    break
                except _queue_mod.Empty:
                    proc.join(1)
                    return {"status": "error",
                            "error": "process exited without returning predictions"}
            if time.monotonic() > deadline:
                break
    if result is None:
        _reap(proc)
        return {"status": "timeout", "error": f"exceeded {timeout:.0f}s"}
    _reap(proc)

    if not result["ok"]:
        return {"status": "error", "error": result["error"]}

    keys = row_keys(input_df)
    preds = dict(zip(keys, result["values"]))
    return {"status": "ok", "predictions": preds, "arch": _read_arch(team_dir)}


def _reap(proc: mp.Process) -> None:
    """Terminate then, if needed, kill a team's worker process."""
    if not proc.is_alive():
        proc.join(1)
        return
    proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join(1)


def _read_arch(team_dir: Path) -> str:
    try:
        return json.loads((team_dir / "config.json").read_text()).get("arch", "?")
    except Exception:  # noqa: BLE001
        return "?"


def _display_name(team_dir: Path, submissions_dir: Path) -> str:
    """Name to show on the leaderboard for one submission.

    Priority: the team name the team typed on the collect.py upload form (kept
    verbatim in its receipt JSON) → the zip/folder filename → the team's own
    config.json "team" field. The form name wins because that is the name the
    rest of the room knows them by, and config.json often still carries the
    notebook template's name.
    """
    from_name = re.sub(
        r"_round\d+$", "",
        team_dir.name.removeprefix("submission_"),
        flags=re.IGNORECASE,
    ) or team_dir.name
    if team_dir.parent == SANDBOX_DIR:
        slug = team_dir.name.removeprefix("submission_")
        for receipt_name in (f"receipt_{team_dir.name}.json", f"receipt_{slug}.json"):
            try:
                name = json.loads((submissions_dir / receipt_name).read_text()).get("team")
            except (OSError, json.JSONDecodeError):
                continue
            if name:
                return str(name)
        return from_name
    try:
        name = json.loads((team_dir / "config.json").read_text()).get("team")
        if name:
            return str(name)
    except Exception:  # noqa: BLE001
        pass
    return from_name


def score(preds: dict[str, float], labels: pd.DataFrame) -> dict:
    """Pooled + per-ticker metrics on the (Date, Ticker) rows that have labels."""
    lab_keys = np.array([f"{d.date()}|{r['Ticker']}" for d, r in labels.iterrows()])
    y = labels["next_day_return"].values
    p = np.array([preds.get(k, np.nan) for k in lab_keys])
    valid = ~np.isnan(p)
    p, y = p[valid], y[valid]
    tickers = np.array([k.split("|")[1] for k in lab_keys])[valid]
    err = p - y
    by_ticker = {}
    for t in pd.unique(tickers):
        m = tickers == t
        by_ticker[t] = {
            "mae": float(np.abs(err[m]).mean()),
            "dir_acc": float(((p[m] > 0) == (y[m] > 0)).mean()),
        }
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "dir_acc": float(((p > 0) == (y > 0)).mean()),
        "n_days": int(len(p)),
        "by_ticker": by_ticker,
    }


def _dnf_stub(dest: Path, reason: str) -> Path:
    """Stage a placeholder team folder whose predict() always fails with `reason`.

    discover_teams() runs BEFORE the per-team scoring loop, so anything it raises
    kills the whole run — no results JSON, and on the portal a stale board at the
    deadline. A single unusable zip must cost exactly one leaderboard row, so we
    hand the failure to the machinery that already turns a broken predict() into
    a DNF instead of raising here.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "predict.py").write_text(
        '"""Placeholder staged by evaluate.py — this submission was not usable."""\n'
        "import pandas as pd\n\n"
        f"_REASON = {reason!r}\n\n\n"
        "def predict(df: pd.DataFrame) -> pd.Series:\n"
        "    raise ValueError(_REASON)\n"
    )
    return dest


def discover_teams(submissions_dir: Path, required: bool = True) -> list[Path]:
    if not submissions_dir.exists():
        if not required:
            return []
        sys.exit(f"No submissions directory at {submissions_dir}. Create it and drop team "
                 f"folders or zips inside (see the docstring at the top of this file).")
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    teams: list[Path] = []
    for entry in sorted(submissions_dir.iterdir()):
        if entry.name.startswith(("_", ".")):
            continue
        if entry.is_dir() and (entry / "predict.py").exists():
            teams.append(entry)
        elif entry.suffix == ".zip":
            dest = SANDBOX_DIR / entry.stem
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)  # external volumes can race here
            if dest.exists():
                shutil.rmtree(dest)
            try:
                with zipfile.ZipFile(entry) as zf:
                    # Cap what we are about to write. collect.py screens uploads, but
                    # a zip can also arrive via --import, a USB stick, or a hand-copied
                    # folder — and this runs while the room is watching the board.
                    unpacked = sum(i.file_size for i in zf.infolist())
                    if unpacked > MAX_UNPACKED_MB * 1_000_000:
                        raise ValueError(
                            f"unzips to {unpacked / 1e6:.0f} MB, over the "
                            f"{MAX_UNPACKED_MB:.0f} MB limit — not a model submission")
                    zf.extractall(dest)  # CPython's zipfile neutralises ../ components
            except (zipfile.BadZipFile, ValueError, OSError, RuntimeError) as exc:
                # Corrupt/truncated/oversized zip: one DNF row, not a dead run.
                reason = f"could not unpack this zip — {type(exc).__name__}: {exc}"
                print(f"  !! {entry.name}: {reason}")
                teams.append(_dnf_stub(dest, reason))
                continue
            shutil.rmtree(dest / "__MACOSX", ignore_errors=True)  # Finder noise
            # Flatten a single nested root folder (submission_X/… -> …); ignore
            # AppleDouble/metadata entries (. and _ prefixes) when counting AND
            # when moving — a zipped ._predict.py sidecar makes rmdir fail, and
            # on an external volume such a file can vanish between listing and
            # rename, which killed the whole scoring run on the first team.
            entries_here = [p for p in dest.iterdir() if not p.name.startswith((".", "_"))]
            if (len(entries_here) == 1 and entries_here[0].is_dir()
                    and not (dest / "predict.py").exists()):
                inner = entries_here[0]
                for child in inner.iterdir():
                    if child.name.startswith((".", "_")):
                        continue
                    target = dest / child.name
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True) if target.is_dir() \
                            else target.unlink(missing_ok=True)
                    try:
                        child.rename(target)
                    except OSError:
                        shutil.copytree(child, target, dirs_exist_ok=True) if child.is_dir() \
                            else shutil.copy2(child, target)
                shutil.rmtree(inner, ignore_errors=True)
            if (dest / "predict.py").exists():
                teams.append(dest)
            else:
                print(f"  !! {entry.name}: no predict.py after unzip — skipped")
    if not teams and required:
        sys.exit("No valid submissions found (each team needs a predict.py).")
    return teams


def main() -> None:
    ap = argparse.ArgumentParser(description="Score team submissions (organiser only).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--round", type=int, choices=[1, 2])
    mode.add_argument("--holdout", action="store_true", help="score against the 2026 holdout")
    ap.add_argument("--year", default=None, help="override eval year (default 2025/2026)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per team")
    ap.add_argument("--submissions", default=None,
                    help="submissions dir (default: submissions/, or the collect.py "
                         "per-round folders submissions/round<N>)")
    args = ap.parse_args()

    year = args.year or ("2026" if args.holdout else "2025")
    if year not in YEARS:
        sys.exit(f"unknown year {year!r} — expected one of {sorted(YEARS)}")
    input_df, labels = build_eval_input(year)
    print(f"Evaluating against private {year} labels: {labels.index[0].date()} .. "
          f"{labels.index[-1].date()}  ({len(labels)} scored rows across "
          f"{labels['Ticker'].nunique()} tickers)")

    # (team_dir, which submissions folder it came from) — the holdout scores a
    # team's Round-1 AND Round-2 model so the reveal can show both, which is the
    # whole point of the "did the AI help, and did either survive?" comparison.
    found: list[tuple[Path, Path, int | None]] = []
    if args.submissions:
        d = Path(args.submissions)
        found += [(t, d, None) for t in discover_teams(d)]
    elif args.holdout:
        for rnd in (1, 2):
            d = SUBMISSIONS_DIR / f"round{rnd}"
            for t in discover_teams(d, required=False):
                found.append((t, d, rnd))
        if not found:  # flat legacy layout — everything in one folder
            found += [(t, SUBMISSIONS_DIR, None)
                      for t in discover_teams(SUBMISSIONS_DIR)]
    else:
        d = (SUBMISSIONS_DIR / f"round{args.round}"
             if (SUBMISSIONS_DIR / f"round{args.round}").exists() else SUBMISSIONS_DIR)
        found += [(t, d, args.round) for t in discover_teams(d)]

    print(f"Found {len(found)} submission(s): "
          f"{', '.join(t.name for t, _, _ in found)}\n")

    # Built ONCE — rebuilding it per prediction item cost ~3 min per team.
    label_keys = {f"{d.date()}|{r['Ticker']}" for d, r in labels.iterrows()}

    entries = []
    for team_dir, submissions_dir, team_round in found:
        print(f"→ {team_dir.name} ...", flush=True)
        result = evaluate_team(team_dir, input_df, args.timeout)
        entry = {"team": _display_name(team_dir, submissions_dir),
                 "dir": team_dir.name, "arch": result.get("arch", "?"),
                 "round": team_round,
                 "status": result["status"]}
        if result["status"] == "ok":
            entry.update(score(result["predictions"], labels))
            entry["predictions"] = {k: round(float(v), 8) for k, v in result["predictions"].items()
                                    if k in label_keys}
            print(f"   OK   MAE={entry['mae']:.6f}  RMSE={entry['rmse']:.6f}  "
                  f"DirAcc={entry['dir_acc']:.3f}  ({entry['n_days']} rows)")
            for t, s in sorted(entry["by_ticker"].items()):
                print(f"        {t:<6} MAE {s['mae']:.6f}  DirAcc {s['dir_acc']:.3f}")
        else:
            entry["error"] = result.get("error", "")[-800:]
            print(f"   {result['status'].upper()}: {entry['error'][:200]}")
        entries.append(entry)

    ok_entries = [e for e in entries if e["status"] == "ok"]
    ok_entries.sort(key=lambda e: e["mae"])
    entries = ok_entries + [e for e in entries if e["status"] != "ok"]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_name = "holdout.json" if args.holdout else f"round{args.round}.json"
    baseline_by_ticker = {
        t: {"mae": float(g["next_day_return"].abs().mean())}
        for t, g in labels.groupby("Ticker")
    }
    out = {
        "eval_year": year,
        "period": [str(labels.index[0].date()), str(labels.index[-1].date())],
        "n_days": int(len(labels)),
        "baseline_zero_mae": float(labels["next_day_return"].abs().mean()),
        "baseline_by_ticker": baseline_by_ticker,
        "teams": entries,
    }
    out_path = RESULTS_DIR / out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{'=' * 64}\nLeaderboard order (pooled MAE):")
    for i, e in enumerate(ok_entries, 1):
        print(f"  {i}. {e['team']:<24} {e['arch']:<14} MAE {e['mae']:.6f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
