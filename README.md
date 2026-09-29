# Can AI Predict the Market? — Workshop Material

A complete material set for a **2-hour, hands-on ML stock-forecasting workshop**.
**Event venue: HKU InnoWing.** Teams build next-day-return models for a five-instrument panel
(**SPY, NVDA, AAPL, MSFT, TSLA**), compete on a **private 2025 test set** in a
*human-only* round and a *human+AI* round, then find out whether AI-assisted
optimisation helped for teams that submitted in both rounds — and whether their
gains survive a 2026 holdout nobody optimised against.

Educational exercise only: historical data, virtual portfolios, no investment advice.

---

## ⚠️ Who is each file for?

**Rule of thumb: participants only ever receive the `participant/` folder. Everything
else stays with the organisers.**

### 👥 PARTICIPANTS — the only folder attendees need

| File | What it is |
|---|---|
| `participant/workshop.ipynb` | The starter notebook — teams' entire workspace. 🟩 `MODIFY` banners = where they compete; 🔒 `LOCKED` cells = shared infrastructure. |
| `participant/GUIDE.md` | Section-by-section walkthrough of the notebook (read alongside it). |
| `participant/CHEATSHEET.md` | One-page competition reference — explicitly allowed during Round 1. |
| `participant/data/workshop_participant.csv` | The dataset: OHLCV panel 2014–2024, 5 tickers (the only data teams see). |
| `participant/README.md` | Quick-start + competition summary for attendees. |
| `participant/SETUP_VENV.md` | Local install guide for VS Code + venv users (macOS caveats: XGBoost excluded there, or use Colab/miniforge). |
| `participant/requirements.txt` | Pinned pip packages for that local install (synced copy of the root file). |
| `participant/requirements-macos.txt` | macOS pip packages for the local install; omits XGBoost because of the torch/OpenMP conflict. |
| `participant/exports/` | Demo submission zips (what "done" looks like). |

### 🎓 TEACHING — presenters and facilitators

| File | What it is |
|---|---|
| `slides/index.html` | The 42-slide deck. Self-contained, zero dependencies — open in any browser, F11, ←/→ to navigate, `R` reveals a chart, `#38r` deep-links slide 38 pre-revealed. |
| `FACILITATOR_GUIDE.md` | Minute-by-minute run of show with per-slide talking points, organiser command sequence, FAQ, fallback plans. |
| `WorkshopOutline.md` | The original design document (source of truth for the pedagogy). |

### 🛠 ORGANISERS — scoring, data, infrastructure

| File | What it is |
|---|---|
| `organizer/README.md` | **Start here** — technical handoff/recap: verification log, reference numbers, day-of commands, known traps. Shareable with co-hosts and agents. |
| `organizer/prepare_data.py` | Downloads the 5-ticker panel from Yahoo → participant CSV + private label CSVs. |
| `organizer/evaluate.py` | Scores team submissions (folder or zip) against the private labels → results JSON. |
| `organizer/paper_trading.py` | $10k LONG/CASH simulation of stored predictions + buy-&-hold benchmark. |
| `organizer/leaderboard.py` | Renders the projector `leaderboard.html` (self-contained, F5 to refresh). |
| `organizer/collect.py` | **The submission portal.** FastAPI app on the organiser laptop — teams upload their zip from a browser on the venue Wi-Fi; it validates, files into `submissions/round<N>/`, re-scores, and serves a live scoreboard. |
| `organizer/build_notebook.py` | Dev tool: regenerates `participant/workshop.ipynb` (embeds `submission_template/predict.py` verbatim). |
| `organizer/private/` | 🔒 **Never distribute**: 2025 labels + 2026 holdout. |
| `organizer/submissions/`, `organizer/results/` | Where team zips land / where scoring output goes (demo contents included). |
| `submission_template/` | Readable reference of the locked submission interface (`predict(df) -> pd.Series`). |
| `environment.yml` / `environment.lock.yml` / `requirements.txt` | Conda env spec, pinned solve, and pip alternative (Linux/Colab tested; Windows documented but not verified; macOS needs conda/miniforge for XGBoost). |

---

## Repository map

```
├── WorkshopOutline.md            🎓 design document
├── README.md                     ← you are here (audience map + quick start)
├── FACILITATOR_GUIDE.md          🎓 run-of-show for presenters/facilitators
├── environment.yml               🛠 conda env spec (name: stocks-ml, all conda-forge)
├── environment.lock.yml          🛠 pinned solve of a known-good environment
├── requirements.txt              🛠 pip equivalent (Linux/Colab tested; Windows not verified)
│
├── slides/                       🎓 teaching
│   └── index.html                42-slide dark trading-terminal deck
│
├── participant/                  👥 distribute this whole folder to teams
│   ├── README.md                 attendee quick start
│   ├── workshop.ipynb            starter notebook (executes end-to-end, about 1–3 min on CPU)
│   ├── GUIDE.md                  notebook walkthrough with the "why"
│   ├── CHEATSHEET.md             in-competition quick reference
│   ├── SETUP_VENV.md             VS Code + venv local install guide (macOS caveats)
│   ├── requirements.txt          synced copy of the root pip file
│   ├── requirements-macos.txt   macOS pip list without XGBoost
│   ├── data/workshop_participant.csv
│   └── exports/                  demo submission zips
│
├── submission_template/          🛠 reference copy of the locked submission interface
│
└── organizer/                    🛠 organisers only — contains private labels
    ├── README.md                 🛠 technical handoff/recap (start here)
    ├── prepare_data.py · evaluate.py · paper_trading.py · leaderboard.py
    ├── build_notebook.py         dev tool: regenerates the participant notebook
    ├── submissions/ · results/   demo runs included
    └── private/                  🔒 2025 labels + 2026 holdout — never distribute
```

## Quick start

### Participants
Open `participant/README.md`. Short version: upload `workshop.ipynb` to
[Colab](https://colab.research.google.com), upload `data/workshop_participant.csv`
when prompted, `Runtime → Run all` (about 1–3 min), read `GUIDE.md` alongside, keep
`CHEATSHEET.md` open during rounds.

### Organisers (one-time setup)
```bash
conda env create -f environment.yml     # or: conda env create -f environment.lock.yml
conda activate stocks-ml
python -m ipykernel install --user --name stocks-ml
cd organizer && python prepare_data.py  # refresh data (writes the 3 CSVs)
```
`environment.yml` installs **everything from conda-forge** so torch and xgboost share
one OpenMP runtime — mixing a conda torch with a pip xgboost can segfault the kernel
when both load in one process. Colab participants are unaffected (preinstalled there).

### Collect submissions (day-of, one command)
```bash
conda activate stocks-ml && cd organizer
python collect.py --pin 4242 --qr
```
Prints the URL to put on the whiteboard; teams upload from any browser on the venue
Wi-Fi, and each round re-scores itself into `results/`. No cloud account, no
upload to a third party, and the private labels never move off your laptop. Campus
Wi-Fi often blocks laptop-to-laptop traffic — test it on the venue network the day
before (`organizer/README.md` §5 has the test and the hotspot/USB fallbacks). If the
portal can't be reached, you can still file zips collected on a USB stick yourself:

```bash
python collect.py --import /Volumes/<STICK>/submission_TeamBlah_round1.zip
```

### Run the workshop day
`FACILITATOR_GUIDE.md` §3–4 has the full run of show; `organizer/README.md` §5 has
the condensed command sequence. Projector: `slides/index.html` +
`organizer/results/leaderboard.html` (F5 to refresh after each evaluation).

## The dataset

| Period | Purpose | Visibility |
|---|---|---|
| 2014–2022 | training | participants |
| 2023–2024 | validation/tuning | participants |
| 2025 | private competition set | not distributed; evaluator passes features to submissions and reveals scores only (labels stay private) |
| 2026 YTD | surprise generalisation holdout | organisers only, revealed at the end |

Data: long-format OHLCV panel (Date × Ticker), split/dividend-adjusted — NVDA (10:1
in 2024), AAPL (4:1, 2020) and TSLA (5:1 + 3:1) all split inside the window, and
unadjusted prices would inject fake ±40–90% "returns" at split dates.

Task: predict each ticker's **next-day return** from OHLCV. Primary metric: **MAE
pooled over all five tickers** ("always predict 0" scores ≈ 0.0162 on 2025);
RMSE, directional accuracy and per-ticker MAE are shown as diagnostics.

## Verified end-to-end (2026-09-29 build)

- `prepare_data.py` → 13,840 participant rows / 1,250 private-2025 rows / 920 holdout
  rows across 5 tickers
- `workshop.ipynb` executes headless: Ridge 0.01537 · Forest 0.01531 · XGB 0.01538 ·
  LSTM 0.01548 · GRU 0.01557 (pooled val MAE; baseline 0.01543) and exports a zip
- `evaluate.py` scores the exported zips on private 2025: forest MAE 0.01598 vs
  baseline 0.01621 pooled (predict-0 spans SPY 0.0075 → TSLA 0.0291) — but the
  forest **loses to the baseline on the 2026 holdout** (0.01517 vs 0.01500): the
  edge didn't generalise
- `paper_trading.py` (equal-weight, zero cost): forest $10,000 → **$12,571 (+25.7%, 33
  trades)** vs buy-and-hold $11,915 (+19.2%) — while the GRU, with the *worst* MAE,
  made **+27.5% on 322 trades**; add `--cost-bps 10` and the GRU's churn drops it to
  +19.6% — accuracy ≠ profit, in both directions
- Adversarially tested (18 cases: crash/NaN/timeout/zip oddities — every failure
  degrades to a graceful DNF row); notebook also executed under pandas 2.2.3
  (Colab's version); slides screenshot-verified. Full log: `organizer/README.md` §8.

Demo results live in `organizer/results/`: `TeamZero` is the baseline, `Team Example`
is a Random Forest submission for Round 1, and `Team Seq` is a GRU submission for
  Round 2. They are different teams, so these scores are not a paired AI comparison.
Wipe the demo files before the event, keeping the folders themselves:
`rm -rf organizer/results/*` and `rm -f organizer/submissions/round*/*`.

## Regenerating / customising

- **Notebook**: edit `organizer/build_notebook.py` and rerun it — it embeds
  `submission_template/predict.py` verbatim into the export cell (single source of
  truth). The shipped `workshop.ipynb` is the distributed artefact.
- **Data window**: `python organizer/prepare_data.py --start 2014-01-01 --end 2026-12-31`.
- **Different instruments**: edit `TICKERS` in `prepare_data.py` — the notebook,
  exporter and evaluator are ticker-agnostic (features and sequence windows are built
  per ticker; the `is_<TICKER>` one-hots follow whatever tickers are present).
