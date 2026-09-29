# 🛠 Organizer Handbook — Can AI Predict the Market?

**Read this if you are:** a co-host picking this up cold, or an agent taking over
preparation. It is a self-contained recap of what exists, what has been verified, how
to run everything, and where the traps are. The full minute-by-minute run of show
lives in [`../FACILITATOR_GUIDE.md`](../FACILITATOR_GUIDE.md) — this file is the
technical handoff.

**Handoff status:** ✅ built, executed end-to-end and adversarially tested
(2026-09-29, conda env `stocks-ml`). One open item: a Colab smoke-test (see §8).

---

## 1 · What this is (2 sentences)

A 2-hour interactive ML workshop for 20–30 students in teams of 3–5. Teams build
next-day-return models for a **5-instrument panel (SPY, NVDA, AAPL, MSFT, TSLA)**,
compete on a **private 2025 test set** in a human-only round and a human+AI round,
then discover whether their gains survive a hidden **2026 holdout** — and whether
prediction accuracy had anything to do with paper-trading profit.

## 2 · File map — who owns what

| Path | Audience | Purpose |
|---|---|---|
| `prepare_data.py` | organiser | Downloads the panel from Yahoo → writes the 3 CSVs (§4) |
| `evaluate.py` | organiser | Scores `submissions/*` (folder or zip) against private labels → `results/round{N}.json` / `holdout.json` |
| `paper_trading.py` | organiser | $10k LONG/CASH simulation per results JSON + buy&hold benchmark |
| `leaderboard.py` | organiser | Renders `results/leaderboard.html` (projector view, F5 to refresh) |
| `collect.py` | organiser | **The submission portal.** FastAPI app on your laptop: teams upload their zip from a browser on the venue Wi-Fi, it files + re-scores the round and serves a live scoreboard (§5) |
| `build_notebook.py` | organiser (dev) | Regenerates `../participant/workshop.ipynb`; embeds `../submission_template/predict.py` verbatim |
| `private/` | 🔒 organiser only | `workshop_2025_full.csv` (labels), `workshop_2026_holdout.csv` — **never distribute** |
| `submissions/round1/`, `submissions/round2/` | organiser | Where submissions land (uploads via `collect.py`, or copied by hand). Demo teams currently inside: `round1/TeamZero` (predicts 0), `round1/submission_Team_Example_round1.zip` (random forest), `round2/submission_Team_Seq_round2.zip` (GRU) |
| `results/` | organiser | Demo outputs from the prepared run — can be wiped before the event |
| `../participant/` | 👥 **participants** | `workshop.ipynb`, `GUIDE.md`, `CHEATSHEET.md`, `data/workshop_participant.csv`, `README.md`. **Distribute this folder and nothing else.** |
| `../slides/index.html` | 🎓 teaching | 42-slide deck (←/→ navigate, `R` reveals a chart, `#38r` deep-links) |
| `../FACILITATOR_GUIDE.md` | 🎓 teaching | Run-of-show, talking points per slide, FAQ, fallback plans |
| `../WorkshopOutline.md` | 🎓 teaching | The design document the material was built from (2-hour edition) |
| `../submission_template/` | organiser reference | Readable copy of the locked `predict.py` contract every submission must expose |
| `../environment.yml` + `environment.lock.yml` + `../requirements.txt` | organiser | Env spec / pinned conda solve / pip alternative (⚠️ pip path is Linux/Colab-only — see §7) |

## 3 · Environment (read this first)

**Use the conda env `stocks-ml` for everything.** All packages come from
conda-forge so torch and xgboost share one OpenMP runtime. Mixing a conda torch with
a pip xgboost (or vice-versa) **segfaults the kernel** on macOS when both load in one
process — this actually happened and cost us a debugging session.

```bash
conda activate stocks-ml          # or: /opt/miniconda3/envs/stocks-ml/bin/python
```

Recreate from scratch if needed:
```bash
conda env create -f environment.yml            # spec
conda env create -f environment.lock.yml       # exact pinned solve (known-good)
python -m ipykernel install --user --name stocks-ml
```

Never `pip install` workshop deps into `base` — if you have, remove them again
(`pip uninstall` the exact packages you added).

## 4 · Data pipeline

```bash
python prepare_data.py            # ~30 s, needs internet
```
Writes (long format: one row per Date × Ticker, `auto_adjust=True` because NVDA/AAPL/
TSLA all split inside the window — unadjusted prices would inject fake ±40–90 %
"returns"):

| File | Rows | Content |
|---|---|---|
| `../participant/data/workshop_participant.csv` | 13,840 | 2014–2024 OHLCV — the only data teams see |
| `private/workshop_2025_full.csv` | 1,250 | 2025 OHLCV **+ `next_day_return` labels** |
| `private/workshop_2026_holdout.csv` | 920 | 2026 YTD — the surprise test |

The script prints a per-ticker sanity table (re-run it T-1 day; the last row of each
private year is dropped so every scored row has a label). Two consecutive runs
produced byte-identical stats.

## 5 · Day-of: the submission portal (`collect.py`)

Teams hand in zips over the venue Wi-Fi instead of emailing them. Start the
collector ~5 minutes before Round 1 and leave it running for the whole session:

```bash
conda activate stocks-ml && cd organizer
python collect.py --pin 4242 --qr
```

It prints a banner with the exact URL to give the room, the scoreboard URL, and
your LAN IP. Teams open the upload page, type the PIN + team name, pick their zip,
and press upload. The zip is validated and filed into `submissions/round<N>/`, and
by default the round is **re-scored automatically** in the background — so the
projector tab just refreshes itself and you never assemble a command under pressure.

| Route | What it does |
|---|---|
| `GET /` | upload page (dark, phone-friendly) |
| `POST /upload` | validates PIN, ≤25 MB, real zip, `predict.py` at the root → saves `submissions/round<N>/submission_<Team>_round<N>.zip` + a `receipt_*.json` with a sha256 |
| `GET /healthz` | **no PIN** — plain "is the line up?" test, also tells a team whether the server is down or the Wi-Fi is blocking them |
| `GET /board` | **the live board** — polls every 3 s, needs no manual refresh |
| `POST /score` | force a re-score of a round (PIN required) |

Guards, all verified live against a running server (24 cases): wrong PIN → 401,
non-zip → 400, zip without `predict.py` → 400, corrupt/half-uploaded zip → 400 (CRC
checked at upload, so a dropped connection is caught on the spot rather than
mid-scoring), >25 MB → 413, round ≠ 1/2 → 400, team name with no letters/digits →
400, a zip whose contents sit in one sub-folder → accepted, re-upload → overwrites
in place (last upload before the deadline wins), and a name that would land on
another team's file (`Team A/B` vs `Team A_B`, or any case-only difference on
macOS) → 409 rather than silently destroying their submission. Four concurrent
uploads of one team leave exactly one zip and one receipt. A scoring run that
*fails* never shows the previous round's numbers as if they were new — the board
raises a red banner instead.

**Deliberately not Streamlit Cloud / Vercel.** The private 2025 labels must never
leave your machine, and a submission is executable code — scoring them on a public
host turns your competition into remote code execution as a service. Serverless
limits (250 MB, 60 s) could not carry a torch scoring stack either.

### Network: will the room actually reach your laptop?

Short answer: **test it the day before, and have a fallback ready.** Signing in to
university Wi-Fi (portal page or eduroam/802.1X) authenticates a device *to the
network*; it says nothing about whether that device may talk to another device on
the same SSID. Most campus WLANs enable AP/client isolation precisely to stop
laptop-to-laptop traffic, so two devices can both be "on HKU Wi-Fi", both logged in,
and still not see each other. Nothing in the collector can fix that — it is a
property of the access points, and only IT can change it.

Check in this order, on the venue's actual network, the day before:

1. Connect the laptop and a phone to the venue Wi-Fi (log in on both).
2. Run `python collect.py --pin 4242`, open the printed URL **on the phone**.
3. Page loads → done, use it. Add `--qr` so people can scan instead of typing an IP.

If the phone cannot load it, the port is probably not the problem — check
`http://<laptop-ip>:8000/healthz` from the phone to separate "server down" from
"network blocks you". Then fall back, in order of preference:

1. **Your own hotspot (best).** macOS: System Settings → General → Sharing →
   Internet Sharing, share the Wi-Fi connection over Wi-Fi, name it something
   obvious, give the room the password on the briefing slide. You get a private
   network with no client isolation, and it survives the venue Wi-Fi failing
   mid-round. Everything stays on your laptop, exactly as intended.
2. **A cheap travel router** in AP mode, if you have one — same idea, more clients
   than a phone hotspot will comfortably hold.
3. **USB stick round-robin.** The oldest, least glamorous option, and it still
   works. Teams hand you the zip and you file it with one command — the portal's
   file picker runs on your own machine, so a stick copy is just another upload:

   ```bash
   python collect.py --import /Volumes/<STICK>/submission_TeamBlah_round1.zip
   # ✅ filed submission_TeamBlah_round1.zip | team='Team Blah' round=1 (205 KB, sha 75b6…)
   ```

   It takes the same validation as an upload (real zip, `predict.py` at the root,
   CRC intact, name-collision guard), writes the same receipt, and exits without
   starting a server. Round and team name are read from the export filename and
   the zip's `config.json`; override with `--round 2` / `--team 'Exact Name'` when
   a team renamed things or saved it as `download.zip`. Then score with
   `python evaluate.py --round <N>` (or leave the portal running and let its
   auto-score pick it up).

   A USB-C stick and a `submissions/round1` folder on your Desktop are enough
   insurance. One USB caveat worth knowing: a copy that drops mid-transfer leaves
   a truncated zip, which the import rejects immediately with a checksum error
   rather than letting it silently vanish from the leaderboard later.

macOS firewall: if it prompts "Do you want the Python application to accept
incoming network connections", answer **Allow**, or nothing will connect while it
looks like a network problem.

### What updates live, and what does not

| Surface | Auto-updates? |
|---|---|
| `http://<laptop>:8000/board` | **Yes** — polls every 3 s. New uploads appear in the *received* table the moment they land; scores appear when the re-score finishes (the header shows `SCORING…` meanwhile) |
| `results/leaderboard.html` | **No** — a frozen snapshot. Re-run `leaderboard.py` and press F5. Add `--live` to inject a meta-refresh, but it still only shows what `leaderboard.py` last rendered |
| `results/*.json` | Written by `evaluate.py`; the source of truth both of the above read |

**Project on `/board` during the rounds.** It is the only surface that needs no
human step. Keep `leaderboard.html` for the holdout reveal, where you want to
control exactly when the room sees the 2026 numbers.

Timing at a deadline: scoring is a single debounced run, not one per upload — 15
teams uploading in the same minute produce **one** full re-score (measured: 15
uploads accepted in 0.2 s, board fully settled 30 s later with all 17 teams).
Expect roughly 20–60 s from the last upload to a settled board, more if models are
slow to load.

### Manual scoring (no portal, or portal down)

```bash
conda activate stocks-ml && cd organizer

# Round 1 — reads submissions/round1/
python evaluate.py --round 1                       # ~2-4 s per team
python paper_trading.py --results results/round1.json
python leaderboard.py                               # renders everything that exists

# Round 2 — reads submissions/round2/
python evaluate.py --round 2
python paper_trading.py --results results/round2.json
python leaderboard.py

# the surprise test — scores each team's LATEST zip (round2 → round1 → root)
python evaluate.py --holdout
python leaderboard.py
```

`evaluate.py` finds the right folder on its own: `--round 1` uses
`submissions/round1/`, `--round 2` uses `submissions/round2/`, and `--holdout`
scores the available submissions from both rounds together (tagged by team and
round). When a team submitted in both rounds, the reveal table lets you compare
its human-only and human+AI models on 2026; rows from different teams are not a
paired AI comparison. Point it anywhere with
`--submissions <dir>`.

Every team is scored in an isolated subprocess (120 s default timeout). Crashes, NaN
predictions, hangs, and hard-killed workers all become **DNF rows** — the pipeline
itself does not fail. Full details, flags and edge-case behaviour: FACILITATOR_GUIDE §4.

## 6 · Reference numbers from the prepared run (2026-09-29 data)

Calibration for what "normal" looks like — they will shift slightly when you
re-download data.

**Validation (2023–24, pooled MAE):** baseline (predict 0) **0.01543** ·
Ridge 0.01537 · **Random Forest 0.01531 (best)** · XGBoost 0.01538 · LSTM 0.01548 · GRU 0.01557

**Private 2025 (pooled MAE):** baseline **0.01621** · forest **0.01598 (beats it)**
· GRU 0.01624 — per-ticker for the forest: SPY 0.0074 · MSFT 0.0101 · AAPL 0.0126 ·
NVDA 0.0211 · TSLA 0.0287.

**2026 holdout** (prepared demo submissions): baseline **0.01500** · forest 0.01517 ·
GRU 0.01534 — both prepared model demos lose to predicting zero on unseen data. The
generalisation lesson is baked into the demo. The forest and GRU figures come from
Team Example R1 and Team Seq R2, respectively; they are different teams and do not
form a paired AI comparison.

**Paper trading** (equal-weight $2k/ticker, threshold 0, zero cost): forest
**+25.71 %** (33 trades) · GRU **+27.46 %** (322 trades, *worst* MAE, best P&L) ·
buy&hold **+19.15 %**. At `--cost-bps 10` the GRU's churn drops it to **+19.63 %** —
the accuracy≠profit demo lives in `results/round2_paper_cost10.json`
(regenerate with `paper_trading.py --results results/round2.json --cost-bps 10`);
`results/round1_paper_cost10.json` is the forest at the same cost (+24.83 %).

## 7 · Known traps (all discovered the hard way)

1. **macOS + pip torch/xgboost = kernel death.** Conda path only on this machine
   (§3). `requirements.txt` documents this; don't point Mac users at the venv path.
2. **`inspect.getsource()` fails for classes in Jupyter** — that's why
   `build_notebook.py` embeds the NN source (`NN_CLASSES_SRC`) as a template in the
   export cell instead of introspecting. Don't "simplify" it back.
3. **Round discipline:** `round1.json` is written once — do **not** re-run
   `--round 1` after Round-2 zips land in `submissions/`, or an AI-assisted model
   leaks into the human-round leaderboard. Move round-1 zips aside before Round 2.
4. **Scan eval output for `!!` lines** — a submission that can't be read (e.g. a zip
   with no `predict.py`) is skipped with one quiet `!! … skipped` line.
5. **Team names** are read from each zip's `config.json` (`team` field), so spaces
   and renamed zips are safe, and the Round-1↔Round-2 Δ join works.
6. **The `_sandbox/` dir and loose export working files** (`predict.py`,
   `config.json`, `model.*`, `team_*.py` appearing next to a notebook) are normal
   byproducts — safe to delete anytime.
7. **Colab users** are on pandas 2.x / Google-built wheels — the notebook was
   executed under pandas 2.2.3 locally and is 2.x-safe, but see §8.

## 8 · Verification log + open items

Verified on 2026-09-29 (conda `stocks-ml`, macOS arm64):

- ✅ `prepare_data.py` ×2 — identical output; per-ticker sanity asserts pass
- ✅ `workshop.ipynb` headless execution ×2 — 0 errors, **identical val MAEs both
  runs** (seeded); export cell produces a valid zip
- ✅ **Tabular path** (random-forest zip) and **sequence path** (GRU zip incl.
  `team_models.py`, torch state-dict, scaler) both scored through `evaluate.py`
  rounds 1/2 + holdout with per-ticker breakdowns
- ✅ `paper_trading.py` corrected-sum math (0 bps and 10 bps variants)
- ✅ `leaderboard.py` full and partial inputs; visual check of rendered HTML
- ✅ Adversarial suite (18 cases): crash / NaN / timeout / stdout-flood / `os._exit`
  / spaces-in-name / nested zip / zip-slip / missing config — all graceful DNFs,
  pipeline never fails; zip-slip contained by CPython's zipfile
- ✅ Notebook executed under **pandas 2.2.3** (Colab's major version) — clean
- ✅ Slides: 32 slides screenshot-verified; numbers match the notebook era
- ✅ `collect.py` portal, 24-case live-HTTP suite against a running server: valid
  upload + receipt, wrong PIN → 401, non-zip → 400, zip without `predict.py` → 400,
  26 MB → 413, bad round → 400, unusable team name → 400, zip nested in one
  sub-folder → accepted, re-upload overwrites in place, `/status` + `/board` +
  `/board-data` + `/healthz` (no PIN). Auto-score round 2 from an upload updated the
  live board with the form-typed team name.
- ✅ `discover_teams()` reads the directory it is given (it silently scanned
  `submissions/` root before), display names prefer the collect.py receipt, and a
  failed scoring run no longer republishes the previous round's numbers as fresh
- ✅ Holdout now scores Round 1 *and* Round 2 per team (entries carry a `round` tag)
  — previously it scored one model per team and printed that same 2026 MAE under
  both the "Human-only" and "Human + AI" headings of the reveal table
- ✅ DNF paths re-checked after that refactor: crashing / all-NaN / wrong-length
  `predict()` all become DNF rows, `evaluate.py` still exits 0 and writes results
- ✅ Busy-port handling: an unrelated local service was squatting port 8765; the
  collector now detects it, walks to the next free port and says so
- ✅ Deadline burst: 15 simultaneous uploads → accepted in 0.2 s, **one** scoring
  run (debounced worker, not one per upload), board settled 30 s later with all 17
  teams and no error
- ✅ `collect.py --import` (USB hand-in): round inferred from the filename, team
  name read from the zip's `config.json`, `--team`/`--round` overrides, and the same
  rejections as an upload (409 on a colliding name, 400 on a non-zip, clear error
  on a missing file)
- ✅ `environment.yml` / `environment.lock.yml` re-solved with the portal deps
  (fastapi, uvicorn, python-multipart, qrcode) from conda-forge — torch 2.13.0 +
  xgboost 3.4.2 still import in the same process afterwards
- ✅ **Slide deck, second pass** (`slides/index.html`, headless Chrome):
  - fixed a 720p overflow — the three opening challenge slides (3/4/5) ran 12 px
    past the bottom and clipped their caption. Added a `max-height:800px` block
    that scales the chart to the height actually available. Now **0 overflow** at
    1280×720, 1366×768, 1024×768, 1440×900, 1600×900, 1920×1080, 1512×982
  - per-element geometry sweep of all 32 slides at 720p and 1080p: no element
    extends past its slide. The only two remaining flags are intentional — the
    title-slide ticker marquee (`overflow:hidden`) and `.big`'s `line-height:1`,
    where ascenders/descenders exceed the box but nothing clips
  - 0 JS errors, 32 slides, all carry `data-title`/`data-chip`, 3 charts × 2
    polylines, reveal + counter + hash deep-links + progress bar all working
  - slide 28's illustrative holdout table replaced with the **real** demo numbers
    (predict-0 0.01621/0.01500, human 0.01598/0.01517, human+AI 0.01624/0.01534) so
    the preview matches the board the facilitator actually reveals
  - the "GRU at 10 bps drops to +19.6 %" claim is now backed by a generated file:
    `results/round2_paper_cost10.json` (+19.63 %). `organizer/README.md` §6 pointed
    at `round1_paper_cost10.json`, which is the *forest* (+24.83 %) — fixed
  - the "Run all ≈ 1 minute" claim (slide 20, GUIDE, participant README,
    facilitator script) softened to **1–3 minutes** — a full timed execution on
    this machine is 79 s, and Colab's free v2 CPU runtime is slower
  - notebook header said the predict-0 bar is "≈ 0.015" on 2025; the real pooled
    figure is **0.0162**. Fixed in `build_notebook.py` and the shipped `.ipynb`

- ✅ **Full end-to-end recheck** (third pass — every documented command executed
  verbatim from the repo root, everything re-derived and re-verified):
  - **Bug: `paper_trading.py` resolved `--results` against the CWD.** The
    documented `python organizer/paper_trading.py --results results/round1.json`
    died with a raw `FileNotFoundError` traceback from the repo root, and the bare
    `python organizer/paper_trading.py` (which the auto-detect exists to support)
    reported "No results JSON found" even though `organizer/results/round1.json`
    was right there. Same class as the `leaderboard.py` bug found earlier — both
    now try CWD → script dir → `results/`, so every invocation works from the
    repo root, from inside `organizer/`, and from `/`. A missing file now exits 1
    with a readable message instead of a traceback.
  - **Bug: the zip-bomb guard in `evaluate.py` could kill an entire scoring run.**
    `discover_teams()` runs *before* the per-team `try/except`, so the guard's
    `raise` — and a corrupt zip arriving by USB — propagated out of `main()` and
    wrote no results JSON at all. One bad USB stick would have taken the
    leaderboard down mid-round. Now caught and degraded to a DNF row via
    `_dnf_stub()`.
  - **Bug: the `_round` suffix was split on its first occurrence.**
    `split("_round")[0]` filed a team called "Alpha_Round_Trio" under the name
    "Alpha" — two different teams on one row, plus a false slug collision. Now
    anchored to the end of the name, in both `collect.py --import` and
    `evaluate.py`'s `_display_name`.
  - **Bug: a non-zero-cost paper-trading run silently overwrote the headline
    result.** The 10 bps demo wrote over `round<N>_paper.json` — the number quoted
    on the projector. Non-zero cost now writes its own `_paper_cost10` file.
  - **Precision: the live board and the leaderboard rendered MAE at 4 dp.** Every
    MAE here is ~0.015x, so the whole competition is decided in the fourth decimal
    and 4 dp rendered genuinely different models as identical. Both are 5 dp now.
  - **Stale figures corrected**: `README.md` attributed the per-ticker spread
    (SPY 0.0075 → TSLA 0.0291) to the forest when those are the predict-0
    baseline's numbers; `CHEATSHEET.md` and `GUIDE.md` still quoted the old
    SPY-only val baseline (0.0075 / "identical ~0.0062") after the move to the
    5-ticker panel, which would have sent teams chasing the wrong target. Both
    now read 0.0154 throughout.
  - **Every documented command re-run from the repo root**: `evaluate.py --round
    1` / `--round 2` / `--holdout`, `paper_trading.py` (both rounds), the bare
    `leaderboard.py`, and `build_notebook.py`. All reproduce the canonical numbers
    exactly (0.015983 / 0.016244 / 0.015002) and the notebook re-validates
    (`nbformat.validate`, 40 cells).
  - **Every number in the prose re-checked against `results/*.json`**: 6 MAE
    claims, 4 paper-trading claims, the per-ticker spread and all 6 notebook val
    MAEs — all match to 5 dp, no stale figures left.
  - **Portal re-smoke-tested after the edits**: `/` 200, `/board` 200, wrong PIN
    401, good upload 200 + receipt, non-zip 400. The probe submission was removed
    afterwards and the demo state restored.
  - **Slide deck, third pass** — geometry sweep of all 32 slides at 1920×1080,
    1600×900, 1366×768 and 1280×720 (real content heights 993/813/681/633 px):
    **0 overflowing elements at every resolution**, in both axes, ignoring
    intentionally-clipped subtrees. The `max-height:800px` block is confirmed
    active at 720p (chart `max-height` resolving to 291 px). The title-slide
    marquee is the only element wider than the viewport, and that is by design
    (`overflow:hidden`).
  - **Repo hygiene**: purged 49 stray `._*` / `.DS_Store` / `__pycache__` entries —
    external-SSD AppleDouble sidecars that would otherwise have shipped *inside*
    `participant/`.
  - Private-data boundary re-confirmed: no participant file references
    `organizer/private/`, and the participant CSV ends 2024-12-31 with zero date
    overlap against either private set.

**Open before the workshop (T-1 day):**
- [ ] **Test the portal on the venue network** (§5): phone must load the printed
      URL. If not, set up the hotspot/travel-router fallback *and rehearse it*
- [ ] Re-run `prepare_data.py` + headless notebook execution (regression pass)
- [ ] **Colab smoke-test**: open `participant/workshop.ipynb` on Colab, `Run all`,
      including the export cell — the one surface this machine cannot reach
- [ ] Wipe demo state if desired: `rm -rf organizer/results/*` and
      `rm -f organizer/submissions/round*/*` (keep the `round1/` `round2/` folders)
- [ ] Distribute **only** `participant/` (never `organizer/private/`)

## 9 · Regenerating things

```bash
python build_notebook.py     # rebuild ../participant/workshop.ipynb (after editing cells here)
python prepare_data.py       # refresh data (any day; picks up new market days)
```

Re-pinning the environment after editing `environment.yml` (install from
conda-forge, then re-export and strip the machine-specific `prefix:` line):

```bash
conda install -y -n stocks-ml -c conda-forge <packages>
conda env export -n stocks-ml --no-builds | grep -v '^prefix:' > environment.lock.yml
```

Instruments live in `TICKERS` at the top of `prepare_data.py` — the notebook,
exporter and evaluator are ticker-agnostic (per-ticker features/windows, one-hot
columns follow whatever tickers are present).

### ✅ Slide expansion for a zero-background room (fourth slide pass)

The teaching section was the deck's real gap: the participant guides define
**nothing** in plain language — no glossary, no explanation of what a model is,
what the training loop does, or what *feature* / *label* / *parameter* /
*hyperparameter* mean. "Overfitting" appeared only as a diagnostic
(`train ≪ val = memorising 2014–22 noise`) with no account of *why*. The slides
were therefore saying "regularisation", "gates" and "hyperparameter" to a room
that had never been told what those words mean.

**Deck is now 42 slides** (was 32): 4 "ML from zero" concept slides, 5 per-model
plain-terms companions, and 1 "Which model do I pick?" decision table.

| Section | Slides | Min | Clock |
|---|---|---|---|
| opening | 7 | 6 | 0:00–0:06 |
| fundamentals | 8 (+4) | 13 | 0:06–0:19 |
| crash course | 13 (+6) | 23 | 0:19–0:42 |
| briefing | 2 | 6 | 0:42–0:48 |
| ROUND 1 | 1 | 20 | 0:48–1:08 |
| evaluation | 1 | 10 | 1:08–1:18 |
| ROUND 2 | 4 | 16 | 1:18–1:34 |
| trading | 2 | 8 | 1:34–1:42 |
| debrief | 2 | 8 | 1:42–1:50 |
| conclusion | 2 | 10 | 1:50–2:00 |

Round 1's 20-minute hands-on block is **unchanged** — it is where the learning
actually lands. The tail (trading/debrief) gave up 6 minutes, which had the most
slack. Metaphors reuse what the participant docs already established ("model
garage", "dials/knobs", "brake" for regularisation, "memorise" for overfitting,
"the bar to clear" for baselines). Deliberately *not* used: "recipe", already
taken by `SETUP_VENV.md` for environment setup.

Verified:

- ✅ **Geometry: 0 overflow on all 42 slides at 1920×1080, 1600×900, 1366×768,
  1280×720, plus 1280×700 and 1024×640 stress cases.** The probe was validated
  first — it reports per-slide measured-element counts, and no slide reported
  fewer than 3, so the zero is a real pass and not a harness artefact. It also
  skips subtrees inside an intentional `overflow:hidden` ancestor, which is why
  the title-slide ticker marquee (2639 px wide in a 1920 px viewport) does not
  register as overflow.
- ✅ Tightest slide is now **S23 "LSTM, in plain terms"** (was 22 px of headroom at
  a 633 px viewport). Fixed by compacting `.edge` line-height/label spacing in the
  `@media (max-height:800px)` tier and the padding in the `max-height:660px` tier
  — content preserved, no teaching text cut. Also **merged the two duplicate
  `@media (max-height:800px)` blocks**, which had been silently re-declaring the
  same five chart rules.
- ✅ Every figure on the new "Which model do I pick?" slide re-checked against
  `organizer/results/*.json` (not from memory): forest MAE 0.015983 → +25.71% on
  33 trades → +24.83% at 10 bps; GRU MAE 0.016244 → +27.46% on 322 trades →
  +19.63% at 10 bps. The MAE-best/worst-P&L inversion and the cost-sensitivity
  are the workshop's thesis, so they had to be exact.
- ✅ `node --check` on the deck's extracted inline JS passes; no hardcoded slide
  count anywhere in the engine (the only `32`/`42` matches are chart path
  coordinates), so inserting slides needs no JS edits.
- ✅ Deep-links and reveal re-tested against the live engine: `#1`, `#9`, `#26`,
  `#42` resolve; `#43` and `#0` clamp correctly; `#3r` opens pre-revealed; the
  Reveal button click works on all 3 chart slides. Counter reads `n / 42`.
- ✅ Docs realigned to the new numbering and clocks: `FACILITATOR_GUIDE.md` §3
  rewritten (every S-reference re-mapped, new section clocks, talking points and
  **MUST / IF TIME** markers on the 10 new slides, plus a revised cut-order row in
  the §6 fallback table); the whiteboard deadlines (1:00/1:28 → 1:08/1:34) and the
  surprise-test window; `README.md` and `organizer/README.md` slide counts and the
  `#27r` → `#38r` deep-link example; `participant/README.md` round times and the
  cheatsheet's export deadlines. `WorkshopOutline.md` now carries the live 2-hour
  schedule; the 60-minute competition block runs 0:42–1:42, with 6 minutes for
  briefing, 36 minutes for model tuning, and 18 minutes for scoring and results.

Open items unchanged: Colab smoke-test of the notebook on the real runtime, and a
portal test on the venue's actual network (a phone must load the printed URL).
