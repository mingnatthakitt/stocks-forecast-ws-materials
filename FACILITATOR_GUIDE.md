# Facilitator Guide — Can AI Predict the Market?

Everything the two workshop leads and the floating facilitators need to run the
2-hour session. Companion docs: `WorkshopOutline.md` (design), `participant/GUIDE.md`
(participants), `participant/CHEATSHEET.md` (in-competition reference).

---

## 1 · Preparation timeline

### T-7 days
- [ ] Confirm venue (projector, team seating, power strips, Wi-Fi credentials visible)
- [ ] Recruit 1 facilitator per 2–3 teams (they debug, explain, and police Round-1 rules)
- [ ] Decide prizes (optional) and the AI-access policy for Round 2 (own laptops' ChatGPT is fine)

### T-1 day
- [ ] `conda env create -f environment.yml && conda activate stocks-ml` (or recreate from `environment.lock.yml` for the exact tested solve)
- [ ] Rerun `python organizer/prepare_data.py` to pull the latest data (SPY, NVDA, AAPL, MSFT, TSLA) → verify the three CSVs print sane date ranges plus the per-ticker table
- [ ] Rebuild + re-execute the notebook headless (fast regression test):
      `python organizer/build_notebook.py && cd participant && jupyter nbconvert --to notebook --execute --inplace workshop.ipynb`
- [ ] Dry-run the full organiser loop (see §4 below) with a dummy submission
- [ ] **Test the submission portal on the venue's actual network** (see §4.1). Campus
      Wi-Fi usually blocks laptop-to-laptop traffic; find out at home-of-the-problem
      today, not at 0:35 with 20 people waiting. If it is blocked, set up your own
      hotspot or travel router now.
- [ ] Open `participant/workshop.ipynb` on Google Colab once and `Run all` — Colab
      is the platform participants actually use, and it is the one surface this
      repo's own tests cannot reach
- [ ] Copy `participant/` (notebook, data CSV, GUIDE, CHEATSHEET) to the distribution channel
      (shared drive / QR code / repo). **Never** the `organizer/private/` folder.

### Day-of, 30 min before
- [ ] Projector: open `slides/index.html` in a browser (F11 fullscreen) **and** a
      second tab with the **live board** `http://<laptop-ip>:8000/board` — it
      refreshes itself, so you never touch the projector mid-round
- [ ] `organizer/submissions/round1/` and `round2/` exist and hold only the demo
      submissions (or wipe them — see §9 of `organizer/README.md`)
- [ ] Start the portal: `python collect.py --pin <4-digit PIN> --qr`, then **open the
      printed URL on your phone** to prove the room can reach it. Write the URL and PIN
      on the whiteboard before anyone asks.
- [ ] If macOS asks whether to accept incoming connections for Python → **Allow**
- [ ] USB-C stick as the fallback — teams hand you their zip, you file it with
      `python collect.py --import /Volumes/<STICK>/<their>.zip`
- [ ] Facilitators have: this guide, the evaluation command list (§4), the cheatsheet
- [ ] Write on the whiteboard: round deadlines (1:08 / 1:34), the portal URL + PIN, and
      the team/model draft board
- [ ] Sanctioned Colab fallback link ready in case local Python fails for someone

---

## 2 · Environment (organiser machine)

All commands assume `conda activate stocks-ml`. Why a dedicated env: torch (conda-forge)
and xgboost must share one OpenMP runtime — mixed conda/pip installs segfault the kernel.
`environment.yml` is the spec, `environment.lock.yml` the pinned solve, `requirements.txt`
the pip alternative for venv users.

The notebook must be *executed by participants* on Colab or the `stocks-ml` kernel.
Colab has everything preinstalled; the notebook's setup cell defensively keeps
xgboost-imported-before-torch for local users.

---

## 3 · Run of show (120 minutes)

> Slide numbers refer to `slides/index.html` (1–42). Split sections between the two
> leads; the non-speaking lead runs the leaderboard/eval pipeline.
>
> **MUST** = the room is lost without it. **IF TIME** = cut these first if you're
> running late; the slide stays in the deck, you just talk over it faster or skip
> to the next. The five `· in plain terms` slides and `Which model do I pick?`
> are the first cuts in a slow room — but see §6: if the room has *no* ML
> background at all, they are the first slides you must **not** cut.

### 0:00–0:06 · Opening challenge — slides 1–7
- **S1–2** Welcome; the two-round contract (human-only, then AI unlocked). *Point at the
  room: "you are the control group."*
- **S3–5** The three vote charts. Run each: show chart → hands up UP vs DOWN →
  press **R** to reveal. Real windows: Q4-2018 crash→recovery (UP +13%), COVID 2020
  (UP after the hidden point), 2023 AI rally (UP). Keep the tempo fast and loud.
- **S6** Debrief: humans are coin-flips here; mention that professionals fail too.
- **S7** State the central question and today's experiment design.

### 0:06–0:19 · Fundamentals — slides 8–15  *(13 min — the biggest block in the first half)*
- **S8** Prices vs returns and the target formula. *Why:* levels drift, returns compare
  across decades and tickers; mention NVDA's raw 10:1 split cliff — adjusted returns
  are the honest, comparable framing. Panel: SPY + NVDA + AAPL + MSFT + TSLA.
- **S9 · MUST** *What a model actually is.* The first time the room hears "feature",
  "label" and "prediction" — define all three in the workshop's own terms
  (`rsi_14` → tomorrow's return). No jargon, no maths. If you skip this, every
  later slide is unreadable to a beginner.
- **S10 · MUST** *How it learns.* Guess → check → adjust, ~10,000 times. The
  dartboard-in-the-dark line does the work: nobody writes the rules, the rules fall
  out of repeated error correction.
- **S11 · MUST** *Memorise vs generalise.* Overfitting from the **why** side, not the
  symptom side — the student who memorised the textbook vs the one who learned it.
  Land the train-vs-val gap now; you'll point at it again on every model slide.
- **S12 · MUST** *Dials vs gears.* Parameter (a number inside the model, learned) vs
  hyperparameter (a dial *you* set, never learned). Define epoch / batch size /
  learning rate / layer here, so the model slides' knob lists are readable.
- **S13** The pipeline. One sentence per box.
- **S14** The timeline — the most important slide of the first half hour. Drill:
  train/validate/private, "2025 labels do not exist for you."
- **S15** Baselines. Land the punchline: **predict-0 is the bar** (≈ 0.0162 pooled on
  2025, ≈ 0.0154 on validation — say the numbers out loud). Mention the difficulty
  gradient: SPY ≈ 0.007, TSLA ≈ 0.029 per-ticker MAE for the same model.

### 0:19–0:42 · Crash course — slides 16–28  *(23 min)*
Don't lecture. Give each model its *personality* (the S-numbered slide), then its
plain-terms translation immediately after. The pattern is always the same three
questions: **what it does → what it cannot express → when to reach for it.**

- **S16** Ridge — the underdog; only as good as its features.
- **S17 · MUST for beginners** *Ridge, in plain terms.* One straight line multiplied
  up by twelve; the "brake" is alpha. Say the honest limit out loud: it can only draw
  a straight line through the points.
- **S18** Random Forest — bagging = averaging independent overfitters.
- **S19 · MUST for beginners** *Forest, in plain terms.* One tree = a flowchart;
  the averaging cancels the individual mistakes. The "each tree sees one day's
  snapshot, like a photograph" line explains the LSTM contrast later.
- **S20** XGBoost — boosting = sequential error-correcting; overfits eagerly.
- **S21 · MUST for beginners** *XGBoost, in plain terms.* A relay race of
  specialists, each correcting the last. "Don't start here" — it overfits eagerly
  and beginners lose an hour to it.
- **S22** LSTM — sequences in, memory, slow, moody.
- **S23 · MUST for beginners** *LSTM, in plain terms.* The sliding window drawn,
  and the three gates spelled out as three yes/no questions per day
  (forget? / write? / use?). Include the memorisation warning — it is the easiest
  model here to memorise with, and that is the honest limit.
- **S24** Wildcard GRU/CNN — one line switches them; capped in the draft.
- **S25 · IF TIME** *GRU vs CNN, in plain terms.* Two gates vs three; the CNN's
  stencil sliding across the window. Fine to compress to one sentence each.
- **S26 · IF TIME (but it's the thesis)** *Which model do I pick?* The decision
  table. Carries the workshop's punchline and it lands **better** here than at the
  end: the GRU has the *worst* MAE (0.01624) yet paper-trades best (+27.5%), while
  the forest has the *best* MAE (0.01598) and makes less (+25.7%). At a realistic
  10bps cost the GRU collapses to +19.6% and the forest holds at +24.8%. If you cut
  one slide in a beginner room, cut S25, not this.
- **S27** Features — "competitions are won here." Commandments: causal only.
- **S28** Leakage — the shuffled-split horror story. *Tell them the notebook's split is
  locked for a reason; if AI suggests shuffling in Round 2, that's a rejection.*

### 0:42–0:48 · Briefing + notebook tour — slides 29–30
- **S29** Teams of 3–5, primary + secondary architecture, draft caps on the whiteboard
  (Ridge 2 / Forest 2 / XGB 3 / LSTM 3 / Wildcard 2). Fill the draft board live.
- **S30** Notebook tour: green banners = playground, locked = infra, the one API call,
  "Run all trains all five models in 1–3 minutes". Teams open the notebook now.

### 0:48–1:08 · ROUND 1 — slide 31
- Kick the timer visible. Facilitators circulate; enforce no-AI.
- Nudge loops worth broadcasting at ~0:53 and ~1:00:
  - "Have you beaten the baseline yet?"
  - "Look at your overfit ratio — what is it telling you?"
  - "Change one thing, rerun, compare."
- **1:08 hard stop:** export cell with `ROUND = 1`, hand zips to facilitators.

### 1:08–1:18 · Private 2025 evaluation — slide 32 + live leaderboard
- **S32** Metrics explainer while organisers run the eval (§4).
- Reveal Round-1 leaderboard. Read the top 3 aloud. *Expected shape:* most teams
  cluster at 0.0070–0.0078; somebody may not beat the baseline — say "normal, and
  educational" out loud.

### 1:18–1:34 · ROUND 2 — slides 33–36
- **S33** AI unlocked rules: prompt cell auto-fills; paste your Round-1 2025 score in.
- **S34** The prompt template; recommend asking the AI for ranked, reasoned suggestions.
- **S35** Change log is mandatory — the debrief feeds on it.
- **S36** Benchmark overfitting warning. Frame: "you now know your 2025 score; tuning
  against it is allowed but will be tested against something it never saw."
- **1:34 hard stop:** export with `ROUND = 2`.

### 1:34–1:42 · Second evaluation + paper trading + holdout — slides 37–38
- Run eval round 2 → refresh leaderboard (Δ column tells the story per team).
- Run paper trading on both rounds' predictions → leaderboard shows the portfolio table.
- **S37** Accuracy ≠ profit — this is where S26's thesis gets cashed out.
- **S38** The 2026 holdout reveal — run
  `python organizer/evaluate.py --holdout` live if time allows (it's fast); the
  leaderboard's holdout table shows whether Round-2 gains transferred.

### 1:42–1:50 · Debrief — slides 39–40
- **S39** Discussion prompts; have the winning team walk their change log.
- **S40** The eight takeaways — don't read them; ask the room to guess each one first.

### 1:50–2:00 · Conclusion — slides 41–42
- **S41** The verdict: patterns findable, persistence unprovable, certainty the enemy.
- **S42** Winner announcement, thanks, disclaimer (read the disclaimer verbatim).

---

## 4 · Organiser command sequence (the only commands you need)

**Start the portal before the room arrives — one command covers collection,
scoring and the scoreboard:**

```bash
conda activate stocks-ml
cd organizer

# 0) (T-1 day) build datasets
python prepare_data.py

# 0b) (day-of, 30 min before) the submission portal — leave it running all session
python collect.py --pin 4242 --qr
#    → banner prints the URL to give the room; uploads land in
#      submissions/round<N>/ and each round is re-scored automatically
#    → projector tab: http://<your-LAN-IP>:8000/board  (refreshes itself)
```

Everything below is the **manual fallback** — use it when the portal is not
available, or when you want to control scoring timing yourself:

```bash
# 1) after collecting zips/folders into submissions/round1/   → Round 1
python evaluate.py --round 1
python paper_trading.py --results results/round1.json
python leaderboard.py
#    → switch projector to leaderboard.html, F5

# 2) after Round 2 zips land into submissions/round2/         → Round 2
python evaluate.py --round 2
python paper_trading.py --results results/round2.json
python leaderboard.py

# 3) the surprise test (run live at 1:34–1:42 or precompute secretly)
python evaluate.py --holdout
python leaderboard.py
```

`evaluate.py`, `paper_trading.py` and `leaderboard.py` all default to the obvious
paths, so the bare commands above are correct — the flags exist for scoring
something unusual.

### 4.1 · Network: getting the room onto your laptop

Run this test on the venue's real network **the day before**, not during the event.

Campus Wi-Fi (portal login, eduroam, 802.1X) authenticates devices *to the network*.
It does not imply devices may talk *to each other*, and most campus WLANs enable
AP/client isolation to prevent exactly that. So "we're all on the same Wi-Fi" does
not guarantee anyone can load `http://<your-ip>:8000/`.

1. Laptop and phone both join the venue Wi-Fi, both log in.
2. `python collect.py --pin 4242`, then open the printed URL **on the phone**.
3. Loads → fine. Post the URL + PIN on the whiteboard, and put `--qr` on the
   projector slide so people can scan rather than type an IP.

If the phone cannot load it, open `http://<laptop-ip>:8000/healthz` on the phone:
that endpoint needs no PIN and exists precisely to separate "the collector is down"
from "this network blocks me". Then, in order of preference:

1. **Your own hotspot.** macOS System Settings → General → Sharing → Internet
   Sharing; share Wi-Fi over Wi-Fi, give it an obvious name, announce the password at
   the briefing. Private network, no client isolation, survives venue Wi-Fi dying
   mid-round. Nothing leaves your laptop.
2. **A travel router in AP mode**, if you own one — same idea with more capacity.
3. **USB stick round-robin** — teams hand you the zip and you run
   `python collect.py --import /Volumes/<STICK>/<their>.zip`, which files it with
   the same checks an upload gets (round and team name read from the filename and
   the zip itself). Then `python evaluate.py --round <N>`. Boring, and it works.

Notes
- **What is live:** the portal's `/board` page updates itself. `results/leaderboard.html`
  does **not** — it is a snapshot until you re-run `leaderboard.py` and press F5, or
  render it with `--live`. Project `/board` during the rounds; use `leaderboard.html`
  for the holdout reveal.
- **Deadline timing:** uploads are coalesced into a single scoring run, so a rush of
  15 teams at 1:00 costs one re-score (~30 s here), not 15. Budget 20–60 s after the
  last upload before the board is final, and say so out loud.
- `evaluate.py` scores each team in an isolated subprocess with a 120 s timeout;
  crashes, NaN predictions, hung and even hard-crashed workers all become DNF rows,
  never pipeline failures. Full round-1 scoring takes seconds per team.
- Teams may submit either a folder or a zip; both are accepted (Finder "Compress"
  zips with `__MACOSX` noise are handled, as is a zip whose files sit in one
  sub-folder).
- **Round discipline:** the portal files uploads into `round1/` and `round2/`
  separately, so an AI-assisted Round-2 model cannot leak into the human round.
  With the manual flow, move Round-1 zips out of the way before Round 2 starts.
- Before showing the leaderboard, scan the eval output for `!!` lines — a team whose
  zip could not be read is skipped with `!! … skipped` and would otherwise vanish
  silently.
- Leaderboard display names come from the team name typed on the upload form, then
  the filename, then the zip's `config.json` — so spaces in names are safe and the
  Round-1↔Round-2 Δ join works.
- `evaluate.py --holdout` uses 2026 YTD data — run it *only* at the reveal.
- Results JSONs contain predictions (never labels); `organizer/private/` never leaves
  the organiser machine.

### Reference numbers from the prepared run (2026-09-29 data)
Useful calibration for what "normal" looks like — your numbers will shift if you
regenerate data after new market days:

| Model | Val MAE 2023–24 (pooled) | 2025 MAE (pooled) | 2026 holdout MAE |
|---|---|---|---|
| predict 0 (TeamZero) | 0.01543 | 0.01621 | 0.01500 |
| Ridge starter | 0.01537 | — | — |
| Random Forest starter | 0.01531 | 0.01598 | 0.01517 |
| XGBoost starter | 0.01538 | — | — |
| LSTM starter | 0.01548 | — | — |
| GRU starter | 0.01557 | 0.01624 | 0.01534 |

`--holdout` scores every model — Round 1 and Round 2 together — and tags each entry
with its round, so the reveal table fills both the *Human-only* and *Human + AI*
column groups. Every model loses to predict-0 on 2026 (0.01500); that is the point
of the reveal.

Per-ticker 2025 MAE for the forest: SPY 0.0074 · MSFT 0.0101 · AAPL 0.0126 ·
NVDA 0.0211 · TSLA 0.0287 — quote the gradient to explain why pooled MAE looks "big".

Paper trading (equal-weight $2k per ticker, threshold 0, zero cost): forest
**+25.7%** (33 trades) vs buy-and-hold **+19.2%** — and the GRU, which had the
*worst* MAE, made **+27.5%** on **322 trades**. Two debrief goldmines:
- **Accuracy ≠ profit cuts both ways:** the worst-MAE model printed the best paper
  P&L — riding a bull stretch is not the same as having an edge.
- **Costs eat churn:** re-run with `--cost-bps 10` and the GRU's +27.5% drops to
  **+19.6%** (7.8 points of its return paid to transaction costs) while the
  forest's 33 trades barely feel it. Then note the 2026 holdout: the forest
  *loses* to the zero baseline there (0.01517 vs 0.01500) — paper profits in one
  year are not generalisation.

---

## 5 · Facilitator FAQ (participant bugs you'll see)

| They say… | It's usually… | Fix |
|---|---|---|
| "My MAE is 0.0000" | feature leaks the target (shift(-1) somewhere) | hunt the feature; explain leakage |
| "Colab can't find the CSV" | file not uploaded | upload `workshop_participant.csv` via the prompt cell |
| "My MAE is ~0.015 and so is everyone's" | correct — pooled over 5 tickers | compare on the 3rd decimal; read the evaluator's per-ticker breakdown |
| "val MAE identical every run" | they're re-running the same cell / caching confusion | restart kernel & run all; check params actually changed |
| "LSTM is worse than Ridge" | correct on daily data | celebrate — that's the lesson, then tune `seq_len`/`epochs` |
| "Can I use walk-forward CV?" | ambition | yes — advanced option, keep the locked split for the final check |
| "predict() returns NaN" | cold-start rows | rows before warm-up must predict `0.0` (predict.py handles gaps if features do) |
| "Can we blend models?" | ensembling | allowed if both models are yours and predict.py stays self-contained |
| "Can we download other data?" | external features | **no** — same-dataset fairness; engineered features from OHLCV only |
| "I'm on a local venv and XGBoost won't install" | macOS pip conflict (torch/xgboost) | expected — they compete with ridge/forest/lstm/wildcard; the notebook explains it. Adjust the draft caps so XGBoost slots go to Colab teams, and remember the workshop is still fair: all teams use the same data and metric |

Round-1 policing: participants closing a ChatGPT tab when you walk by is the norm —
a friendly "this is the human round" suffices. Log nothing, shame no one.

---

## 6 · Fallback plans

| Failure | Plan B |
|---|---|
| Venue Wi-Fi dies | notebook runs fully offline; submissions go on a **USB stick**, filed with `collect.py --import` (§4.1) |
| Room can't reach your laptop (client isolation) | hotspot or travel router (§4.1) — rehearse this at T-1, it is not fixable in the room |
| Portal dies mid-round | USB stick + the manual commands; nothing else changes |
| Someone's laptop dies | pair them into another team; 4 is still fine |
| Evaluation crashes for one team | it becomes a DNF row; the workshop continues — fix after |
| Colab quota / slow runtime | CPU-only is enough; if desperate, Ridge/Forest/XGB cells alone are a complete workshop |
| Odd team count / draft collision | wildcard caps are flexible; two teams may share an arch with different feature sets |
| Running 10 min late | Cut in this order: **S25** (GRU vs CNN) → **S13** (the pipeline, one sentence per box) → the holdout live-run (precompute it). **Never cut the debrief.** If the room has no ML background, cut nothing in the 0:06–0:19 block — that's the block doing the teaching. |

---

## 7 · After the workshop

- [ ] Archive `organizer/results/` (leaderboard.html + JSONs) and the submissions folder
- [ ] Send teams their change logs back with the holdout numbers — the follow-up email writes itself
- [ ] Note which models won; adjust next cohort's draft caps (e.g. if XGBoost sweeps, cap it at 2)
- [ ] If 2026 data has grown by the next run, `prepare_data.py` regenerates everything
