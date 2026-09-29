# 👥 Participant Kit — Can AI Predict the Market?

**This folder is everything you need as a workshop attendee.** Everything else in the
repository belongs to the organisers — you can ignore it.

## What's in here

| File | What it is | When to open it |
|---|---|---|
| `workshop.ipynb` | The starter notebook — your workspace for the whole workshop. All training plumbing is done; you compete by editing clearly-marked 🟩 sections. | From 0:42 (briefing) onward |
| `GUIDE.md` | Section-by-section walkthrough of the notebook, with the *why* behind every part + a troubleshooting table | Skim before Round 1; keep open while you work |
| `CHEATSHEET.md` | One-page quick reference: feature menu, hyperparameter dials, rules, export checklist | **During Round 1 & 2** (it's competition-legal) |
| `SETUP_VENV.md` | Local install guide for VS Code + venv users (macOS caveats included) | Only if not using Colab |
| `requirements.txt` | Pinned packages for the local venv install | Used by SETUP_VENV.md |
| `data/workshop_participant.csv` | The dataset: OHLCV for **SPY, NVDA, AAPL, MSFT, TSLA**, 2014–2024, split-adjusted | Loaded by the notebook automatically |
| `exports/` | Example submission zips produced by the export cell (so you know what "done" looks like) | Reference only |

**Not in this folder (and never will be):** the private 2025 labels your model will be
scored against, and the 2026 holdout. That's the whole point.

## Quick start

### Google Colab (recommended — nothing to install)
1. Go to [colab.research.google.com](https://colab.research.google.com) → `File → Upload notebook` → pick `workshop.ipynb`.
2. Run the first cell. When it asks for the data file, upload `data/workshop_participant.csv`.
3. `Runtime → Run all` (1–3 minutes on the free CPU runtime — it trains all five models). You should see a scoreboard appear.

### Local VS Code + venv (no conda needed)
Follow **[SETUP_VENV.md](SETUP_VENV.md)** — a step-by-step guide for creating a
`.venv`, pointing VS Code at it, and running the notebook locally.

> ⚠️ One honest caveat: on **macOS**, a pip venv cannot run XGBoost alongside torch
> (packaging conflict). The guide explains it — macOS-venv teams compete fully with
> Ridge / Random Forest / LSTM / Wildcard, or switch to Colab for all five models.

## The competition in 60 seconds

- **Task:** predict each ticker's **next-day return**. Primary metric: **MAE**, pooled
  over all five tickers (lower = better).
- **The bar:** "always predict 0" scores ≈ **0.0162** on the private 2025 set. Beating
  that is the whole game.
- **Round 1 (0:48–1:08) — human only.** No ChatGPT/Claude/Gemini/Copilot. Slides,
  cheatsheet, docs, teammates and facilitators are allowed.
- **Round 2 (1:18–1:34) — AI unlocked.** Use AI, but log what it suggested and what
  you actually changed. Re-export with `ROUND = 2`.
- **Hand-in:** run the 📦 export cell → upload the zip it produces at the **submission
  link and PIN on the whiteboard** (a page on the facilitator's laptop; you can use
  your phone). Type your team name exactly as agreed, pick your round, press upload.
  You get a ✓ with a receipt hash — that's your proof of hand-in. You can re-upload
  until the deadline; the last one counts. If the link doesn't load, the Wi-Fi is
  probably blocking device-to-device traffic: ask a facilitator for the hotspot
  password, or hand the zip over on a USB stick. **No zip = no leaderboard row.**

## When you're stuck

1. Check the troubleshooting table at the end of [`GUIDE.md`](GUIDE.md).
2. Re-read the 🟩 banner comments above the cell you're editing.
3. Ask a facilitator — that's what they're floating around for.
