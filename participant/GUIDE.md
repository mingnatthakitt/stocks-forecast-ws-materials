# Participant Guide — Can AI Predict the Market?

A section-by-section walkthrough of `workshop.ipynb`: what each part does, **why**
it exists, and where you're expected to change things. Read this next to the notebook.

> 🟩 `MODIFY THIS SECTION` = your playground · 🔒 `LOCKED` = shared infrastructure —
> if you break a locked cell, your submission zip breaks with it.

---

## 0 · Setup (locked)

Loads the OHLCV panel for 2014–2024 — **SPY, NVDA, AAPL, MSFT, TSLA** — fixes random seeds (fairness — everyone's
Random Forest sees the same randomness), and imports everything.

**Colab flow:** `Runtime → Run all` takes 1–3 minutes (it trains all five models). If the data cell can't find
`workshop_participant.csv`, it will ask you to upload it — grab the file from the workshop
folder.

**One quirk worth knowing:** xgboost is imported *before* torch in the setup cell. On
some local Python installs the reverse order crashes the kernel (two conflicting
OpenMP libraries). Colab doesn't care, but the notebook is defensive for local users.

---

## 1 · Meet the data (locked)

Two plots and a list of statistics. What you should take away:

1. **Price levels are not stationary and not comparable** — NVDA ×100 vs SPY ×3 over
   the window, and raw charts contain split cliffs (NVDA's 10:1 in 2024). The data is
   split-adjusted; returns are the common unit.
2. **Returns hover around zero for everyone** — but with very different widths
   (TSLA's daily σ ≈ 3× SPY's — see the per-ticker table the cell prints).
3. The **autocorrelation cell** is the workshop's opening thesis: the correlation
   between today's return and tomorrow's is nearly zero *for every ticker*. Whatever
   signal exists is small, which is exactly why *beating the baseline is hard*.

---

## 2 · The target + your features 🟩

The heart of your competitive advantage.

**The target (locked):** row *(date, ticker)* predicts the return from that day's
close to that ticker's **next** day's close — computed *within* each ticker, never
across tickers:

```
target[(t, g)] = Close_g[t+1] / Close_g[t] − 1
```

`make_dataset()` glues features and target together and drops rows with missing
values (the first ~50 rows per ticker while rolling windows warm up, plus each
ticker's final row whose "next day" doesn't exist in the data). It returns
`(X, y, tickers)` — the ticker series lets the sequence models build windows per
ticker so no window ever crosses a ticker boundary.

**`add_features()` 🟩 — the starter menu:**

| Family | Features | What they try to capture |
|---|---|---|
| Returns | `return_1d/5d/10d`, `lag_return_1d/2d` | recent performance, momentum |
| Moving averages | `sma_ratio_5_20`, `price_vs_sma50`, `ema_gap_12_26` | short-term vs medium-term trend |
| Momentum/position | `momentum_20`, `dist_high_60`, `dist_low_60` | where price sits in its recent range |
| RSI | `rsi_14` | overbought / oversold pressure |
| Volatility | `volatility_10/20` | how violent recent days were |
| Volume & range | `volume_change_1d`, `volume_ratio_5`, `range_pct` | participation and intraday choppiness |

**How to compete here:**

- **Prune** — `FEATURES` lists what the model sees. Fewer, better features often beat
  the full menu (less noise for the model to memorise).
- **Add** — your own ideas at the marked spot. Ideas: day-of-week one-hot, consecutive
  up-days, Bollinger-band position, overnight gap vs intraday move.
- **Stay causal** — rolling windows and `shift(k)` with `k ≥ 0` only. `shift(-1)` is
  the future. If you use tomorrow's information in a feature, your validation score
  becomes fiction and the private test will expose you.

**The leak-discipline detail people miss:** features must be computable from the OHLCV
the evaluators hand you — they will be, because your exact `add_features()` is exported
into your submission zip. What must NOT differ is what the *feature depends on*: nothing
beyond day *t*'s close *for that ticker*.

**Why the `is_<TICKER>` one-hots are in `FEATURES`:** one model serves five
personalities. SPY's daily σ ≈ 1.1%, TSLA's ≈ 3.6% — the dummies let the model learn
a different error scale (and different patterns) per ticker. They are derived from the
`Ticker` column, so they are causal and travel safely through the export.

---

## 3 · The time-based split 🔒

`TRAIN = 2014–2022`, `VALIDATION = 2023–2024`, `2025 = 🔒 private`.

Why not shuffle-split? A shuffled split lets the model train on 2021 and be tested on
2019 — it effectively sees the future of its own test set. Scores look fantastic and
mean nothing. The split cell is locked precisely so this can't happen by accident.

The chart shows the three eras on one price line. The 2025 test period is evaluated
privately: you see your scores on the leaderboard, but not the market rows or target values.

---

## 4 · Metrics & baselines (locked)

The shared metric function every model and every baseline is judged with:

- **MAE** — mean absolute error of predicted returns. *The competition metric.*
- **RMSE** — same, but punishes big misses (crash days).
- **DirAcc** — did you call the direction right? Coin flip = 50%.

Then three baselines are scored on validation:

| Baseline | What it says |
|---|---|
| predict 0.0 | "tomorrow ≈ today" — on daily returns this is *shockingly* strong (pooled 2023–24 MAE ≈ 0.0154) |
| yesterday's return | naive momentum |
| train mean | the long-run drift |

Internalize the numbers this cell prints: **your model's entire job is to beat the
first row.** Most serious attempts on daily equity data do not.

---

## 5 · The model garage (one uniform API 🔒 + your params 🟩)

Everything funnels through one call:

```python
train_and_validate(arch, params={...}, label="my-label")
```

It trains on 2014–22, evaluates on 2023–24, prints val MAE + the **overfit ratio**
(`val MAE / train MAE`) and registers your run. Best run per architecture is kept for
the export.

| `arch` | Family | Knobs that matter |
|---|---|---|
| `"ridge"` | linear + L2 | `alpha` |
| `"random_forest"` | bagged trees | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| `"xgboost"` | boosted trees | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_lambda`, optional `early_stopping_rounds` |
| `"lstm"` | recurrent net | `seq_len`, `hidden_size`, `num_layers`, `dropout`, `epochs`, `lr`, `batch_size` |
| `"wildcard"` | GRU *or* 1D-CNN (set `WILDCARD_ARCH`) | GRU knobs or CNN knobs (`channels`, `kernel_size`) |

**How to read the overfit ratio** (`validation MAE / training MAE`):

- Around **1×** — train and validation errors are similar; still check their absolute values.
- **Above 1.3×** — validation error is much higher. For example, train MAE 0.003 and
  validation MAE 0.009 gives a 3× ratio. This can indicate overfitting or a market
  regime change. Check the features and try a simpler or more regularised model.
- **Below 1×** — validation error is lower than training error. Different periods and
  training randomness can cause this; it does not by itself mean the model is over-regularised.

**Sequence models (LSTM/GRU/CNN)** receive the last `seq_len` days *as a sequence*
rather than one feature row — a genuinely different hypothesis about what information
matters. Scaling is handled inside (fit on training data only — leakage discipline).

---

## 6 · Scoreboard (locked)

A table of every run plus a bar chart with the baseline drawn as a dashed gold line.
Bars under the line = you learned something; bars above it = you learned noise.

The scatter cell plots predicted vs actual for your best model. Expect a narrow
vertical band of predictions vs a wide spread of reality — that's what forecasting a
near-random-walk honestly looks like. (If the band looks *wide*, be suspicious.)

---

## 7 · 📦 Export (team identity 🟩 + machinery 🔒)

Set `TEAM_NAME` and `ROUND`, run the cell, get `exports/submission_<TEAM>_round<N>.zip`:

```
predict.py          locked interface — reads config.json, rebuilds features, predicts
config.json         arch / features / params / scaler / val MAE snapshot
model.pkl|.pt       your model, RETRAINED on all visible data (2014–2024)
team_features.py    your add_features(), copied verbatim from this notebook
team_models.py      NN classes (only for sequence models)
```

Why retrain on train+validation for the final model? It gives the final fit more
visible observations after you have chosen its settings. More data can help, but
markets change, so it does not guarantee a better forecast. Freeze your hyperparameters
before exporting: that model snapshot is what plays on 2025.

**Checklist before handing in:** correct `ROUND`, correct `TEAM_NAME` (identical both
rounds), export cell re-run *after* your last change, zip contains `predict.py` +
`config.json` + a model file + `team_features.py`.

---

## 8 · 🤖 Round 2 annex

**The auto-filled prompt** — the cell builds your AI prompt from your actual best run
(architecture, features, params, validation MAE vs baseline). Paste your Round-1 2025
score into the marked line before sending. Suggested additions to the prompt if you
want better answers: ask the AI to rank its suggestions by expected impact, and to flag
which of its ideas risk time-series leakage.

**The change log** — fill the ✓/✗ table during Round 2. It stays in the notebook and
is not included in the model ZIP, so keep the notebook available for the debrief.
The workshop's real question is what you accepted, what you rejected, and whether your
judgement worked.

**Reminder:** you now know your Round-1 2025 score. Using it as the *only* signal to
tune against is leaderboard overfitting — the 2026 holdout exists precisely to test
whether your Round-2 gains were real or benchmark-specific.

---

## Fast troubleshooting

| Symptom | Likely cause |
|---|---|
| `xgboost missing` on Colab | `%pip install xgboost` — usually preinstalled |
| Kernel dies when XGBoost trains (local only) | import-order issue; keep the setup cell's xgboost-before-torch order |
| `FEATURES contains columns add_features() does not make` | typo in `FEATURES`, or you removed the feature from the function |
| All val MAEs identical ~0.0154 | that's *correct* — that's the "predict 0" pooled val baseline; daily returns are near-random, so now compete in the fourth decimal |
| Export raises `No trained models` | run the model cells above before exporting |
| Submission evaluates as DNF | your `predict()` returned NaN — rows without enough history must be predicted `0.0` |
