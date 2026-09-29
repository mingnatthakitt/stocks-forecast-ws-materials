# 📋 Round-1 Cheatsheet — Can AI Predict the Market?

The one-pager you're allowed to open during **Round 1 (human only)**.
Everything here is also in the slides and the notebook — this is just faster.

---

## The competition in one line

> Train on **2014–22**, validate on **2023–24**, export a zip, organisers score your
> model on **private 2025** — a 5-ticker panel (SPY, NVDA, AAPL, MSFT, TSLA).
> Primary metric: **MAE pooled over all tickers** (lower better).
> The bar: "always predict 0" ≈ **0.0162 pooled** (SPY alone ≈ 0.0075 — NVDA/TSLA are much harder).

## Workflow loop (repeat until time is called)

```
edit FEATURES / params → run train_and_validate(...) → read val MAE + overfit ratio
        ↑                                                    │
        └──────── keep what helped, revert what didn't ◄──────┘
```

Change **one thing at a time** (or two, but log both). One run = seconds, so experiment freely.

---

## Feature menu (what's available / worth trying)

| Try | Because |
|---|---|
| **Prune FEATURES** to 5–8 strong ones | every useless feature is noise the model can overfit |
| day-of-week one-hot | calendar effects |
| keep the `is_<TICKER>` one-hots | NVDA/TSLA need wider error allowances than SPY |
| consecutive up/down day count | streak behaviour |
| Bollinger position `(close − sma20) / (2·std20)` | normalised deviation |
| overnight gap `open/prev_close − 1` | sentiment between sessions |
| distance to 52-week high | anchoring / drawdown |
| `volume_ratio_5` × direction interaction | conviction behind moves |
| rolling skew/kurtosis of returns | tail-risk regime |

**Forbidden:** anything using `shift(-1)`, the target, or 2023–24 outcomes *inside* a
feature. Rolling windows are fine.

---

## Hyperparameter dial guide

### Ridge (`arch="ridge"`)
| Knob | Try | Effect |
|---|---|---|
| `alpha` | 0.1 → 1 → 10 → 100 | bigger = flatter, safer, duller |

### Random Forest (`arch="random_forest"`)
| Knob | Try | Effect |
|---|---|---|
| `n_estimators` | 100–500 | more = stabler, slower |
| `max_depth` | 3 / 6 / 10 | deep = memorises (watch overfit ratio) |
| `min_samples_leaf` | 1 / 5 / 20 | bigger = smoother leaves |
| `max_features` | 0.5 / 0.8 | fewer per split = more diverse trees |

### XGBoost (`arch="xgboost"`)
| Knob | Try | Effect |
|---|---|---|
| `max_depth` | 2–6 | the #1 overfit dial |
| `learning_rate` | 0.01–0.2 | lower needs more `n_estimators` |
| `n_estimators` | 200–800 | with `early_stopping_rounds: 20` it self-limits |
| `subsample` / `colsample_bytree` | 0.7–1.0 | randomness = regularisation |
| `reg_lambda` | 0.5–5 | L2 brake on leaf values |

### LSTM / GRU / CNN (`arch="lstm"|"wildcard"`)
| Knob | Try | Effect |
|---|---|---|
| `seq_len` | 5 / 20 / 60 | how much history per window |
| `hidden_size` / `channels` | 8–64 | capacity — small often wins |
| `dropout` | 0.1–0.4 | the NN regulariser |
| `epochs` | 10–60 | more = more memorisation risk |
| `lr` | 1e-4–3e-3 | too high = diverges, too low = underfits |
| `batch_size` | 32 / 64 / 128 | mostly speed |

---

## Reading your results

| Observation | Diagnosis | Move |
|---|---|---|
| train ≪ val MAE (ratio > 1.3×) | overfitting | shrink model, add regularisation, prune features |
| both terrible | underfitting | more/better features, bigger model, more epochs |
| val MAE ≫ 0.0154 (pooled val baseline) | broken or pure noise | check features are causal; simplify |
| val MAE slightly < baseline | **this is winning territory** | squeeze: features > hyperparameters |
| DirAcc ≈ 50% | no directional edge | normal for daily returns — MAE is the game |
| one run looks amazing | be suspicious | leakage? luck? rerun with a different seed only if time permits |

---

## Rules (both rounds)

**Round 1 — human only.** ✅ slides, cheatsheet, docs, teammates, facilitators.
❌ ChatGPT, Claude, Gemini, Copilot, any generative AI.

**Round 2 — AI unlocked.** Document every AI suggestion: ✓ accepted (why) /
✗ rejected (why). Re-export with `ROUND = 2`.

**Always:** same dataset, same split (locked in the notebook), fixed seeds, no
touching `organizer/` files, no NaN predictions (predict `0.0` for cold-start rows).

---

## Export checklist (the 60 seconds that decide everything)

1. `TEAM_NAME = "…"` — spelled identically both rounds
2. `ROUND = 1` (before 1:08) then `ROUND = 2` (before 1:34)
3. Re-run the export cell **after your final change**
4. Zip contains: `predict.py`, `config.json`, `model.pkl`/`model.pt`, `team_features.py`
5. Upload it at the link + PIN on the whiteboard (phone is fine) and type your team
   name — keep the ✓ receipt, it's your proof of hand-in. Link dead? Use the
   facilitator's hotspot, or hand the zip over on a USB stick.
   **No zip, no leaderboard row.**
