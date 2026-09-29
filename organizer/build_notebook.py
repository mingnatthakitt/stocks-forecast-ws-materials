#!/usr/bin/env python3
"""
build_notebook.py — generates participant/workshop.ipynb (the teams' starter notebook).

The .ipynb is the distributed artefact; edit cells either here (rerun this script
to regenerate) or directly in Jupyter. This builder exists so that:

  * the locked predict.py template embedded in the notebook is ALWAYS byte-identical
    to submission_template/predict.py (it is read from that file at build time), and
  * the notebook source lives in reviewable Python instead of hand-written JSON.

    python3 build_notebook.py            # writes ../participant/workshop.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PREDICT_PY_PATH = PROJECT_ROOT / "submission_template" / "predict.py"
OUT_PATH = PROJECT_ROOT / "participant" / "workshop.ipynb"

# The neural-net source lives here ONCE: it fills both the notebook cell and the
# exported team_models.py template. (inspect.getsource() works for functions in
# Jupyter but NOT for classes — __main__ has no __file__ — so the export cell must
# embed this as a literal template instead of introspecting.)
NN_CLASSES_SRC = """class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):                      # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])        # state after the most recent day


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class CNN1DRegressor(nn.Module):
    \"\"\"Reads the window with local conv filters — pattern-spotter, not memoriser.\"\"\"
    def __init__(self, n_features: int, channels: int = 32, kernel_size: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):                      # (batch, seq_len, n_features)
        z = self.conv(x.permute(0, 2, 1))      # -> (batch, channels, 1)
        return self.head(z.flatten(1))


def build_torch_model(arch: str, params: dict, n_features: int) -> nn.Module:
    torch.manual_seed(SEED)
    if arch == "lstm":
        return LSTMRegressor(n_features, params.get("hidden_size", 32),
                             params.get("num_layers", 1), params.get("dropout", 0.2))
    if arch == "gru":
        return GRURegressor(n_features, params.get("hidden_size", 32),
                            params.get("num_layers", 1), params.get("dropout", 0.2))
    if arch == "cnn":
        return CNN1DRegressor(n_features, params.get("channels", 32),
                              params.get("kernel_size", 5), params.get("dropout", 0.2))
    raise ValueError(f"unknown sequence architecture: {arch}")
"""

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ===========================================================================
# SECTION 0 — Setup
# ===========================================================================
md("""# 📈 Can AI Predict the Market? — Team Notebook

**Your mission:** build a model that predicts the **next-day return** for a panel of
five instruments — **SPY, NVDA, AAPL, MSFT, TSLA** — survive the private 2025 test
set, and find out whether AI actually makes you better.

| Period | What it is | Who sees it |
|---|---|---|
| **2014–2022** | training | you |
| **2023–2024** | validation / tuning | you |
| **2025** | 🔒 private competition set | organisers only |

**Competition rules (short version)**
- **Round 1 — human only.** No ChatGPT / Claude / Gemini / Copilot. Slides, cheatsheet,
  docs, teammates and facilitators are allowed.
- **Round 2 — AI unlocked.** Use AI however you like, but log what it suggested and
  what you actually changed (last section of this notebook).
- Primary metric: **MAE** of next-day returns, pooled over all five tickers
  (lower = better). Directional accuracy is shown as a diagnostic. The bar:
  *always predict 0.0* scores MAE ≈ **0.0162** pooled on 2025 (SPY alone ≈ 0.0075 —
  NVDA and TSLA are much harder).

**How to read the banners in the code**

| Banner | Meaning |
|---|---|
| 🟩 `==== MODIFY THIS SECTION ====` | your playground — this is where you compete |
| 🔒 `==== LOCKED ====` | shared infrastructure — changing it can break your submission |

*This is an educational exercise. Historical data, virtual money, not investment advice.*
""")

code("""# 🔒 ==== LOCKED: setup ====
# NOTE: xgboost is imported BEFORE torch on purpose — on some local Python installs
# the reverse order crashes the kernel (duplicate OpenMP runtimes). Keep this order.
import sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.precision", 6)

SEED = 42                      # fixed for fairness — do not change
np.random.seed(SEED)

try:
    import xgboost
    print("xgboost", xgboost.__version__)
except ImportError:
    print("!! xgboost missing — run: %pip install xgboost")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    print("torch", torch.__version__)
except ImportError:
    print("!! torch missing — LSTM/GRU/CNN sections will not work (pip install torch)")

import sklearn
print("pandas", pd.__version__, "| numpy", np.__version__, "| sklearn", sklearn.__version__)
print("setup OK")""")

code("""# 🔒 ==== LOCKED: load the participant dataset (OHLCV panel, 2014–2024) ====
DATA_URL = ""   # organisers may paste a direct-download link here

def load_participant_data() -> pd.DataFrame:
    candidates = [
        Path("data/workshop_participant.csv"),        # repo layout
        Path("participant/data/workshop_participant.csv"),
        Path("/content/workshop_participant.csv"),    # Colab upload
        Path("workshop_participant.csv"),             # same folder as notebook
    ]
    for c in candidates:
        if c.exists():
            print(f"loaded {c}")
            return pd.read_csv(c, index_col="Date", parse_dates=["Date"])
    if DATA_URL:
        print("downloading from DATA_URL ...")
        return pd.read_csv(DATA_URL, index_col="Date", parse_dates=["Date"])
    try:  # last resort on Colab: ask for the file
        from google.colab import files
        print("Select workshop_participant.csv to upload ...")
        up = files.upload()
        return pd.read_csv(next(iter(up)), index_col="Date", parse_dates=["Date"])
    except ImportError:
        raise FileNotFoundError(
            "workshop_participant.csv not found. Download it from the workshop folder and put "
            "it next to this notebook (or in data/)."
        )""")

code("""RAW = load_participant_data()

expected = {"Ticker", "Open", "High", "Low", "Close", "Volume"}
missing = expected - set(RAW.columns)
assert not missing, f"data file is missing columns: {missing}"
assert RAW.index.is_monotonic_increasing, "dates must be sorted"
TICKERS = sorted(RAW["Ticker"].unique())
print(f"{len(RAW)} rows · {len(TICKERS)} tickers {TICKERS} · "
      f"{RAW.index[0].date()} .. {RAW.index[-1].date()}")
RAW.head()""")

# ===========================================================================
# SECTION 1 — Explore
# ===========================================================================
md("""## 1 · Meet the data

One panel, five instruments — **SPY** (the S&P 500 ETF), plus four names everyone
knows: **NVDA, AAPL, MSFT, TSLA**. Prices are split/dividend-adjusted (NVDA's raw
chart contains a 10:1 cliff in 2024 — unadjusted "returns" there would be fiction).

Two things to notice:

1. **Price levels** trend and diverge wildly (NVDA ×100, TSLA ×20, SPY ×3…) — the
   *level* is not stationary and not comparable across tickers.
2. **Returns** jitter around zero for everyone — comparable across tickers and
   decades. This is what we model.

> 💡 Why returns and not prices? A model trained on 2014 price levels is useless at
> 2024 levels, and a single model across five tickers needs a common unit. Returns
> are that unit.
""")

code("""# 🔒 ==== LOCKED: first look ====
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for t, g in RAW.groupby("Ticker"):
    g = g.sort_index()
    axes[0].plot(g.index, g["Close"] / g["Close"].iloc[0] * 100.0, lw=0.9, label=t)
axes[0].set_title("price, normalised to 100 at the start of 2014")
axes[0].legend(fontsize=8)

rets_all = RAW.groupby("Ticker")["Close"].transform(lambda s: s / s.shift(1) - 1.0)
axes[1].hist(rets_all.dropna(), bins=160, color="#2dd4a7")
axes[1].set_title("distribution of daily returns (all tickers)")
axes[1].axvline(0, color="#ff5d73", lw=0.8)
for ax in axes:
    ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

r = RAW.assign(ret=rets_all).dropna(subset=["ret"])
summary = r.groupby("Ticker")["ret"].agg(
    mean="mean", std="std", up_share=lambda x: (x > 0).mean(), zero_MAE=lambda x: x.abs().mean())
display(summary)
print("zero-MAE = the MAE of just predicting 0.0 for that ticker.")""")

code("""# 🔒 ==== LOCKED: the hard truth about predictability ====
# Autocorrelation of daily returns: correlation between today's return and the
# return `lag` days later. If markets were easily predictable, these would be big.
print("return autocorrelation by lag:")
for lag in (1, 2, 3, 5, 10):
    ac = RAW.groupby("Ticker")["Close"].apply(
        lambda s: s.pct_change().dropna().autocorr(lag))
    print(f"  lag {lag:>2}: " + "  ".join(f"{t} {v:+.4f}" for t, v in ac.items()))
print("\\n~0 everywhere means yesterday's move tells you almost nothing about")
print("tomorrow — for calm SPY and wild TSLA alike. Whatever signal exists is")
print("small; that's the challenge.")""")

# ===========================================================================
# SECTION 2 — Target & features
# ===========================================================================
md("""## 2 · The prediction task + your features

**Target:** the *next-day return*
$R_{t+1} = (P_{t+1} - P_t) / P_t$, computed from closes **within each ticker**.
Row *(date, ticker)* holds: features built from information available **up to that
day's close**, and the target **that close → next day's close**.

> ⚠️ **The two rules of time-series ML:** row *(t, ticker)*'s features may only
> contain information that existed at *t*'s close **for that ticker** (no peeking at
> *t+1*, no borrowing another ticker's future). Anything else is leakage — it
> inflates validation scores and then dies on the private test set.

### 🟩 The two knobs you will spend your time on
1. **`FEATURES`** — which columns the model sees (edit the list below).
2. **`add_features()`** — how columns are built (add your own indicators here).
""")

code("""# 🟩 ==== MODIFY THIS SECTION: feature engineering ====
# Starter menu — computed PER TICKER, and only from information up to day t's close.
# Add your own indicators at the marked spot. Keep every computation causal (rolling
# windows that end at t, shifts, diffs). NEVER use shift(-1) or the target itself.
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"df: long OHLCV panel (Date index + Ticker column). Returns the same rows
    plus feature columns. Early rows may contain NaN (warm-up of rolling windows).\"\"\"
    out = df.sort_index(kind="stable").copy()

    # Per-ticker computation, explicitly (pandas ≥3 excludes grouping columns from
    # groupby.apply, so a plain loop over tickers is the robust way).
    pieces = []
    for t in out["Ticker"].unique():
        g = out[out["Ticker"] == t].copy()
        c, v = g["Close"], g["Volume"]

        # --- returns & lags -------------------------------------------------
        g["return_1d"] = c / c.shift(1) - 1.0
        g["return_5d"] = c / c.shift(5) - 1.0
        g["return_10d"] = c / c.shift(10) - 1.0
        g["lag_return_1d"] = g["return_1d"].shift(1)
        g["lag_return_2d"] = g["return_1d"].shift(2)

        # --- moving averages --------------------------------------------------
        sma5, sma20, sma50 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(50).mean()
        g["sma_ratio_5_20"] = sma5 / sma20 - 1.0          # short-term vs medium-term trend
        g["price_vs_sma50"] = c / sma50 - 1.0             # where price sits vs its 50d anchor
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        g["ema_gap_12_26"] = ema12 / ema26 - 1.0          # MACD-flavoured momentum

        # --- momentum & position ----------------------------------------------
        g["momentum_20"] = c / c.shift(20) - 1.0
        hi60, lo60 = c.rolling(60).max(), c.rolling(60).min()
        g["dist_high_60"] = c / hi60 - 1.0                # pullback from recent top
        g["dist_low_60"] = c / lo60 - 1.0

        # --- RSI (14) ------------------------------------------------------
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0.0, 1e-10)
        g["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / loss)

        # --- volatility -------------------------------------------------------
        g["volatility_10"] = g["return_1d"].rolling(10).std()
        g["volatility_20"] = g["return_1d"].rolling(20).std()

        # --- volume & intraday range -----------------------------------------
        g["volume_change_1d"] = v / v.shift(1) - 1.0
        g["volume_ratio_5"] = v / v.rolling(5).mean() - 1.0
        g["range_pct"] = (g["High"] - g["Low"]) / c       # intraday choppiness

        # --- 🟩 your features here --------------------------------------------
        # Ideas: day-of-week one-hot, distance to 52w high, consecutive up-days,
        # Bollinger position, overnight vs intraday return split, ...
        # g["my_feature"] = ...
        pieces.append(g)

    out = pd.concat(pieces).sort_index(kind="stable")

    # --- ticker identity one-hots (helps one model serve five personalities) ---
    for t in sorted(out["Ticker"].unique()):
        out[f"is_{t}"] = (out["Ticker"] == t).astype(float)
    return out""")

code("""# 🟩 ==== MODIFY THIS SECTION: choose the feature set ====
FEATURES = [
    "return_1d", "return_5d", "lag_return_1d",
    "sma_ratio_5_20", "ema_gap_12_26",
    "rsi_14", "volatility_20",
    "volume_ratio_5", "range_pct", "momentum_20",
    "is_NVDA", "is_AAPL", "is_MSFT", "is_TSLA", "is_SPY",
]

# 🔒 ==== LOCKED: build the modellable dataset ====
def make_dataset(raw: pd.DataFrame, features: list[str] | None = None):
    \"\"\"Returns (X, y, tickers): feature rows, next-day-return targets, and the
    ticker of each row — aligned, with warm-up rows (incomplete features) and
    rows without a next-day target removed.\"\"\"
    if features is None:
        features = FEATURES          # read at CALL time, so edits apply immediately
    feats = add_features(raw)
    target = feats.groupby("Ticker")["Close"].transform(lambda s: s.shift(-1) / s - 1.0)
    missing = [f for f in features if f not in feats.columns]
    if missing:
        raise ValueError(f"FEATURES contains columns add_features() does not make: {missing}")
    data = pd.concat(
        [feats[features].astype(float), feats["Ticker"], target.rename("target")],
        axis=1,
    ).dropna()
    return data[features], data["target"], data["Ticker"]

X_all, y_all, tick_all = make_dataset(RAW)
print(f"{len(X_all)} usable rows ({X_all.index[0].date()} .. {X_all.index[-1].date()}), "
      f"{X_all.shape[1]} features, {tick_all.nunique()} tickers")
X_all.tail(3)""")

# ===========================================================================
# SECTION 3 — Split
# ===========================================================================
md("""## 3 · The time-based split 🔒

We **train on 2014–2022** and **validate on 2023–2024** — the same eras for **every
ticker** in the panel. The 2025 private set is untouchable — it plays the role of
"the future".

> 🚫 **Never** use `train_test_split` with shuffling on this data. A shuffled split
> lets 2019 rows sit next to 2021 rows; the model memorises the market regime and
> validation looks brilliant until real tomorrow arrives. (Try it during Round 2 with
> AI and watch the validation MAE halve — and mean nothing.)

Walk-forward validation (retraining on expanding windows) is the fancier cousin of
this single split — ask a facilitator if curious.
""")

code("""# 🔒 ==== LOCKED: chronological split ====
TRAIN_END = "2022-12-31"     # training  = 2014 .. 2022-12-31
VAL_END = "2024-12-31"       # validation = 2023-01-01 .. 2024-12-31

def time_split(X, y, tickers):
    tr = X.index <= TRAIN_END
    va = (X.index > TRAIN_END) & (X.index <= VAL_END)
    return X[tr], X[va], y[tr], y[va]

fig, ax = plt.subplots(figsize=(12, 2.4))
spy = RAW[RAW["Ticker"] == "SPY"].sort_index()
ax.plot(spy.index, spy["Close"], lw=0.8, color="#7d8b9b", label="SPY (proxy for the panel)")
ax.axvspan(RAW.index[0], pd.Timestamp(TRAIN_END), color="#2dd4a7", alpha=0.15, label="train 2014–22")
ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(VAL_END), color="#f5c451", alpha=0.18, label="val 2023–24")
ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), color="#ff5d73", alpha=0.12, label="🔒 private 2025")
ax.set_title("the only split that is allowed — identical for all five tickers")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()

Xtr, Xva, ytr_, yva_ = time_split(*make_dataset(RAW))
print(f"train: {len(Xtr)} rows | val: {len(Xva)} rows")""")

# ===========================================================================
# SECTION 4 — Metrics & baselines
# ===========================================================================
code("""# 🔒 ==== LOCKED: the shared metric ====
def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    \"\"\"MAE (competition metric), RMSE, and directional accuracy.\"\"\"
    err = y_pred - y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "DirAcc": float(((y_pred > 0) == (y_true > 0)).mean()),
    }

def show_metrics(name: str, metrics: dict) -> None:
    print(f"{name:<34} MAE {metrics['MAE']:.6f} | RMSE {metrics['RMSE']:.6f} "
          f"| DirAcc {metrics['DirAcc']:.3f}")""")

md("""## 4 · Baselines — the bar to clear

> 📏 **A model is only good if it beats the trivial ones.** Every score you print
> today should be read next to these numbers.

| Baseline | Idea |
|---|---|
| **predict 0** | "tomorrow ≈ today" — surprisingly hard to beat on daily returns |
| **predict yesterday's return** | momentum: yesterday repeats |
| **predict the average return** | the long-run drift |
""")

code("""# 🔒 ==== LOCKED: baseline scores on validation ====
baseline_rows = []
for name, pred in {
    "predict 0.0": pd.Series(0.0, index=yva_.index),
    "predict yesterday's return": Xva["lag_return_1d"] if "lag_return_1d" in Xva else Xva["return_1d"].shift(1),
    "predict train mean": pd.Series(ytr_.mean(), index=yva_.index),
}.items():
    p = pred.fillna(0.0)
    baseline_rows.append({"model": name, **evaluate_predictions(yva_, p)})

BASELINE_VAL_MAE = baseline_rows[0]["MAE"]
pd.DataFrame(baseline_rows).set_index("model")""")

# ===========================================================================
# SECTION 5 — Models
# ===========================================================================
md("""## 5 · The model garage

One uniform API for everything — you never touch training plumbing, only knobs:

```python
train_and_validate(arch, params=..., label="...")   # trains on 2014–22, scores on 2023–24
```

| `arch` | Family | Typical personality |
|---|---|---|
| `"ridge"` | linear regression + L2 | fast, honest, hard to overfit — the underdog |
| `"random_forest"` | bagged trees | robust, low-effort, weak on pure sequence info |
| `"xgboost"` | boosted trees | strongest tabular learner, loves good features, overfits eagerly |
| `"lstm"` | recurrent net | reads the last N days as a sequence, slow, moody |
| `"wildcard"` | GRU **or** 1D-CNN (your pick) | the gambit |

> 🔍 **Reading your results:** compare **train MAE** vs **val MAE**. Train ≪ val is
> **overfitting** — the model memorised 2014–22 noise. Both terrible = underfitting.
> Overfitting is fixed with regularisation (shallower trees, more dropout, fewer
> estimators) or better features, not with more training data alone.
""")

code("""# 🔒 ==== LOCKED: shared training infrastructure ====
import inspect
import joblib

SEQ_ARCHS = {"lstm", "gru", "cnn"}
RESULTS: list[dict] = []            # every training run lands here
BEST: dict[str, dict] = {}          # best run per resolved architecture

def scale(Xtr_: pd.DataFrame, Xother_: pd.DataFrame):
    \"\"\"Standardise with statistics from TRAINING data only (leakage discipline!).\"\"\"
    mu, sd = Xtr_.mean(), Xtr_.std().replace(0.0, 1.0)
    return (Xtr_ - mu) / sd, (Xother_ - mu) / sd, mu, sd

def make_sequences(X_scaled: np.ndarray, seq_len: int, tickers: np.ndarray):
    \"\"\"Per-ticker sliding windows. Window i covers ticker g's rows
    [e-seq_len+1 .. e] and predicts the target at row e. Windows NEVER cross
    ticker boundaries.\"\"\"
    wins, ends = [], []
    for t in pd.unique(tickers):
        pos = np.where(tickers == t)[0]          # date-ascending within the ticker
        if len(pos) < seq_len:
            continue
        sub = X_scaled[pos]
        for j in range(seq_len - 1, len(pos)):
            wins.append(sub[j - seq_len + 1 : j + 1])
            ends.append(pos[j])
    return np.stack(wins).astype(np.float32), np.array(ends)

def train_torch_model(model, Xtr_seq, ytr_seq, epochs: int, lr: float, batch_size: int):
    \"\"\"Small, deterministic CPU training loop (MSE loss, Adam).\"\"\"
    loader = DataLoader(TensorDataset(torch.tensor(Xtr_seq), torch.tensor(ytr_seq)),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   epoch {epoch + 1:>3}/{epochs}  train MSE {loss.item():.6f}")
    model.eval()
    return model""")

code("# 🔒 ==== LOCKED: neural network architectures (shared with your submission) ====\n"
     + NN_CLASSES_SRC)

code("""# 🔒 ==== LOCKED: the uniform trainer ====
def _resolve_arch(arch: str) -> str:
    return WILDCARD_ARCH if arch == "wildcard" else arch


def _fit_tabular(arch, params, Xtr_, ytr_, Xva_=None, yva_=None):
    # Early stopping only makes sense when a validation set is available (training
    # path); the export path retrains on everything, so it is dropped there.
    es = params.get("early_stopping_rounds") if Xva_ is not None else None
    if arch == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=params.get("alpha", 1.0), random_state=SEED)
    elif arch == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", 0.8), random_state=SEED, n_jobs=-1)
    elif arch == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError(
                "xgboost is not installed in this environment. This is common on macOS "
                "pip/venv setups (torch + xgboost wheels conflict there). You can still "
                "compete fully with: \\"ridge\\", \\"random_forest\\", \\"lstm\\" or the "
                "\\"wildcard\\" (GRU/CNN) — or switch to Colab. To try installing here: "
                "%pip install xgboost"
            ) from e
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 300), max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 1.0), eval_metric="mae",
            early_stopping_rounds=es,
            random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(f"unknown architecture: {arch}")
    if Xva_ is not None and es:
        model.fit(Xtr_.values, ytr_.values, eval_set=[(Xva_.values, yva_.values)], verbose=False)
    else:
        model.fit(Xtr_.values, ytr_.values)
    return model


def train_and_validate(arch: str, params: dict | None = None,
                       features: list[str] | None = None,
                       label: str | None = None) -> dict:
    \"\"\"Train `arch` on 2014–22, score on 2023–24, register the run.

    Returns {"model", "arch", "params", "metrics", "label", "val_pred"} and keeps
    the best run per architecture in BEST for the export cell.\"\"\"
    t0 = time.time()
    params = dict(params or {})
    arch_r = _resolve_arch(arch)
    feats = features if features is not None else FEATURES
    X_full, y_full, tick_full = make_dataset(RAW, feats)
    Xtr, Xva, ytr_, yva_ = time_split(X_full, y_full, tick_full)
    label = label or arch_r

    if arch_r in SEQ_ARCHS:
        seq_len = int(params.get("seq_len", 20))
        Xs_all, _, mu, sd = scale(X_full, X_full)     # stats from visible data only
        wins, ends_pos = make_sequences(Xs_all.values, seq_len, tick_full.values)
        dates = X_full.index[ends_pos]
        tr_m = dates <= pd.Timestamp(TRAIN_END)
        va_m = (dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))
        y_at = y_full.values[ends_pos]                # target at each window's last row
        model = build_torch_model(arch_r, params, X_full.shape[1])
        model = train_torch_model(model, wins[tr_m], y_at[tr_m].astype(np.float32),
                                  params.get("epochs", 30), params.get("lr", 1e-3),
                                  params.get("batch_size", 64))
        with torch.no_grad():
            val_pred = model(torch.tensor(wins[va_m])).squeeze(-1).numpy()
            train_pred = model(torch.tensor(wins[tr_m])).squeeze(-1).numpy()
        pred_val = pd.Series(val_pred, index=dates[va_m])
        m_tr = evaluate_predictions(
            pd.Series(y_at[tr_m], index=dates[tr_m]),
            pd.Series(train_pred, index=dates[tr_m]))
        extra = {"seq_len": seq_len, "scaler": {"mean": mu[feats].tolist(), "std": sd[feats].tolist()}}
    else:
        model = _fit_tabular(arch_r, params, Xtr, ytr_, Xva, yva_)
        pred_val = pd.Series(model.predict(Xva.values), index=Xva.index)
        m_tr = evaluate_predictions(ytr_, pd.Series(model.predict(Xtr.values), index=Xtr.index))
        extra = {}

    # Seq-path predictions are ordered by ticker (window groups), yva_ by panel
    # row order — both indexes hold duplicate dates, so evaluate POSITIONALLY.
    if arch_r in SEQ_ARCHS:
        m_va = evaluate_predictions(pd.Series(y_at[va_m], index=dates[va_m]), pred_val)
    else:
        m_va = evaluate_predictions(yva_, pred_val)
    run = {"label": label, "arch": arch_r, "params": params, "features": list(feats),
           "train_MAE": m_tr["MAE"], "val_MAE": m_va["MAE"], "val_RMSE": m_va["RMSE"],
           "val_DirAcc": m_va["DirAcc"], "seconds": round(time.time() - t0, 1),
           "model": model, "val_pred": pred_val, "val_y": yva_, **extra}
    RESULTS.append({k: v for k, v in run.items() if k not in ("model", "val_pred")})
    prev = BEST.get(arch_r)
    if prev is None or m_va["MAE"] < prev["val_MAE"]:
        BEST[arch_r] = run
    show_metrics(f"[{label}]", m_va)
    print(f"{'':<34} train MAE {m_tr['MAE']:.6f}  ->  overfit ratio "
          f"{m_va['MAE'] / max(m_tr['MAE'], 1e-12):.1f}x  ({run['seconds']}s)")
    return run""")

md("""### Model A — Ridge (the underdog)""")

code("""# 🟩 ==== MODIFY THIS SECTION: Model A — Ridge ====
# alpha = regularisation strength. Bigger alpha = flatter, more conservative model.
# It cannot learn interactions — its power comes entirely from YOUR features.
run_ridge = train_and_validate("ridge", params={
    "alpha": 1.0,
}, label="ridge v1")

# ideas: alpha 0.1 / 1 / 10 / 100 — watch train MAE and val MAE move together""")

md("""### Model B — Random Forest""")

code("""# 🟩 ==== MODIFY THIS SECTION: Model B — Random Forest ====
# n_estimators: more trees = stabler but slower.  max_depth / min_samples_leaf:
# the anti-overfitting dials.  max_features: features tried per split (<1 decorrelates trees).
run_rf = train_and_validate("random_forest", params={
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "max_features": 0.8,
}, label="forest v1")

# ideas: depth 3 vs 6 vs 12 (overfit watch!); min_samples_leaf 1 vs 20; 500 trees""")

md("""### Model C — XGBoost (the tabular king)""")

code("""# 🟩 ==== MODIFY THIS SECTION: Model C — XGBoost ====
# learning_rate: step size — lower = gentler, needs more trees.  max_depth: tree complexity.
# subsample / colsample_bytree: randomness that fights overfitting.  reg_lambda: L2 brake.
# early_stopping_rounds: stop adding trees when validation stops improving (set an int to enable).
run_xgb = train_and_validate("xgboost", params={
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
}, label="xgb v1")

# ideas: depth 2–6; lr 0.01–0.2; early_stopping_rounds 20 + eval against val""")

md("""### Model D — LSTM (the sequence reader)

Feeds the model the last `seq_len` days **as a sequence** instead of a single
feature row. Slow to train, sensitive to scaling (handled for you), loves to
overfit — keep `hidden_size` and `epochs` modest.
""")

code("""# 🟩 ==== MODIFY THIS SECTION: Model D — LSTM ====
run_lstm = train_and_validate("lstm", params={
    "seq_len": 20,        # days of history per window
    "hidden_size": 32,    # capacity
    "num_layers": 1,      # stack more only if you have time to wait
    "dropout": 0.2,       # regularisation
    "epochs": 30,         # more epochs = more memorisation risk
    "lr": 1e-3,
    "batch_size": 64,
}, label="lstm v1")

# ideas: seq_len 5 / 20 / 60; hidden 8 vs 64 (capacity vs overfit); epochs 10 vs 60""")

md("""### Model E — Wildcard: GRU **or** 1D-CNN

Set `WILDCARD_ARCH` to your pick. **GRU** = the LSTM's leaner cousin (fewer gates,
faster). **1D-CNN** = convolution filters sliding over the window — spots local
patterns (3-day dips, volume spikes) rather than carrying a memory.
""")

code("""# 🟩 ==== MODIFY THIS SECTION: Model E — Wildcard ====
WILDCARD_ARCH = "gru"      # 🟩 pick: "gru" or "cnn"

run_wild = train_and_validate("wildcard", params={
    "seq_len": 20,
    "hidden_size": 32, "num_layers": 1, "dropout": 0.2,       # GRU knobs
    "channels": 32, "kernel_size": 5,                          # CNN knobs
    "epochs": 30, "lr": 1e-3, "batch_size": 64,
}, label="wildcard v1")""")

md("""## 6 · Scoreboard

Compare every run. Remember: **val MAE below the baseline line is the whole game.**
""")

code("""# 🔒 ==== LOCKED: the scoreboard ====
summary = pd.DataFrame(RESULTS)[
    ["label", "arch", "train_MAE", "val_MAE", "val_RMSE", "val_DirAcc", "seconds"]
]
display(summary)

fig, ax = plt.subplots(figsize=(9, 0.5 * max(len(summary), 2) + 1.5))
colors = ["#2dd4a7" if v < BASELINE_VAL_MAE else "#ff5d73" for v in summary["val_MAE"]]
ax.barh(summary["label"], summary["val_MAE"], color=colors)
ax.axvline(BASELINE_VAL_MAE, color="#f5c451", ls="--", lw=1.2,
           label=f"baseline (predict 0): {BASELINE_VAL_MAE:.6f}")
ax.invert_yaxis(); ax.set_xlabel("validation MAE (lower = better)")
ax.set_title("your models vs the baseline"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()""")

code("""# 🔒 ==== LOCKED (optional): what the best model actually predicts ====
# Scatter of predicted vs actual next-day returns on validation — reality check:
# real models predict a *narrow band* around zero while reality swings wide.
best_label = min(RESULTS, key=lambda r: r["val_MAE"])["label"]
run_model = BEST[min(RESULTS, key=lambda r: r["val_MAE"])["arch"]]
pred = run_model["val_pred"]
actual = run_model["val_y"]          # stored alongside, already row-aligned
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].scatter(pred, actual, s=6, alpha=0.5, color="#2dd4a7")
lims = [min(pred.min(), actual.min()), max(pred.max(), actual.max())]
axes[0].plot(lims, lims, color="#7d8b9b", lw=0.8, ls="--")
axes[0].set(title=f"{best_label}: predicted vs actual", xlabel="predicted", ylabel="actual")
axes[1].plot(pred.cumsum().values, label="cumulative predicted", color="#f5c451")
axes[1].plot(actual.cumsum().values, label="cumulative actual", color="#7d8b9b")
axes[1].legend(fontsize=8); axes[1].set_title("cumulative view (panel row order)")
for ax in axes: ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()""")

# ===========================================================================
# SECTION 6 — Export
# ===========================================================================
md("""## 7 · 📦 Export your submission

The cell below packages **everything the organisers need** into one zip:

```
submission_<TEAM>_round<N>.zip
├── predict.py          # locked evaluation interface (same for every team)
├── team_features.py    # ← your add_features(), copied verbatim from this notebook
├── team_models.py      # neural-net classes (only used by sequence models)
├── config.json         # architecture, features, hyperparameters, scaler snapshot
└── model file          # your trained model, retrained on ALL visible data (2014–24)
```

Notes
- The final model is **retrained on 2014–2024** (train + validation) with your best
  settings — no reason to leave 2 years of data on the table for the final model.
- **Round 1:** run this with `ROUND = 1`. **Round 2:** after AI-assisted changes,
  run it again with `ROUND = 2`. Both zips coexist — name your team the same.
""")

code("""# 🟩 ==== MODIFY THIS SECTION: your team identity ====
TEAM_NAME = "Team Example"   # 🟩 your team name — must be identical in Round 2
ROUND = 1                    # 🟩 1 = human-only submission, 2 = AI-assisted submission
ARCH_FOR_SUBMISSION = None   # 🟩 None = automatically your best val-MAE model,
                             #    or force one: "ridge" | "random_forest" | "xgboost" | "lstm" | "gru" | "cnn"  # noqa: E501""")

code("""# 🔒 ==== LOCKED: the exporter ====
PREDICT_PY = r\'\'\'__PREDICT_PY__\'\'\'
TEAM_MODELS_PY = r\'\'\'__TEAM_MODELS_PY__\'\'\'

EXPORT_DIR = Path("exports")

def _resolved_best_arch() -> str:
    ok = {a: r["val_MAE"] for a, r in BEST.items()}
    if not ok:
        raise RuntimeError("No trained models found — run the training cells above first.")
    return min(ok, key=ok.get)

def _write(path: Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")

def export_submission() -> Path:
    arch = _resolve_arch(ARCH_FOR_SUBMISSION or _resolved_best_arch())
    best = BEST[arch]
    params = best["params"]
    feats = best["features"]
    is_seq = arch in SEQ_ARCHS

    # ---- retrain on ALL visible data (2014–2024) with the winning settings ----
    Xf, yf, tickf = make_dataset(RAW, feats)
    if is_seq:
        Xs, _, mu, sd = scale(Xf, Xf)                       # stats from all visible rows
        seq_len = int(params.get("seq_len", 20))
        wins, ends_pos = make_sequences(Xs.values, seq_len, tickf.values)
        model = build_torch_model(arch, params, Xf.shape[1])
        model = train_torch_model(model, wins, yf.values[ends_pos].astype(np.float32),
                                  params.get("epochs", 30), params.get("lr", 1e-3),
                                  params.get("batch_size", 64))
        torch.save(model.state_dict(), "model.pt")
        model_file, scaler, cfg_seq = "model.pt", {"mean": mu[feats].tolist(), "std": sd[feats].tolist()}, seq_len
    else:
        model = _fit_tabular(arch, params, Xf, yf)
        joblib.dump(model, "model.pkl")
        model_file, scaler, cfg_seq = "model.pkl", None, None

    # ---- carry your feature code and the NN classes with the submission ----
    _write("team_features.py",
           "# Auto-generated from the team notebook — the feature pipeline.\\n"
           "# (Regenerated by the export cell; do not hand-edit the copy inside the zip.)\\n\\n"
           "import numpy as np\\nimport pandas as pd\\n\\n\\n"
           + inspect.getsource(add_features))
    if is_seq:
        # NN classes ship as a verbatim template — inspect.getsource() cannot read
        # classes defined in Jupyter cells (no __file__ on __main__).
        _write("team_models.py", TEAM_MODELS_PY)

    config = {
        "team": TEAM_NAME, "round": ROUND, "arch": arch, "features": list(feats),
        "params": {k: v for k, v in params.items()}, "seq_len": cfg_seq,
        "scaler": scaler, "model_file": model_file,
        "trained_on": f"{Xf.index[0].date()}..{Xf.index[-1].date()}",
        "val_mae": best["val_MAE"],
    }
    _write("config.json", json.dumps(config, indent=2))
    _write("predict.py", PREDICT_PY)

    # ---- zip it ----
    import zipfile
    EXPORT_DIR.mkdir(exist_ok=True)
    zip_path = EXPORT_DIR / f"submission_{TEAM_NAME.replace(' ', '_')}_round{ROUND}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("predict.py", "predict.py")
        zf.write("config.json", "config.json")
        zf.write(model_file, model_file)
        zf.write("team_features.py", "team_features.py")
        if is_seq:
            zf.write("team_models.py", "team_models.py")
    print(f"✅ {zip_path}  |  arch={arch}  val MAE={best['val_MAE']:.6f}  "
          f"features={len(feats)}  round={ROUND}")
    print("   → upload this zip at the submission link + PIN on the whiteboard "
          "(or hand it to a facilitator). No zip, no leaderboard row.")
    return zip_path

submission_path = export_submission()""")

md("""### ✅ Pre-submission checklist

- [ ] `TEAM_NAME` set (identical spelling in both rounds)
- [ ] `ROUND` correct (`1` before Round 1 deadline, `2` after the AI round)
- [ ] The export cell ran **after your last change** (the zip is a snapshot!)
- [ ] Zip contains `predict.py`, `config.json`, a model file, `team_features.py`
- [ ] Upload the zip at the submission link + PIN on the whiteboard (phone is fine),
      typing your team name exactly; keep the ✓ receipt as your proof of hand-in.
      Link not loading? Use the facilitator's hotspot or a USB stick.

*Rest easy: the evaluator gives every team the same warm-up history, so sequence
models are not penalised at the start of 2025.*
""")

# ===========================================================================
# SECTION 7 — Round 2 annex
# ===========================================================================
md("""## 8 · 🤖 Round 2 — AI unlocked

Copy the prompt below into your AI tool **after the Round 1 leaderboard** (it is
auto-filled with your numbers). Then implement the suggestions **selectively** —
you are the engineer, the AI is the consultant.

> ⚠️ **Benchmark-overfitting warning:** you now know your 2025 score. If you tune
> *against that number*, you are optimising for the benchmark, not for the market —
> ask for improvements that are justified by **validation** behaviour and ML
> reasoning, not by the 2025 score alone.
""")

code("""# 🔒 ==== LOCKED: build your Round 2 AI prompt (auto-filled) ====
arch = _resolve_arch(ARCH_FOR_SUBMISSION or _resolved_best_arch())
cfg = BEST[arch]
print(f\"\"\"Model: {arch}

Training period: 2014-2022
Validation: 2023-2024

Validation MAE: {cfg['val_MAE']:.6f}
Validation MAE of the trivial baseline (predict 0): {BASELINE_VAL_MAE:.6f}
2025 competition MAE: <PASTE YOUR ROUND-1 SCORE FROM THE LEADERBOARD>

Current features:
{cfg['features']}

Current hyperparameters:
{json.dumps(cfg['params'], indent=2)}

How would you improve this model WITHOUT using the 2025 labels? Suggest concrete
changes to features, hyperparameters and regularisation, explain the ML reasoning
for each, and warn me where a change risks overfitting. Do not suggest looking at
the 2025 data.\"\"\")""")

md("""### 📋 AI change log — fill this in BEFORE exporting Model V2

*The workshop is not measuring whether AI can write your model. It is measuring
**what you changed and why**. Every accepted/rejected suggestion goes here.*

```text
AI suggested:
✓ <suggestion>            — accepted because <reason / val result>
✓ <suggestion>            — accepted because ...
✗ <suggestion>            — rejected because <time / suspicious / overfitting risk>

What we actually changed:
- <change 1>              — validation MAE: <before> → <after>
- <change 2>              — validation MAE: <before> → <after>
```

### 🧠 Before the final leaderboard, think about:

1. Did the AI's changes improve **validation**, the **2025 score**, or both?
   Where they disagree, which one would you trust for the *real* future?
2. If organisers secretly evaluate on **2026 data no one has seen** — would your
   Round-2 improvement survive? Why or why not?
3. Did any team's fancy architecture beat a simple model with great features?
""")

md("""---

*Educational exercise only. Historical data, virtual portfolios, no investment
advice. Past patterns do not guarantee future returns — today proved it.*
""")

# ===========================================================================
# Assemble
# ===========================================================================
def main() -> None:
    predict_src = PREDICT_PY_PATH.read_text()
    assert "'''" not in predict_src, "predict.py must not contain triple single quotes"
    team_models_src = ("import torch\nimport torch.nn as nn\nSEED = 42\n\n\n"
                       + NN_CLASSES_SRC)
    assert "'''" not in team_models_src, "NN source must not contain triple single quotes"
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "None",
    }
    for kind, src in CELLS:
        source = src.replace("__PREDICT_PY__", predict_src)
        source = source.replace("__TEAM_MODELS_PY__", team_models_src)
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_PATH)
    print(f"wrote {OUT_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
