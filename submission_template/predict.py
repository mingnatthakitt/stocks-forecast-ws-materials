"""
predict.py — THE SUBMISSION INTERFACE (do not modify)
=====================================================

Every team's submission.zip contains this exact predict.py together with:

    config.json        your architecture, feature list, hyperparameters, scaler
    model.pkl/.pt      your trained model (written by the export cell)
    team_features.py   your feature-engineering function (auto-copied from the notebook)
    team_models.py     the neural-network class definitions (auto-copied)

The organiser's evaluation system imports THIS file and calls:

    predict(df)  ->  pd.Series

Contract
--------
* `df` is a raw OHLCV **panel** (Date index; columns Ticker, Open, High, Low, Close,
  Volume — one row per Date × Ticker, sorted by Date). It includes warm-up history
  BEFORE the scored period, so rolling features and sequence windows have data.
* You must return one predicted NEXT-DAY RETURN per row of `df`, as a Series with
  the same index. Rows are scored only where the organiser has private labels.
* Rows your pipeline cannot handle (e.g. before enough warm-up history) must get
  the prediction 0.0 — NEVER NaN. This evaluator rejects NaN predictions.
* Features and sequence windows are built PER TICKER — never across tickers.
* Everything your model needs must be inside your zip: the evaluator only gives
  you this DataFrame. No internet, no reading the labels (they are not here).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())

SEQ_ARCHS = ("lstm", "gru", "cnn")


def _predict_tabular(model, X: np.ndarray) -> np.ndarray:
    """Ridge / Random Forest / XGBoost: model saved with joblib."""
    return np.asarray(model.predict(X), dtype=float)


def _predict_sequence(model, X: np.ndarray, tickers: np.ndarray, seq_len: int,
                      ok: np.ndarray) -> np.ndarray:
    """LSTM / GRU / 1D-CNN: window of `seq_len` feature rows -> next-day return.

    Windows are built PER TICKER (never across tickers) and evaluated in one
    batch per ticker. A window is scored only when every row inside it is
    complete; anything else stays at the safe 0.0.
    """
    import torch

    preds = np.zeros(len(X), dtype=float)
    with torch.no_grad():
        for ticker in pd.unique(tickers):
            pos = np.where(tickers == ticker)[0]        # date-ascending within ticker
            if len(pos) < seq_len:
                continue
            sub, sub_ok = X[pos], ok[pos]
            windows, dst = [], []
            for j in range(seq_len - 1, len(pos)):
                if sub_ok[j - seq_len + 1 : j + 1].all():
                    windows.append(sub[j - seq_len + 1 : j + 1])
                    dst.append(pos[j])
            if not windows:
                continue
            batch = torch.tensor(np.stack(windows), dtype=torch.float32)
            out = model(batch).squeeze(-1).numpy()
            preds[np.array(dst)] = out
    return preds


def predict(df: pd.DataFrame) -> pd.Series:
    # 1) Rebuild the features exactly like during training (per ticker).
    from team_features import add_features

    feats = add_features(df.copy())
    missing = [c for c in CONFIG["features"] if c not in feats.columns]
    if missing:
        raise RuntimeError(
            f"add_features() did not produce the features listed in config.json: {missing}. "
            "The feature function and the FEATURES list must stay in sync."
        )
    X = feats[CONFIG["features"]].astype(float)
    tickers = feats["Ticker"].values

    # 2) Apply the scaler fitted on training data (stored in config.json).
    scaler = CONFIG.get("scaler")
    if scaler is not None:
        mean = np.array(scaler["mean"], dtype=float)
        std = np.array(scaler["std"], dtype=float)
        Xv = (X.values - mean) / std
    else:
        Xv = X.values

    # 3) Predict wherever the feature matrix is complete; 0.0 elsewhere.
    ok = ~np.isnan(Xv).any(axis=1)
    preds = np.zeros(len(Xv), dtype=float)
    if ok.any():
        arch = CONFIG["arch"]
        if arch in SEQ_ARCHS:
            import torch

            from team_models import build_torch_model

            seq_len = int(CONFIG.get("seq_len") or 20)
            model = build_torch_model(arch, CONFIG["params"], X.shape[1])
            state = torch.load(HERE / CONFIG["model_file"], map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            preds = _predict_sequence(model, Xv, tickers, seq_len, ok)
        else:
            model = joblib.load(HERE / CONFIG["model_file"])
            preds[ok] = _predict_tabular(model, Xv[ok])

    return pd.Series(preds, index=df.index, name="predicted_next_day_return")


# The evaluation system does roughly this:
#
#     import sys; sys.path.insert(0, <unzipped team folder>)
#     import predict
#     preds = predict(panel_df)      # OHLCV panel incl. warm-up history
#     # then scores `preds` against the private labels on the 2025 (Date, Ticker) rows
#
# Run a local self-test (uses whatever CSV sits next to this file):
if __name__ == "__main__":
    csv = HERE / "local_check.csv"
    if csv.exists():
        frame = pd.read_csv(csv, index_col="Date", parse_dates=True)
        out = predict(frame)
        print(out.tail(10))
        assert not out.isna().any(), "NaN predictions found — fix before submitting!"
        print("Self-test OK:", len(out), "predictions, no NaNs.")
    else:
        print("Put an OHLCV csv named local_check.csv next to predict.py to self-test.")
