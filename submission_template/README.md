# Submission Package (reference copy)

Participants never assemble this folder by hand — **the export cell at the end of
`participant/workshop.ipynb` builds a complete, ready-to-submit zip automatically.**

This folder holds the readable reference copy of the two fixed files that every
submission contains:

| File | Role |
|---|---|
| `predict.py` | The locked evaluation interface. Exposes `predict(df) -> pd.Series` (next-day-return predictions aligned to the input OHLCV index). Reads everything it needs from `config.json`. |
| `config.json` | Example of the metadata snapshot the export cell writes: team name, architecture, feature list, hyperparameters, scaler statistics, validation MAE. |

A real submission zip looks like:

```
submission_<TEAM>_round<N>.zip
├── predict.py          # locked interface (same for every team)
├── team_features.py    # auto-generated from the notebook's add_features()
├── team_models.py      # auto-generated neural-net classes (only used by seq models)
├── config.json         # architecture / features / params / scaler snapshot
└── model.pkl or .pt    # the trained model artifact
```

The organisers unzip each submission, import `predict.py`, feed the same OHLCV
data to every team, and score the returned predictions against the private
2025 labels (see `organizer/evaluate.py`).
