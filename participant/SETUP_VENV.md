# 💻 VS Code + venv Setup Guide — Can AI Predict the Market?

For attendees who want to run the notebook **locally in VS Code** with a plain Python
`venv` (no conda). If you're happy with **Google Colab**, skip this file entirely —
Colab needs zero installation and is still the recommended path.

> **TL;DR by operating system**
>
> | OS | Plain venv + pip | What you get |
> |---|---|---|
> | **Linux** | ✅ works | all 5 models |
> | **Windows** | ✅ works | all 5 models (standard combo, not lab-tested here) |
> | **macOS** | ⚠️ partial | **all models except XGBoost** — the pip wheels of torch and xgboost crash when loaded together on macOS (OpenMP conflict). The notebook tells you this politely and you can compete fully with the other four. |
>
> On macOS and you want XGBoost too? Two options: run on **Colab**, or create the env
> with **miniforge** instead of venv — VS Code works with conda envs through the exact
> same interpreter picker, so you lose nothing. (Full recipe at the bottom.)

---

## 1 · Prerequisites

- **VS Code** with the **Python** and **Jupyter** extensions (install from the
  Extensions panel).
- **Python 3.11 or 3.12** on your PATH (`python3 --version`). 3.10+ is required.

## 2 · Get the workshop folder

You need the `participant/` folder the organisers shared, containing:
`workshop.ipynb`, `requirements.txt`, `data/workshop_participant.csv`, `GUIDE.md`,
`CHEATSHEET.md`.

## 3 · Create the venv and install

Open the folder in VS Code (`File → Open Folder…`), then open a terminal **inside
VS Code** (`` Ctrl+` ``) — it should already be in the folder.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the packages (the pinned versions the workshop was tested with):

```bash
# Linux only — grab the CPU build of torch first, so pip doesn't pull a
# multi-GB CUDA download you don't need:
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Installation takes a few minutes (torch is the big one).

## 4 · Point VS Code at the venv

1. Open `workshop.ipynb`.
2. Top-right **kernel picker → "Select Another Kernel… → Python Environments…"**.
3. Choose **`.venv (Python 3.x)`** — VS Code lists every venv it can find; if it's
   missing, click the refresh icon or `Python: Select Interpreter` → enter the path.
4. Run the first cell. Expected output: version numbers ending in `setup OK`
   (a yellow note about xgboost is fine on macOS — see below).

Then `Runtime`: `Run All` (~1–2 minutes). You should see five models train and a
scoreboard appear.

## 5 · macOS users: the XGBoost situation

On macOS, the pip wheels of **torch** and **xgboost** each bundle their own OpenMP
runtime; loaded into the same process they crash — this is a known packaging problem,
not something you did wrong. The notebook handles it gracefully:

- The setup cell prints a yellow `!! xgboost missing` note.
- If you run the XGBoost cell you get a clear message explaining your options —
  no kernel crash, no lost work.

**You can compete fully without XGBoost**: Ridge, Random Forest, LSTM and the
Wildcard (GRU/CNN) all train and export exactly the same way. Tell your facilitator
you're an "XGBoost-free" team so the model draft accounts for it.

Want the full five? Either run on **Colab**, or use miniforge instead of venv —
then keep using VS Code exactly as above:

```bash
# one-time: install miniforge (https://github.com/conda-forge/miniforge), then:
conda create -n stocks-ml python=3.12 -c conda-forge
conda activate stocks-ml
python -m pip install torch xgboost scikit-learn pandas matplotlib yfinance joblib ipykernel
```
conda-forge builds torch and xgboost against one shared OpenMP, so they coexist
happily. In VS Code, pick the `stocks-ml` interpreter — identical workflow.

## 6 · Troubleshooting

| Symptom | Fix |
|---|---|
| Kernel picker doesn't show `.venv` | Install `ipykernel` in the venv (`pip install ipykernel`), then `Developer: Reload Window`. |
| `Kernel died` when the XGBoost cell runs (macOS) | You installed both pip wheels — that's the known conflict. Remove xgboost (`pip uninstall xgboost`) and compete with the other four models, or move to Colab/miniforge. |
| `pip install torch...` downloads 2+ GB on Linux | You skipped the CPU index URL from step 3. Cancel, run that line first, then `pip install -r requirements.txt`. |
| `ModuleNotFoundError: xgboost` in one cell, everything else fine | Expected on macOS venvs — see §5. |
| Notebook can't find `workshop_participant.csv` | The file must sit in `data/` next to the notebook (or the notebook's folder). |
| Cells run but plots don't show | Make sure you're running the `.ipynb` in VS Code's notebook editor with the Jupyter extension, not as a plain Python file. |

## 7 · Sanity check before Round 1

With the venv active:

```bash
python -c "import pandas, sklearn, torch; print('core ok')"
python -c "import xgboost; print('xgboost ok')"   # macOS venv: expected to FAIL — that's fine
```

Both green (or only the xgboost line red on macOS) → you're ready. On Colab? You're
always ready.
