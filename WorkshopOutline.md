# Can AI Predict the Market?
## A Hands-On ML Stock Forecasting Challenge

### 1. Workshop Overview

**Proposed format:** Interactive machine-learning workshop + team competition  
**Estimated duration:** 2 hours  
**Target participants:** Approximately 20–30 students, split into teams of 3–5  
**Recommended background:** Basic Python is helpful; prior machine-learning or finance knowledge is not required  
**Platform:** Google Colab / Jupyter Notebook  
**Venue requirements:** Projector, stable Wi-Fi, tables suitable for team work, and accessible power outlets

### Core idea

Instead of running a conventional tutorial where participants simply follow a prepared stock-prediction notebook, participants will learn several machine-learning approaches and then apply them in a controlled forecasting competition.

The workshop has two competition rounds:

1. **Human-only optimisation**
2. **Human + AI optimisation**

Round 1 and Round 2 submissions are evaluated against the **same private 2025
competition dataset**. Teams that submit in both rounds can compare their own
scores on the board; comparisons across different teams do not isolate AI's effect.

The workshop focuses on machine learning, time-series validation, model development, feature engineering, hyperparameter tuning, AI-assisted coding, and critical evaluation.

**No real-money trading will take place.** Any trading component will use historical data and simulated/paper portfolios only.

---

# 2. Workshop Objectives

By the end of the workshop, participants should be able to:

- Understand how stock forecasting can be formulated as a machine-learning problem.
- Understand the distinction between training, validation, and test/competition data.
- Explain why time-series data cannot be treated exactly like ordinary randomly shuffled datasets.
- Recognise and avoid common forms of data leakage.
- Understand the basic ideas behind several ML architectures.
- Train and tune machine-learning models for a real dataset.
- Perform basic financial feature engineering.
- Compare different model architectures based on empirical results.
- Understand the effect of hyperparameters and preprocessing.
- Evaluate models using appropriate forecasting metrics.
- Understand why prediction accuracy does not automatically imply trading profitability.
- Use AI coding tools critically rather than simply accepting generated code.
- Compare human-only optimisation with AI-assisted optimisation.
- Understand how repeatedly optimising against a benchmark can cause benchmark/leaderboard overfitting.

---

# 3. Central Question

The entire workshop revolves around:

> **Can AI predict the market?**

Rather than giving participants an answer at the beginning, the workshop allows them to investigate the question experimentally.

Participants will build models, evaluate them on unseen historical market data, improve them, compare architectures, and observe whether AI assistance genuinely improves their models.

---

# 4. Dataset Design

All teams receive the same historical dataset: daily, adjusted OHLCV data for
**SPY, NVDA, AAPL, MSFT, and TSLA**. The shipped workshop uses these fixed periods:

| Period | Purpose | Visible to participants? |
|---|---|---|
| **2014–2022** | Model training | Yes |
| **2023–2024** | Validation and tuning | Yes |
| **2025** | Private competition benchmark | No; organisers pass features to the submission for scoring and keep labels private |
| **2026 YTD** | Surprise generalisation holdout | No during either round; revealed at the end |

The dataset will primarily contain standard OHLCV market data:

- Date
- Open
- High
- Low
- Close
- Volume

From these, participants can derive additional features.

---

# 5. Prediction Task

The main task will be to predict **next-day stock returns**.

For example:

$$
R_{t+1} = \frac{P_{t+1} - P_t}{P_t}
$$


The model receives information available up to day \(t\) and attempts to estimate the return on day \(t+1\).

From the same predictions we can evaluate both:

### Regression performance

How close was the predicted return to the actual return?

Possible metrics:

- MAE
- RMSE

### Directional performance

Did the model correctly predict whether the market would move up or down?

$$
Direction = \begin{cases} 1 & r_{t+1}>0 \\ 0 & r_{t+1}\le0 \end{cases}
$$


The primary competition metric is **MAE pooled across all five tickers** (lower is better). RMSE, directional accuracy, and per-ticker MAE are diagnostics.

---

# 6. Workshop Flow

## Part 1 — Opening Challenge
**~6 minutes**

Start by showing participants several historical stock charts with the future portion hidden.

Ask:

> Will the next period go up or down?

Participants vote before the answer is revealed.

This introduces the workshop question:

> If humans struggle to predict the next movement, can a machine learn useful patterns from historical data?

We then introduce the day's challenge.

---

# 7. Part 2 — What Does “Predicting a Stock” Mean?
**~13 minutes**

Brief introduction to:

- Time-series data
- Prices vs. returns
- Regression vs. classification
- Features and targets
- Training data
- Validation data
- Test data

Example pipeline:

```text
Historical Market Data
        ↓
OHLCV
        ↓
Feature Engineering
        ↓
Machine-Learning Model
        ↓
Predicted Next-Day Return
```

We also establish an important principle:

> A machine-learning model should be compared against simple baselines.

Possible baseline models:

- Previous return
- Mean historical return
- Always predict approximately zero return

This allows participants to determine whether their more complicated model actually learned something useful.

---

# 8. Part 3 — Model Crash Course
**~16 minutes**

Participants receive a short conceptual introduction to several model families.

The intention is not to provide a full mathematical lecture on each architecture, but to give participants enough understanding to make informed modelling decisions.

## Model A — Linear / Ridge Regression

Concept:

$$
\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

### Strengths

- Fast
- Simple
- Interpretable
- Difficult to massively overfit compared with larger models
- Strong baseline

### Weaknesses

- Limited ability to model complex nonlinear relationships

This serves as the workshop's potential **underdog model**.

---

## Model B — Random Forest

Introduce decision trees and ensemble learning.

```text
             Dataset
                ↓
       ┌────────┼────────┐
     Tree      Tree      Tree
       └────────┼────────┘
                ↓
        Combined Prediction
```

### Strengths

- Handles nonlinear relationships
- Relatively robust
- Easy to train
- Works well with tabular data

### Weaknesses

- Does not naturally understand sequential dependencies
- Can become large and computationally inefficient

---

## Model C — XGBoost / Gradient Boosting

Explain the difference between bagging and boosting.

**Random Forest:**

> Multiple trees independently make predictions.

**Gradient Boosting:**

> Each new tree attempts to correct errors made by previous trees.

Example tunable parameters:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- regularisation parameters

### Strengths

- Very strong on structured/tabular datasets
- Flexible
- Highly tunable

### Weaknesses

- Can overfit
- Requires good feature engineering
- Many interacting hyperparameters

---

## Model D — LSTM

Introduce sequence modelling.

```text
Day t-4 → Day t-3 → Day t-2 → Day t-1 → Today
                                        ↓
                                     Tomorrow
```

Potential tunable parameters:

- sequence length
- hidden size
- number of layers
- dropout
- learning rate
- batch size
- epochs

### Strengths

- Designed for sequential information
- Can learn temporal dependencies

### Weaknesses

- Slower to train
- More sensitive to preprocessing
- More opportunities for overfitting

---

## Model E — Wildcard Architecture

One additional architecture can be provided as an experimental option.

Possible choices:

- GRU
- 1D CNN
- MLP
- Support Vector Regression
- ExtraTrees

This gives teams an opportunity to test less obvious approaches.

---

# 9. Part 4 — Feature Engineering and Data Leakage
**~7 minutes**

Participants are introduced to basic financial features.

Starter features may include:

```text
1-day return
5-day return
Lagged returns

SMA 5
SMA 20

EMA 12
EMA 26

RSI

Rolling volatility

Volume change

Momentum indicators
```

Participants can modify or add features during the competition.

### Key lesson: more complex models are not automatically better.

A simple model using good features may outperform a sophisticated neural network using poor inputs.

---

## Time-Series Data Leakage

A major teaching point will be why random train/test splitting is inappropriate.

### Incorrect

```text
2017 ─ TRAIN
2021 ─ TRAIN
2019 ─ TEST
2024 ─ TRAIN
2020 ─ TEST
```

The model indirectly gains information from the future.

### Correct

```text
2014 ───────────── 2022 | 2023 ── 2024 | 2025
         TRAIN              VALIDATION      TEST
```

This introduces realistic time-series evaluation.

Walk-forward validation can also be briefly discussed.

---

# 10. Part 5 — Competition Briefing
**~6 minutes**

Participants form teams of approximately **3–5 people**.

Teams choose a **primary** and **secondary** model to focus their tuning on. Focus
slots guide exploration; they do not restrict which trained model a team may
submit. The notebook includes five starter candidates: Ridge, Random Forest,
XGBoost, LSTM, and one wildcard selected as GRU or CNN. The standard Colab/Conda
setup trains all five; a macOS pip venv skips XGBoost. Any available candidate may
be submitted if it has the best validation MAE.

Example:

| Architecture | Suggested primary focus slots |
|---|---:|
| Linear / Ridge | 2 |
| Random Forest | 2 |
| XGBoost | 3 |
| LSTM | 3 |
| Wildcard | 2 |

Each team selects:

- **Primary model**
- **Secondary model**

These slots only guide which models teams focus on; they do not limit what a team
can train or submit.

---

# 11. Starter Code

Every team receives a prepared notebook containing the full ML pipeline.

Example structure:

```text
workshop.ipynb

├── Load dataset
├── Explore data
├── Feature engineering
├── Train/validation split
├── Model definitions
│   ├── Linear / Ridge
│   ├── Random Forest
│   ├── XGBoost
│   ├── LSTM
│   └── Wildcard
├── Training
├── Validation
├── Evaluation
└── Export model
```

Participants do **not** need to build an entire ML system from scratch.

Instead, clearly labelled sections will be editable.

Example:

```python
# ==============================
# MODIFY THIS SECTION
# ==============================

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 200
}
```

They may also modify:

```python
FEATURES = [
    "return_1d",
    "return_5d",
    "sma_5",
    "sma_20",
    "rsi",
    "volatility"
]
```

Advanced teams can implement additional features or modify permitted parts of the architecture.

---

# 12. ROUND 1 — Human-Only Optimisation
**~20 minutes**

During Round 1:

> **Generative AI tools are not allowed.**

Participants may use:

- Workshop slides
- Documentation
- Provided cheatsheet
- Their teammates
- Workshop facilitators

They may experiment with:

- Hyperparameters
- Feature selection
- New features
- Scaling
- Lookback windows
- Sequence lengths
- Model depth
- Regularisation
- Training duration
- Early stopping
- Preprocessing

The objective is to make participants reason about the behaviour of their model before delegating the problem to an AI assistant.

---

# 13. Round 1 Submission

At the deadline, every team exports its model.

Rather than requesting only neural-network weights, each team will submit a reproducible model package because different architectures may require different preprocessing.

Conceptually:

```text
submission/
│
├── model/
│   └── trained_model
│
├── config.json
├── preprocessing
├── feature_config
└── predict.py
```

Every submission must expose the same prediction interface.

Conceptually:

```python
def predict(data):
    return predictions
```

The organiser evaluation system therefore does not need to know whether the submission internally uses:

- Ridge
- Random Forest
- XGBoost
- LSTM
- GRU
- CNN

It simply provides the same market data and collects predictions.

---

# 14. Private 2025 Evaluation
**~10 minutes**

Teams **do not receive the 2025 target values**.

The organiser system runs:

```text
Team Model
    ↓
Private 2025 Dataset
    ↓
Predictions
    ↓
Private Ground Truth
    ↓
Evaluation Metrics
```

A Round 1 leaderboard is shown.

Example:

| Team | Architecture | MAE ↓ | RMSE ↓ | Direction Accuracy ↑ |
|---|---|---:|---:|---:|
| Team A | XGBoost | ... | ... | ... |
| Team B | LSTM | ... | ... | ... |
| Team C | Ridge | ... | ... | ... |

Participants see their scores but **not the actual 2025 labels**.

---

# 15. ROUND 2 — AI Unlocked
**~16 minutes**

After the first evaluation:

> **AI tools are now allowed.**

Teams receive a fixed optimisation period.

They can provide the AI with:

- Their code
- Architecture
- Training performance
- Validation performance
- 2025 competition score
- Existing features
- Current hyperparameters

For example:

```text
Model: XGBoost

Training period:
2014–2022

Validation:
2023–2024

Validation MAE:
0.XXXX

2025 competition MAE:
0.XXXX

Current features:
[...]

Current hyperparameters:
[...]

How would you improve this model without using the 2025 labels?
```

Teams then decide which AI recommendations to implement.

---

# 16. AI Change Log

Before submitting Model V2, each team records what the AI suggested and what they actually changed.

Example:

```text
AI suggested:

✓ Reduce max_depth from 8 → 4
✓ Reduce learning rate
✓ Add early stopping
✓ Remove redundant moving-average features
✓ Increase lookback period
✗ Replace model with Transformer — rejected because of time
```

This is important because the workshop should not simply measure whether an AI-produced model receives a higher score.

We want participants to understand **what the AI changed and why**.

---

# 17. ROUND 2 — Same 2025 Evaluation
**~8 minutes**

Teams submit their AI-assisted model.

The models are evaluated against **exactly the same private 2025 competition dataset**.

Illustrative paired example only; compare each team's own Round 1 and Round 2
scores on the live board. Teams that submit in only one round cannot be paired.

Example:

| Team | Architecture | Human | Human + AI | Change |
|---|---|---:|---:|---:|
| Team A | XGBoost | 0.0131 | 0.0119 | -9.2% MAE |
| Team B | LSTM | 0.0127 | 0.0122 | -3.9% MAE |
| Team C | Ridge | 0.0124 | 0.0126 | +1.6% MAE |

This allows the workshop to investigate:

> **Did AI actually improve the model?**

Potential outcomes are all educational:

- AI significantly improves the model.
- AI makes only a minor difference.
- AI improves validation but hurts competition performance.
- AI makes the model worse.
- AI proposes changes that participants correctly reject.

---

# 18. Important Lesson — Benchmark Feedback

Because teams receive their first 2025 score before Round 2, the second optimisation phase has indirectly gained information about the competition benchmark.

The labels remain private, but teams now know whether their modelling decisions performed well or poorly against 2025.

This introduces another real ML concept:

## Leaderboard / Benchmark Overfitting

Repeatedly adjusting a system based on test performance can gradually make the model specialised to that benchmark.

Therefore:

> A better 2025 Round 2 score does not automatically prove that the model will generalise better to future market conditions.

---

# 19. 2026 Holdout Reveal

After both rounds, reveal the prepared **2026 year-to-date holdout**. It is hidden
from teams during optimisation. Run the evaluation live if time allows; otherwise
precompute it and reveal the result on slide 38.

Teams would not know about its results during either optimisation phase.

```text
Human Model ───────┐
                   ├──→ Hidden 2026 data
AI Model ──────────┘
```

The prepared demo rows are from different teams and models, so they illustrate
the board format but are **not a paired AI comparison**:

```text
                   2025        2026 holdout

Predict 0           0.01621     0.01500
Team Example · R1   0.01598     0.01517
Team Seq · R2       0.01624     0.01534
```

For a within-team comparison, use the Round 1 and Round 2 rows with the same team
name when both are present.

This creates an important discussion:

> Did AI make the model genuinely better, or did our optimisation process make it better specifically at the benchmark?

The 2026 evaluation would primarily be a **teaching demonstration**, rather than changing the announced 2025 competition result.

---

# 20. Historical Paper-Trading Simulation
**Included in the 8-minute results block**

After evaluating forecasting performance, models can be passed through a simple historical paper-trading simulation.

For example:

```text
Predicted return > threshold
        ↓
       LONG

Predicted return ≈ 0
        ↓
       CASH
```

Every strategy begins with the same simulated amount, for example:

```text
Starting portfolio: $10,000 virtual money
```

The simulation can display:

```text
Team A
$10,000 → $10,420

Team B
$10,000 → $9,870

Team C
$10,000 → $10,180
```

A simple **buy-and-hold strategy** can also be included as a benchmark.

This demonstrates another important concept:

> **Prediction accuracy and trading profitability are not the same objective.**

A model can achieve lower prediction error while still producing a worse trading strategy because of:

- Small prediction margins
- Excessive trading
- Volatility
- Transaction costs
- Risk
- Incorrect directional calls at important moments

The exercise will remain entirely historical and simulated.

**No participants will be encouraged to invest real money based on the models.**

---

# 21. Final Model Comparison

After the competition, the workshop returns to the architectures introduced earlier.

Participants compare their actual observations.

| Architecture | Typical Strength | Typical Weakness |
|---|---|---|
| Linear / Ridge | Fast, simple, interpretable | Limited nonlinear relationships |
| Random Forest | Robust and accessible | Weak temporal modelling |
| XGBoost | Strong on engineered tabular features | Depends heavily on features/tuning |
| LSTM | Models temporal sequences | Slower and easier to overfit |
| Wildcard | Different modelling assumptions | May require more experimentation |

Discussion questions:

- Did the most complex model perform best?
- Did feature engineering matter more than architecture?
- Which hyperparameters had the greatest effect?
- Which models appeared to overfit?
- What did AI change?
- Did AI suggestions consistently help?
- Were there AI suggestions that teams rejected?
- Did improvements on 2025 transfer to unseen data?
- Did the forecasting winner also produce the strongest trading simulation?

---

# 22. Main Takeaways

The workshop concludes by returning to:

# Can AI Predict the Market?

The intended conclusion is not simply **yes** or **no**.

Instead:

Machine-learning systems can identify patterns in historical financial data, but building a model that performs well historically is substantially easier than proving that those patterns will continue to work under unseen market conditions.

Participants should leave understanding that:

### 1. Architecture matters.

Different models have different inductive biases and strengths.

### 2. Data matters.

Poor or leaked data can make evaluation meaningless.

### 3. Features matter.

Good representations can allow relatively simple models to outperform sophisticated ones.

### 4. Validation matters.

Time-series evaluation must preserve temporal order.

### 5. Hyperparameters matter.

The same model architecture can produce very different results depending on how it is configured.

### 6. AI can help — but it is not automatically correct.

AI-generated optimisation still needs human judgement.

### 7. Benchmark performance is not the same as generalisation.

Repeatedly optimising against the same benchmark can create misleading improvements.

### 8. Prediction accuracy is not equivalent to profit.

Real financial systems must also consider risk, costs, market changes, and uncertainty.

---

# 23. Live 2-Hour Schedule

| Time | Activity |
|---|---|
| **0:00–0:06** | Opening market prediction challenge |
| **0:06–0:19** | Forecasting and ML fundamentals |
| **0:19–0:42** | Model and feature crash course |
| **0:42–0:48** | Team briefing and notebook tour |
| **0:48–1:08** | Round 1 — human-only optimisation |
| **1:08–1:18** | Private 2025 evaluation and leaderboard |
| **1:18–1:34** | Round 2 — AI-assisted optimisation |
| **1:34–1:42** | Round 2 evaluation, paper trading, and 2026 holdout reveal |
| **1:42–1:50** | Debrief |
| **1:50–2:00** | Conclusion and winner announcement |

This is the live run of show; see `FACILITATOR_GUIDE.md` §3 for delivery notes.

**Competition block: 0:42–1:42 (60 minutes total)** — 6 minutes for briefing,
36 minutes of model optimisation (20 + 16), and 18 minutes for evaluation and
results (10 + 8).

---

# 24. Competition Rules

### Round 1 — Human Only

Allowed:

- Workshop materials
- Official documentation
- Provided cheatsheet
- Discussion within team
- Assistance from workshop facilitators

Not allowed:

- ChatGPT
- Claude
- Gemini
- Codex
- Copilot Chat
- Other generative-AI assistants

### Round 2 — AI Assisted

AI tools become allowed for all teams.

Teams may:

- Ask for model suggestions
- Ask AI to inspect code
- Generate or modify features
- Tune hyperparameters
- Modify preprocessing
- Debug models
- Refactor permitted code

The competition remains time-limited.

If resources permit, organisers may provide teams with equivalent AI access or API credits.

If this is not practical, the same fixed AI optimisation window will still keep the workshop structure consistent.

---

# 25. Competition Fairness

All teams receive:

- The same source dataset
- The same hidden competition period
- The same starter notebook
- The same initial model implementations
- The same compute environment where possible
- The same Round 1 time
- The same Round 2 time
- The same evaluation system

Random seeds should be fixed where practical to reduce differences caused purely by training randomness.

Model size or training-time limits may also be introduced if necessary to prevent compute availability from determining the result.

---

# 26. Technical Preparation

Organisers will prepare:

### Dataset

- Clean historical OHLCV dataset
- 2014–2024 participant dataset
- Private 2025 targets
- Private 2026 YTD holdout, kept hidden until the final reveal

### Starter notebook

- Data loading
- Feature engineering
- Models
- Evaluation functions
- Visualisations
- Submission export

### Evaluation system

Receives submitted models and:

1. Loads model
2. Applies required preprocessing
3. Generates 2025 predictions
4. Calculates metrics
5. Updates leaderboard

### Competition leaderboard

Displays:

- Team
- Model architecture
- MAE
- RMSE
- Direction accuracy
- Human vs AI difference

### Paper-trading simulator

Uses the model's historical predictions to simulate a simple strategy using virtual capital.

---

# 27. Staffing

Recommended:

### 2 workshop leads / speakers

Responsibilities:

- Present concepts
- Explain competition
- Run evaluations
- Lead final discussion

Speaking sections can be divided between the two organisers rather than assigning one person as the sole presenter.

### Additional facilitators

Ideally several event and student facilitators can circulate between teams during the practical sessions.

Responsibilities:

- Help participants debug
- Explain model parameters
- Answer conceptual questions
- Ensure Round 1 AI restrictions are followed

Approximately one facilitator per 2–3 teams would be ideal, although the workshop can operate with fewer.

---

# 28. Venue Requirements

For a workshop held at HKU InnoWing, plan for:

- Room for approximately **20–30 participants**
- Team-style seating
- Projector / presentation display
- Stable Wi-Fi
- Power outlets or extension cables
- Space for approximately **4–10 teams**
- Approximately **2 hours of venue access**, preferably with some setup/cleanup time around the session

Participants should bring their own laptops.

Since training will primarily use Google Colab or lightweight models, specialised on-site GPU hardware is not required.

---

# 29. Budget

The workshop can be conducted with **little to no required budget**.

Core requirements:

- Venue
- Internet access
- Participant laptops
- Free Google Colab resources

Optional expenses:

- Small AI API allowance for teams
- Competition prizes
- Snacks / refreshments

AI API credits are optional rather than required for the workshop.

---

# 30. Safety and Financial Disclaimer

This workshop is an **educational machine-learning exercise**, not investment advice.

Participants will:

- Use historical market data
- Train experimental ML models
- Evaluate forecasting performance
- Use virtual portfolios only

There will be:

- **No real-money trading**
- **No request for participants to purchase securities**
- **No representation that workshop models can generate future profits**

The financial-market setting is used as an engaging real-world dataset for teaching machine learning, time-series modelling, model evaluation, and AI-assisted development.

---

# 31. Overall Workshop Flow

```text
               CAN AI PREDICT THE MARKET?
                          │
                          ▼
                What are we predicting?
                          │
                          ▼
                 Understand the data
                          │
                          ▼
                   Learn ML models
                          │
                          ▼
             Features + Time-Series Leakage
                          │
                          ▼
               TEAM MODEL FOCUS SLOTS
                          │
                          ▼
               ROUND 1 — HUMAN ONLY
                          │
                          ▼
                   SUBMIT MODEL V1
                          │
                          ▼
                PRIVATE 2025 TEST
                          │
                          ▼
                 ROUND 1 LEADERBOARD
                          │
                          ▼
                   🤖 AI UNLOCKED
                          │
                          ▼
              HUMAN + AI OPTIMISATION
                          │
                          ▼
                   SUBMIT MODEL V2
                          │
                          ▼
                 SAME 2025 TEST SET
                          │
                          ▼
                 ROUND 2 LEADERBOARD
                          │
                          ▼
                 WHAT DID AI CHANGE?
                          │
                          ▼
            HISTORICAL PAPER TRADING
                          │
                          ▼
          OPTIONAL UNSEEN GENERALISATION
                          │
                          ▼
                COMPARE ARCHITECTURES
                          │
                          ▼
                  FINAL DISCUSSION
                          │
                          ▼
       ACCURACY ≠ GUARANTEED PROFITABILITY
```

# 32. Workshop Summary

**Can AI Predict the Market? — A Hands-On ML Stock Forecasting Challenge** combines a short introduction to machine learning with an interactive model-development competition.

Rather than having participants passively copy code, teams must understand, modify, train, and evaluate their own models.

The Human vs. Human+AI structure also makes the workshop relevant to current AI-assisted development workflows by allowing participants to directly investigate whether AI tools improve their work and, more importantly, **why**.

The intended outcome is not to produce a profitable trading algorithm.

The intended outcome is for participants to leave with a practical understanding of:

**machine learning + time series + model selection + feature engineering + hyperparameter tuning + validation + AI-assisted coding + critical evaluation.**
