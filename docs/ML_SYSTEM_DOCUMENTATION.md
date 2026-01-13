# NBA Player Prop ML System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Leakage Prevention](#data-leakage-prevention-critical)
3. [Data Engineering Pipeline](#data-engineering-pipeline)
4. [Label Engineering](#label-engineering)
5. [Model Architecture](#model-architecture)
6. [Cross-Validation Strategy](#cross-validation-strategy)
7. [Training the Models](#training-the-models)
8. [Evaluating Performance](#evaluating-performance)
9. [Meta-Ensemble Training](#meta-ensemble-training)
10. [Directory Structure](#directory-structure)
11. [Quick Reference Commands](#quick-reference-commands)

---

## System Overview

The ML system consists of:

1. **4 Prop-Specific XGBoost Models**: Separate models for PTS, REB, AST, 3PM
2. **Probability Calibrators**: Isotonic regression to convert raw outputs to true probabilities
3. **Meta-Ensemble (Optional)**: Stacking model that learns when to trust each base model
4. **LLM Integration**: ML signals are injected into Gemini prompts for enhanced reasoning

### Architecture Diagram

```
Raw Data (Kaggle CSVs)
        │
        ▼
┌─────────────────────────────────┐
│   Feature Engineering           │
│   (nba_data_builder.py)         │
│   • Rolling averages (L5, L10)  │
│   • Uses ONLY PREVIOUS games    │  ◄── DATA LEAKAGE PREVENTION
│   • rowsBetween(-5, -1)         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Label Engineering             │
│   (data_preprocessor.py)        │
│   • Join features with actuals  │
│   • outcome = actual > line     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Temporal Train/Test Split     │
│   (train_all_models.py)         │
│   • 80% train (earlier dates)   │  ◄── DATA LEAKAGE PREVENTION
│   • 20% test (later dates)      │
└───────────────┬─────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌─────────────┐   ┌─────────────┐
│ TimeSeriesCV│   │ Final Train │
│ (validation)│   │ + Calibrate │
└─────────────┘   └─────────────┘
```

---

## Data Leakage Prevention (CRITICAL)

**Data leakage** occurs when information from the future (outcomes) bleeds into training features. This is the #1 cause of overly optimistic model performance that fails in production.

### How This System Prevents Leakage:

#### 1. Feature Engineering Uses ONLY Previous Games

In `nba_data_builder.py`, all rolling statistics use `rowsBetween(-5, -1)`:

```python
# This ONLY looks at the 5 games BEFORE the current game
window_5 = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-5, -1))  # Rows -5 to -1 (excludes row 0 = current game)

# L5_Avg_PTS is the average of the PREVIOUS 5 games
df = df.withColumn("L5_Avg_PTS", avg("points").over(window_5))
```

**Key insight**: When predicting for game on 2024-12-15, L5_Avg_PTS uses games from 2024-12-10 through 2024-12-14, NOT including 2024-12-15.

#### 2. Temporal Train/Test Split (NOT Random)

In `train_all_models.py`:

```python
def temporal_train_test_split(df, test_ratio=0.2):
    """
    CRITICAL: Splits by DATE, not randomly.
    
    Train: 2024-12-01 to 2024-12-29
    Test:  2024-12-30 to 2025-01-07
    """
    df = df.sort_values('game_date')
    split_idx = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx]  # Earlier dates
    test_df = df.iloc[split_idx:]   # Later dates
```

**Why this matters**: Random splits would leak information. If model trains on games from Jan 1-7 and tests on Dec 25, it has "seen the future."

#### 3. TimeSeriesSplit Cross-Validation

In `model_trainer.py`:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Fold 1: Train on games 1-20,    Validate on 21-30
# Fold 2: Train on games 1-40,    Validate on 41-50
# Fold 3: Train on games 1-60,    Validate on 61-70
# etc.
```

**Key insight**: TimeSeriesSplit ensures validation data ALWAYS comes after training data chronologically.

#### 4. Label Generation Joins on Date + Player

In `data_preprocessor.py`:

```python
# Join features with actuals ONLY for matching dates
labeled = features_df.merge(
    actuals_df,
    on=['personId', 'game_date'],  # Exact match required
    how='inner'
)

# Label is created from ACTUAL game results
labeled['outcome'] = (labeled['actual_value'] > labeled['Line']).astype(int)
```

**Key insight**: The outcome (OVER=1, UNDER=0) is derived from actual game results that happened AFTER the features were generated.

---

## Data Engineering Pipeline

### Source Data (kaggle_data/)

| File | Contents | Records |
|------|----------|---------|
| `PlayerStatistics.csv` | Per-game player stats | 1.6M+ |
| `TeamStatistics.csv` | Per-game team stats | ~2K |
| `Games.csv` | Game metadata | ~2K |

### Feature Generation Process

**Location**: `process/nba_data_builder.py`

1. **Load Raw Data**: Player and team statistics
2. **Calculate Rolling Averages**:
   - `L5_Avg_*`: Average of last 5 games
   - `L10_Avg_*`: Average of last 10 games
   - `Season_Avg_*`: Season-to-date average
   - `L5_StdDev_*`: Variance in last 5 games
3. **Calculate Matchup Features**:
   - `Opp_L5_Allowed_*`: What opponent allows to this position
   - `Opp_Rank_Allows_*`: Opponent's defensive ranking
4. **Add Context**:
   - `is_b2b`: Back-to-back game flag
   - `rest_days`: Days since last game
   - `home`: Home/away indicator
5. **Add Market Signals**:
   - `Edge`: (L5_Avg - Line)
   - `EV`: Expected value calculation
   - `Kelly`: Kelly criterion sizing
   - `quality_score`: Combined ranking metric

### Total Features: 143

---

## Label Engineering

**Location**: `ml_pipeline/data_preprocessor.py`

### Label Definition

```python
# For each prop, the label is binary:
outcome = 1  # if actual_value > Line (OVER hit)
outcome = 0  # if actual_value <= Line (UNDER hit)

# Example:
# Player: LeBron James
# Prop: PTS, Line: 25.5
# Actual: 28 points
# Label: 1 (OVER)
```

### Prop-to-Actual Mapping

| Prop Type | Actual Column |
|-----------|---------------|
| PTS | `points` |
| REB | `reboundsTotal` |
| AST | `assists` |
| 3PM | `threePointersMade` |

### Label Balance

The current dataset shows relatively balanced labels:
- PTS: 48.9% OVER
- REB: 56.0% OVER
- AST: 46.3% OVER
- 3PM: 44.4% OVER

---

## Model Architecture

### Base Models: XGBoost Classifiers

**Location**: `ml_pipeline/model_trainer.py`

```python
DEFAULT_PARAMS = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Optimize for probability accuracy
    'n_estimators': 500,             # Number of trees
    'max_depth': 6,                  # Tree depth (prevent overfitting)
    'learning_rate': 0.05,           # Conservative learning rate
    'subsample': 0.8,                # Row sampling
    'colsample_bytree': 0.8,         # Column sampling
    'min_child_weight': 3,           # Minimum samples per leaf
    'reg_alpha': 0.1,                # L1 regularization
    'reg_lambda': 1.0,               # L2 regularization
}
```

### Prop-Specific Feature Sets

Each model uses a tailored feature set:

**PTS Model (27 features):**
- Core features (context, market signals)
- Recent form: L5/L10/Season PTS averages
- Efficiency: TS%, eFG%, PTS_per100
- Matchup: Opp_Rank_Allows_PTS, Opp_Def_Rating
- Pace: Team and opponent pace

**REB Model (25 features):**
- Core features
- Recent form: L5/L10/Season REB averages
- Matchup: Opp_Rank_Allows_REB

**AST Model (25 features):**
- Core features
- Recent form: L5/L10/Season AST averages
- Matchup: Opp_Rank_Allows_AST

**3PM Model (25 features):**
- Core features
- Recent form: L5/L10/Season 3PM averages
- Shooting profile: 3PA_Rate, 3PM_per100

### Probability Calibration

**Location**: `ml_pipeline/calibrator.py`

XGBoost outputs are NOT well-calibrated probabilities. We use **isotonic regression** to fix this:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(
    y_min=0.01,  # Avoid 0% predictions
    y_max=0.99,  # Avoid 100% predictions
)
calibrator.fit(raw_probabilities, actual_outcomes)

# Now 70% prediction = ~70% actual hit rate
calibrated_prob = calibrator.predict(raw_prob)
```

---

## Cross-Validation Strategy

### TimeSeriesSplit (Prevents Leakage)

```python
from sklearn.model_selection import TimeSeriesSplit

# Example with 100 samples, 5 splits:
# 
# Fold 1: Train [0:20],   Validate [20:40]
# Fold 2: Train [0:40],   Validate [40:60]
# Fold 3: Train [0:60],   Validate [60:80]
# Fold 4: Train [0:80],   Validate [80:100]
#
# Note: Training set GROWS, validation is always FUTURE data
```

### Metrics Tracked

| Metric | Purpose | Target |
|--------|---------|--------|
| Log Loss | Probability accuracy | < 0.65 |
| Brier Score | Calibration quality | < 0.22 |
| ROC-AUC | Discrimination | > 0.58 |
| ECE | Expected Calibration Error | < 0.05 |

---

## Training the Models

### Option 1: Full Retraining

```bash
cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Step 1: Regenerate labeled dataset (if new training data available)
python -c "
from ml_pipeline.data_preprocessor import prepare_labeled_dataset
prepare_labeled_dataset()
"

# Step 2: Train all models
python ml_pipeline/train_all_models.py

# Step 3: Train meta-ensemble (optional)
python ml_pipeline/ensemble.py
```

### Option 2: Incremental Training

To add new data and retrain:

```bash
# 1. Ensure new training CSVs are in datasets/training_inputs/
# 2. Regenerate labeled dataset
python ml_pipeline/data_preprocessor.py

# 3. Retrain with new data
python ml_pipeline/train_all_models.py
```

### Option 3: Retrain Single Model

```python
from ml_pipeline.model_trainer import PropModelTrainer
from ml_pipeline.calibrator import ProbabilityCalibrator
import pandas as pd

# Load data
df = pd.read_csv('datasets/ml_training/labeled_data_LATEST.csv')

# Filter to one prop
pts_df = df[df['Prop_Type_Normalized'] == 'PTS']

# Train
trainer = PropModelTrainer('PTS')
X, y = trainer.prepare_data(pts_df)
trainer.train(X, y)
trainer.save('ml_models/prop_models/pts_model.pkl')
```

---

## Evaluating Performance

### Quick Evaluation

```python
from ml_pipeline.evaluator import ModelEvaluator
from ml_pipeline.inference import get_ml_predictor
import pandas as pd

# Load labeled data
df = pd.read_csv('datasets/ml_training/labeled_data_20251211_204331.csv')

# Get predictions
predictor = get_ml_predictor()
predictions = predictor.predict(df)

# Evaluate
y_true = predictions['outcome'].values
y_prob = predictions['ml_prob_over'].values
y_pred = (predictions['ml_prediction'] == 'OVER').astype(int).values

metrics = ModelEvaluator.evaluate(y_true, y_prob, y_pred)
```

### Evaluation by Prop Type

```python
prop_results = ModelEvaluator.evaluate_by_prop_type(predictions)

for prop, metrics in prop_results.items():
    print(f"{prop}: Accuracy={metrics['accuracy']:.1%}, ROC-AUC={metrics['roc_auc']:.3f}")
```

### Generate Full Report

```python
report = ModelEvaluator.generate_report(
    predictions,
    save_path='training_reports/ml_evaluation_report.md'
)
```

### Backtest Over Time

```python
daily_results = ModelEvaluator.backtest(
    predictions,
    min_confidence=0.60  # Only bet when P > 60%
)

print(daily_results[['date', 'n_bets', 'n_wins', 'profit', 'cumulative_profit']])
```

---

## Meta-Ensemble Training

The meta-ensemble learns when to trust each base model by combining:
- Base model predictions (`ml_prob_over`)
- Context features (tier, reliability, edge, rest days)

### Train Meta-Ensemble

```bash
python ml_pipeline/ensemble.py
```

### What the Meta-Ensemble Learns

From training output:
```
Feature weights:
   ml_prob_over: +1.4708          # Trust the base model
   quality_score: +0.1548         # Higher quality = more confident
   Reliability_Tag_encoded: -0.1255  # VOLATILE players get downweighted
   home: +0.1061                  # Slight home advantage
   rest_days: +0.1051             # More rest = better
```

### Use Meta-Ensemble in Predictions

```python
from ml_pipeline.ensemble import MetaEnsemble
from ml_pipeline.inference import get_ml_predictor

# Get base predictions
predictor = get_ml_predictor()
predictions = predictor.predict(features_df)

# Enhance with meta-ensemble
ensemble = MetaEnsemble.load()
enhanced = ensemble.enhance_predictions(predictions)

# Now has: meta_prob_over, meta_prediction, meta_confidence
```

---

## Directory Structure

```
NBA Player Prop Analyst/
├── datasets/                    # All data files
│   ├── ml_training/            # Labeled data for ML
│   ├── training_inputs/        # Raw training CSVs
│   └── previews/               # Preview outputs
├── kaggle_data/                # Source data from Kaggle
├── ml_models/                  # Trained model artifacts
│   ├── prop_models/           # XGBoost models (.pkl)
│   ├── calibrators/           # Probability calibrators
│   ├── feature_config.py      # Feature definitions
│   └── meta_model.pkl         # Stacking ensemble
├── ml_pipeline/                # ML training & inference code
│   ├── data_preprocessor.py   # Label engineering
│   ├── model_trainer.py       # XGBoost training
│   ├── calibrator.py          # Probability calibration
│   ├── inference.py           # Production predictions
│   ├── evaluator.py           # Performance metrics
│   ├── ensemble.py            # Meta-ensemble
│   └── train_all_models.py    # Training orchestrator
├── process/                    # Core pipeline code
│   ├── nba_ai_pipeline.py     # Main LLM pipeline
│   ├── nba_data_builder.py    # Feature engineering
│   ├── shared_config.py       # Configuration
│   └── training_reporter.py   # Report generation
├── training_reports/           # Generated reports
└── logs/                       # Runtime logs
```

---

## Quick Reference Commands

```bash
# Activate environment
cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Generate labeled dataset
python ml_pipeline/data_preprocessor.py

# Train all models
python ml_pipeline/train_all_models.py

# Train meta-ensemble
python ml_pipeline/ensemble.py

# Test inference
python ml_pipeline/inference.py

# Quick evaluation
python -c "
from ml_pipeline.evaluator import ModelEvaluator
from ml_pipeline.inference import get_ml_predictor
import pandas as pd

df = pd.read_csv('datasets/ml_training/labeled_data_20251211_204331.csv')
predictor = get_ml_predictor()
predictions = predictor.predict(df)
ModelEvaluator.evaluate(predictions['outcome'], predictions['ml_prob_over'])
"
```

---

## Warnings & Limitations

### Current Data Size
- Only 178 labeled samples
- Models may be overfitting
- Need more historical data for robust training

### Known Issues
- SHAP analysis disabled (install with `pip install shap`)
- 3PM model has low feature importance (small sample)

### Recommendations
1. **Collect more data**: Run historical training for more dates
2. **Monitor calibration**: Check ECE regularly
3. **A/B test**: Compare ML+LLM vs LLM-only predictions
4. **Retrain monthly**: As new data accumulates
