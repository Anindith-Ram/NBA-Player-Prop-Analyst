# NBA Player Prop Analyst - Complete System Guide

> **Last Updated**: December 11, 2025  
> **System Version**: 3.0 (PySpark Edition with ML Pipeline)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [DATA LEAKAGE PREVENTION (CRITICAL)](#data-leakage-prevention-critical)
3. [Directory Structure (Reorganized)](#directory-structure-reorganized)
4. [Data Engineering Pipeline](#data-engineering-pipeline)
5. [Label Engineering](#label-engineering)
6. [Cross-Validation Strategy](#cross-validation-strategy)
7. [Training the Models](#training-the-models)
8. [Training the Ensemble Model](#training-the-ensemble-model)
9. [Performance Tracking](#performance-tracking)
10. [Quick Reference Commands](#quick-reference-commands)

---

## Executive Summary

This system consists of:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Engineering** | PySpark | 143 features from Kaggle CSVs + Odds API |
| **Base Models** | XGBoost (4 models) | PTS, REB, AST, 3PM binary classifiers |
| **Calibrators** | Isotonic Regression | Convert raw outputs to true probabilities |
| **Meta-Ensemble** | Logistic Regression | Learns when to trust each base model |
| **LLM Integration** | Google Gemini | ML signals enhance reasoning |

### Current Performance Summary

From `training_reports/historical_training_20251211_192532.json`:
- **178 labeled samples** (need more data!)
- **Overall Accuracy**: ~35% (reflects small sample size)
- **Recommendations**: Collect 1,000+ samples for robust training

---

## DATA LEAKAGE PREVENTION (CRITICAL)

**Data leakage** is the #1 killer of ML models in production. When information from the future (outcomes) bleeds into training features, you get overly optimistic metrics that fail in real betting.

### ✅ How This System Prevents Leakage

#### 1. Feature Engineering Uses ONLY Previous Games

In `process/nba_data_builder.py`, ALL rolling statistics use `rowsBetween(-5, -1)`:

```python
# This ONLY looks at the 5 games BEFORE the current game
window_5 = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-5, -1))  # Rows -5 to -1 (excludes row 0 = current game)

# L5_Avg_PTS is the average of the PREVIOUS 5 games
df = df.withColumn("L5_Avg_PTS", avg("points").over(window_5))
```

**Key insight**: When predicting for game on 2024-12-15, `L5_Avg_PTS` uses games from 2024-12-10 through 2024-12-14, **NOT including 2024-12-15**.

#### 2. Temporal Train/Test Split (NOT Random)

In `ml_pipeline/model_trainer.py`:

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

**Why this matters**: Random splits would leak information. If the model trains on games from Jan 1-7 and tests on Dec 25, it has "seen the future."

#### 3. TimeSeriesSplit Cross-Validation

In `ml_pipeline/model_trainer.py`:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Fold 1: Train on games 1-20,    Validate on 21-30
# Fold 2: Train on games 1-40,    Validate on 41-50
# Fold 3: Train on games 1-60,    Validate on 61-70
# etc.
```

**Key insight**: `TimeSeriesSplit` ensures validation data ALWAYS comes after training data chronologically.

#### 4. Label Generation Joins on Date + Player

In `ml_pipeline/data_preprocessor.py`:

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

### ⚠️ Potential Leakage Risks to Monitor

| Risk | Status | Mitigation |
|------|--------|------------|
| Rolling window includes current game | ✅ SAFE | `rowsBetween(-5, -1)` excludes row 0 |
| Random train/test split | ✅ SAFE | Uses `temporal_train_test_split` |
| Season averages include current game | ✅ SAFE | `rowsBetween(unboundedPreceding, -1)` |
| Cross-validation shuffles data | ✅ SAFE | Uses `TimeSeriesSplit` |
| Feature from future opponent stats | ⚠️ CHECK | Ensure opponent features use pre-game data |

---

## Directory Structure (Reorganized)

### Current Structure (Cluttered)
```
NBA Player Prop Analyst/
├── archive/                    # Old files (keep for reference)
├── datasets/
│   ├── ml_training/           # Labeled data
│   ├── previews/              # Preview outputs
│   └── training_inputs/       # Raw training CSVs (30+ files!)
├── kaggle_data/               # Source data
├── ml_models/                 # Trained models
├── ml_pipeline/               # ML code
├── process/                   # Core pipeline code
├── training_reports/          # JSON reports
└── [various files at root]
```

### Recommended Reorganization

```
NBA Player Prop Analyst/
│
├── config/                    # All configuration files
│   ├── shared_config.py      # API keys, paths, caching
│   ├── feature_config.py     # Feature definitions
│   └── trained_dates.json    # Tracking which dates are trained
│
├── data/                      # All data files (clean separation)
│   ├── raw/                  # Immutable source data
│   │   ├── kaggle/          # PlayerStatistics.csv, TeamStatistics.csv
│   │   └── odds/            # Historical odds snapshots
│   │
│   ├── processed/            # Intermediate processed data
│   │   ├── features/        # Features without labels
│   │   └── training_inputs/ # Training CSVs (consolidated)
│   │
│   └── labeled/              # Final labeled datasets
│       └── labeled_data_YYYYMMDD.csv
│
├── models/                    # All trained model artifacts
│   ├── base/                 # XGBoost prop models
│   │   ├── pts_model.pkl
│   │   ├── reb_model.pkl
│   │   ├── ast_model.pkl
│   │   └── 3pm_model.pkl
│   │
│   ├── calibrators/          # Probability calibrators
│   │   ├── pts_calibrator.pkl
│   │   ├── reb_calibrator.pkl
│   │   ├── ast_calibrator.pkl
│   │   └── 3pm_calibrator.pkl
│   │
│   └── ensemble/             # Meta-ensemble
│       └── meta_model.pkl
│
├── src/                       # All source code
│   ├── data/                 # Data engineering
│   │   ├── __init__.py
│   │   ├── builder.py       # nba_data_builder.py
│   │   └── preprocessor.py  # data_preprocessor.py
│   │
│   ├── models/               # Model training
│   │   ├── __init__.py
│   │   ├── trainer.py       # model_trainer.py
│   │   ├── calibrator.py
│   │   ├── ensemble.py
│   │   └── inference.py
│   │
│   ├── evaluation/           # Performance tracking
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── reporter.py      # training_reporter.py
│   │
│   └── pipeline/             # End-to-end pipelines
│       ├── __init__.py
│       ├── ai_pipeline.py   # nba_ai_pipeline.py
│       └── optimizer.py
│
├── reports/                   # All generated reports
│   ├── training/             # Historical training reports
│   ├── evaluation/           # Model evaluation reports
│   └── predictions/          # Daily predictions
│
├── logs/                      # Runtime logs
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── scripts/                   # Entry point scripts
│   ├── train_models.py       # Train all models
│   ├── train_ensemble.py     # Train meta-ensemble
│   ├── evaluate.py           # Run evaluation
│   └── predict.py            # Make predictions
│
├── tests/                     # Unit tests
│   ├── test_data_leakage.py  # Critical: verify no leakage
│   ├── test_features.py
│   └── test_models.py
│
├── requirements.txt
├── README.md
├── SYSTEM_GUIDE.md           # This file
└── .gitignore
```

---

## Data Engineering Pipeline

### Overview

```
Raw Data (Kaggle CSVs)
        │
        ▼
┌─────────────────────────────────┐
│   PySpark Feature Engineering   │
│   (src/data/builder.py)         │
│   • Rolling averages (L5, L10)  │
│   • Uses ONLY PREVIOUS games    │  ◄── DATA LEAKAGE PREVENTION
│   • rowsBetween(-5, -1)         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Label Engineering             │
│   (src/data/preprocessor.py)    │
│   • Join features with actuals  │
│   • outcome = actual > line     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Temporal Train/Test Split     │
│   (src/models/trainer.py)       │
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

### Feature Categories (143 Total)

| Category | Features | Example |
|----------|----------|---------|
| **Context** | 3 | `home`, `is_b2b`, `rest_days` |
| **Market Signals** | 7 | `Edge`, `Z_Score`, `EV`, `Kelly`, `quality_score` |
| **Player Classification** | 3 | `Player_Tier_encoded`, `Reliability_Tag_encoded`, `CV_Score` |
| **Rolling Averages** | 21 | `L5_Avg_PTS`, `L10_Avg_REB`, `Season_Avg_AST` |
| **Home/Away Splits** | 8 | `L5_Avg_PTS_SameVenue`, `Season_Avg_REB_SameVenue` |
| **Volatility** | 8 | `L5_StdDev_PTS`, `L10_StdDev_3PM` |
| **Form Indicators** | 4 | `Form_PTS`, `Form_REB`, `Form_AST`, `Form_3PM` |
| **Efficiency Metrics** | 6 | `Player_TS_Pct`, `Player_eFG_Pct`, `Player_PTS_per100` |
| **Matchup Defense** | 20+ | `Opp_Rank_Allows_PTS`, `Opp_L5_Allowed_REB` |
| **Pace** | 8 | `L5_Team_Pace`, `Opp_L10_Team_Pace` |
| **Team Context** | 20+ | `Team_Def_Rating`, `Opp_Def_Rating` |

### Rolling Window Logic (NO LEAKAGE)

```python
# Window definitions in nba_data_builder.py

# Last 5 games (EXCLUDES current game with -1 ending)
window_5 = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-5, -1))

# Last 10 games
window_10 = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-10, -1))

# Season to date (all previous games)
window_season = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(Window.unboundedPreceding, -1))

# Home/Away splits (partitioned by venue type)
window_5_home = (Window
    .partitionBy("personId", "home")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-5, -1))
```

---

## Label Engineering

### Label Definition

```python
# Binary classification: OVER (1) or UNDER (0)

outcome = 1  # if actual_value > Line (OVER hit)
outcome = 0  # if actual_value <= Line (UNDER hit)

# Example:
# Player: LeBron James
# Prop: PTS, Line: 25.5
# Actual: 28 points
# Label: 1 (OVER)
```

### Prop-to-Actual Mapping

| Prop Type | Actual Column in CSV |
|-----------|---------------------|
| PTS | `points` |
| REB | `reboundsTotal` |
| AST | `assists` |
| 3PM | `threePointersMade` |

### Label Creation Process

```python
# ml_pipeline/data_preprocessor.py

def create_outcome_labels_pandas(features_df, actuals_df):
    # 1. Join features with actuals on personId + game_date
    labeled = features_df.merge(
        actuals_df,
        on=['personId', 'game_date'],
        how='inner'
    )
    
    # 2. Map prop type to actual column
    def get_actual_value(row):
        prop_type = row['Prop_Type_Normalized']
        if prop_type == 'PTS':
            return row['points']
        elif prop_type == 'REB':
            return row['reboundsTotal']
        # ... etc
    
    labeled['actual_value'] = labeled.apply(get_actual_value, axis=1)
    
    # 3. Create binary outcome
    labeled['outcome'] = (labeled['actual_value'] > labeled['Line']).astype(int)
    
    return labeled
```

### Current Label Balance

From your data:
- **PTS**: 48.9% OVER (well balanced)
- **REB**: 56.0% OVER (slight imbalance)
- **AST**: 46.3% OVER (well balanced)
- **3PM**: 44.4% OVER (well balanced)

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

### Implementation in model_trainer.py

```python
def cross_validate(self, X, y, n_splits=5):
    """
    Performs time-series cross-validation.
    
    CRITICAL: Uses TimeSeriesSplit to prevent data leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    log_losses = []
    brier_scores = []
    aucs = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**self.params)
        model.fit(X_train, y_train, verbose=False)
        
        probs = model.predict_proba(X_val)[:, 1]
        
        log_losses.append(log_loss(y_val, probs))
        brier_scores.append(brier_score_loss(y_val, probs))
        if len(set(y_val)) > 1:
            aucs.append(roc_auc_score(y_val, probs))
    
    return {
        'log_loss': np.mean(log_losses),
        'brier_score': np.mean(brier_scores),
        'roc_auc': np.mean(aucs),
    }
```

### Why NOT Use k-Fold or Random Splits

| Method | Problem |
|--------|---------|
| **k-Fold CV** | Shuffles data, train set can contain future games |
| **Random Split** | Test set can be chronologically before train set |
| **Stratified CV** | Still shuffles within strata, causes leakage |
| **TimeSeriesSplit** | ✅ Guarantees train < test chronologically |

---

## Training the Models

### Full Training Pipeline

```bash
# Step 1: Activate environment
cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Step 2: Generate labeled dataset (if new training data available)
python ml_pipeline/data_preprocessor.py

# Step 3: Train all 4 prop models + calibrators
python ml_pipeline/train_all_models.py

# Step 4: Train meta-ensemble (optional, needs ML predictions)
python ml_pipeline/ensemble.py
```

### Training a Single Model

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

# Calibrate
calibrator = ProbabilityCalibrator()
y_probs = trainer.predict_proba(X)
calibrator.fit(y.values, y_probs)

# Save
trainer.save('ml_models/prop_models/pts_model.pkl')
calibrator.save('ml_models/calibrators/pts_calibrator.pkl')
```

### Incremental Training (Adding New Data)

```bash
# 1. Generate new training CSVs for new dates
# (Run nba_data_builder.py for new date range)

# 2. Regenerate labeled dataset (combines all CSVs)
python ml_pipeline/data_preprocessor.py

# 3. Retrain with expanded dataset
python ml_pipeline/train_all_models.py
```

### Hyperparameter Tuning (Optional)

The default hyperparameters in `model_trainer.py`:

```python
DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 500,           # Number of trees
    'max_depth': 6,                # Tree depth (prevents overfitting)
    'learning_rate': 0.05,         # Conservative learning rate
    'subsample': 0.8,              # Row sampling
    'colsample_bytree': 0.8,       # Column sampling
    'min_child_weight': 3,         # Minimum samples per leaf
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 1.0,             # L2 regularization
}
```

For hyperparameter tuning with Optuna:

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    
    trainer = PropModelTrainer('PTS', params)
    X, y = trainer.prepare_data(df)
    cv_metrics = trainer.cross_validate(X, y, n_splits=5)
    
    return cv_metrics['log_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
```

---

## Training the Ensemble Model

The meta-ensemble is a **stacking model** that learns when to trust each base model.

### Architecture

```
Level 1 (Base Models):           Level 2 (Meta-Model):
┌──────────────┐
│ PTS XGBoost  │──┐
└──────────────┘  │
┌──────────────┐  │    ┌────────────────────┐
│ REB XGBoost  │──┼───▶│ Logistic Regression│──▶ Final P(OVER)
└──────────────┘  │    │ (meta_model.pkl)   │
┌──────────────┐  │    └────────────────────┘
│ AST XGBoost  │──┤            ▲
└──────────────┘  │            │
┌──────────────┐  │    ┌───────┴────────┐
│ 3PM XGBoost  │──┘    │ Context Features│
└──────────────┘       │ • Player Tier   │
                       │ • Reliability   │
                       │ • Edge, EV      │
                       │ • Rest days     │
                       └────────────────┘
```

### Training the Meta-Ensemble

```bash
# Requires ML predictions with known outcomes
python ml_pipeline/ensemble.py
```

Or programmatically:

```python
from ml_pipeline.data_preprocessor import load_training_csvs, load_actuals_pandas, create_outcome_labels_pandas, encode_categoricals
from ml_pipeline.inference import get_ml_predictor
from ml_pipeline.ensemble import train_meta_ensemble

# 1. Load and label data
features_df = load_training_csvs()
actuals_df = load_actuals_pandas()
labeled_df = create_outcome_labels_pandas(features_df, actuals_df)
labeled_df = encode_categoricals(labeled_df)

# 2. Get base model predictions
predictor = get_ml_predictor()
predictions_df = predictor.predict(labeled_df)

# 3. Train meta-ensemble
ensemble = train_meta_ensemble(predictions_df, save=True)
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

### Using the Meta-Ensemble

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
print(enhanced[['Player', 'Prop_Type_Normalized', 'ml_prob_over', 'meta_prob_over']])
```

---

## Performance Tracking

### Metrics to Track

| Metric | Purpose | Target |
|--------|---------|--------|
| **Log Loss** | Probability accuracy | < 0.65 |
| **Brier Score** | Calibration quality | < 0.22 |
| **ROC-AUC** | Discrimination ability | > 0.58 |
| **ECE** | Expected Calibration Error | < 0.05 |
| **Hit Rate @60%** | Betting edge at high confidence | > 55% |
| **Simulated ROI** | Profitability at -110 odds | > -2% |

### Quick Evaluation Script

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

### Performance by Prop Type

```python
prop_results = ModelEvaluator.evaluate_by_prop_type(predictions)

for prop, metrics in prop_results.items():
    print(f"{prop}: Accuracy={metrics['accuracy']:.1%}, ROC-AUC={metrics['roc_auc']:.3f}")
```

### Backtest Over Time

```python
daily_results = ModelEvaluator.backtest(
    predictions,
    min_confidence=0.60  # Only bet when P > 60%
)

print(daily_results[['date', 'n_bets', 'n_wins', 'profit', 'cumulative_profit']])
```

### Generate Full Report

```python
report = ModelEvaluator.generate_report(
    predictions,
    save_path='reports/evaluation/ml_evaluation_report.md'
)
```

### Training Reports Location

Your historical training results are stored in `training_reports/`:

```json
{
  "training_completed_at": "2025-12-11T19:25:32.156213",
  "dates_evaluated": 7,
  "total_predictions": 26,
  "total_hits": 9,
  "overall_accuracy": 34.61538461538461,
  "new_case_studies": 8
}
```

---

## Quick Reference Commands

### Environment Setup

```bash
cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

### Data Generation (NEW - Easy Interface)

```bash
# Generate training data for a single date
python scripts/generate_data.py --date 2024-12-15

# Generate for a date range
python scripts/generate_data.py --range 2024-12-01 2024-12-31

# Generate for last N days
python scripts/generate_data.py --days 7

# List available dates in dataset
python scripts/generate_data.py --list

# Show existing training data
python scripts/generate_data.py --existing

# Skip odds fetching (faster but less data)
python scripts/generate_data.py --date 2024-12-15 --no-odds
```

### Alternative: Direct nba_data_builder.py

```bash
# Training data generation
python process/nba_data_builder.py training --date 2024-12-15
python process/nba_data_builder.py training --start 2024-12-01 --end 2024-12-31
python process/nba_data_builder.py training --days 7

# Preview top props (for testing)
python process/nba_data_builder.py preview --date 2024-12-15

# List available dates
python process/nba_data_builder.py list-dates
```

### Full Training Pipeline

```bash
# Option A: One command (recommended)
python scripts/train.py all

# Option B: Step by step
python scripts/generate_data.py --days 30     # 1. Generate data
python ml_pipeline/data_preprocessor.py       # 2. Create labeled dataset
python scripts/train.py models                # 3. Train models
python scripts/train.py ensemble              # 4. Train ensemble
python scripts/train.py evaluate              # 5. Evaluate
```

### One-Liner Evaluation

```bash
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

### Complete Workflow Example

```bash
# Full workflow from scratch:
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# 1. Generate training data for a month
python scripts/generate_data.py --range 2024-12-01 2024-12-31

# 2. Train all models
python scripts/train.py all

# 3. Verify no data leakage
python tests/test_data_leakage.py
```

---

## Recommendations

### Immediate Actions

1. **Collect More Data**: Your 178 samples is critically low. Target 1,000+ for robust training.
2. **Monitor Calibration**: Check ECE after each training run
3. **Track by Prop Type**: Some props (3PM) may need more data

### Medium-Term Improvements

1. **Hyperparameter Tuning**: Use Optuna for automated tuning
2. **Feature Selection**: Use SHAP values to prune unimportant features
3. **A/B Testing**: Compare ML+LLM vs LLM-only predictions

### Long-Term Goals

1. **Retrain Monthly**: As new data accumulates
2. **Drift Detection**: Monitor feature distributions over time
3. **Expand Prop Types**: Add steals, blocks, double-doubles

---

## Troubleshooting

### XGBoost Import Error

```bash
# Install libomp for macOS
brew install libomp

# Set library path
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

### Not Enough Data for Cross-Validation

```
⚠️ Not enough data for cross-validation (X samples)
```

Solution: Collect more training data. Minimum ~50 samples per prop type.

### Missing Features Warning

```
⚠️ Missing features for PTS: ['Opp_L5_Allowed_PTS', ...]
```

Solution: The model will use available features. These are likely opponent features that weren't computed for all games. Not critical but indicates incomplete data.

---

*Last Updated: December 11, 2025*
