# NBA Prop Analyst Toolkit - Quick Reference

## Overview

The `nba_toolkit.py` provides a unified interface for all NBA prop analysis operations. It consolidates functionality from multiple scripts into a single, easy-to-use toolkit.

## Installation

No installation needed - it's part of the project. Just import it:

```python
from nba_toolkit import NBAPropToolkit
```

## Quick Start

### Daily Workflow (Update Data + Predict)

```python
from nba_toolkit import quick_update_and_predict

# Update training data (adds yesterday) and predict today
predictions = quick_update_and_predict()
```

### Using the Toolkit Class

```python
from nba_toolkit import NBAPropToolkit

toolkit = NBAPropToolkit()
```

## Data Management

### 1. Update Kaggle Data

```python
# Update player stats, team stats, games, schedules
success = toolkit.update_kaggle_data()
```

### 2. Fetch Odds

```python
# Fetch odds for a single date (yesterday)
odds_df = toolkit.fetch_odds(season="2025-26", single_date="2025-12-13")

# Fetch odds for a date range
odds_df = toolkit.fetch_odds(season="2025-26", start_date="2025-12-01", end_date="2025-12-13")

# Fetch all missing odds for a season
odds_df = toolkit.fetch_odds(season="2025-26")
```

### 3. Update Training Data

```python
# Incremental update (adds yesterday's data) - RECOMMENDED
training_path = toolkit.update_training_data(incremental=True)

# Full rebuild (rebuilds entire dataset)
training_path = toolkit.update_training_data(incremental=False)

# Skip Kaggle update (if already updated)
training_path = toolkit.update_training_data(incremental=True, skip_kaggle=True)

# Skip odds fetching (if already fetched)
training_path = toolkit.update_training_data(incremental=True, skip_odds=True)
```

### 4. Get Training Data Path

```python
# Get path to current training data
path = toolkit.get_training_data_path()
```

## Model Training

### Train All Models

```python
# Train with hyperparameter tuning (default)
results = toolkit.train_models()

# Train without tuning (faster)
results = toolkit.train_models(tune=False)

# Custom data path
results = toolkit.train_models(data_path="datasets/ml_training/training_data.parquet")

# Custom tuning iterations
results = toolkit.train_models(n_iter=50)
```

**Note**: Models are automatically saved only if they beat the current best (based on calibrated accuracy).

### Get Best Models Summary

```python
# Get summary of best models
summary = toolkit.get_best_models_summary()

# Print summary
toolkit.print_status()  # Shows best models in status
```

## Predictions

### Generate Predictions for Today

```python
# Predict today's props (default)
predictions = toolkit.predict_today()

# Predict specific date
predictions = toolkit.predict_today(target_date="2025-12-14")

# Custom confidence threshold
predictions = toolkit.predict_today(confidence_threshold=8)

# ML-only (skip LLM)
predictions = toolkit.predict_today(ml_only=True)

# Custom max props per game (default: 15)
predictions = toolkit.predict_today(max_props_per_game=10)
```

### Custom Predictions

```python
# Get ML predictor for custom predictions
predictor = toolkit.get_model_predictor()

# Use predictor directly
results = predictor.predict(features_df)
```

## System Status

### Check Status

```python
# Print comprehensive status
toolkit.print_status()

# Get status as dictionary
status = toolkit.get_data_status()
```

## Command Line Usage

```bash
# Show system status
python nba_toolkit.py --status

# Update training data incrementally
python nba_toolkit.py --update-data

# Train models
python nba_toolkit.py --train

# Generate predictions
python nba_toolkit.py --predict

# Quick workflow (update + predict)
python nba_toolkit.py --quick

# Predict specific date
python nba_toolkit.py --predict --date 2025-12-14
```

## Complete Workflow Examples

### Daily Workflow

```python
from nba_toolkit import quick_update_and_predict

# One-liner: update data and predict
predictions = quick_update_and_predict()
```

### Training Workflow

```python
from nba_toolkit import NBAPropToolkit

toolkit = NBAPropToolkit()

# 1. Update training data
toolkit.update_training_data(incremental=True)

# 2. Train models
results = toolkit.train_models()

# 3. Check best models
toolkit.print_status()
```

### Custom Prediction Workflow

```python
from nba_toolkit import NBAPropToolkit

toolkit = NBAPropToolkit()

# 1. Fetch odds for specific date
odds_df = toolkit.fetch_odds(season="2025-26", single_date="2025-12-15")

# 2. Update training data
toolkit.update_training_data(incremental=True)

# 3. Generate predictions
predictions = toolkit.predict_today(
    target_date="2025-12-15",
    confidence_threshold=8,
    max_props_per_game=15
)

# 4. Filter to high confidence
high_conf = predictions[predictions['confidence'] >= 8]
print(f"High confidence picks: {len(high_conf)}")
```

## Key Features

### Automatic Best Model Tracking
- Models are saved only if they beat the current best
- Based on calibrated accuracy (most reliable metric)
- Prevents regression (worse models don't replace better ones)

### Edge-Based Filtering
- Automatically calculates Edge for each prop
- Filters to props with significant edge (OVER or UNDER)
- Only passes edge props to model for predictions

### DraftKings-Only Odds
- Uses DraftKings as single source of truth
- Prevents duplicate lines for same player-prop
- Limits to 15 props per game (configurable)

### Incremental Updates
- Efficient daily updates (only processes yesterday)
- Automatically concatenates with existing data
- Saves API credits by reusing historical odds

## Function Reference

### Data Management
- `update_kaggle_data()` - Update Kaggle data
- `fetch_odds()` - Fetch historical odds
- `update_training_data()` - Build/update training dataset
- `get_training_data_path()` - Get path to training data

### Training
- `train_models()` - Train all prop models
- `get_best_models_summary()` - Get best models info

### Predictions
- `predict_today()` - Generate predictions with edge filtering
- `get_model_predictor()` - Get ML predictor instance

### Utilities
- `get_data_status()` - Get system status
- `print_status()` - Print comprehensive status

## Convenience Functions

- `quick_update_and_predict()` - Daily workflow (update + predict)
- `train_and_evaluate()` - Train models and return metrics

## Error Handling

All functions return appropriate values or raise exceptions:
- Data functions return DataFrames (empty if no data) or paths
- Training functions return results dictionaries
- Prediction functions return DataFrames with predictions
- Errors are printed and exceptions raised for critical failures

## Best Practices

1. **Daily Updates**: Use `quick_update_and_predict()` for daily workflow
2. **Training**: Train after significant data updates (weekly/monthly)
3. **Edge Filtering**: Always use edge filtering for predictions (automatic)
4. **Best Models**: System automatically uses best models (no action needed)
5. **Status Checks**: Use `print_status()` to verify data/model state

