# 🏀 NBA Player Prop Analyst

A hybrid **Machine Learning + LLM** system for predicting NBA player props (Points, Rebounds, Assists, Three-Pointers Made). Built with PySpark for scalable feature engineering and XGBoost for probability prediction.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [ML Models](#ml-models)
- [Data Leakage Prevention](#data-leakage-prevention)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project combines traditional machine learning with Large Language Models (LLMs) to predict NBA player prop outcomes. The system:

1. **Ingests** player statistics from Kaggle datasets + live odds from The Odds API
2. **Engineers** 143 features using PySpark (rolling averages, matchup data, market signals)
3. **Trains** 4 prop-specific XGBoost models with probability calibration
4. **Enhances** predictions using Google Gemini for contextual reasoning
5. **Outputs** final OVER/UNDER predictions with confidence scores

### Why Hybrid ML + LLM?

| Component | Strength | Weakness |
|-----------|----------|----------|
| **ML Models** | Consistent, fast, handles structured data | Can't interpret news, injuries, narratives |
| **LLM (Gemini)** | Contextual reasoning, handles uncertainty | Inconsistent, slower, no historical calibration |
| **Combined** | Best of both worlds | - |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Kaggle CSVs              → Raw player/team statistics          │
│  The Odds API             → Live betting lines & odds           │
│  PySpark Processing       → 143 engineered features             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML LAYER                                 │
│  4 XGBoost Models         → PTS, REB, AST, 3PM classifiers      │
│  Isotonic Calibrators     → Convert outputs to probabilities    │
│  Meta-Ensemble (Optional) → Learns when to trust each model     │
│                                                                 │
│  Output: P(OVER), confidence score, edge vs market              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLM LAYER (Gemini)                       │
│  • Receives ML signals + feature data                           │
│  • Applies reasoning from case study library                    │
│  • Considers news/injury context ML can't capture               │
│  • Outputs final prediction + reasoning                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Feature Engineering (143 Total Features)

| Category | Count | Examples |
|----------|-------|----------|
| **Context** | 3 | `home`, `is_b2b`, `rest_days` |
| **Market Signals** | 7 | `Edge`, `Z_Score`, `EV`, `Kelly`, `quality_score` |
| **Player Classification** | 3 | `Player_Tier`, `Reliability_Tag`, `CV_Score` |
| **Rolling Averages** | 21 | `L5_Avg_PTS`, `L10_Avg_REB`, `Season_Avg_AST` |
| **Home/Away Splits** | 8 | `L5_Avg_PTS_SameVenue`, `Season_Avg_REB_SameVenue` |
| **Volatility Metrics** | 8 | `L5_StdDev_PTS`, `L10_StdDev_3PM` |
| **Form Indicators** | 4 | `Form_PTS`, `Form_REB`, `Form_AST`, `Form_3PM` |
| **Efficiency** | 6 | `Player_TS_Pct`, `Player_eFG_Pct`, `Player_PTS_per100` |
| **Matchup Defense** | 20+ | `Opp_Rank_Allows_PTS`, `Opp_L5_Allowed_REB` |
| **Pace** | 8 | `L5_Team_Pace`, `Opp_L10_Team_Pace` |
| **Team Context** | 20+ | `Team_Def_Rating`, `Opp_Def_Rating` |

### Key Capabilities

- ⚡ **PySpark** for scalable feature engineering on 1.6M+ game records
- 🎯 **Prop-specific models** optimized for PTS, REB, AST, 3PM
- 📊 **Probability calibration** using isotonic regression
- 🧠 **LLM integration** with Google Gemini for contextual reasoning
- 🔒 **Zero data leakage** design (see [Data Leakage Prevention](#data-leakage-prevention))
- 📈 **Comprehensive evaluation** with backtesting support

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Java 8+ (for PySpark)
- [Homebrew](https://brew.sh/) (macOS) for libomp

### Step 1: Clone the Repository

```bash
git clone https://github.com/Anindith-Ram/nba-player-prop-analyst.git
cd nba-player-prop-analyst
```

### Step 2: Create Virtual Environment

```bash
python -m venv nba_env
source nba_env/bin/activate  # On Windows: nba_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install libomp (macOS only)

XGBoost requires OpenMP for parallel processing:

```bash
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

Add this export to your `~/.zshrc` or `~/.bashrc` to make it permanent.

### Step 5: Download Source Data

Download the [NBA Player Statistics dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) from Kaggle and place files in `kaggle_data/`:

```
kaggle_data/
├── Games.csv
├── Players.csv
├── PlayerStatistics.csv
├── TeamStatistics.csv
├── LeagueSchedule24_25.csv
└── LeagueSchedule25_26.csv
```

---

## ⚙️ Configuration

### Step 1: Set Up API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required
GEMINI_API_KEY=your-gemini-api-key    # https://makersuite.google.com/app/apikey
ODDS_API_KEY=your-odds-api-key        # https://the-odds-api.com/

# Optional
KAGGLE_API_TOKEN=your-kaggle-token    # https://www.kaggle.com/settings
```

### Step 2: Load Environment Variables

```bash
export $(cat .env | xargs)
```

Or add to your shell profile for persistence.

---

## 📖 Usage

### Quick Start

```bash
# Activate environment
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# 1. Build training dataset
python scripts/build_combined_dataset.py

# 2. Train ML models
python scripts/train_with_tuning.py

# 3. Run predictions for today's games
python scripts/predict_today.py
```

### Training Pipeline

```bash
# Full pipeline: data → models → evaluate
python scripts/train_with_tuning.py --full

# Just train models (if data exists)
python scripts/train_with_tuning.py --train-only

# Evaluate model performance
python scripts/honest_evaluation.py
```

### Generate Training Data

```bash
# Build combined dataset from Kaggle CSVs + odds
python scripts/build_combined_dataset.py

# Fetch historical odds (requires ODDS_API_KEY)
python scripts/fetch_historical_odds.py --season 2024-25
```

### Make Predictions

```bash
# Today's predictions with live odds
python scripts/predict_today.py

# Hybrid evaluation (ML + context)
python scripts/evaluate_hybrid.py
```

### Testing

```bash
# Critical: Verify no data leakage in feature engineering
python tests/test_data_leakage.py
```

---

## 📁 Project Structure

```
nba-player-prop-analyst/
│
├── config/                     # Configuration files
│   ├── shared_config.py       # API keys, paths, caching (singleton pattern)
│   ├── feature_config.py      # Feature definitions per prop type
│   └── trained_dates.json     # Tracking which dates have been processed
│
├── process/                    # Core data processing
│   ├── nba_data_builder.py    # PySpark feature engineering (143 features)
│   ├── nba_ai_pipeline.py     # Main LLM prediction pipeline
│   ├── shared_config.py       # Shared utilities and API helpers
│   └── optimizer.py           # Hyperparameter optimization
│
├── ml_pipeline/                # Machine learning components
│   ├── model_trainer.py       # XGBoost training with TimeSeriesCV
│   ├── calibrator.py          # Isotonic regression calibration
│   ├── ensemble.py            # Meta-ensemble (stacking model)
│   ├── inference.py           # Production prediction API
│   ├── evaluator.py           # Performance metrics & backtesting
│   └── hybrid_predictor.py    # ML + LLM hybrid predictions
│
├── scripts/                    # Entry point scripts
│   ├── predict_today.py       # Run predictions for today's games
│   ├── build_combined_dataset.py  # Build training data
│   ├── train_with_tuning.py   # Train models with hyperparameter search
│   ├── fetch_historical_odds.py   # Download odds history
│   ├── honest_evaluation.py   # Rigorous model evaluation
│   └── evaluate_hybrid.py     # Evaluate hybrid system
│
├── tests/                      # Test suite
│   └── test_data_leakage.py   # Verify no data leakage (CRITICAL)
│
├── docs/                       # Documentation
│   ├── SYSTEM_GUIDE.md        # Complete system documentation
│   ├── ML_SYSTEM_DOCUMENTATION.md  # ML-specific documentation
│   └── HYBRID_SYSTEM_ROADMAP.md    # Future development plans
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Directories Created at Runtime

```
kaggle_data/                    # Source data (user downloads)
datasets/                       # Processed training data
ml_models/                      # Trained model artifacts
predictions/                    # Generated predictions
training_reports/               # Evaluation reports
logs/                          # Runtime logs
```

---

## 📊 Data Pipeline

### 1. Source Data (Kaggle)

The system uses NBA player statistics from Kaggle:

| File | Description | Records |
|------|-------------|---------|
| `PlayerStatistics.csv` | Per-game player stats | 1.6M+ |
| `TeamStatistics.csv` | Per-game team stats | ~2K/season |
| `Games.csv` | Game metadata | ~2K/season |

### 2. Feature Engineering (PySpark)

The `nba_data_builder.py` generates 143 features using rolling windows:

```python
# Example: Last 5 games average (EXCLUDES current game)
window_5 = (Window
    .partitionBy("personId")
    .orderBy("gameDateTimeEst")
    .rowsBetween(-5, -1))  # -5 to -1, NOT including row 0

L5_Avg_PTS = avg("points").over(window_5)
```

### 3. Odds Integration

Live and historical odds from The Odds API:

```python
# Fetches player props for specified sport/market
response = requests.get(
    "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/",
    params={"apiKey": ODDS_API_KEY, "markets": "player_points"}
)
```

---

## 🤖 ML Models

### Architecture

| Model | Algorithm | Features | Output |
|-------|-----------|----------|--------|
| PTS Model | XGBoost | 27 features | P(OVER points line) |
| REB Model | XGBoost | 25 features | P(OVER rebounds line) |
| AST Model | XGBoost | 25 features | P(OVER assists line) |
| 3PM Model | XGBoost | 25 features | P(OVER 3PM line) |

### Hyperparameters

```python
DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}
```

### Probability Calibration

Raw XGBoost outputs are calibrated using isotonic regression:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(y_min=0.01, y_max=0.99)
calibrator.fit(raw_probabilities, actual_outcomes)

# Now 70% prediction ≈ 70% actual hit rate
calibrated_prob = calibrator.predict(raw_prob)
```

---

## 🔒 Data Leakage Prevention

**Data leakage** is the #1 killer of ML models in production. This system implements multiple safeguards:

### 1. Rolling Windows Exclude Current Game

```python
# rowsBetween(-5, -1) = games -5, -4, -3, -2, -1 (NOT 0)
window_5 = Window.partitionBy("personId").orderBy("date").rowsBetween(-5, -1)
```

### 2. Temporal Train/Test Split

```python
# Split by DATE, not randomly
df = df.sort_values('game_date')
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]   # Earlier dates
test = df.iloc[split_idx:]    # Later dates
```

### 3. TimeSeriesSplit Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Validation data ALWAYS comes after training data chronologically
```

### 4. Label Generation After Features

Labels (OVER/UNDER outcomes) are joined AFTER features are computed, ensuring no future information leaks into features.

### Verify No Leakage

```bash
python tests/test_data_leakage.py
```

---

## 🔌 API Reference

### Inference API

```python
from ml_pipeline.inference import get_ml_predictor

# Initialize predictor (loads all models)
predictor = get_ml_predictor()

# Make predictions on feature DataFrame
predictions = predictor.predict(features_df)

# Output columns:
# - ml_prob_over: Calibrated probability of OVER
# - ml_prediction: 'OVER' or 'UNDER'
# - ml_confidence: Prediction confidence
```

### Feature Builder API

```python
from process.nba_data_builder import NBADataBuilder

builder = NBADataBuilder()

# Generate features for a date range
features = builder.build_features(
    start_date='2024-12-01',
    end_date='2024-12-31',
    include_odds=True
)
```

### Evaluation API

```python
from ml_pipeline.evaluator import ModelEvaluator

# Evaluate predictions
metrics = ModelEvaluator.evaluate(
    y_true=outcomes,
    y_prob=probabilities,
    y_pred=predictions
)

# Returns: accuracy, log_loss, brier_score, roc_auc, ECE
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Run `tests/test_data_leakage.py` before submitting PRs
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update documentation for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kaggle NBA Statistics Dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) by Sumitro Datta
- [The Odds API](https://the-odds-api.com/) for betting lines data
- [Google Gemini](https://deepmind.google/technologies/gemini/) for LLM capabilities
- [XGBoost](https://xgboost.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/) teams

---

## 📧 Contact

**Anindith Ram** - [@Anindith-Ram](https://github.com/Anindith-Ram)

Project Link: [https://github.com/Anindith-Ram/nba-player-prop-analyst](https://github.com/Anindith-Ram/nba-player-prop-analyst)
