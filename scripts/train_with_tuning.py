#!/usr/bin/env python3
"""
TRAINING WITH HYPERPARAMETER TUNING
====================================
Trains models with GridSearch optimization on max_depth and learning_rate.

Usage:
    python scripts/train_with_tuning.py
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List

# Set library path for XGBoost (macOS)
os.environ['DYLD_LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import joblib

from ml_pipeline.model_trainer import PropModelTrainer, temporal_train_test_split
from ml_pipeline.calibrator import ProbabilityCalibrator
from ml_pipeline.best_model_tracker import BestModelTracker


# ============================================================================
# PATHS
# ============================================================================

MODELS_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'prop_models')
CALIBRATORS_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'calibrators')
ML_TRAINING_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'ml_training')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'training_reports')


# ============================================================================
# COMPREHENSIVE HYPERPARAMETER GRID
# ============================================================================

# EXPANDED grid - covers wide range of configurations
# Total combinations: 8 × 9 × 4 × 5 × 5 × 5 × 5 × 4 = 720,000
# We use RandomizedSearchCV to sample efficiently from this space

PARAM_GRID = {
    # Tree depth: shallow (3) to deep (10)
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    
    # Learning rate: very slow (0.005) to aggressive (0.2)
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    
    # Number of trees
    'n_estimators': [100, 200, 300, 500],
    
    # Row sampling
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Column sampling
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Minimum samples in leaf
    'min_child_weight': [1, 3, 5, 7, 10],
    
    # L1 regularization
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    
    # L2 regularization
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
}

# How many random configurations to try (higher = better but slower)
N_ITER = 500  # Sample 500 from ~720k possibilities

# Fixed parameters (not tuned)
FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
}


# ============================================================================
# TUNING FUNCTIONS
# ============================================================================

def tune_hyperparameters(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    prop_type: str,
    n_splits: int = 5,
    n_iter: int = None
) -> Dict:
    """
    COMPREHENSIVE hyperparameter tuning using RandomizedSearchCV.
    
    Explores a large parameter space efficiently using random sampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        prop_type: Prop type for logging
        n_splits: Number of CV splits
        n_iter: Number of random configurations to try
        
    Returns:
        Best parameters dict
    """
    if n_iter is None:
        n_iter = N_ITER
    
    # Calculate total grid size
    total_combinations = 1
    for values in PARAM_GRID.values():
        total_combinations *= len(values)
    
    print(f"\n   🔧 COMPREHENSIVE HYPERPARAMETER TUNING for {prop_type}")
    print(f"      ╔════════════════════════════════════════════════════════╗")
    print(f"      ║  Parameter Space: {total_combinations:,} total combinations          ║")
    print(f"      ║  Sampling: {n_iter} random configurations                  ║")
    print(f"      ║  Cross-validation: {n_splits} time-series folds                 ║")
    print(f"      ╚════════════════════════════════════════════════════════╝")
    
    print(f"\n      Parameters being tuned:")
    for param, values in PARAM_GRID.items():
        print(f"         • {param}: {values}")
    
    # Create base model
    base_model = xgb.XGBClassifier(**FIXED_PARAMS)
    
    # TimeSeriesSplit for proper temporal CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # RandomizedSearchCV for efficient exploration
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=42
    )
    
    print(f"\n      🏃 Running {n_iter} iterations (this may take a while)...")
    
    # Run random search
    random_search.fit(X_train, y_train)
    
    # Get results
    best_params = random_search.best_params_
    best_score = -random_search.best_score_
    
    # Print best configuration with emphasis
    print(f"\n      ╔══════════════════════════════════════════════════════════╗")
    print(f"      ║ 🏆 BEST CONFIGURATION FOR {prop_type:<28}    ║")
    print(f"      ╠══════════════════════════════════════════════════════════╣")
    for param, value in best_params.items():
        if isinstance(value, float):
            param_str = f"      ║   {param:<20}: {value:.4f}"
        else:
            param_str = f"      ║   {param:<20}: {value}"
        print(f"{param_str:<62}║")
    print(f"      ╠══════════════════════════════════════════════════════════╣")
    print(f"      ║   Best CV Log Loss: {best_score:.4f}                          ║")
    print(f"      ╚══════════════════════════════════════════════════════════╝")
    
    # Show top 5 configurations
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df['mean_log_loss'] = -results_df['mean_test_score']
    top_5 = results_df.nsmallest(5, 'mean_log_loss')[['params', 'mean_log_loss', 'std_test_score', 'rank_test_score']]
    
    print(f"\n      📊 Top 5 configurations:")
    for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n         {rank}. Log Loss: {row['mean_log_loss']:.4f} (±{-row['std_test_score']:.4f})")
        for param, value in row['params'].items():
            if isinstance(value, float):
                print(f"            • {param}: {value:.4f}")
            else:
                print(f"            • {param}: {value}")
    
    # Merge with fixed params
    full_params = FIXED_PARAMS.copy()
    full_params.update(best_params)
    
    return full_params, best_score, top_5.to_dict('records')


def train_model_with_tuning(
    prop_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tune: bool = True
) -> Tuple[PropModelTrainer, ProbabilityCalibrator, Dict]:
    """
    Train a single prop model with optional hyperparameter tuning.
    
    Args:
        prop_type: One of 'PTS', 'REB', 'AST', '3PM'
        train_df: Training data
        val_df: Validation data
        tune: Whether to run hyperparameter tuning
        
    Returns:
        Tuple of (trainer, calibrator, metrics)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {prop_type} MODEL")
    print(f"{'='*60}")
    
    # Initialize trainer with default params
    trainer = PropModelTrainer(prop_type)
    
    # Prepare data
    try:
        X_train, y_train = trainer.prepare_data(train_df)
        X_val, y_val = trainer.prepare_data(val_df)
    except ValueError as e:
        print(f"   ⚠️ Skipping {prop_type}: {e}")
        return None, None, {}
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Features: {len(trainer.features)}")
    
    # Skip if not enough data
    if len(X_train) < 100:
        print(f"   ⚠️ Not enough training data for {prop_type}")
        return None, None, {}
    
    # Hyperparameter tuning
    tuning_results = None
    if tune:
        best_params, best_score, top_configs = tune_hyperparameters(X_train, y_train, prop_type)
        trainer.params = best_params
        tuning_results = {
            'best_params': {k: v for k, v in best_params.items() if k not in FIXED_PARAMS},
            'best_cv_log_loss': best_score,
            'top_5_configs': top_configs
        }
    
    # Train with best params
    print(f"\n   🏋️ Training {prop_type} model...")
    trainer.train(X_train, y_train, X_val if len(X_val) > 5 else None, y_val if len(y_val) > 5 else None)
    
    # Get probabilities for calibration
    y_prob_train = trainer.predict_proba(X_train)
    y_prob_val = trainer.predict_proba(X_val) if len(X_val) > 0 else np.array([])
    
    # Calibrate
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_train.values, y_prob_train)
    
    print(f"\n   📊 Calibration metrics:")
    print(f"      ECE: {calibrator.calibration_metrics['ece']:.4f}")
    print(f"      MCE: {calibrator.calibration_metrics['mce']:.4f}")
    
    # Feature importance (top 10)
    print(f"\n   🔑 Top 10 features:")
    for idx, row in trainer.feature_importance.head(10).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    # Validation metrics
    if len(X_val) > 0:
        y_pred_val = trainer.predict(X_val)
        y_prob_val_calibrated = calibrator.calibrate(y_prob_val)
        
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_accuracy_cal = accuracy_score(y_val, (y_prob_val_calibrated >= 0.5).astype(int))
        val_roc_auc = roc_auc_score(y_val, y_prob_val)
        val_log_loss = log_loss(y_val, y_prob_val)
        
        print(f"\n   📈 Validation Results:")
        print(f"      Accuracy: {val_accuracy:.1%}")
        print(f"      Accuracy (calibrated): {val_accuracy_cal:.1%}")
        print(f"      ROC-AUC: {val_roc_auc:.4f}")
        print(f"      Log Loss: {val_log_loss:.4f}")
        
        metrics = {
            'accuracy': val_accuracy,
            'accuracy_calibrated': val_accuracy_cal,
            'roc_auc': val_roc_auc,
            'log_loss': val_log_loss,
            'best_params': {k: v for k, v in trainer.params.items() if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']},
            'tuning_results': tuning_results,
        }
    else:
        metrics = {
            'best_params': {k: v for k, v in trainer.params.items() if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']},
        }
    
    # Check if this is the best model based on calibrated accuracy
    tracker = BestModelTracker()
    
    # Get calibrated accuracy (use validation if available, otherwise training)
    if len(X_val) > 0 and 'accuracy_calibrated' in metrics:
        calibrated_accuracy = metrics['accuracy_calibrated']
    else:
        # Fallback: use training calibrated accuracy
        y_pred_train_cal = (calibrator.calibrate(y_prob_train) >= 0.5).astype(int)
        calibrated_accuracy = accuracy_score(y_train, y_pred_train_cal)
        metrics['accuracy_calibrated'] = calibrated_accuracy
    
    # Prepare model paths
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CALIBRATORS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, f'{prop_type.lower()}_model.pkl')
    calibrator_path = os.path.join(CALIBRATORS_DIR, f'{prop_type.lower()}_calibrator.pkl')
    
    # Check if this model is better than the current best
    is_best = tracker.is_better(prop_type, calibrated_accuracy)
    best_accuracy = tracker.get_best_accuracy(prop_type)
    
    if is_best:
        print(f"\n   🏆 NEW BEST MODEL! (Calibrated Accuracy: {calibrated_accuracy:.1%})")
        if best_accuracy is not None:
            improvement = calibrated_accuracy - best_accuracy
            print(f"   📈 Improvement: {improvement:+.1%} over previous best ({best_accuracy:.1%})")
        
        # Save the new best model
        trainer.save(model_path)
        calibrator.save(calibrator_path)
        print(f"   ✅ Saved best model: {model_path}")
        
        # Update tracker
        tracker.update_best_model(
            prop_type=prop_type,
            calibrated_accuracy=calibrated_accuracy,
            model_path=model_path,
            calibrator_path=calibrator_path,
            metrics=metrics,
            training_info={
                'params': {k: v for k, v in trainer.params.items() 
                          if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']},
                'n_train': len(X_train),
                'n_val': len(X_val),
                'tuning_results': tuning_results
            }
        )
    else:
        print(f"\n   ⚠️  Model not better than current best")
        print(f"   Current: {calibrated_accuracy:.1%} | Best: {best_accuracy:.1%}")
        print(f"   Keeping existing best model. This model was not saved.")
        
        # Still update tracker with training history
        tracker.update_best_model(
            prop_type=prop_type,
            calibrated_accuracy=calibrated_accuracy,
            model_path=model_path,  # Won't be used since not best
            calibrator_path=calibrator_path,  # Won't be used since not best
            metrics=metrics,
            training_info={
                'params': {k: v for k, v in trainer.params.items() 
                          if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']},
                'n_train': len(X_train),
                'n_val': len(X_val),
                'tuning_results': tuning_results
            }
        )
    
    return trainer, calibrator, metrics


def train_all_with_tuning(data_path: str, tune: bool = True, n_iter: int = None) -> Dict:
    """
    Train all prop models with hyperparameter tuning.
    
    Args:
        data_path: Path to cleaned training data
        tune: Whether to run hyperparameter tuning
        n_iter: Number of random configurations to try (overrides N_ITER)
        
    Returns:
        Results dict with metrics for each prop type
    """
    # Override global N_ITER if provided
    global N_ITER
    if n_iter is not None:
        N_ITER = n_iter
    print("="*60)
    print("NBA PROP MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print(f"Started: {datetime.now()}")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"   ✅ Loaded {len(df):,} samples")
    
    # Distribution
    print(f"\n   Distribution by prop type:")
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        prop_df = df[df['Prop_Type_Normalized'] == prop_type]
        if len(prop_df) > 0:
            over_rate = (prop_df['outcome'] == 1).mean() * 100
            print(f"      {prop_type}: {len(prop_df):,} samples, OVER rate: {over_rate:.1f}%")
    
    # Temporal split
    print(f"\n📊 Splitting data temporally (80/20)...")
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)
    
    # Train each model
    results = {}
    trainers = {}
    calibrators = {}
    
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        trainer, calibrator, metrics = train_model_with_tuning(
            prop_type, train_df, test_df, tune=tune
        )
        
        if trainer is not None:
            trainers[prop_type] = trainer
            calibrators[prop_type] = calibrator
            results[prop_type] = metrics
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nModels trained: {len(trainers)}")
    
    # Performance table
    print(f"\n{'Prop':<6} {'Accuracy':<12} {'Acc (cal)':<12} {'ROC-AUC':<10} {'Log Loss':<10}")
    print("-" * 70)
    
    for prop_type, metrics in results.items():
        acc = f"{metrics.get('accuracy', 0):.1%}"
        acc_cal = f"{metrics.get('accuracy_calibrated', 0):.1%}"
        roc = f"{metrics.get('roc_auc', 0):.3f}"
        ll = f"{metrics.get('log_loss', 0):.3f}"
        print(f"{prop_type:<6} {acc:<12} {acc_cal:<12} {roc:<10} {ll:<10}")
    
    # Show best models summary
    tracker = BestModelTracker()
    tracker.print_summary()
    
    # Best parameters for each model
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS BY MODEL")
    print("="*70)
    
    for prop_type, metrics in results.items():
        best_params = metrics.get('best_params', {})
        if best_params:
            print(f"\n   🏆 {prop_type}:")
            for param, value in best_params.items():
                if isinstance(value, float):
                    print(f"      • {param}: {value:.4f}")
                else:
                    print(f"      • {param}: {value}")
    
    print("\n" + "="*70)
    
    # Save results
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORTS_DIR, f'tuning_results_{timestamp}.json')
    
    import json
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'data_path': data_path,
            'samples': len(df),
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Results saved: {report_path}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models with hyperparameter tuning')
    parser.add_argument('--data', type=str, 
                        default='datasets/ml_training/super_dataset_2024_25_clean.parquet',
                        help='Path to cleaned training data')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip hyperparameter tuning (use defaults)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode - use fewer tuning iterations (50 instead of 500)')
    
    args = parser.parse_args()
    
    data_path = os.path.join(PROJECT_ROOT, args.data) if not args.data.startswith('/') else args.data
    
    results = train_all_with_tuning(data_path, tune=not args.no_tune)

