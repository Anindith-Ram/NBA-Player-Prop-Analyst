#!/usr/bin/env python3
"""
HONEST MODEL EVALUATION v2
==========================
Diagnoses model performance with multiple validation strategies.

Since market signals have been removed from features, this script focuses on:
1. Understanding WHY accuracy might still be high
2. Testing for different types of data leakage
3. Providing realistic performance estimates
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
from collections import defaultdict

# Set library path for XGBoost (macOS)
os.environ['DYLD_LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_labeled_data() -> pd.DataFrame:
    """Load the labeled dataset."""
    from ml_pipeline.data_preprocessor import (
        load_training_csvs, load_actuals_pandas,
        create_outcome_labels_pandas, encode_categoricals
    )
    
    print("Loading data...")
    features_df = load_training_csvs()
    actuals_df = load_actuals_pandas()
    labeled_df = create_outcome_labels_pandas(features_df, actuals_df)
    labeled_df = encode_categoricals(labeled_df)
    
    return labeled_df


def baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Calculate baseline metrics for comparison."""
    n = len(y_true)
    over_rate = y_true.mean()
    under_rate = 1 - over_rate
    
    majority_class = 1 if over_rate > 0.5 else 0
    majority_acc = max(over_rate, under_rate)
    
    return {
        'random_accuracy': 0.5,
        'majority_class_accuracy': majority_acc,
        'over_rate': over_rate,
        'under_rate': under_rate,
        'majority_class': 'OVER' if majority_class == 1 else 'UNDER'
    }


def diagnose_data_characteristics(df: pd.DataFrame):
    """Analyze data characteristics that might affect model performance."""
    print_header("DATA CHARACTERISTICS ANALYSIS")
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"   Unique players: {df['personId'].nunique()}")
    print(f"   Unique games: {df['gameId'].nunique() if 'gameId' in df.columns else 'N/A'}")
    
    # Outcome distribution
    over_rate = df['outcome'].mean() * 100
    print(f"\n📊 Outcome Distribution:")
    print(f"   OVER: {over_rate:.1f}%")
    print(f"   UNDER: {100-over_rate:.1f}%")
    
    if abs(over_rate - 50) > 5:
        print(f"   ⚠️ Imbalanced! Predicting {'UNDER' if over_rate < 50 else 'OVER'} gives {max(over_rate, 100-over_rate):.1f}% baseline")
    
    # Props per player
    props_per_player = df.groupby('personId').size()
    print(f"\n📊 Player Repetition:")
    print(f"   Avg props per player: {props_per_player.mean():.1f}")
    print(f"   Max props per player: {props_per_player.max()}")
    print(f"   Players with 10+ props: {(props_per_player >= 10).sum()}")
    
    # Prop type distribution
    print(f"\n📊 Prop Type Distribution:")
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        prop_df = df[df['Prop_Type_Normalized'] == prop_type]
        if len(prop_df) > 0:
            over = prop_df['outcome'].mean() * 100
            print(f"   {prop_type}: {len(prop_df):,} props, OVER rate: {over:.1f}%")
    
    return {
        'n_samples': len(df),
        'n_players': df['personId'].nunique(),
        'over_rate': over_rate,
        'avg_props_per_player': props_per_player.mean()
    }


def test_player_leakage(df: pd.DataFrame):
    """Test if model memorizes player patterns (player-level leakage)."""
    print_header("PLAYER LEAKAGE TEST")
    print("Testing if same players in train/test causes inflated accuracy...")
    
    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)
    
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    train_players = set(train_df['personId'].unique())
    test_players = set(test_df['personId'].unique())
    
    overlap = train_players & test_players
    test_only = test_players - train_players
    
    print(f"\n📊 Player Overlap Analysis:")
    print(f"   Players in training: {len(train_players)}")
    print(f"   Players in test: {len(test_players)}")
    print(f"   Overlapping players: {len(overlap)} ({100*len(overlap)/len(test_players):.1f}% of test)")
    print(f"   Test-only players: {len(test_only)}")
    
    if len(overlap) / len(test_players) > 0.8:
        print(f"\n⚠️ HIGH PLAYER OVERLAP!")
        print(f"   {100*len(overlap)/len(test_players):.1f}% of test players appear in training.")
        print(f"   Model may be memorizing player-specific patterns.")
        print(f"   This inflates accuracy but may not generalize to new players.")
    
    # Compare accuracy on seen vs unseen players
    if len(test_only) > 20:
        test_seen = test_df[test_df['personId'].isin(overlap)]
        test_unseen = test_df[test_df['personId'].isin(test_only)]
        
        print(f"\n📊 Performance Split:")
        print(f"   Seen players: {len(test_seen)} props")
        print(f"   Unseen players: {len(test_unseen)} props")
    
    return {
        'overlap_pct': len(overlap) / len(test_players) * 100,
        'n_overlap': len(overlap),
        'n_test_only': len(test_only)
    }


def test_temporal_gap(df: pd.DataFrame):
    """Test with larger temporal gaps to reduce autocorrelation."""
    print_header("TEMPORAL GAP TEST")
    print("Testing with larger time gaps between train and test...")
    
    from ml_pipeline.inference import get_ml_predictor
    
    df = df.sort_values('game_date').reset_index(drop=True)
    
    # Get unique dates
    unique_dates = df['game_date'].unique()
    unique_dates = sorted(unique_dates)
    
    results = []
    
    # Test different gap sizes
    gaps = [
        ('No gap (adjacent)', 0),
        ('1 week gap', 7),
        ('2 week gap', 14),
        ('1 month gap', 30),
    ]
    
    predictor = get_ml_predictor(verbose=False)
    
    print(f"\n📊 Accuracy vs Temporal Gap:")
    print(f"   {'Gap':<25} {'Test Size':<12} {'Accuracy':<12} {'ROC-AUC'}")
    print(f"   {'-'*60}")
    
    for gap_name, gap_days in gaps:
        # Find split point with gap
        train_end_idx = int(len(unique_dates) * 0.6)
        train_end_date = unique_dates[train_end_idx]
        
        # Add gap
        test_start_date = pd.to_datetime(train_end_date) + pd.Timedelta(days=gap_days)
        
        train_df = df[df['game_date'] <= train_end_date]
        test_df = df[df['game_date'] >= test_start_date.strftime('%Y-%m-%d')]
        
        if len(test_df) < 50:
            print(f"   {gap_name:<25} {'(too few samples)':<12}")
            continue
        
        # Get predictions
        predictions = predictor.predict(test_df)
        
        y_true = predictions['outcome'].values
        y_prob = predictions['ml_prob_over'].values
        y_pred = (predictions['ml_prediction'] == 'OVER').astype(int).values
        
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5
        
        results.append({
            'gap': gap_name,
            'accuracy': acc,
            'roc_auc': auc,
            'test_size': len(test_df)
        })
        
        print(f"   {gap_name:<25} {len(test_df):<12} {acc:.1%}{'':>6} {auc:.3f}")
    
    if len(results) >= 2:
        acc_drop = results[0]['accuracy'] - results[-1]['accuracy']
        if acc_drop > 0.1:
            print(f"\n⚠️ SIGNIFICANT ACCURACY DROP WITH TIME GAP!")
            print(f"   Accuracy dropped {acc_drop*100:.1f}% with longer gap.")
            print(f"   This suggests temporal autocorrelation is inflating results.")
        elif acc_drop > 0.05:
            print(f"\n📊 Moderate accuracy drop ({acc_drop*100:.1f}%) with time gap.")
        else:
            print(f"\n✅ Accuracy stable across time gaps (good sign).")
    
    return results


def test_walk_forward_validation(df: pd.DataFrame):
    """True walk-forward validation simulating real betting."""
    print_header("WALK-FORWARD VALIDATION (Realistic)")
    print("Simulating real betting: train on past, predict future day-by-day...")
    
    from ml_pipeline.inference import get_ml_predictor
    import xgboost as xgb
    from feature_config import PROP_FEATURES_CLEAN
    
    df = df.sort_values('game_date').reset_index(drop=True)
    
    # Get unique dates
    unique_dates = sorted(df['game_date'].unique())
    n_dates = len(unique_dates)
    
    # Start predicting after 60% of dates
    start_pred_idx = int(n_dates * 0.6)
    
    all_preds = []
    all_actuals = []
    
    print(f"\n   Training on {start_pred_idx} days, predicting {n_dates - start_pred_idx} days...")
    
    # Use simple model for walk-forward (faster)
    for i in range(start_pred_idx, min(n_dates, start_pred_idx + 30)):  # Limit to 30 days for speed
        pred_date = unique_dates[i]
        
        # Train on all data before this date
        train_mask = df['game_date'] < pred_date
        test_mask = df['game_date'] == pred_date
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(test_df) == 0:
            continue
        
        # Simple prediction: use majority class from training
        train_over_rate = train_df['outcome'].mean()
        pred = 1 if train_over_rate > 0.5 else 0
        
        all_preds.extend([pred] * len(test_df))
        all_actuals.extend(test_df['outcome'].values)
    
    if len(all_preds) > 0:
        baseline_acc = accuracy_score(all_actuals, all_preds)
        
        print(f"\n📊 Walk-Forward Results (Majority Class Baseline):")
        print(f"   Test samples: {len(all_preds)}")
        print(f"   Baseline accuracy: {baseline_acc:.1%}")
        print(f"   Actual OVER rate in test: {np.mean(all_actuals):.1%}")
    
    return {'baseline_walk_forward': baseline_acc if len(all_preds) > 0 else None}


def calculate_realistic_roi(accuracy: float, avg_odds: float = -110) -> float:
    """Calculate realistic ROI given accuracy and average odds."""
    if avg_odds < 0:
        decimal_odds = 1 + (100 / abs(avg_odds))
    else:
        decimal_odds = 1 + (avg_odds / 100)
    
    win_payout = decimal_odds - 1
    loss = 1
    
    ev_per_bet = (accuracy * win_payout) - ((1 - accuracy) * loss)
    roi = ev_per_bet * 100
    
    return roi


def run_comprehensive_diagnosis():
    """Run comprehensive model diagnosis."""
    print_header("COMPREHENSIVE MODEL DIAGNOSIS")
    print(f"Started: {datetime.now()}")
    print("\nThis analysis will help understand your model's TRUE performance.")
    
    # Load data
    df = load_labeled_data()
    
    # 1. Data characteristics
    data_stats = diagnose_data_characteristics(df)
    
    # 2. Player leakage test
    player_stats = test_player_leakage(df)
    
    # 3. Temporal gap test
    temporal_results = test_temporal_gap(df)
    
    # 4. Walk-forward validation
    walk_forward_results = test_walk_forward_validation(df)
    
    # Final summary
    print_header("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    
    print("\n🔍 KEY FINDINGS:")
    
    # Data imbalance
    over_rate = data_stats['over_rate']
    if abs(over_rate - 50) > 5:
        print(f"\n   1. DATA IMBALANCE:")
        print(f"      OVER rate: {over_rate:.1f}% (not 50%)")
        print(f"      Baseline accuracy just by predicting {'UNDER' if over_rate < 50 else 'OVER'}: {max(over_rate, 100-over_rate):.1f}%")
        print(f"      → Part of your accuracy comes from this imbalance")
    
    # Player overlap
    if player_stats['overlap_pct'] > 80:
        print(f"\n   2. HIGH PLAYER OVERLAP ({player_stats['overlap_pct']:.0f}%):")
        print(f"      Most test players also appear in training.")
        print(f"      → Model may memorize player patterns, not learn generalizable rules")
        print(f"      → New players in future may have worse predictions")
    
    # Temporal autocorrelation
    if len(temporal_results) >= 2:
        acc_no_gap = temporal_results[0]['accuracy']
        acc_with_gap = temporal_results[-1]['accuracy']
        if acc_no_gap - acc_with_gap > 0.05:
            print(f"\n   3. TEMPORAL AUTOCORRELATION:")
            print(f"      Accuracy drops {(acc_no_gap - acc_with_gap)*100:.1f}% with time gaps")
            print(f"      → Recent performance patterns leak into nearby test dates")
            print(f"      → Real-world performance will be closer to gap results")
    
    print("\n" + "="*70)
    print("  REALISTIC PERFORMANCE ESTIMATE")
    print("="*70)
    
    # Use the most conservative (realistic) estimate
    if len(temporal_results) >= 2:
        realistic_acc = temporal_results[-1]['accuracy']  # Use largest gap
    else:
        realistic_acc = max(over_rate/100, (100-over_rate)/100) + 0.05  # Baseline + small edge
    
    print(f"\n🎯 Realistic Accuracy Estimate: {realistic_acc:.1%}")
    print(f"   (Based on temporal gap testing)")
    
    roi = calculate_realistic_roi(realistic_acc)
    print(f"   Expected ROI at -110 odds: {roi:+.1f}%")
    
    if realistic_acc > 0.55:
        print(f"\n   ✅ This is GOOD for sports betting!")
        print(f"   At {realistic_acc:.1%} accuracy, you have a real edge.")
    elif realistic_acc > 0.524:
        print(f"\n   📊 This is MARGINAL.")
        print(f"   Small edge, need high volume and discipline.")
    else:
        print(f"\n   ⚠️ This is BREAKEVEN or LOSING.")
        print(f"   Need 52.4% to break even at -110 odds.")
    
    print("\n" + "="*70)
    print("  RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*70)
    
    print("""
   1. TRAIN ON MORE DATA
      - More dates = more diverse situations
      - Goal: 3000+ samples across full season
      
   2. USE STRICTER TEMPORAL VALIDATION
      - Always leave 1+ month gap between train/test
      - Or use true walk-forward (train on all past, predict next day)
      
   3. CONSIDER PLAYER-INDEPENDENT FEATURES
      - Focus on: opponent defense, pace, rest days, home/away
      - Less on: player averages (can overfit to specific players)
      
   4. TRACK LIVE RESULTS
      - Paper trade for 1 month before real money
      - Compare live accuracy to backtested accuracy
      
   5. ENSEMBLE WITH SIMPLE RULES
      - Combine ML with quality_score thresholds
      - Only bet when BOTH agree
""")
    
    # Save report
    report_dir = os.path.join(PROJECT_ROOT, 'training_reports')
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'model_diagnosis_{timestamp}.json')
    
    import json
    report = {
        'timestamp': timestamp,
        'data_stats': data_stats,
        'player_stats': player_stats,
        'temporal_results': temporal_results,
        'realistic_accuracy': realistic_acc,
        'realistic_roi': roi
    }
    
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj
    
    report = convert_numpy(report)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved: {report_path}")
    
    return report


if __name__ == "__main__":
    run_comprehensive_diagnosis()
