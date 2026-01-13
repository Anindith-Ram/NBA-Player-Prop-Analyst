#!/usr/bin/env python3
"""
HYBRID ML+LLM EVALUATION SCRIPT (PROPER HOLDOUT)
=================================================
Evaluates the hybrid ML+Gemini prediction system using TRUE holdout data.

CRITICAL: Uses time-based split to avoid data leakage. Only evaluates on games
the model has NEVER seen during training.

Usage:
    python scripts/evaluate_hybrid.py --n 50                      # Test 50 holdout props
    python scripts/evaluate_hybrid.py --n 100 --blind             # Gemini without ML hints
    python scripts/evaluate_hybrid.py --holdout-pct 20            # Use last 20% of data
    python scripts/evaluate_hybrid.py --help
"""
import os
import sys
import argparse
from datetime import datetime
import json

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

import pandas as pd
import numpy as np

from ml_pipeline.hybrid_predictor import HybridPredictor


# ============================================================================
# DATA SPLITTING - MATCHES TRAINING SPLIT EXACTLY
# ============================================================================

def get_holdout_data(df: pd.DataFrame, holdout_pct: float = 20.0, exclude_playoffs: bool = True) -> tuple:
    """
    Split data into training and holdout sets using TIME-BASED split.
    
    CRITICAL: Uses the EXACT same 80/20 split as the training pipeline.
    This ensures we evaluate on the same test data used during training validation.
    
    Args:
        df: Full dataset with 'Game_Date' or 'game_date' column
        holdout_pct: Percentage of data to hold out (default 20% = same as training)
        exclude_playoffs: If True, exclude games after April 15 (playoffs)
        
    Returns:
        (train_df, holdout_df, split_info dict)
    """
    # Find date column (different datasets use different names)
    date_col = None
    for col in ['Game_Date', 'game_date', 'GAME_DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("Dataset must have a date column (Game_Date, game_date, or GAME_DATE)")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # EXCLUDE PLAYOFF DATA - regular season only (ends mid-April)
    if exclude_playoffs:
        # NBA regular season typically ends around April 13-15
        playoff_cutoff = pd.Timestamp('2025-04-15')
        original_len = len(df)
        df = df[df[date_col] < playoff_cutoff]
        excluded = original_len - len(df)
        if excluded > 0:
            print(f"\n   ⚠️ Excluded {excluded} playoff/post-season games (after {playoff_cutoff.date()})")
    
    df = df.sort_values(date_col)
    
    # Calculate split point - EXACTLY as done in training
    n_total = len(df)
    n_holdout = int(n_total * (holdout_pct / 100))
    n_train = n_total - n_holdout
    
    # Split
    train_df = df.iloc[:n_train]
    holdout_df = df.iloc[n_train:]
    
    train_start = train_df[date_col].min()
    train_end = train_df[date_col].max()
    test_start = holdout_df[date_col].min()
    test_end = holdout_df[date_col].max()
    
    split_info = {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'train_samples': len(train_df),
        'test_samples': len(holdout_df),
        'holdout_pct': holdout_pct,
    }
    
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  📊 TEMPORAL SPLIT (MATCHES TRAINING EXACTLY)                    ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Training Set ({100-holdout_pct:.0f}%):                                            ║")
    print(f"║    • {len(train_df):,} samples                                          ║")
    print(f"║    • Dates: {train_start.date()} → {train_end.date()}                     ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Test Set ({holdout_pct:.0f}%): ⚠️ ML has NEVER seen this data            ║")
    print(f"║    • {len(holdout_df):,} samples                                          ║")
    print(f"║    • Dates: {test_start.date()} → {test_end.date()}                       ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    
    return train_df, holdout_df, split_info


def get_diverse_sample(df: pd.DataFrame, n_props: int = 20) -> pd.DataFrame:
    """
    Get a diverse sample of props across different players and prop types.
    """
    props_per_type = max(1, n_props // 4)
    sample_props = []
    
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        prop_df = df[df['Prop_Type_Normalized'] == prop_type]
        if len(prop_df) == 0:
            continue
            
        # Get unique players for this prop type
        unique_players = prop_df.groupby('personId').first().reset_index()
        # Sample different players
        n_sample = min(props_per_type, len(unique_players))
        sampled = unique_players.sample(n=n_sample, random_state=42)
        sample_props.append(sampled)
    
    if not sample_props:
        return df.sample(n=min(n_props, len(df)), random_state=42)
    
    diverse_sample = pd.concat(sample_props)
    
    # If we need more samples, add randomly
    if len(diverse_sample) < n_props:
        remaining_needed = n_props - len(diverse_sample)
        remaining_pool = df[~df.index.isin(diverse_sample.index)]
        if len(remaining_pool) > 0:
            additional = remaining_pool.sample(n=min(remaining_needed, len(remaining_pool)), random_state=42)
            diverse_sample = pd.concat([diverse_sample, additional])
    
    return diverse_sample


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(results: pd.DataFrame) -> dict:
    """
    Evaluate predictions against actual outcomes.
    """
    if 'outcome' not in results.columns:
        return {'error': 'No outcome column found - cannot evaluate accuracy'}
    
    # Get predictions and actuals
    actuals = results['outcome'].values
    
    # ML predictions accuracy
    ml_preds = (results['ml_prediction'] == 'OVER').astype(int).values
    ml_correct = (ml_preds == actuals).sum()
    ml_accuracy = ml_correct / len(actuals) * 100
    
    # Gemini predictions accuracy  
    gemini_preds = (results['gemini_prediction'] == 'OVER').astype(int).values
    gemini_correct = (gemini_preds == actuals).sum()
    gemini_accuracy = gemini_correct / len(actuals) * 100
    
    # Where they disagreed - who was right?
    disagreements = results[~results['agrees_with_ml']]
    if len(disagreements) > 0:
        disagree_ml_preds = (disagreements['ml_prediction'] == 'OVER').astype(int).values
        disagree_gemini_preds = (disagreements['gemini_prediction'] == 'OVER').astype(int).values
        disagree_actuals = disagreements['outcome'].values
        
        ml_right_on_disagree = (disagree_ml_preds == disagree_actuals).sum()
        gemini_right_on_disagree = (disagree_gemini_preds == disagree_actuals).sum()
    else:
        ml_right_on_disagree = 0
        gemini_right_on_disagree = 0
    
    # Confidence analysis
    high_conf = results[results['gemini_confidence'] >= 7]
    if len(high_conf) > 0:
        high_conf_preds = (high_conf['gemini_prediction'] == 'OVER').astype(int).values
        high_conf_actuals = high_conf['outcome'].values
        high_conf_accuracy = (high_conf_preds == high_conf_actuals).mean() * 100
    else:
        high_conf_accuracy = None
    
    low_conf = results[results['gemini_confidence'] <= 4]
    if len(low_conf) > 0:
        low_conf_preds = (low_conf['gemini_prediction'] == 'OVER').astype(int).values
        low_conf_actuals = low_conf['outcome'].values
        low_conf_accuracy = (low_conf_preds == low_conf_actuals).mean() * 100
    else:
        low_conf_accuracy = None
    
    # By prop type
    prop_type_metrics = {}
    for prop_type in results['Prop_Type_Normalized'].unique():
        subset = results[results['Prop_Type_Normalized'] == prop_type]
        if len(subset) == 0:
            continue
        subset_actuals = subset['outcome'].values
        ml_acc = ((subset['ml_prediction'] == 'OVER').astype(int).values == subset_actuals).mean() * 100
        gemini_acc = ((subset['gemini_prediction'] == 'OVER').astype(int).values == subset_actuals).mean() * 100
        prop_type_metrics[prop_type] = {
            'count': len(subset),
            'ml_accuracy': ml_acc,
            'gemini_accuracy': gemini_acc
        }
    
    return {
        'total_props': len(results),
        'ml_accuracy': ml_accuracy,
        'ml_correct': ml_correct,
        'gemini_accuracy': gemini_accuracy,
        'gemini_correct': gemini_correct,
        'disagreements': len(disagreements),
        'ml_right_on_disagree': ml_right_on_disagree,
        'gemini_right_on_disagree': gemini_right_on_disagree,
        'high_conf_count': len(high_conf),
        'high_conf_accuracy': high_conf_accuracy,
        'low_conf_count': len(low_conf),
        'low_conf_accuracy': low_conf_accuracy,
        'by_prop_type': prop_type_metrics,
    }


def print_detailed_results(results: pd.DataFrame, metrics: dict, blind_mode: bool = False):
    """Print detailed evaluation results."""
    
    mode_str = "BLIND (no ML hints)" if blind_mode else "STANDARD (Gemini sees ML)"
    
    print("\n" + "="*75)
    print(f"📊 DETAILED PREDICTION RESULTS ({mode_str})")
    print("="*75)
    
    print(f"\n{'Player':<22} {'Prop':<6} {'Line':<6} {'ML':<7} {'Gemini':<7} {'Actual':<7} {'✓/✗':<4} {'Conf':<4}")
    print("-"*75)
    
    for idx, row in results.iterrows():
        name = str(row.get('FullName', 'Unknown'))[:20]
        prop = row.get('Prop_Type_Normalized', '')
        line = row.get('Line', 0)
        ml = row.get('ml_prediction', '')
        gemini = row.get('gemini_prediction', '')
        actual_val = row.get('outcome', -1)
        actual = 'OVER' if actual_val == 1 else 'UNDER'
        conf = row.get('gemini_confidence', 0)
        
        # Check if Gemini was correct
        gemini_correct = (gemini == actual)
        result_symbol = '✅' if gemini_correct else '❌'
        
        # Highlight disagreements
        agree_marker = '' if row.get('agrees_with_ml', True) else '*'
        
        print(f"{name:<22} {prop:<6} {line:<6.1f} {ml:<7} {gemini:<7}{agree_marker} {actual:<7} {result_symbol:<4} {conf:<4}")
    
    print("\n* = Gemini disagreed with ML")
    
    # Summary metrics
    print("\n" + "="*75)
    print("📈 ACCURACY SUMMARY")
    print("="*75)
    
    print(f"\n┌{'─'*35}┬{'─'*15}┬{'─'*15}┐")
    print(f"│ {'Metric':<33} │ {'ML':<13} │ {'Gemini':<13} │")
    print(f"├{'─'*35}┼{'─'*15}┼{'─'*15}┤")
    print(f"│ {'Overall Accuracy':<33} │ {metrics['ml_accuracy']:>12.1f}% │ {metrics['gemini_accuracy']:>12.1f}% │")
    print(f"│ {'Correct Predictions':<33} │ {metrics['ml_correct']:>13} │ {metrics['gemini_correct']:>13} │")
    print(f"└{'─'*35}┴{'─'*15}┴{'─'*15}┘")
    
    # By prop type
    if metrics.get('by_prop_type'):
        print(f"\n📊 ACCURACY BY PROP TYPE")
        print("-"*50)
        for prop_type, data in sorted(metrics['by_prop_type'].items()):
            winner = "ML" if data['ml_accuracy'] > data['gemini_accuracy'] else "Gemini" if data['gemini_accuracy'] > data['ml_accuracy'] else "TIE"
            print(f"   {prop_type:<6}: ML {data['ml_accuracy']:.1f}% | Gemini {data['gemini_accuracy']:.1f}% ({data['count']} props) → {winner}")
    
    # Disagreement analysis
    if metrics['disagreements'] > 0:
        print(f"\n📊 DISAGREEMENT ANALYSIS ({metrics['disagreements']} props where Gemini ≠ ML)")
        print("-"*50)
        print(f"   ML was right:     {metrics['ml_right_on_disagree']}/{metrics['disagreements']}")
        print(f"   Gemini was right: {metrics['gemini_right_on_disagree']}/{metrics['disagreements']}")
        
        if metrics['gemini_right_on_disagree'] > metrics['ml_right_on_disagree']:
            print(f"   → 🎯 Gemini ADDED VALUE on disagreements!")
        elif metrics['gemini_right_on_disagree'] < metrics['ml_right_on_disagree']:
            print(f"   → ⚠️ ML was better on disagreements")
        else:
            print(f"   → 🤝 Tied on disagreements")
    
    # Confidence analysis
    print(f"\n📊 CONFIDENCE ANALYSIS")
    print("-"*50)
    if metrics['high_conf_accuracy'] is not None:
        print(f"   High confidence (7-10): {metrics['high_conf_accuracy']:.1f}% accuracy ({metrics['high_conf_count']} props)")
    if metrics['low_conf_accuracy'] is not None:
        print(f"   Low confidence (1-4):   {metrics['low_conf_accuracy']:.1f}% accuracy ({metrics['low_conf_count']} props)")
    
    # Bottom line
    print("\n" + "="*75)
    print("💰 BOTTOM LINE")
    print("="*75)
    
    diff = metrics['gemini_accuracy'] - metrics['ml_accuracy']
    if diff > 0:
        print(f"   🚀 Gemini improved accuracy by +{diff:.1f}%")
    elif diff < 0:
        print(f"   ⚠️ ML alone was better by +{-diff:.1f}%")
    else:
        print(f"   🤝 ML and Gemini achieved the same accuracy")
    
    # Realistic expectation
    print(f"\n   📊 REALISTIC BENCHMARK:")
    print(f"   Random baseline: 50%")
    print(f"   Break-even at -110 odds: 52.4%")
    
    # ROI estimate (assuming -110 odds)
    gemini_roi = (metrics['gemini_accuracy'] / 100 * 1.91 - 1) * 100
    ml_roi = (metrics['ml_accuracy'] / 100 * 1.91 - 1) * 100
    print(f"\n   Estimated ROI (at -110 odds):")
    print(f"   ML:     {ml_roi:+.1f}%")
    print(f"   Gemini: {gemini_roi:+.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Hybrid ML+Gemini Prediction System (Proper Holdout)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard evaluation on holdout data
    python scripts/evaluate_hybrid.py --n 50
    
    # Blind mode - Gemini makes independent predictions
    python scripts/evaluate_hybrid.py --n 50 --blind
    
    # Use more holdout data (30% instead of 20%)
    python scripts/evaluate_hybrid.py --n 100 --holdout-pct 30
    
    # Compare blind vs standard
    python scripts/evaluate_hybrid.py --n 50 --compare
        """
    )
    
    parser.add_argument('--n', type=int, default=200,
                        help='Number of props to evaluate (default: 200 for statistical significance)')
    parser.add_argument('--mode', choices=['fast', 'balanced', 'premium'],
                        default='premium',
                        help='Prediction mode (default: premium)')
    parser.add_argument('--data', type=str, 
                        default='datasets/ml_training/super_dataset_2024_25_clean.parquet',
                        help='Path to dataset with outcomes')
    parser.add_argument('--holdout-pct', type=float, default=20.0,
                        help='Percentage of data to hold out for testing (default: 20)')
    parser.add_argument('--include-playoffs', action='store_true',
                        help='Include playoff games (default: exclude them for fair comparison)')
    parser.add_argument('--blind', action='store_true',
                        help='Blind mode: Gemini does NOT see ML predictions')
    parser.add_argument('--compare', action='store_true',
                        help='Run both blind and standard mode for comparison')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*75)
    print("🏀 HYBRID ML+GEMINI EVALUATION (PROPER HOLDOUT)")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   Props: {args.n}")
    print(f"   Holdout: {args.holdout_pct}%")
    print(f"   Blind: {'YES (independent Gemini)' if args.blind else 'NO (Gemini sees ML)'}")
    print(f"   Started: {datetime.now()}")
    print("="*75)
    
    # Load data
    data_path = os.path.join(PROJECT_ROOT, args.data) if not args.data.startswith('/') else args.data
    print(f"\n📂 Loading data from: {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"   ✅ Loaded {len(df):,} total samples")
    
    # Check for required columns
    if 'outcome' not in df.columns:
        print("   ⚠️ No 'outcome' column found - accuracy cannot be calculated")
        return
    
    # Find date column
    date_col = None
    for col in ['Game_Date', 'game_date', 'GAME_DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("   ⚠️ No date column found - using random split (less reliable)")
        train_df = df.sample(frac=1-args.holdout_pct/100, random_state=42)
        holdout_df = df.drop(train_df.index)
        split_info = {'test_start': 'unknown', 'test_end': 'unknown'}
    else:
        # PROPER TIME-BASED SPLIT (matches training exactly)
        # Exclude playoffs by default for fair comparison
        train_df, holdout_df, split_info = get_holdout_data(
            df, args.holdout_pct, 
            exclude_playoffs=not args.include_playoffs
        )
    
    # Get diverse sample from HOLDOUT only
    print(f"\n📊 Sampling {args.n} diverse props from HOLDOUT data...")
    sample = get_diverse_sample(holdout_df, args.n)
    print(f"   ✅ Sampled {len(sample)} props across {sample['FullName'].nunique()} unique players")
    print(f"   ⚠️  These are games the ML model has NEVER seen")
    
    # Run evaluation
    def run_evaluation(blind_mode: bool):
        predictor = HybridPredictor(mode=args.mode, blind_mode=blind_mode)
        results = predictor.predict(sample.copy())
        metrics = evaluate_predictions(results)
        return results, metrics, predictor
    
    if args.compare:
        # Run both modes
        print("\n" + "="*75)
        print("🔬 COMPARISON: BLIND vs STANDARD")
        print("="*75)
        
        print("\n--- STANDARD MODE (Gemini sees ML) ---")
        std_results, std_metrics, std_pred = run_evaluation(blind_mode=False)
        
        print("\n--- BLIND MODE (Independent Gemini) ---")
        blind_results, blind_metrics, blind_pred = run_evaluation(blind_mode=True)
        
        print("\n" + "="*75)
        print("📊 COMPARISON RESULTS")
        print("="*75)
        print(f"\n{'Metric':<30} {'Standard':<15} {'Blind':<15}")
        print("-"*60)
        print(f"{'Gemini Accuracy':<30} {std_metrics['gemini_accuracy']:>12.1f}% {blind_metrics['gemini_accuracy']:>12.1f}%")
        print(f"{'ML Accuracy':<30} {std_metrics['ml_accuracy']:>12.1f}% {blind_metrics['ml_accuracy']:>12.1f}%")
        print(f"{'Disagreements':<30} {std_metrics['disagreements']:>12} {blind_metrics['disagreements']:>12}")
        print(f"{'Gemini wins on disagree':<30} {std_metrics['gemini_right_on_disagree']:>12} {blind_metrics['gemini_right_on_disagree']:>12}")
        
        if blind_metrics['gemini_accuracy'] > std_metrics['gemini_accuracy']:
            print(f"\n🎯 BLIND mode was better by +{blind_metrics['gemini_accuracy'] - std_metrics['gemini_accuracy']:.1f}%")
            print("   → Showing ML prediction to Gemini may be HURTING performance!")
        else:
            print(f"\n📊 STANDARD mode was better by +{std_metrics['gemini_accuracy'] - blind_metrics['gemini_accuracy']:.1f}%")
        
        results = blind_results
        metrics = blind_metrics
        predictor = blind_pred
    else:
        results, metrics, predictor = run_evaluation(blind_mode=args.blind)
    
    # Print results
    if not args.quiet and not args.compare:
        print_detailed_results(results, metrics, args.blind)
    
    # Save if requested
    if args.save:
        save_path = os.path.join(PROJECT_ROOT, args.save) if not args.save.startswith('/') else args.save
        
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'mode': args.mode,
            'blind_mode': args.blind,
            'holdout_pct': args.holdout_pct,
            'n_props': len(results),
            'metrics': metrics,
            'predictions': results[['FullName', 'Prop_Type_Normalized', 'Line', 
                                    'ml_prediction', 'gemini_prediction', 'gemini_confidence',
                                    'outcome', 'agrees_with_ml', 'reasoning']].to_dict('records')
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {save_path}")
    
    predictor.print_stats()
    print("\n" + "="*75)
    print("✅ Evaluation complete!")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
