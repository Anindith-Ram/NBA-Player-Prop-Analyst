"""
EVALUATOR
Backtesting and performance monitoring for ML models.

This module provides:
1. Standard ML metrics (accuracy, log loss, ROC-AUC, Brier score)
2. Betting-specific metrics (hit rate at confidence thresholds, ROI simulation)
3. Calibration analysis
4. Performance breakdown by prop type, player tier, etc.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix
)

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


class ModelEvaluator:
    """
    Evaluates ML model performance with betting-relevant metrics.
    
    Provides comprehensive analysis of model predictions including:
    - Standard classification metrics
    - Betting profitability simulations
    - Hit rate at various confidence thresholds
    - Performance breakdown by prop type
    """
    
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray = None,
        verbose: bool = True
    ) -> Dict:
        """
        Computes comprehensive evaluation metrics.
        
        Args:
            y_true: Actual outcomes (0 or 1)
            y_prob: Predicted probabilities for class 1 (OVER)
            y_pred: Binary predictions (computed from y_prob if None)
            verbose: Whether to print results
            
        Returns:
            Dictionary with all evaluation metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        if y_pred is None:
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = np.asarray(y_pred).flatten()
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Probabilistic metrics
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except Exception:
            metrics['log_loss'] = np.nan
            
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # ROC-AUC (requires both classes)
        try:
            if len(set(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['roc_auc'] = np.nan
        except Exception:
            metrics['roc_auc'] = np.nan
        
        # Hit rate at different confidence levels
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            high_conf_mask = y_prob >= threshold
            n_high_conf = high_conf_mask.sum()
            
            if n_high_conf > 0:
                # For high confidence OVER predictions
                hits = y_true[high_conf_mask].sum()
                metrics[f'hit_rate_{int(threshold*100)}'] = hits / n_high_conf
                metrics[f'n_bets_{int(threshold*100)}'] = int(n_high_conf)
            else:
                metrics[f'hit_rate_{int(threshold*100)}'] = np.nan
                metrics[f'n_bets_{int(threshold*100)}'] = 0
        
        # Simulated ROI (flat betting on all predictions)
        # Assumes -110 odds (bet 110 to win 100)
        wins = (y_pred == y_true).sum()
        losses = (y_pred != y_true).sum()
        total_bets = wins + losses
        
        if total_bets > 0:
            total_wagered = total_bets * 110
            total_returned = wins * 210  # win back 110 + 100 profit
            metrics['simulated_roi'] = (total_returned - total_wagered) / total_wagered
            metrics['profit_units'] = (wins * 100 - losses * 110) / 100  # in units
        else:
            metrics['simulated_roi'] = 0.0
            metrics['profit_units'] = 0.0
        
        metrics['total_bets'] = int(total_bets)
        metrics['wins'] = int(wins)
        metrics['losses'] = int(losses)
        
        # Confusion matrix breakdown
        if len(set(y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
        
        if verbose:
            ModelEvaluator.print_report(metrics)
        
        return metrics
    
    @staticmethod
    def print_report(metrics: Dict, title: str = "Model Evaluation"):
        """Prints formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        print(f"\n📊 Core Metrics:")
        print(f"   Accuracy:    {metrics.get('accuracy', 0):.1%}")
        print(f"   Log Loss:    {metrics.get('log_loss', 0):.4f}")
        print(f"   Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"   ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
        
        print(f"\n💰 Betting Metrics:")
        for thresh in [55, 60, 65, 70]:
            hr_key = f'hit_rate_{thresh}'
            n_key = f'n_bets_{thresh}'
            if hr_key in metrics and not np.isnan(metrics.get(hr_key, np.nan)):
                print(f"   P>{thresh}%: {metrics[hr_key]:.1%} hit rate ({metrics[n_key]} bets)")
        
        print(f"\n📈 Simulated Performance (flat betting at -110):")
        print(f"   Total Bets:  {metrics.get('total_bets', 0)}")
        print(f"   Wins/Losses: {metrics.get('wins', 0)}/{metrics.get('losses', 0)}")
        print(f"   ROI:         {metrics.get('simulated_roi', 0):+.1%}")
        print(f"   Profit:      {metrics.get('profit_units', 0):+.2f} units")
        
        print(f"{'='*60}")
    
    @staticmethod
    def evaluate_by_prop_type(
        df: pd.DataFrame,
        prob_col: str = 'ml_prob_over',
        pred_col: str = 'ml_prediction',
        outcome_col: str = 'outcome',
        prop_col: str = 'Prop_Type_Normalized'
    ) -> Dict[str, Dict]:
        """
        Evaluates model performance broken down by prop type.
        
        Args:
            df: DataFrame with predictions and outcomes
            prob_col: Column name for predicted probabilities
            pred_col: Column name for predictions
            outcome_col: Column name for actual outcomes
            prop_col: Column name for prop type
            
        Returns:
            Dictionary mapping prop type to metrics
        """
        results = {}
        
        for prop_type in df[prop_col].unique():
            prop_df = df[df[prop_col] == prop_type]
            
            if len(prop_df) < 5:
                continue
            
            y_true = prop_df[outcome_col].values
            y_prob = prop_df[prob_col].values
            y_pred = (prop_df[pred_col] == 'OVER').astype(int).values
            
            results[prop_type] = ModelEvaluator.evaluate(
                y_true, y_prob, y_pred, verbose=False
            )
        
        return results
    
    @staticmethod
    def backtest(
        df: pd.DataFrame,
        prob_col: str = 'ml_prob_over',
        outcome_col: str = 'outcome',
        date_col: str = 'game_date',
        min_confidence: float = 0.55
    ) -> pd.DataFrame:
        """
        Performs backtesting simulation over time.
        
        Args:
            df: DataFrame with predictions and outcomes
            prob_col: Column name for predicted probabilities
            outcome_col: Column name for actual outcomes
            date_col: Column name for dates
            min_confidence: Minimum probability threshold for betting
            
        Returns:
            DataFrame with daily profit/loss and cumulative returns
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Only bet on confident predictions
        df['bet'] = df[prob_col] >= min_confidence
        df['bet_direction'] = np.where(df[prob_col] >= 0.5, 'OVER', 'UNDER')
        
        # Calculate wins/losses
        df['hit'] = (
            ((df['bet_direction'] == 'OVER') & (df[outcome_col] == 1)) |
            ((df['bet_direction'] == 'UNDER') & (df[outcome_col] == 0))
        )
        
        # Profit per bet (assuming -110 odds)
        df['profit'] = np.where(
            df['bet'],
            np.where(df['hit'], 100, -110),
            0
        )
        
        # Daily aggregation
        daily = df.groupby(date_col).agg({
            'bet': 'sum',
            'hit': lambda x: x[df.loc[x.index, 'bet']].sum(),
            'profit': 'sum'
        }).reset_index()
        
        daily.columns = ['date', 'n_bets', 'n_wins', 'profit']
        daily['cumulative_profit'] = daily['profit'].cumsum()
        daily['win_rate'] = daily['n_wins'] / daily['n_bets'].replace(0, 1)
        
        return daily
    
    @staticmethod
    def generate_report(
        df: pd.DataFrame,
        prob_col: str = 'ml_prob_over',
        pred_col: str = 'ml_prediction',
        outcome_col: str = 'outcome',
        save_path: str = None
    ) -> str:
        """
        Generates a comprehensive evaluation report.
        
        Args:
            df: DataFrame with predictions and outcomes
            prob_col: Column for probabilities
            pred_col: Column for predictions
            outcome_col: Column for outcomes
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        lines = []
        lines.append("# ML Model Evaluation Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Samples: {len(df)}")
        lines.append("")
        
        # Overall metrics
        y_true = df[outcome_col].values
        y_prob = df[prob_col].values
        y_pred = (df[pred_col] == 'OVER').astype(int).values
        
        metrics = ModelEvaluator.evaluate(y_true, y_prob, y_pred, verbose=False)
        
        lines.append("## Overall Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Accuracy | {metrics['accuracy']:.1%} |")
        lines.append(f"| ROC-AUC | {metrics['roc_auc']:.4f} |")
        lines.append(f"| Log Loss | {metrics['log_loss']:.4f} |")
        lines.append(f"| Brier Score | {metrics['brier_score']:.4f} |")
        lines.append(f"| Simulated ROI | {metrics['simulated_roi']:+.1%} |")
        lines.append("")
        
        # By prop type
        if 'Prop_Type_Normalized' in df.columns:
            lines.append("## Performance by Prop Type")
            lines.append("")
            lines.append("| Prop | Accuracy | ROC-AUC | Bets | Profit |")
            lines.append("|------|----------|---------|------|--------|")
            
            prop_results = ModelEvaluator.evaluate_by_prop_type(df)
            for prop_type, m in sorted(prop_results.items()):
                acc = m.get('accuracy', 0)
                auc = m.get('roc_auc', 0)
                bets = m.get('total_bets', 0)
                profit = m.get('profit_units', 0)
                lines.append(f"| {prop_type} | {acc:.1%} | {auc:.3f} | {bets} | {profit:+.1f}u |")
        
        lines.append("")
        
        report = "\n".join(lines)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Report saved: {save_path}")
        
        return report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_predictions(predictions_df: pd.DataFrame) -> Dict:
    """
    Convenience function to evaluate a predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with ml_prob_over, ml_prediction, outcome columns
        
    Returns:
        Dictionary of evaluation metrics
    """
    required_cols = ['ml_prob_over', 'ml_prediction', 'outcome']
    missing = [c for c in required_cols if c not in predictions_df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    y_true = predictions_df['outcome'].values
    y_prob = predictions_df['ml_prob_over'].values
    y_pred = (predictions_df['ml_prediction'] == 'OVER').astype(int).values
    
    return ModelEvaluator.evaluate(y_true, y_prob, y_pred)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("ModelEvaluator module loaded successfully")
    print("\nAvailable methods:")
    print("  - ModelEvaluator.evaluate(y_true, y_prob, y_pred)")
    print("  - ModelEvaluator.evaluate_by_prop_type(df)")
    print("  - ModelEvaluator.backtest(df)")
    print("  - ModelEvaluator.generate_report(df)")
