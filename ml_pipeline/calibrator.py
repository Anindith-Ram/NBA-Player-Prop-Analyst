"""
PROBABILITY CALIBRATOR
Calibrates XGBoost outputs to true probabilities using isotonic regression.

Why calibration matters:
- XGBoost outputs are NOT well-calibrated probabilities
- A predicted 70% should win ~70% of the time
- Calibration ensures Kelly criterion calculations are accurate
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from datetime import datetime

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import joblib

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression.
    
    Isotonic regression is preferred over Platt scaling (logistic) because:
    - It makes no assumptions about the shape of the calibration curve
    - It works well for tree-based models like XGBoost
    - It can handle non-monotonic miscalibration patterns
    """
    
    def __init__(self):
        self.calibrator = None
        self.calibration_metrics = {}
        self.fitted = False
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fits isotonic regression to calibrate probabilities.
        
        Args:
            y_true: Actual outcomes (0 or 1)
            y_prob: Raw predicted probabilities from XGBoost
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        self.calibrator = IsotonicRegression(
            y_min=0.01,  # Avoid 0% probabilities
            y_max=0.99,  # Avoid 100% probabilities
            out_of_bounds='clip'
        )
        self.calibrator.fit(y_prob, y_true)
        self.fitted = True
        
        # Compute calibration metrics
        self._compute_metrics(y_true, y_prob)
        
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrates raw probabilities.
        
        Args:
            y_prob: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_prob = np.asarray(y_prob).flatten()
        return self.calibrator.predict(y_prob)
    
    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Computes calibration metrics (ECE, MCE).
        
        ECE (Expected Calibration Error): Average miscalibration across bins
        MCE (Maximum Calibration Error): Worst-case miscalibration
        """
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        bin_details = []
        
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            n_samples = mask.sum()
            
            if n_samples > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                bin_error = abs(bin_accuracy - bin_confidence)
                ece += (n_samples / len(y_true)) * bin_error
                mce = max(mce, bin_error)
                
                bin_details.append({
                    'bin': i,
                    'range': f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}",
                    'n_samples': n_samples,
                    'accuracy': bin_accuracy,
                    'confidence': bin_confidence,
                    'error': bin_error
                })
        
        self.calibration_metrics = {
            'ece': ece,
            'mce': mce,
            'n_bins_with_data': len(bin_details),
            'bin_details': bin_details
        }
    
    def plot_calibration_curve(
        self, 
        y_true: np.ndarray, 
        y_prob_raw: np.ndarray,
        y_prob_calibrated: np.ndarray = None,
        title: str = "Calibration Curve",
        save_path: str = None
    ):
        """
        Plots before/after calibration curves.
        
        Args:
            y_true: Actual outcomes
            y_prob_raw: Raw probabilities before calibration
            y_prob_calibrated: Calibrated probabilities (optional)
            title: Plot title
            save_path: Path to save the plot
        """
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not installed. Cannot plot calibration curve.")
            return
        
        n_plots = 2 if y_prob_calibrated is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        plots_data = [(y_prob_raw, 'Before Calibration')]
        if y_prob_calibrated is not None:
            plots_data.append((y_prob_calibrated, 'After Calibration'))
        
        for ax, (probs, subtitle) in zip(axes, plots_data):
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, probs, n_bins=10
                )
                ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
                ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                       label='Model', markersize=8)
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'{title}\n{subtitle}')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved calibration plot: {save_path}")
        
        plt.close()
    
    def get_metrics_summary(self) -> str:
        """Returns a formatted summary of calibration metrics."""
        if not self.calibration_metrics:
            return "No calibration metrics available"
        
        return (
            f"ECE: {self.calibration_metrics['ece']:.4f} "
            f"(lower is better, <0.05 is good)\n"
            f"MCE: {self.calibration_metrics['mce']:.4f} "
            f"(worst-case bin error)"
        )
    
    def save(self, path: str):
        """Saves the calibrator to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'calibrator': self.calibrator,
            'metrics': self.calibration_metrics,
            'fitted': self.fitted,
            'saved_at': datetime.now().isoformat()
        }, path)
        
        print(f"   ✅ Saved calibrator: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        """Loads a calibrator from disk."""
        data = joblib.load(path)
        calibrator = cls()
        calibrator.calibrator = data['calibrator']
        calibrator.calibration_metrics = data.get('metrics', {})
        calibrator.fitted = data.get('fitted', True)
        return calibrator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate_calibration(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Evaluates calibration quality without fitting a calibrator.
    
    Args:
        y_true: Actual outcomes
        y_prob: Predicted probabilities
        verbose: Whether to print results
        
    Returns:
        Dictionary with calibration metrics
    """
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    
    if verbose:
        print("\nCalibration Analysis:")
        print("-" * 60)
        print(f"{'Bin':<12} {'N':<8} {'Accuracy':<12} {'Confidence':<12} {'Error':<10}")
        print("-" * 60)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_samples = mask.sum()
        
        if n_samples > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_error = abs(bin_accuracy - bin_confidence)
            ece += (n_samples / len(y_true)) * bin_error
            mce = max(mce, bin_error)
            
            if verbose:
                print(f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}  "
                      f"{n_samples:<8} {bin_accuracy:<12.3f} {bin_confidence:<12.3f} {bin_error:<10.3f}")
    
    if verbose:
        print("-" * 60)
        print(f"ECE: {ece:.4f}")
        print(f"MCE: {mce:.4f}")
    
    return {'ece': ece, 'mce': mce}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("ProbabilityCalibrator module loaded successfully")
    print(f"Matplotlib available: {HAS_MATPLOTLIB}")
