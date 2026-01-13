"""
META-ENSEMBLE
Second-level model that learns when to trust which base model.

This stacking ensemble combines predictions from the 4 prop-specific XGBoost models
with contextual features to make a final prediction. The meta-model learns:
- When to trust PTS model vs REB model vs AST model vs 3PM model
- How player tier and reliability affect model accuracy
- How market signals correlate with model accuracy
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))

from feature_config import TIER_ENCODING, RELIABILITY_ENCODING

# ============================================================================
# PATHS
# ============================================================================

META_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_models', 'meta_model.pkl')


class MetaEnsemble:
    """
    Stacking ensemble that combines base model predictions with context.
    
    Level 1: Prop-specific XGBoost models (PTS, REB, AST, 3PM)
    Level 2: Logistic regression on base predictions + context
    
    The meta-model learns patterns like:
    - "When PTS model says 70% OVER but player is VOLATILE, reduce confidence"
    - "When ML edge is large AND market edge is large, trust the prediction more"
    - "When rest_days = 0 (B2B), models tend to overpredict"
    """
    
    def __init__(self):
        self.meta_model = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = []
        
        # Features from base models
        self.base_features = ['ml_prob_over']
        
        # Context features to include
        self.context_features = [
            'Player_Tier_encoded',
            'Reliability_Tag_encoded',
            'Edge',
            'EV',
            'Kelly',
            'quality_score',
            'Implied_Over_Pct',
            'rest_days',
            'is_b2b',
            'home',
        ]
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares feature matrix for meta-model."""
        df = df.copy()
        
        # Encode categoricals if needed
        if 'Player_Tier' in df.columns and 'Player_Tier_encoded' not in df.columns:
            df['Player_Tier_encoded'] = df['Player_Tier'].map(TIER_ENCODING).fillna(1)
        
        if 'Reliability_Tag' in df.columns and 'Reliability_Tag_encoded' not in df.columns:
            df['Reliability_Tag_encoded'] = df['Reliability_Tag'].map(RELIABILITY_ENCODING).fillna(1)
        
        # Collect available features
        available_features = []
        
        # Base model predictions
        for f in self.base_features:
            if f in df.columns:
                available_features.append(f)
        
        # Context features
        for f in self.context_features:
            if f in df.columns:
                available_features.append(f)
        
        self.feature_names = available_features
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Fill NaN with defaults
        X = X.fillna(0)
        
        # Convert booleans
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        return X
    
    def fit(
        self, 
        df: pd.DataFrame, 
        outcome_col: str = 'outcome'
    ):
        """
        Fits the meta-model on predictions with known outcomes.
        
        Args:
            df: DataFrame with ML predictions and outcomes
            outcome_col: Name of outcome column (0/1)
        """
        print("🎯 Training Meta-Ensemble...")
        
        # Prepare features
        X = self._prepare_features(df)
        y = df[outcome_col].values
        
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Samples: {len(X)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit meta-model
        self.meta_model.fit(X_scaled, y)
        self.fitted = True
        
        # Show feature importance
        print(f"\n   Feature weights:")
        coefs = self.meta_model.coef_[0]
        for name, coef in sorted(zip(self.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
            print(f"      {name}: {coef:+.4f}")
        
        print(f"\n   ✅ Meta-ensemble trained")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns ensemble-calibrated probabilities.
        
        Args:
            df: DataFrame with ML predictions
            
        Returns:
            Array of P(OVER) probabilities
        """
        if not self.fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        return self.meta_model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions."""
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)
    
    def enhance_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds meta-ensemble predictions to DataFrame.
        
        Args:
            df: DataFrame with ML predictions
            
        Returns:
            DataFrame with meta_prob_over, meta_prediction, meta_confidence added
        """
        df = df.copy()
        
        probs = self.predict_proba(df)
        
        df['meta_prob_over'] = probs
        df['meta_prediction'] = np.where(probs > 0.5, 'OVER', 'UNDER')
        df['meta_confidence'] = np.clip(np.abs(probs - 0.5) * 20, 1, 10).astype(int)
        
        # Combined edge (average of ML and meta)
        if 'ml_edge' in df.columns:
            df['combined_edge'] = (df['ml_edge'] + (probs - 0.5)) / 2
        
        return df
    
    def save(self, path: str = None):
        """Saves the meta-ensemble to disk."""
        path = path or META_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'base_features': self.base_features,
            'context_features': self.context_features,
            'fitted': self.fitted,
            'saved_at': datetime.now().isoformat()
        }, path)
        
        print(f"   ✅ Meta-ensemble saved: {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'MetaEnsemble':
        """Loads a meta-ensemble from disk."""
        path = path or META_MODEL_PATH
        
        data = joblib.load(path)
        ensemble = cls()
        ensemble.meta_model = data['meta_model']
        ensemble.scaler = data['scaler']
        ensemble.feature_names = data['feature_names']
        ensemble.base_features = data.get('base_features', ensemble.base_features)
        ensemble.context_features = data.get('context_features', ensemble.context_features)
        ensemble.fitted = data.get('fitted', True)
        
        return ensemble


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_meta_ensemble(
    labeled_df: pd.DataFrame,
    save: bool = True
) -> MetaEnsemble:
    """
    Trains the meta-ensemble on labeled data with ML predictions.
    
    IMPORTANT: Uses temporal split to avoid data leakage!
    - Trains on first 80% of data (by date)
    - Evaluates on last 20% (held-out test set)
    
    Args:
        labeled_df: DataFrame with ML predictions (ml_prob_over) and outcomes
        save: Whether to save the trained model
        
    Returns:
        Trained MetaEnsemble
    """
    print("="*60)
    print("TRAINING META-ENSEMBLE")
    print("="*60)
    
    # Verify required columns
    required = ['ml_prob_over', 'outcome']
    missing = [c for c in required if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # CRITICAL: Temporal split to avoid data leakage
    df = labeled_df.copy()
    if 'game_date' in df.columns:
        df = df.sort_values('game_date')
    
    total_samples = len(df)
    train_size = int(total_samples * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"🎯 Training Meta-Ensemble...")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Initialize and train on TRAINING data only
    ensemble = MetaEnsemble()
    ensemble.fit(train_df, 'outcome')
    
    # Evaluate on HELD-OUT TEST data
    test_base_probs = test_df['ml_prob_over'].values
    test_meta_probs = ensemble.predict_proba(test_df)
    y_test = test_df['outcome'].values
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    test_base_preds = (test_base_probs >= 0.5).astype(int)
    test_meta_preds = (test_meta_probs >= 0.5).astype(int)
    
    base_acc = accuracy_score(y_test, test_base_preds)
    meta_acc = accuracy_score(y_test, test_meta_preds)
    
    try:
        base_auc = roc_auc_score(y_test, test_base_probs)
        meta_auc = roc_auc_score(y_test, test_meta_probs)
    except:
        base_auc = 0.5
        meta_auc = 0.5
    
    print(f"\n📊 Performance on HELD-OUT TEST SET ({len(test_df)} samples):")
    print(f"   Base Model:   Accuracy={base_acc:.1%}, ROC-AUC={base_auc:.4f}")
    print(f"   Meta-Ensemble: Accuracy={meta_acc:.1%}, ROC-AUC={meta_auc:.4f}")
    
    improvement = (meta_acc - base_acc) * 100
    print(f"   Improvement: {improvement:+.1f}% accuracy")
    
    if save:
        ensemble.save()
    
    print("="*60)
    
    return ensemble


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
    
    # Set library path for XGBoost
    os.environ['DYLD_LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    
    from data_preprocessor import load_training_csvs, load_actuals_pandas, create_outcome_labels_pandas, encode_categoricals
    from inference import get_ml_predictor
    
    print("Loading data and generating ML predictions...")
    
    # Load and prepare data
    features_df = load_training_csvs()
    actuals_df = load_actuals_pandas()
    labeled_df = create_outcome_labels_pandas(features_df, actuals_df)
    labeled_df = encode_categoricals(labeled_df)
    
    # Get ML predictions
    predictor = get_ml_predictor(verbose=False)
    predictions_df = predictor.predict(labeled_df)
    
    # Train meta-ensemble
    ensemble = train_meta_ensemble(predictions_df, save=True)
