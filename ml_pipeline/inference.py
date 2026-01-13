"""
INFERENCE PIPELINE
Production prediction using trained ML models.

Usage:
    from ml_pipeline.inference import get_ml_predictor
    predictor = get_ml_predictor()
    predictions = predictor.predict(features_df)
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))  # Add ml_pipeline to path

# Set library path for XGBoost
os.environ['DYLD_LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Import from ml_pipeline (same directory) - use relative imports
from ml_pipeline.model_trainer import PropModelTrainer
from ml_pipeline.calibrator import ProbabilityCalibrator
from config.feature_config import TIER_ENCODING, RELIABILITY_ENCODING  # ml_models is in path

# ============================================================================
# PATHS
# ============================================================================

MODELS_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'prop_models')
CALIBRATORS_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'calibrators')


# ============================================================================
# ML PREDICTOR CLASS
# ============================================================================

class MLPredictor:
    """
    Production ML predictor for NBA player props.
    
    This class:
    1. Loads trained XGBoost models for each prop type
    2. Loads probability calibrators
    3. Provides unified prediction interface
    4. Adds ML signals to existing feature DataFrames
    
    Usage:
        predictor = MLPredictor()
        predictions = predictor.predict(features_df)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the ML predictor and load all models.
        
        Args:
            verbose: Whether to print loading messages
        """
        self.trainers: Dict[str, PropModelTrainer] = {}
        self.calibrators: Dict[str, ProbabilityCalibrator] = {}
        self.verbose = verbose
        self._load_models()
    
    def _load_models(self):
        """Loads all trained models and calibrators."""
        for prop_type in ['PTS', 'REB', 'AST', '3PM']:
            model_path = os.path.join(MODELS_DIR, f'{prop_type.lower()}_model.pkl')
            calibrator_path = os.path.join(CALIBRATORS_DIR, f'{prop_type.lower()}_calibrator.pkl')
            
            if os.path.exists(model_path):
                try:
                    self.trainers[prop_type] = PropModelTrainer.load(model_path)
                    if self.verbose:
                        print(f"   ✅ Loaded {prop_type} model")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {prop_type} model: {e}")
            
            if os.path.exists(calibrator_path):
                try:
                    self.calibrators[prop_type] = ProbabilityCalibrator.load(calibrator_path)
                    if self.verbose:
                        print(f"   ✅ Loaded {prop_type} calibrator")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {prop_type} calibrator: {e}")
        
        if self.verbose:
            print(f"   Loaded {len(self.trainers)} models, {len(self.calibrators)} calibrators")
    
    def is_ready(self) -> bool:
        """Returns True if at least one model is loaded."""
        return len(self.trainers) > 0
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features for ML models."""
        df = df.copy()
        
        # Player Tier encoding
        if 'Player_Tier' in df.columns:
            df['Player_Tier_encoded'] = df['Player_Tier'].map(TIER_ENCODING).fillna(1)
        elif 'Player_Tier_encoded' not in df.columns:
            df['Player_Tier_encoded'] = 1
        
        # Reliability Tag encoding
        if 'Reliability_Tag' in df.columns:
            df['Reliability_Tag_encoded'] = df['Reliability_Tag'].map(RELIABILITY_ENCODING).fillna(1)
        elif 'Reliability_Tag_encoded' not in df.columns:
            df['Reliability_Tag_encoded'] = 1
        
        return df
    
    def _impute_missing(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Imputes missing values with column medians."""
        df = df.copy()
        for col in features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        return df
    
    def predict_single_prop(
        self, 
        features_df: pd.DataFrame, 
        prop_type: str
    ) -> pd.DataFrame:
        """
        Generates predictions for a single prop type.
        
        Args:
            features_df: DataFrame with features
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            
        Returns:
            DataFrame with ML predictions added
        """
        if prop_type not in self.trainers:
            return features_df
        
        trainer = self.trainers[prop_type]
        calibrator = self.calibrators.get(prop_type)
        
        # Filter to this prop type
        mask = features_df['Prop_Type_Normalized'] == prop_type
        if not mask.any():
            return features_df
        
        prop_df = features_df[mask].copy()
        
        # Encode categoricals
        prop_df = self._encode_categoricals(prop_df)
        
        # Impute missing values
        prop_df = self._impute_missing(prop_df, trainer.features)
        
        # Get available features
        available_features = [f for f in trainer.features if f in prop_df.columns]
        
        if len(available_features) == 0:
            return features_df
        
        # Prepare feature matrix
        X = prop_df[available_features].copy()
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        try:
            # Get raw probabilities
            raw_probs = trainer.model.predict_proba(X)[:, 1]
            
            # Calibrate if available
            if calibrator and calibrator.fitted:
                calibrated_probs = calibrator.calibrate(raw_probs)
            else:
                calibrated_probs = raw_probs
            
            # Add predictions
            prop_df['ml_prob_over'] = calibrated_probs
            prop_df['ml_prediction'] = np.where(calibrated_probs > 0.5, 'OVER', 'UNDER')
            
            # Confidence on 1-10 scale (based on distance from 50%)
            prop_df['ml_confidence'] = np.clip(
                np.abs(calibrated_probs - 0.5) * 20, 1, 10
            ).astype(int)
            
            # Edge vs market (if Implied_Over_Pct available)
            if 'Implied_Over_Pct' in prop_df.columns:
                prop_df['ml_edge'] = calibrated_probs - (prop_df['Implied_Over_Pct'] / 100)
            else:
                prop_df['ml_edge'] = calibrated_probs - 0.5
            
            # Update the original DataFrame
            features_df.loc[mask, 'ml_prob_over'] = prop_df['ml_prob_over'].values
            features_df.loc[mask, 'ml_prediction'] = prop_df['ml_prediction'].values
            features_df.loc[mask, 'ml_confidence'] = prop_df['ml_confidence'].values
            features_df.loc[mask, 'ml_edge'] = prop_df['ml_edge'].values
            
        except Exception as e:
            print(f"   ⚠️ Prediction error for {prop_type}: {e}")
        
        return features_df
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for all props in the features DataFrame.
        
        Args:
            features_df: DataFrame with features (from your existing pipeline)
            
        Returns:
            DataFrame with predictions added:
            - ml_prob_over: Calibrated P(OVER)
            - ml_prediction: 'OVER' or 'UNDER'
            - ml_confidence: 1-10 scale
            - ml_edge: Deviation from implied probability
        """
        if not self.is_ready():
            print("⚠️ No ML models loaded. Returning original DataFrame.")
            return features_df
        
        # Initialize ML columns
        features_df = features_df.copy()
        features_df['ml_prob_over'] = 0.5
        features_df['ml_prediction'] = 'UNKNOWN'
        features_df['ml_confidence'] = 5
        features_df['ml_edge'] = 0.0
        
        # Predict for each prop type
        for prop_type in ['PTS', 'REB', 'AST', '3PM']:
            if prop_type in self.trainers:
                features_df = self.predict_single_prop(features_df, prop_type)
        
        return features_df
    
    def predict_from_dict(self, features_list: List[Dict]) -> List[Dict]:
        """
        Generates predictions from a list of feature dictionaries.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of dictionaries with ML predictions added
        """
        if not features_list:
            return features_list
        
        df = pd.DataFrame(features_list)
        df_with_predictions = self.predict(df)
        
        return df_with_predictions.to_dict('records')
    
    def get_model_summary(self) -> Dict:
        """Returns a summary of loaded models."""
        summary = {
            'models_loaded': list(self.trainers.keys()),
            'calibrators_loaded': list(self.calibrators.keys()),
            'model_details': {}
        }
        
        for prop_type, trainer in self.trainers.items():
            summary['model_details'][prop_type] = trainer.get_summary()
        
        return summary


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_predictor: Optional[MLPredictor] = None

def get_ml_predictor(verbose: bool = False) -> MLPredictor:
    """
    Returns singleton ML predictor instance.
    
    Args:
        verbose: Whether to print loading messages (only on first call)
        
    Returns:
        MLPredictor instance
    """
    global _predictor
    if _predictor is None:
        print("🤖 Initializing ML Predictor...")
        _predictor = MLPredictor(verbose=verbose)
    return _predictor


def reset_ml_predictor():
    """Resets the singleton predictor (forces reload on next get)."""
    global _predictor
    _predictor = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def add_ml_predictions(features_list: List[Dict]) -> List[Dict]:
    """
    Convenience function to add ML predictions to feature dictionaries.
    
    Args:
        features_list: List of feature dictionaries from your pipeline
        
    Returns:
        List with ML predictions added to each dictionary
    """
    predictor = get_ml_predictor()
    return predictor.predict_from_dict(features_list)


def get_ml_prediction_summary(features_df: pd.DataFrame) -> str:
    """
    Returns a formatted summary of ML predictions.
    
    Args:
        features_df: DataFrame with ML predictions
        
    Returns:
        Formatted string summary
    """
    if 'ml_prob_over' not in features_df.columns:
        return "No ML predictions in DataFrame"
    
    lines = []
    lines.append("\n📊 ML PREDICTION SUMMARY")
    lines.append("=" * 50)
    
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        prop_df = features_df[features_df['Prop_Type_Normalized'] == prop_type]
        if len(prop_df) > 0:
            avg_prob = prop_df['ml_prob_over'].mean()
            avg_conf = prop_df['ml_confidence'].mean()
            n_over = (prop_df['ml_prediction'] == 'OVER').sum()
            n_under = (prop_df['ml_prediction'] == 'UNDER').sum()
            
            lines.append(f"\n{prop_type}:")
            lines.append(f"  Count: {len(prop_df)}")
            lines.append(f"  Avg P(OVER): {avg_prob:.1%}")
            lines.append(f"  Avg Confidence: {avg_conf:.1f}/10")
            lines.append(f"  Predictions: {n_over} OVER, {n_under} UNDER")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("ML Inference Pipeline")
    print("=" * 50)
    
    predictor = get_ml_predictor(verbose=True)
    
    if predictor.is_ready():
        print("\n✅ ML Predictor ready for inference")
        print(f"\nLoaded models: {list(predictor.trainers.keys())}")
    else:
        print("\n⚠️ No models loaded. Train models first with scripts/train_with_tuning.py")
