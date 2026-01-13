"""
MODEL TRAINER
Trains prop-specific XGBoost models with hyperparameter tuning.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ xgboost not installed. Run: pip install xgboost")

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from scipy.stats import uniform, randint
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ shap not installed. SHAP analysis disabled. Run: pip install shap")

from feature_config import PROP_FEATURES, PROP_FEATURES_CLEAN, TIER_ENCODING, RELIABILITY_ENCODING, USE_CLEAN_FEATURES


class PropModelTrainer:
    """Trains and evaluates XGBoost models for each prop type."""
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.0,  # Adjust for class imbalance
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # EXPANDED Hyperparameter grid for comprehensive tuning
    # This covers a wide range of configurations for maximum accuracy
    PARAM_GRID = {
        # Tree depth: shallow (3) to deep (10) - affects complexity
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        
        # Learning rate: very slow (0.005) to aggressive (0.2)
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        
        # Number of trees: more trees = more capacity
        'n_estimators': [100, 200, 300, 500],
        
        # Row sampling: controls overfitting
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        
        # Column sampling: feature randomization per tree
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        
        # Minimum samples in leaf: regularization
        'min_child_weight': [1, 3, 5, 7, 10],
        
        # L1 regularization
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        
        # L2 regularization
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }
    
    def __init__(self, prop_type: str, params: Dict = None, use_clean_features: bool = None):
        """
        Initialize a PropModelTrainer for a specific prop type.
        
        Args:
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            params: Optional XGBoost parameters override
            use_clean_features: If True, exclude market signal features (Edge, EV, Kelly, etc.)
                               Recommended when training on quality-filtered data.
                               Defaults to USE_CLEAN_FEATURES from feature_config.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Run: pip install xgboost")
            
        self.prop_type = prop_type
        
        # Use clean features by default (no market signals)
        if use_clean_features is None:
            use_clean_features = USE_CLEAN_FEATURES
        
        self.use_clean_features = use_clean_features
        features_dict = PROP_FEATURES_CLEAN if use_clean_features else PROP_FEATURES
        self.features = features_dict.get(prop_type, [])
        
        if not self.features:
            raise ValueError(f"Unknown prop type: {prop_type}. Use PTS, REB, AST, or 3PM")
        
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts features and labels for this prop type.
        
        Args:
            df: DataFrame with all features and 'outcome' column
            
        Returns:
            Tuple of (X features DataFrame, y labels Series)
        """
        # Filter to this prop type
        prop_df = df[df['Prop_Type_Normalized'] == self.prop_type].copy()
        
        if len(prop_df) == 0:
            raise ValueError(f"No data found for prop type: {self.prop_type}")
        
        # Handle categorical encoding
        prop_df = self._encode_categoricals(prop_df)
        
        # Handle missing values
        prop_df = self._impute_missing(prop_df)
        
        # Get available features (some may be missing from data)
        available_features = [f for f in self.features if f in prop_df.columns]
        missing_features = [f for f in self.features if f not in prop_df.columns]
        
        if missing_features:
            print(f"   ⚠️ Missing features for {self.prop_type}: {missing_features[:5]}...")
        
        # Update features list to only include available ones
        self.features = available_features
        
        X = prop_df[self.features].copy()
        y = prop_df['outcome'].copy()
        
        return X, y
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features as integers."""
        df = df.copy()
        
        # Player Tier encoding
        if 'Player_Tier' in df.columns:
            df['Player_Tier_encoded'] = df['Player_Tier'].map(TIER_ENCODING).fillna(1)
        elif 'Player_Tier_encoded' not in df.columns:
            df['Player_Tier_encoded'] = 1  # Default to ROTATION
        
        # Reliability Tag encoding
        if 'Reliability_Tag' in df.columns:
            df['Reliability_Tag_encoded'] = df['Reliability_Tag'].map(RELIABILITY_ENCODING).fillna(1)
        elif 'Reliability_Tag_encoded' not in df.columns:
            df['Reliability_Tag_encoded'] = 1  # Default to STANDARD
        
        return df
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values with column medians."""
        df = df.copy()
        for col in self.features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        return df
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> 'xgb.XGBClassifier':
        """
        Trains the XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels
            
        Returns:
            Trained XGBClassifier
        """
        print(f"   Training {self.prop_type} model with {len(X_train)} samples...")
        
        self.model = xgb.XGBClassifier(**self.params)
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Performs time-series cross-validation.
        
        CRITICAL: Uses TimeSeriesSplit to prevent data leakage.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        # Reduce splits if not enough data
        n_splits = min(n_splits, len(X) // 10)
        if n_splits < 2:
            print(f"   ⚠️ Not enough data for cross-validation ({len(X)} samples)")
            return {
                'log_loss': np.nan,
                'log_loss_std': np.nan,
                'brier_score': np.nan,
                'brier_score_std': np.nan,
                'roc_auc': np.nan,
                'roc_auc_std': np.nan,
            }
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        log_losses = []
        brier_scores = []
        aucs = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Skip if validation set is too small
            if len(y_val) < 5:
                continue
            
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train, y_train, verbose=False)
            
            probs = model.predict_proba(X_val)[:, 1]
            
            try:
                log_losses.append(log_loss(y_val, probs))
                brier_scores.append(brier_score_loss(y_val, probs))
                # ROC-AUC requires both classes present
                if len(set(y_val)) > 1:
                    aucs.append(roc_auc_score(y_val, probs))
            except Exception as e:
                print(f"   ⚠️ CV fold error: {e}")
                continue
        
        return {
            'log_loss': np.mean(log_losses) if log_losses else np.nan,
            'log_loss_std': np.std(log_losses) if log_losses else np.nan,
            'brier_score': np.mean(brier_scores) if brier_scores else np.nan,
            'brier_score_std': np.std(brier_scores) if brier_scores else np.nan,
            'roc_auc': np.mean(aucs) if aucs else np.nan,
            'roc_auc_std': np.std(aucs) if aucs else np.nan,
        }
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict = None,
        n_splits: int = 5,
        n_iter: int = 200,
        verbose: bool = True
    ) -> Dict:
        """
        Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
        
        Uses random search to efficiently explore a large parameter space.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Optional custom parameter grid. Defaults to PARAM_GRID.
            n_splits: Number of CV splits for tuning
            n_iter: Number of random parameter combinations to try
            verbose: Whether to print progress
            
        Returns:
            Best parameters dict (merged with DEFAULT_PARAMS)
        """
        if param_grid is None:
            param_grid = self.PARAM_GRID
        
        if verbose:
            n_combinations = 1
            for values in param_grid.values():
                n_combinations *= len(values)
            print(f"\n   🔧 COMPREHENSIVE HYPERPARAMETER TUNING for {self.prop_type}")
            print(f"      Parameter space: {n_combinations:,} total combinations")
            print(f"      Sampling: {n_iter} random configurations")
            print(f"      Cross-validation: {n_splits} time-series folds")
            print(f"      Parameters being tuned:")
            for param, values in param_grid.items():
                print(f"         • {param}: {values}")
        
        # Create base model with fixed params (only objective and eval_metric)
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
        }
        base_model = xgb.XGBClassifier(**base_params)
        
        # TimeSeriesSplit for proper temporal CV
        n_splits = min(n_splits, len(X_train) // 100)  # Ensure enough data per fold
        if n_splits < 2:
            if verbose:
                print(f"      ⚠️ Not enough data for tuning, using defaults")
            return self.DEFAULT_PARAMS.copy()
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # RandomizedSearchCV for efficient exploration of large param space
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_log_loss',  # Negative because sklearn maximizes
            n_jobs=-1,
            verbose=1 if verbose else 0,
            refit=True,
            random_state=42
        )
        
        if verbose:
            print(f"\n      🏃 Running {n_iter} iterations...")
        
        # Run random search
        random_search.fit(X_train, y_train)
        
        # Get results
        best_params = random_search.best_params_
        best_score = -random_search.best_score_  # Convert back to positive log_loss
        
        if verbose:
            print(f"\n      ╔══════════════════════════════════════════════════════════╗")
            print(f"      ║ 🏆 BEST CONFIGURATION FOUND                               ║")
            print(f"      ╠══════════════════════════════════════════════════════════╣")
            for param, value in best_params.items():
                param_str = f"      ║   {param:<20}: {value}"
                print(f"{param_str:<62}║")
            print(f"      ╠══════════════════════════════════════════════════════════╣")
            print(f"      ║   Best CV Log Loss: {best_score:.4f}                          ║")
            print(f"      ╚══════════════════════════════════════════════════════════╝")
        
        # Get top 5 configurations for reference
        results_df = pd.DataFrame(random_search.cv_results_)
        results_df['mean_log_loss'] = -results_df['mean_test_score']
        top_5 = results_df.nsmallest(5, 'mean_log_loss')[['params', 'mean_log_loss', 'std_test_score']]
        
        if verbose:
            print(f"\n      📊 Top 5 configurations:")
            for idx, row in top_5.iterrows():
                print(f"         {idx+1}. Log Loss: {row['mean_log_loss']:.4f} (±{-row['std_test_score']:.4f})")
                params_str = ", ".join([f"{k}={v}" for k, v in list(row['params'].items())[:4]])
                print(f"            {params_str}...")
        
        # Merge with required params
        full_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
        }
        full_params.update(best_params)
        
        # Update instance params
        self.params = full_params
        
        # Store tuning results
        self.training_metrics['tuning'] = {
            'best_params': best_params,
            'best_cv_log_loss': best_score,
            'n_iter': n_iter,
            'n_splits': n_splits,
            'top_5_configs': top_5.to_dict('records') if len(top_5) > 0 else []
        }
        
        return full_params
    
    def cross_validate_player_holdout(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        player_ids: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Performs player-held-out cross-validation.
        
        CRITICAL: Splits by PLAYER to test generalization to unseen players.
        All props for a player are either ALL in train or ALL in test.
        
        This prevents:
        - Player memorization (model learning "LeBron always goes OVER")
        - Inflated accuracy from seeing same players in train/test
        
        Data leakage is STILL PREVENTED because:
        - Rolling features (L5, L10, Season) are computed from past games only
        - Labels are from actual game outcomes
        - The split is by player GROUP, not by random row
        
        Args:
            X: Feature matrix
            y: Labels
            player_ids: Series of player IDs for grouping
            n_splits: Number of CV folds
            test_size: Fraction of players held out per fold
            
        Returns:
            Dictionary with cross-validation metrics
        """
        from sklearn.model_selection import GroupShuffleSplit
        
        # Ensure we have enough unique players
        n_players = player_ids.nunique()
        if n_players < 10:
            print(f"   ⚠️ Not enough unique players for player-holdout CV ({n_players} players)")
            return self.cross_validate(X, y, n_splits)
        
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        
        log_losses = []
        brier_scores = []
        aucs = []
        accuracies = []
        
        print(f"   🧪 Player-Holdout CV: {n_splits} folds, {test_size*100:.0f}% players held out per fold")
        
        for fold_idx, (train_idx, val_idx) in enumerate(gss.split(X, y, groups=player_ids)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Skip if validation set is too small or missing a class
            if len(y_val) < 10 or len(set(y_val)) < 2:
                continue
            
            # Count unique players in each set
            train_players = player_ids.iloc[train_idx].nunique()
            val_players = player_ids.iloc[val_idx].nunique()
            
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train, y_train, verbose=False)
            
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            try:
                log_losses.append(log_loss(y_val, probs))
                brier_scores.append(brier_score_loss(y_val, probs))
                aucs.append(roc_auc_score(y_val, probs))
                accuracies.append((preds == y_val).mean())
            except Exception as e:
                print(f"   ⚠️ Fold {fold_idx+1} error: {e}")
                continue
        
        if accuracies:
            print(f"      Avg accuracy on unseen players: {np.mean(accuracies):.1%} (+/- {np.std(accuracies):.1%})")
        
        return {
            'log_loss': np.mean(log_losses) if log_losses else np.nan,
            'log_loss_std': np.std(log_losses) if log_losses else np.nan,
            'brier_score': np.mean(brier_scores) if brier_scores else np.nan,
            'brier_score_std': np.std(brier_scores) if brier_scores else np.nan,
            'roc_auc': np.mean(aucs) if aucs else np.nan,
            'roc_auc_std': np.std(aucs) if aucs else np.nan,
            'accuracy': np.mean(accuracies) if accuracies else np.nan,
            'accuracy_std': np.std(accuracies) if accuracies else np.nan,
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability of OVER for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of P(OVER) probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Returns binary predictions (1=OVER, 0=UNDER).
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for OVER prediction
            
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def compute_shap_values(
        self, 
        X: pd.DataFrame, 
        sample_size: int = 1000
    ) -> Optional[Tuple]:
        """
        Computes SHAP values for feature importance analysis.
        
        Args:
            X: Feature matrix
            sample_size: Max samples to use for SHAP (for efficiency)
            
        Returns:
            Tuple of (explainer, shap_values, X_sample) or None if SHAP unavailable
        """
        if not HAS_SHAP:
            print("   ⚠️ SHAP not installed. Skipping SHAP analysis.")
            return None
            
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Sample for efficiency
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X.copy()
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        return explainer, shap_values, X_sample
    
    def save(self, path: str):
        """
        Saves the trained model to disk.
        
        Args:
            path: File path for saving (should end in .pkl)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'features': self.features,
            'feature_importance': self.feature_importance,
            'params': self.params,
            'prop_type': self.prop_type,
            'training_metrics': self.training_metrics,
            'saved_at': datetime.now().isoformat()
        }, path)
        
        print(f"   ✅ Saved model: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PropModelTrainer':
        """
        Loads a trained model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            PropModelTrainer instance with loaded model
        """
        data = joblib.load(path)
        trainer = cls(data['prop_type'], data['params'])
        trainer.model = data['model']
        trainer.features = data['features']
        trainer.feature_importance = data.get('feature_importance')
        trainer.training_metrics = data.get('training_metrics', {})
        return trainer
    
    def get_summary(self) -> Dict:
        """Returns a summary of the trained model."""
        return {
            'prop_type': self.prop_type,
            'n_features': len(self.features),
            'top_features': self.feature_importance.head(5).to_dict('records') if self.feature_importance is not None else [],
            'training_metrics': self.training_metrics
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_labeled_data(path: str) -> pd.DataFrame:
    """
    Loads labeled training data from parquet or CSV.
    
    Args:
        path: Path to data file
        
    Returns:
        DataFrame with labeled data
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def temporal_train_test_split(
    df: pd.DataFrame, 
    test_ratio: float = 0.2,
    date_col: str = 'game_date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data temporally (NOT randomly) to prevent leakage.
    
    Train: Earlier dates
    Test: Later dates
    
    Args:
        df: DataFrame with date column
        test_ratio: Fraction of data for test set
        date_col: Name of date column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"   Train: {train_df[date_col].min()} to {train_df[date_col].max()} ({len(train_df)} samples)")
    print(f"   Test:  {test_df[date_col].min()} to {test_df[date_col].max()} ({len(test_df)} samples)")
    
    return train_df, test_df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Quick test with sample data
    print("PropModelTrainer module loaded successfully")
    print(f"XGBoost available: {HAS_XGBOOST}")
    print(f"SHAP available: {HAS_SHAP}")
