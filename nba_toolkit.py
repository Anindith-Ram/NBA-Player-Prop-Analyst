"""
NBA PROP ANALYST TOOLKIT
========================
Consolidated interface for all NBA prop analysis operations.

This toolkit provides a unified API for:
1. Data Management (Kaggle updates, odds fetching, dataset building)
2. Model Training (with best model tracking)
3. Predictions (with edge calculation and filtering)

Usage:
    from nba_toolkit import NBAPropToolkit
    
    toolkit = NBAPropToolkit()
    
    # Update data incrementally
    toolkit.update_training_data()
    
    # Train models
    toolkit.train_models(data_path="datasets/ml_training/training_data.parquet")
    
    # Make predictions
    predictions = toolkit.predict_today(target_date="2025-12-14")
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

from scripts.update_kaggle_data import main as update_kaggle_main
from scripts.fetch_historical_odds import fetch_season_odds
from scripts.build_combined_dataset import run_pipeline as build_dataset_pipeline
from scripts.train_with_tuning import train_all_with_tuning
from scripts.predict_today import (
    fetch_todays_player_props,
    build_features_for_props,
    calculate_edge_for_props,
    run_predictions
)
from ml_pipeline.best_model_tracker import BestModelTracker
from ml_pipeline.inference import MLPredictor
from process.shared_config import DATASETS_DIR, PREDICTIONS_DIR


class NBAPropToolkit:
    """
    Unified toolkit for NBA prop analysis operations.
    
    Provides high-level interfaces for:
    - Data management (updates, fetching, building)
    - Model training (with best model tracking)
    - Predictions (with edge calculation)
    """
    
    def __init__(self):
        """Initialize the toolkit."""
        self.ml_training_dir = os.path.join(DATASETS_DIR, 'ml_training')
        self.odds_dir = os.path.join(DATASETS_DIR, 'odds')
        self.models_dir = os.path.join(PROJECT_ROOT, 'ml_models', 'prop_models')
        self.calibrators_dir = os.path.join(PROJECT_ROOT, 'ml_models', 'calibrators')
        self.predictions_dir = PREDICTIONS_DIR
        
        # Ensure directories exist
        os.makedirs(self.ml_training_dir, exist_ok=True)
        os.makedirs(self.odds_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.calibrators_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
    
    # ============================================================================
    # DATA MANAGEMENT
    # ============================================================================
    
    def update_kaggle_data(self) -> bool:
        """
        Update Kaggle data (player stats, team stats, games, schedules).
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*70)
        print("UPDATING KAGGLE DATA")
        print("="*70)
        
        try:
            result = update_kaggle_main()
            return result == 0
        except Exception as e:
            print(f"   ❌ Error updating Kaggle data: {e}")
            return False
    
    def fetch_odds(
        self,
        season: str = "2025-26",
        start_date: str = None,
        end_date: str = None,
        single_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical odds for a season or date range.
        
        Args:
            season: Season string (e.g., "2025-26")
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            single_date: Single date to fetch (YYYY-MM-DD), optional
            
        Returns:
            DataFrame with odds data
        """
        print("\n" + "="*70)
        print("FETCHING ODDS")
        print("="*70)
        
        try:
            if single_date:
                df = fetch_season_odds(
                    season,
                    force_refresh=False,
                    start_date_override=single_date,
                    end_date_override=single_date
                )
            elif start_date and end_date:
                df = fetch_season_odds(
                    season,
                    force_refresh=False,
                    start_date_override=start_date,
                    end_date_override=end_date
                )
            else:
                df = fetch_season_odds(season, force_refresh=False)
            
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"   ❌ Error fetching odds: {e}")
            return pd.DataFrame()
    
    def update_training_data(
        self,
        incremental: bool = True,
        skip_kaggle: bool = False,
        skip_odds: bool = False
    ) -> str:
        """
        Update training data (incremental or full rebuild).
        
        Args:
            incremental: If True, only add yesterday's data (default: True)
            skip_kaggle: Skip Kaggle data update
            skip_odds: Skip odds fetching
            
        Returns:
            Path to training data file
        """
        print("\n" + "="*70)
        print("UPDATING TRAINING DATA")
        print("="*70)
        
        try:
            output_path = build_dataset_pipeline(
                target_date=None,
                force_rebuild=not incremental,
                skip_kaggle=skip_kaggle,
                skip_odds=skip_odds,
                incremental=incremental
            )
            return output_path
        except Exception as e:
            print(f"   ❌ Error updating training data: {e}")
            raise
    
    def get_training_data_path(self) -> Optional[str]:
        """
        Get path to the current training data file.
        
        Returns:
            Path to training_data.parquet, or None if not found
        """
        training_data_path = os.path.join(self.ml_training_dir, 'training_data.parquet')
        if os.path.exists(training_data_path):
            return training_data_path
        return None
    
    # ============================================================================
    # MODEL TRAINING
    # ============================================================================
    
    def train_models(
        self,
        data_path: str = None,
        tune: bool = True,
        n_iter: int = None
    ) -> Dict:
        """
        Train all prop models with hyperparameter tuning.
        
        Models are automatically saved only if they beat the current best
        (based on calibrated accuracy).
        
        Args:
            data_path: Path to training data (defaults to training_data.parquet)
            tune: Whether to run hyperparameter tuning
            n_iter: Number of random configurations to try
            
        Returns:
            Dictionary with training results for each prop type
        """
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        if data_path is None:
            data_path = self.get_training_data_path()
            if data_path is None:
                raise ValueError("No training data found. Run update_training_data() first.")
        
        try:
            results = train_all_with_tuning(
                data_path=data_path,
                tune=tune,
                n_iter=n_iter
            )
            
            # Show best models summary
            tracker = BestModelTracker()
            tracker.print_summary()
            
            return results
        except Exception as e:
            print(f"   ❌ Error training models: {e}")
            raise
    
    def get_best_models_summary(self) -> Dict:
        """
        Get summary of best models per prop type.
        
        Returns:
            Dictionary with best model info for each prop type
        """
        tracker = BestModelTracker()
        return tracker.get_summary()
    
    # ============================================================================
    # PREDICTIONS
    # ============================================================================
    
    def predict_today(
        self,
        target_date: str = None,
        confidence_threshold: int = 7,
        mode: str = 'premium',
        ml_only: bool = False,
        max_props: int = 10,
        max_props_per_game: int = 15,
        historical_data_path: str = None
    ) -> pd.DataFrame:
        """
        Generate predictions for today's props with edge filtering.
        
        This function:
        1. Fetches today's props (max 15 per game, DraftKings only)
        2. Calculates Edge for each prop
        3. Filters to props with significant edge (OVER or UNDER)
        4. Runs ML predictions
        5. Returns results with edge information
        
        Args:
            target_date: Date to predict (YYYY-MM-DD), defaults to today
            confidence_threshold: Minimum confidence for recommendations (default: 7)
            mode: Prediction mode ('fast', 'balanced', 'premium')
            ml_only: Use ML only, skip LLM
            max_props: Max props for Gemini to analyze (default: 10)
            max_props_per_game: Max props to fetch per game (default: 15)
            historical_data_path: Path to historical data (auto-detected if None)
            
        Returns:
            DataFrame with predictions and edge information
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print("\n" + "="*70)
        print(f"GENERATING PREDICTIONS FOR {target_date}")
        print("="*70)
        
        # Step 1: Fetch today's props
        props_df = fetch_todays_player_props(
            target_date=target_date,
            max_props_per_game=max_props_per_game
        )
        
        if len(props_df) == 0:
            print("   ⚠️ No props available for this date")
            return pd.DataFrame()
        
        # Step 2: Load historical data
        if historical_data_path is None:
            historical_data_path = self.get_training_data_path()
            if historical_data_path is None:
                # Try alternative paths
                candidates = [
                    os.path.join(self.ml_training_dir, f'training_data_{target_date.replace("-", "")}.parquet'),
                    os.path.join(self.ml_training_dir, 'super_dataset_2024_25_regular_season.parquet'),
                ]
                for path in candidates:
                    if os.path.exists(path):
                        historical_data_path = path
                        break
        
        if historical_data_path is None or not os.path.exists(historical_data_path):
            raise ValueError("No historical data found. Run update_training_data() first.")
        
        print(f"\n📂 Loading historical data from: {historical_data_path}")
        historical_df = pd.read_parquet(historical_data_path)
        print(f"   ✅ Loaded {len(historical_df):,} historical records")
        
        # Step 3: Build features
        features_df = build_features_for_props(props_df, historical_df)
        
        if len(features_df) == 0:
            print("   ⚠️ Could not build features for any props")
            return pd.DataFrame()
        
        # Step 4: Calculate Edge and filter
        print("\n" + "="*70)
        print("EDGE CALCULATION & FILTERING")
        print("="*70)
        features_df = calculate_edge_for_props(features_df, historical_df, lookback_games=10)
        
        # Filter to props with significant edge
        under_edge_threshold = -0.05
        features_df_with_edge = features_df[
            (features_df['Edge'] > 0) | (features_df['Edge'] < under_edge_threshold)
        ].copy()
        
        over_edge_count = (features_df_with_edge['Edge'] > 0).sum()
        under_edge_count = (features_df_with_edge['Edge'] < under_edge_threshold).sum()
        
        print(f"\n   🔍 Filtered to {len(features_df_with_edge)} props with significant edge")
        print(f"      - OVER edge: {over_edge_count} props")
        print(f"      - UNDER edge: {under_edge_count} props")
        
        if len(features_df_with_edge) == 0:
            print("   ⚠️ No props with significant edge found")
            return pd.DataFrame()
        
        # Step 5: Run predictions
        results = run_predictions(
            features_df_with_edge,
            confidence_threshold=confidence_threshold,
            mode=mode,
            ml_only=ml_only,
            max_props=max_props
        )
        
        # Ensure Edge is in results
        if 'Edge' not in results.columns and 'Edge' in features_df_with_edge.columns:
            edge_map = features_df_with_edge.set_index(['FullName', 'Prop_Type_Normalized', 'Line'])['Edge'].to_dict()
            results['Edge'] = results.apply(
                lambda row: edge_map.get((row.get('FullName'), row.get('Prop_Type_Normalized'), row.get('Line')), 0),
                axis=1
            )
        
        return results
    
    def get_model_predictor(self) -> MLPredictor:
        """
        Get the ML predictor instance (for custom predictions).
        
        Returns:
            MLPredictor instance
        """
        return MLPredictor()
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_data_status(self) -> Dict:
        """
        Get status of all data files.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'training_data': None,
            'odds_data': {},
            'best_models': {}
        }
        
        # Check training data
        training_path = self.get_training_data_path()
        if training_path:
            df = pd.read_parquet(training_path)
            status['training_data'] = {
                'path': training_path,
                'records': len(df),
                'date_range': (
                    str(df['game_date'].min()) if 'game_date' in df.columns else 'N/A',
                    str(df['game_date'].max()) if 'game_date' in df.columns else 'N/A'
                )
            }
        
        # Check odds data
        for season in ['2024_25', '2025_26']:
            odds_path = os.path.join(self.odds_dir, f'historical_odds_{season}.parquet')
            if os.path.exists(odds_path):
                df = pd.read_parquet(odds_path)
                status['odds_data'][season] = {
                    'records': len(df),
                    'dates': df['game_date'].nunique() if 'game_date' in df.columns else 0
                }
        
        # Check best models
        tracker = BestModelTracker()
        status['best_models'] = tracker.get_summary()
        
        return status
    
    def print_status(self):
        """Print comprehensive status of all data and models."""
        status = self.get_data_status()
        
        print("\n" + "="*70)
        print("NBA PROP ANALYST - SYSTEM STATUS")
        print("="*70)
        
        print("\n📊 Training Data:")
        if status['training_data']:
            td = status['training_data']
            print(f"   Path: {td['path']}")
            print(f"   Records: {td['records']:,}")
            print(f"   Date Range: {td['date_range'][0]} → {td['date_range'][1]}")
        else:
            print("   ⚠️ No training data found")
        
        print("\n📈 Odds Data:")
        if status['odds_data']:
            for season, info in status['odds_data'].items():
                print(f"   {season}: {info['records']:,} records, {info['dates']} dates")
        else:
            print("   ⚠️ No odds data found")
        
        print("\n🏆 Best Models:")
        for prop_type, info in status['best_models'].items():
            if info:
                print(f"   {prop_type}: {info['calibrated_accuracy']:.1%} accuracy")
                print(f"      Updated: {info['updated_at']}")
            else:
                print(f"   {prop_type}: No model saved yet")
        
        print("="*70)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_update_and_predict(target_date: str = None) -> pd.DataFrame:
    """
    Quick workflow: Update data incrementally, then predict today.
    
    This is the main daily workflow:
    1. Update training data (add yesterday)
    2. Generate predictions for today
    
    Args:
        target_date: Date to predict (defaults to today)
        
    Returns:
        DataFrame with predictions
    """
    toolkit = NBAPropToolkit()
    
    # Update training data incrementally
    toolkit.update_training_data(incremental=True)
    
    # Predict today
    predictions = toolkit.predict_today(target_date=target_date)
    
    return predictions


def train_and_evaluate(data_path: str = None, tune: bool = True) -> Dict:
    """
    Train models and return evaluation metrics.
    
    Args:
        data_path: Path to training data (defaults to training_data.parquet)
        tune: Whether to run hyperparameter tuning
        
    Returns:
        Dictionary with training results
    """
    toolkit = NBAPropToolkit()
    return toolkit.train_models(data_path=data_path, tune=tune)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Prop Analyst Toolkit')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--update-data', action='store_true', help='Update training data incrementally')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions for today')
    parser.add_argument('--quick', action='store_true', help='Quick workflow: update data + predict')
    parser.add_argument('--date', type=str, default=None, help='Target date for predictions (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    toolkit = NBAPropToolkit()
    
    if args.status:
        toolkit.print_status()
    elif args.quick:
        predictions = quick_update_and_predict(target_date=args.date)
        if len(predictions) > 0:
            print(f"\n✅ Generated {len(predictions)} predictions")
    elif args.update_data:
        toolkit.update_training_data(incremental=True)
    elif args.train:
        toolkit.train_models()
    elif args.predict:
        predictions = toolkit.predict_today(target_date=args.date)
        if len(predictions) > 0:
            print(f"\n✅ Generated {len(predictions)} predictions")
    else:
        parser.print_help()

