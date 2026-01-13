"""
DATA PREPROCESSOR
Converts raw features + actuals into labeled training data for XGBoost.

This module:
1. Joins features with actual game results
2. Creates binary outcome labels (1=OVER, 0=UNDER)
3. Handles missing values and categorical encoding
4. Exports labeled data for ML training
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, to_date, to_timestamp, date_format,
    coalesce
)

# Import from ml_models
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))
from feature_config import PROP_TO_ACTUAL, TIER_ENCODING, RELIABILITY_ENCODING

# Import shared config
from shared_config import (
    get_spark_session,
    stop_spark_session,
    PLAYER_STATS_CSV,
    DATASETS_DIR
)

# ============================================================================
# CONFIGURATION
# ============================================================================

ML_TRAINING_DIR = os.path.join(DATASETS_DIR, 'ml_training')
TRAINING_INPUTS_DIR = os.path.join(DATASETS_DIR, 'training_inputs')


# ============================================================================
# SPARK-BASED LABELING
# ============================================================================

def load_actuals_spark(spark: SparkSession) -> DataFrame:
    """
    Loads actual player statistics from Kaggle CSV.
    
    Returns:
        DataFrame with actual game results (points, rebounds, assists, 3PM)
    """
    print("📊 Loading PlayerStatistics.csv for actuals...")
    
    df = spark.read.csv(
        PLAYER_STATS_CSV,
        header=True,
        inferSchema=True
    )
    
    # Parse timestamp and extract date
    df = df.withColumn(
        "game_date",
        to_date(to_timestamp(col("gameDateTimeEst"), "yyyy-MM-dd HH:mm:ss"))
    )
    
    # Select only needed columns for actuals
    actuals = df.select(
        col("personId"),
        col("game_date"),
        col("points"),
        col("reboundsTotal"),
        col("assists"),
        col("threePointersMade")
    )
    
    print(f"   ✅ Loaded {actuals.count():,} actual game records")
    return actuals


def create_outcome_labels_spark(
    features_df: DataFrame, 
    actuals_df: DataFrame
) -> DataFrame:
    """
    Joins features with actuals to create binary outcome labels (PySpark version).
    
    y = 1 if Actual > Line (OVER hit)
    y = 0 if Actual <= Line (UNDER hit)
    
    Args:
        features_df: Features DataFrame with Line column
        actuals_df: Actuals DataFrame with points, assists, etc.
        
    Returns:
        Labeled DataFrame with 'outcome' column (0 or 1)
    """
    # Rename actuals columns to avoid confusion
    actuals_renamed = actuals_df.select(
        col("personId").alias("actual_personId"),
        col("game_date").alias("actual_game_date"),
        col("points").alias("actual_points"),
        col("reboundsTotal").alias("actual_reboundsTotal"),
        col("assists").alias("actual_assists"),
        col("threePointersMade").alias("actual_threePointersMade")
    )
    
    # Join on personId and game_date
    joined = features_df.join(
        actuals_renamed,
        (features_df["personId"] == actuals_renamed["actual_personId"]) &
        (features_df["game_date"] == actuals_renamed["actual_game_date"]),
        how="inner"
    )
    
    # Create actual_value based on prop type
    labeled = joined.withColumn(
        "actual_value",
        when(col("Prop_Type_Normalized") == "PTS", col("actual_points"))
        .when(col("Prop_Type_Normalized") == "REB", col("actual_reboundsTotal"))
        .when(col("Prop_Type_Normalized") == "AST", col("actual_assists"))
        .when(col("Prop_Type_Normalized") == "3PM", col("actual_threePointersMade"))
        .otherwise(lit(None))
    )
    
    # Binary outcome: 1 = OVER, 0 = UNDER
    labeled = labeled.withColumn(
        "outcome",
        when(col("actual_value") > col("Line"), lit(1)).otherwise(lit(0))
    )
    
    # Drop duplicate columns
    labeled = labeled.drop("actual_personId", "actual_game_date")
    
    return labeled


# ============================================================================
# PANDAS-BASED LABELING (for CSV files)
# ============================================================================

def load_training_csvs(training_dir: str = None) -> pd.DataFrame:
    """
    Loads all training input CSVs and combines them.
    
    Args:
        training_dir: Directory containing training CSV files
        
    Returns:
        Combined DataFrame with all training features
    """
    training_dir = training_dir or TRAINING_INPUTS_DIR
    
    print(f"📂 Loading training CSVs from {training_dir}...")
    
    csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {training_dir}")
    
    dfs = []
    for csv_file in sorted(csv_files):
        filepath = os.path.join(training_dir, csv_file)
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"   Loaded {csv_file}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Total: {len(combined):,} rows from {len(csv_files)} files")
    
    return combined


def load_actuals_pandas(csv_path: str = None) -> pd.DataFrame:
    """
    Loads actual player statistics from Kaggle CSV (Pandas version).
    
    Args:
        csv_path: Path to PlayerStatistics.csv
        
    Returns:
        DataFrame with actual game results
    """
    csv_path = csv_path or PLAYER_STATS_CSV
    
    print(f"📊 Loading actuals from {csv_path}...")
    
    actuals = pd.read_csv(csv_path, low_memory=False)
    
    # Parse date - handle mixed formats with timezone
    actuals['game_date'] = pd.to_datetime(
        actuals['gameDateTimeEst'], 
        format='mixed',
        utc=True
    ).dt.strftime('%Y-%m-%d')
    
    # Select needed columns
    actuals = actuals[[
        'personId', 'game_date', 'points', 'reboundsTotal', 
        'assists', 'threePointersMade'
    ]]
    
    print(f"   ✅ Loaded {len(actuals):,} actual game records")
    return actuals


def create_outcome_labels_pandas(
    features_df: pd.DataFrame,
    actuals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins features with actuals to create binary outcome labels (Pandas version).
    
    Args:
        features_df: Features DataFrame with Line column
        actuals_df: Actuals DataFrame with points, assists, etc.
        
    Returns:
        Labeled DataFrame with 'outcome' column (0 or 1)
    """
    print("🔗 Joining features with actuals...")
    
    # Ensure game_date is string format for both
    features_df = features_df.copy()
    actuals_df = actuals_df.copy()
    
    features_df['game_date'] = pd.to_datetime(features_df['game_date']).dt.strftime('%Y-%m-%d')
    actuals_df['game_date'] = pd.to_datetime(actuals_df['game_date']).dt.strftime('%Y-%m-%d')
    
    # Merge on personId and game_date
    labeled = features_df.merge(
        actuals_df,
        on=['personId', 'game_date'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    print(f"   Matched {len(labeled):,} rows (from {len(features_df):,} features)")
    
    # Create actual_value based on prop type
    def get_actual_value(row):
        prop_type = row.get('Prop_Type_Normalized', '')
        if prop_type == 'PTS':
            return row.get('points', np.nan)
        elif prop_type == 'REB':
            return row.get('reboundsTotal', np.nan)
        elif prop_type == 'AST':
            return row.get('assists', np.nan)
        elif prop_type == '3PM':
            return row.get('threePointersMade', np.nan)
        return np.nan
    
    labeled['actual_value'] = labeled.apply(get_actual_value, axis=1)
    
    # Binary outcome: 1 = OVER, 0 = UNDER
    labeled['outcome'] = (labeled['actual_value'] > labeled['Line']).astype(int)
    
    # Log outcome distribution
    over_count = (labeled['outcome'] == 1).sum()
    under_count = (labeled['outcome'] == 0).sum()
    print(f"   Outcome distribution: OVER={over_count}, UNDER={under_count}")
    
    return labeled


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features as integers.
    
    Args:
        df: DataFrame with Player_Tier and Reliability_Tag columns
        
    Returns:
        DataFrame with encoded columns added
    """
    df = df.copy()
    
    # Player Tier encoding
    if 'Player_Tier' in df.columns:
        df['Player_Tier_encoded'] = df['Player_Tier'].map(TIER_ENCODING).fillna(1)
    else:
        df['Player_Tier_encoded'] = 1  # Default to ROTATION
    
    # Reliability Tag encoding  
    if 'Reliability_Tag' in df.columns:
        df['Reliability_Tag_encoded'] = df['Reliability_Tag'].map(RELIABILITY_ENCODING).fillna(1)
    else:
        df['Reliability_Tag_encoded'] = 1  # Default to STANDARD
    
    return df


def prepare_labeled_dataset(
    features_df: pd.DataFrame = None,
    actuals_df: pd.DataFrame = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Full pipeline to prepare labeled dataset for ML training.
    
    Args:
        features_df: Optional pre-loaded features DataFrame
        actuals_df: Optional pre-loaded actuals DataFrame
        output_path: Path to save the labeled dataset
        
    Returns:
        Labeled and encoded DataFrame ready for ML training
    """
    print("="*60)
    print("PREPARING LABELED DATASET FOR ML TRAINING")
    print(f"Started: {datetime.now()}")
    print("="*60)
    
    # Load data if not provided
    if features_df is None:
        features_df = load_training_csvs()
    
    if actuals_df is None:
        actuals_df = load_actuals_pandas()
    
    # Create outcome labels
    labeled_df = create_outcome_labels_pandas(features_df, actuals_df)
    
    # Encode categoricals
    labeled_df = encode_categoricals(labeled_df)
    
    # Remove rows with missing outcomes
    before_count = len(labeled_df)
    labeled_df = labeled_df.dropna(subset=['outcome', 'actual_value'])
    after_count = len(labeled_df)
    
    if before_count > after_count:
        print(f"   Dropped {before_count - after_count} rows with missing outcomes")
    
    # Ensure output directory exists
    os.makedirs(ML_TRAINING_DIR, exist_ok=True)
    
    # Save to parquet or CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_path is None:
        # Try parquet first, fall back to CSV
        try:
            import pyarrow
            output_path = os.path.join(ML_TRAINING_DIR, f'labeled_data_{timestamp}.parquet')
            labeled_df.to_parquet(output_path, index=False)
        except ImportError:
            print("   ⚠️ pyarrow not installed, saving as CSV instead")
            output_path = os.path.join(ML_TRAINING_DIR, f'labeled_data_{timestamp}.csv')
            labeled_df.to_csv(output_path, index=False)
    else:
        if output_path.endswith('.parquet'):
            try:
                labeled_df.to_parquet(output_path, index=False)
            except ImportError:
                output_path = output_path.replace('.parquet', '.csv')
                labeled_df.to_csv(output_path, index=False)
        else:
            labeled_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved labeled dataset: {output_path}")
    print(f"   Total samples: {len(labeled_df):,}")
    
    # Summary by prop type
    print("\n📊 Summary by prop type:")
    for prop_type in ['PTS', 'REB', 'AST', '3PM']:
        prop_df = labeled_df[labeled_df['Prop_Type_Normalized'] == prop_type]
        if len(prop_df) > 0:
            over_rate = (prop_df['outcome'] == 1).mean() * 100
            print(f"   {prop_type}: {len(prop_df):,} samples, OVER rate: {over_rate:.1f}%")
    
    print("="*60)
    
    return labeled_df


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare labeled dataset for ML training')
    parser.add_argument('--training-dir', type=str, default=None,
                        help='Directory containing training CSV files')
    parser.add_argument('--actuals-csv', type=str, default=None,
                        help='Path to PlayerStatistics.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for labeled dataset')
    
    args = parser.parse_args()
    
    # Run the pipeline
    prepare_labeled_dataset(
        output_path=args.output
    )
