#!/usr/bin/env python3
"""
CONSOLIDATED DATA ENGINEERING PIPELINE
======================================
Single entry point for all NBA prop data engineering.

This script handles:
1. Checking existing data (incremental updates)
2. Updating Kaggle data (if needed)
3. Fetching missing odds (if needed)
4. Building super datasets using existing pipeline
5. Combining and filtering to exact date ranges
6. Outputting clean, organized datasets

Date Ranges (Regular Season, 10+ games played):
- 2024-25: Nov 10, 2024 → April 13, 2025
- 2025-26: Nov 13, 2025 → Yesterday (dynamic)

Usage:
    python scripts/build_combined_dataset.py                    # Full pipeline
    python scripts/build_combined_dataset.py --force            # Force rebuild
    python scripts/build_combined_dataset.py --status           # Check existing data
    python scripts/build_combined_dataset.py --skip-kaggle      # Skip Kaggle update
    python scripts/build_combined_dataset.py --skip-odds        # Skip odds fetch
    python scripts/build_combined_dataset.py --incremental      # Incremental update (only yesterday)
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_yesterday() -> str:
    """Get yesterday's date as YYYY-MM-DD string."""
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


def get_today() -> str:
    """Get today's date as YYYY-MM-DD string."""
    return datetime.now().strftime('%Y-%m-%d')


def get_training_date_range(season: str) -> Tuple[str, str]:
    """
    Get the training date range for a season.
    
    Args:
        season: Season string like "2024-25"
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    from process.shared_config import TRAINING_DATE_RANGES
    
    start, end = TRAINING_DATE_RANGES.get(season, (None, None))
    
    if start is None:
        raise ValueError(f"Unknown season: {season}")
    
    # If end is None, use yesterday
    if end is None:
        end = get_yesterday()
    
    return start, end


# ============================================================================
# STATUS CHECKING
# ============================================================================

def get_existing_dataset_info(output_dir: str) -> Dict:
    """
    Check what data already exists in the output directory.
    
    Returns:
        Dict with info about existing datasets
    """
    from process.shared_config import DATASETS_DIR
    
    if output_dir is None:
        output_dir = os.path.join(DATASETS_DIR, 'ml_training')
    
    info = {
        'output_dir': output_dir,
        'seasons': {},
        'combined': None
    }
    
    if not os.path.exists(output_dir):
        return info
    
    # Check for season-specific files
    for season in ['2024-25', '2025-26']:
        season_file = os.path.join(output_dir, f'season_{season.replace("-", "_")}_regular.parquet')
        if os.path.exists(season_file):
            df = pd.read_parquet(season_file)
            df['game_date'] = pd.to_datetime(df['game_date'])
            info['seasons'][season] = {
                'path': season_file,
                'records': len(df),
                'date_min': df['game_date'].min().strftime('%Y-%m-%d'),
                'date_max': df['game_date'].max().strftime('%Y-%m-%d'),
                'modified': datetime.fromtimestamp(os.path.getmtime(season_file)).strftime('%Y-%m-%d %H:%M')
            }
    
    # Check for combined file
    combined_files = [f for f in os.listdir(output_dir) if f.startswith('combined_') and f.endswith('.parquet')]
    if combined_files:
        # Get most recent
        combined_files.sort(reverse=True)
        combined_path = os.path.join(output_dir, combined_files[0])
        df = pd.read_parquet(combined_path)
        df['game_date'] = pd.to_datetime(df['game_date'])
        info['combined'] = {
            'path': combined_path,
            'records': len(df),
            'date_min': df['game_date'].min().strftime('%Y-%m-%d'),
            'date_max': df['game_date'].max().strftime('%Y-%m-%d'),
            'modified': datetime.fromtimestamp(os.path.getmtime(combined_path)).strftime('%Y-%m-%d %H:%M')
        }
    
    return info


def print_status(output_dir: str = None):
    """Print status of existing data."""
    info = get_existing_dataset_info(output_dir)
    
    print("\n" + "="*70)
    print("DATA PIPELINE STATUS")
    print("="*70)
    print(f"Output directory: {info['output_dir']}")
    print(f"Today: {get_today()}")
    print(f"Yesterday: {get_yesterday()}")
    
    print("\n--- Season Datasets ---")
    for season, data in info['seasons'].items():
        print(f"\n  {season}:")
        print(f"    Path: {data['path']}")
        print(f"    Records: {data['records']:,}")
        print(f"    Date range: {data['date_min']} → {data['date_max']}")
        print(f"    Modified: {data['modified']}")
    
    if not info['seasons']:
        print("  No season datasets found.")
    
    print("\n--- Combined Dataset ---")
    if info['combined']:
        print(f"  Path: {info['combined']['path']}")
        print(f"  Records: {info['combined']['records']:,}")
        print(f"  Date range: {info['combined']['date_min']} → {info['combined']['date_max']}")
        print(f"  Modified: {info['combined']['modified']}")
    else:
        print("  No combined dataset found.")
    
    # Check expected date ranges
    print("\n--- Expected Training Ranges ---")
    for season in ['2024-25', '2025-26']:
        try:
            start, end = get_training_date_range(season)
            print(f"  {season}: {start} → {end}")
        except ValueError:
            print(f"  {season}: Not configured")
    
    print("="*70)


def get_existing_odds_dates(season: str) -> set:
    """Get set of dates that already have odds data."""
    from process.shared_config import DATASETS_DIR
    
    odds_path = os.path.join(DATASETS_DIR, 'odds', f'historical_odds_{season.replace("-", "_")}.parquet')
    
    if not os.path.exists(odds_path):
        return set()
    
    df = pd.read_parquet(odds_path)
    return set(df['game_date'].unique())


def get_existing_training_dates(training_data_path: str) -> set:
    """
    Get set of dates that already exist in training_data.parquet.
    
    Args:
        training_data_path: Path to training_data.parquet
        
    Returns:
        Set of date strings (YYYY-MM-DD)
    """
    if not os.path.exists(training_data_path):
        return set()
    
    try:
        df = pd.read_parquet(training_data_path)
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            dates = set(df['game_date'].dt.strftime('%Y-%m-%d').unique())
            return dates
        return set()
    except Exception as e:
        print(f"   ⚠️  Error reading existing training data: {e}")
        return set()


# ============================================================================
# DATA UPDATES
# ============================================================================

def update_kaggle_data() -> bool:
    """
    Update Kaggle data using existing update script.
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("STEP 1: Updating Kaggle Data")
    print("="*70)
    
    try:
        # Save original sys.argv and replace with just the script name
        # This prevents passing --incremental or other flags to update_kaggle_data
        import sys
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]  # Only pass script name, no arguments
        
        from scripts.update_kaggle_data import main as update_kaggle
        result = update_kaggle()
        
        # Restore original argv
        sys.argv = original_argv
        
        return result == 0
    except Exception as e:
        # Restore original argv in case of error
        if 'original_argv' in locals():
            sys.argv = original_argv
        print(f"   Error updating Kaggle data: {e}")
        return False


def fetch_missing_odds(season: str, start_date: str, end_date: str, single_date: str = None) -> bool:
    """
    Fetch any missing odds for a date range or a single date.
    
    Uses the existing fetch_historical_odds.py which already supports resuming.
    
    Args:
        season: Season string
        start_date: Start date (ignored if single_date is provided)
        end_date: End date (ignored if single_date is provided)
        single_date: If provided, only fetch odds for this single date
        
    Returns:
        True if successful, False otherwise
    """
    if single_date:
        print(f"\n   Fetching odds for {season}: {single_date} (single date)")
        date_range = (single_date, single_date)
    else:
        print(f"\n   Fetching odds for {season}: {start_date} → {end_date}")
        date_range = (start_date, end_date)
    
    try:
        from scripts.fetch_historical_odds import fetch_season_odds
        
        # If single_date is provided, use start/end overrides
        if single_date:
            df = fetch_season_odds(
                season, 
                force_refresh=False,
                start_date_override=single_date,
                end_date_override=single_date
            )
        else:
            # The existing function already handles resume/incremental
            df = fetch_season_odds(season, force_refresh=False)
        
        if df is not None and len(df) > 0:
            print(f"   ✅ Odds data ready: {len(df):,} records")
            return True
        else:
            print(f"   ⚠️ No odds fetched (may already be complete or no data)")
            return True
            
    except Exception as e:
        print(f"   Error fetching odds: {e}")
        return False


# ============================================================================
# DATASET BUILDING
# ============================================================================

def build_season_dataset(
    spark,
    season: str,
    output_dir: str,
    force: bool = False,
    date_filter: Tuple[str, str] = None
) -> Optional[pd.DataFrame]:
    """
    Build super dataset for a single season using existing pipeline.
    
    Args:
        spark: SparkSession
        season: Season string like "2024-25"
        output_dir: Output directory
        force: Force rebuild even if exists
        date_filter: Optional (start_date, end_date) tuple to filter results
        
    Returns:
        DataFrame with season data, or None if failed
    """
    from process.nba_data_builder import build_super_dataset
    
    output_path = os.path.join(output_dir, f'season_{season.replace("-", "_")}_regular.parquet')
    
    # Check if we need to build
    if os.path.exists(output_path) and not force:
        print(f"   Loading cached: {output_path}")
        df = pd.read_parquet(output_path)
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Apply date filter if provided
        if date_filter:
            start_date, end_date = date_filter
            mask = (df['game_date'] >= start_date) & (df['game_date'] <= end_date)
            df = df[mask]
            print(f"   Filtered to date range: {start_date} → {end_date}: {len(df):,} records")
        
        return df
    
    print(f"   Building from scratch...")
    
    try:
        spark_df = build_super_dataset(
            spark,
            season=season,
            use_local_odds=True,
            output_path=output_path
        )
        
        if spark_df is None:
            print(f"   ⚠️ No data returned for {season}")
            return None
        
        df = spark_df.toPandas()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Apply date filter if provided
        if date_filter:
            start_date, end_date = date_filter
            mask = (df['game_date'] >= start_date) & (df['game_date'] <= end_date)
            df = df[mask]
            print(f"   Filtered to date range: {start_date} → {end_date}: {len(df):,} records")
        
        return df
        
    except Exception as e:
        print(f"   Error building {season}: {e}")
        return None


def filter_to_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter DataFrame to exact date range."""
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    mask = (df['game_date'] >= start_date) & (df['game_date'] <= end_date)
    return df[mask]


def organize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organize DataFrame columns into a clean, consistent schema.
    
    Column order:
    1. Identifiers
    2. Prop info
    3. Rolling stats
    4. Context
    5. Team/Opponent
    6. Advanced
    """
    # Define column groups
    id_cols = ['personId', 'gameId', 'FullName', 'TeamAbbrev', 'OpponentAbbrev', 'game_date']
    prop_cols = ['Prop_Type_Normalized', 'Line', 'actual_value', 'outcome']
    
    # Rolling features (dynamic based on what exists)
    rolling_patterns = ['L5_Avg_', 'L10_Avg_', 'Season_Avg_', 'L5_StdDev_', 'L10_StdDev_', 'Form_']
    rolling_cols = [c for c in df.columns if any(c.startswith(p) for p in rolling_patterns)]
    
    # Context columns
    context_cols = ['is_home', 'home', 'is_b2b', 'rest_days', 'Player_Tier', 'Reliability_Tag']
    
    # Team/Opponent columns
    team_patterns = ['Team_', 'Opp_', 'L5_Team_', 'L10_Team_']
    team_cols = [c for c in df.columns if any(c.startswith(p) for p in team_patterns)]
    
    # Advanced player stats
    advanced_patterns = ['Player_TS_', 'Player_eFG_', 'Player_PTS_per', 'Player_REB_per', 
                         'Player_AST_per', 'Player_3PM_per', 'Player_Poss']
    advanced_cols = [c for c in df.columns if any(p in c for p in advanced_patterns)]
    
    # Build ordered list
    ordered = []
    for group in [id_cols, prop_cols, rolling_cols, context_cols, team_cols, advanced_cols]:
        for col in group:
            if col in df.columns and col not in ordered:
                ordered.append(col)
    
    # Add remaining columns
    for col in df.columns:
        if col not in ordered:
            ordered.append(col)
    
    return df[ordered]


def combine_seasons(
    datasets: Dict[str, pd.DataFrame],
    min_games_current_season: int = 10,
    is_incremental: bool = False
) -> pd.DataFrame:
    """
    Combine multiple season datasets with proper filtering.
    
    Args:
        datasets: Dict mapping season -> DataFrame
        min_games_current_season: Min games required in most recent season
        is_incremental: If True, skip min_games filter (for single-day updates)
        
    Returns:
        Combined DataFrame
    """
    if not datasets:
        raise ValueError("No datasets to combine")
    
    # Sort seasons chronologically
    seasons = sorted(datasets.keys())
    
    combined_parts = []
    
    for i, season in enumerate(seasons):
        df = datasets[season]
        is_current_season = (i == len(seasons) - 1)  # Last season is "current"
        
        # Skip min_games filter in incremental mode (single day won't have 10+ games per player)
        if is_current_season and min_games_current_season > 0 and not is_incremental:
            # Filter to players with enough games
            player_game_counts = df.groupby('personId')['gameId'].nunique()
            qualified_players = player_game_counts[player_game_counts >= min_games_current_season].index
            df = df[df['personId'].isin(qualified_players)]
            print(f"   {season}: Filtered to {len(qualified_players)} players with {min_games_current_season}+ games")
        elif is_incremental:
            print(f"   {season}: Incremental mode - skipping min_games filter (single day update)")
        
        combined_parts.append(df)
    
    # Combine
    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined.sort_values('game_date')
    
    return combined


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    target_date: str = None,
    force_rebuild: bool = False,
    skip_kaggle: bool = False,
    skip_odds: bool = False,
    output_dir: str = None,
    min_games_current_season: int = 10,
    incremental: bool = False
) -> str:
    """
    Main entry point for data engineering pipeline.
    
    Args:
        target_date: Date predictions are for (default: today)
        force_rebuild: Force full rebuild even if data exists
        skip_kaggle: Skip Kaggle data update
        skip_odds: Skip odds fetching
        output_dir: Output directory
        min_games_current_season: Min games required in current season
        incremental: If True, incrementally update training_data.parquet (only process yesterday)
        
    Returns:
        Path to the combined dataset
    """
    from process.shared_config import get_spark_session, stop_spark_session, DATASETS_DIR
    
    # Setup
    if target_date is None:
        target_date = get_today()
    
    if output_dir is None:
        output_dir = os.path.join(DATASETS_DIR, 'ml_training')
    
    os.makedirs(output_dir, exist_ok=True)
    
    yesterday = get_yesterday()
    
    # Check for incremental mode
    training_data_path = os.path.join(output_dir, 'training_data.parquet')
    existing_dates = set()
    
    if incremental:
        existing_dates = get_existing_training_dates(training_data_path)
        if existing_dates:
            latest_date = max(existing_dates)
            print(f"\n📊 Incremental mode: Found existing training_data.parquet")
            print(f"   Latest date in existing data: {latest_date}")
            print(f"   Will process: {yesterday}")
            if yesterday in existing_dates:
                print(f"   ⚠️  Yesterday ({yesterday}) already exists in training data")
                print(f"   Will rebuild data for {yesterday} anyway")
        else:
            print(f"\n📊 Incremental mode: No existing training_data.parquet found")
            print(f"   Will create new training_data.parquet")
    
    print("\n" + "="*70)
    print("CONSOLIDATED DATA ENGINEERING PIPELINE")
    print("="*70)
    print(f"Target date: {target_date}")
    print(f"Yesterday: {yesterday}")
    print(f"Incremental mode: {incremental}")
    print(f"Force rebuild: {force_rebuild}")
    print(f"Skip Kaggle: {skip_kaggle}")
    print(f"Skip Odds: {skip_odds}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Step 1: Update Kaggle data
    if not skip_kaggle:
        update_kaggle_data()
    else:
        print("\n[SKIPPED] Kaggle data update")
    
    # Step 2: Fetch odds for each season
    if not skip_odds:
        print("\n" + "="*70)
        print("STEP 2: Fetching Historical Odds")
        print("="*70)
        
        if incremental:
            # In incremental mode, only fetch odds for yesterday
            # Determine which season yesterday belongs to
            yesterday_dt = datetime.strptime(yesterday, '%Y-%m-%d')
            if yesterday_dt < datetime(2025, 10, 1):
                season = '2024-25'
            else:
                season = '2025-26'
            
            print(f"   Incremental mode: Fetching odds for {yesterday} (season: {season})")
            fetch_missing_odds(season, None, None, single_date=yesterday)
        else:
            for season in ['2024-25', '2025-26']:
                try:
                    start, end = get_training_date_range(season)
                    fetch_missing_odds(season, start, end)
                except ValueError as e:
                    print(f"   Skipping {season}: {e}")
    else:
        print("\n[SKIPPED] Odds fetching")
    
    # Step 3: Build season datasets
    print("\n" + "="*70)
    print("STEP 3: Building Season Datasets")
    print("="*70)
    
    spark = get_spark_session()
    datasets = {}
    
    try:
        if incremental:
            # In incremental mode, only build for yesterday's date
            # Determine which season yesterday belongs to
            yesterday_dt = datetime.strptime(yesterday, '%Y-%m-%d')
            if yesterday_dt < datetime(2025, 10, 1):
                season = '2024-25'
            else:
                season = '2025-26'
            
            print(f"\n--- {season} (incremental: {yesterday}) ---")
            
            try:
                # Check if cached season file has yesterday's data
                season_cache_path = os.path.join(output_dir, f'season_{season.replace("-", "_")}_regular.parquet')
                needs_rebuild = force_rebuild
                
                if not needs_rebuild and os.path.exists(season_cache_path):
                    # Check if yesterday exists in cached file
                    cached_df = pd.read_parquet(season_cache_path)
                    cached_df['game_date'] = pd.to_datetime(cached_df['game_date'])
                    yesterday_in_cache = (cached_df['game_date'] == yesterday_dt).any()
                    
                    if not yesterday_in_cache:
                        print(f"   Yesterday ({yesterday}) not in cached file, will rebuild season dataset")
                        needs_rebuild = True
                    else:
                        print(f"   Yesterday ({yesterday}) found in cached file")
                
                # Build dataset but filter to only yesterday
                df = build_season_dataset(
                    spark, 
                    season, 
                    output_dir, 
                    force=needs_rebuild,
                    date_filter=(yesterday, yesterday)
                )
                
                if df is not None and len(df) > 0:
                    print(f"   After filtering: {len(df):,} records")
                    print(f"   Actual range: {df['game_date'].min().strftime('%Y-%m-%d')} → {df['game_date'].max().strftime('%Y-%m-%d')}")
                    
                    if len(df) > 0:
                        datasets[season] = df
                else:
                    print(f"   ⚠️  No data found for {yesterday}")
                    print(f"   This might mean:")
                    print(f"      - No games were played on {yesterday}")
                    print(f"      - Kaggle data hasn't been updated yet")
                    print(f"      - Games haven't been processed yet")
                    
            except ValueError as e:
                print(f"   Skipping {season}: {e}")
        else:
            for season in ['2024-25', '2025-26']:
                print(f"\n--- {season} ---")
                
                try:
                    start, end = get_training_date_range(season)
                    print(f"   Training range: {start} → {end}")
                    
                    df = build_season_dataset(spark, season, output_dir, force=force_rebuild)
                    
                    if df is not None and len(df) > 0:
                        # Filter to exact training range
                        df = filter_to_date_range(df, start, end)
                        print(f"   After filtering: {len(df):,} records")
                        print(f"   Actual range: {df['game_date'].min().strftime('%Y-%m-%d')} → {df['game_date'].max().strftime('%Y-%m-%d')}")
                        
                        if len(df) > 0:
                            datasets[season] = df
                        
                except ValueError as e:
                    print(f"   Skipping {season}: {e}")
        
        # Step 4: Combine datasets
        print("\n" + "="*70)
        print("STEP 4: Combining Datasets")
        print("="*70)
        
        if not datasets:
            if incremental:
                print("   ⚠️  No new data to add. Training data is up to date.")
                return training_data_path
            raise ValueError("No datasets were built!")
        
        combined = combine_seasons(datasets, min_games_current_season, is_incremental=incremental)
        
        # Step 5: Organize schema
        print("\n" + "="*70)
        print("STEP 5: Organizing Schema")
        print("="*70)
        
        combined = organize_schema(combined)
        print(f"   Columns: {len(combined.columns)}")
        
        # Step 6: Save (incremental mode: concatenate with existing, otherwise save new)
        print("\n" + "="*70)
        print("STEP 6: Saving Combined Dataset")
        print("="*70)
        
        if incremental and os.path.exists(training_data_path):
            print(f"   Incremental mode: Loading existing training_data.parquet...")
            existing_df = pd.read_parquet(training_data_path)
            existing_df['game_date'] = pd.to_datetime(existing_df['game_date'])
            
            # Remove yesterday's data from existing (in case we're re-processing)
            yesterday_dt = pd.to_datetime(yesterday)
            existing_df = existing_df[existing_df['game_date'] != yesterday_dt]
            print(f"   Removed existing data for {yesterday}: {len(existing_df):,} records remain")
            
            # Concatenate new data with existing
            print(f"   Concatenating: {len(existing_df):,} existing + {len(combined):,} new = ", end="")
            combined = pd.concat([existing_df, combined], ignore_index=True)
            combined = combined.sort_values('game_date')
            print(f"{len(combined):,} total")
            
            output_path = training_data_path
        else:
            # Full rebuild mode: save with date suffix or as training_data.parquet
            if incremental:
                output_filename = 'training_data.parquet'
            else:
                output_filename = f'training_data_{target_date.replace("-", "")}.parquet'
            output_path = os.path.join(output_dir, output_filename)
        
        combined.to_parquet(output_path, index=False)
        print(f"   ✅ Saved to: {output_path}")
        
        # Print summary
        print_dataset_summary(combined, output_path)
        
        return output_path
        
    finally:
        stop_spark_session()


def print_dataset_summary(df: pd.DataFrame, output_path: str):
    """Print detailed summary of the dataset."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\n   Path: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    print(f"\n   Date range: {df['game_date'].min().strftime('%Y-%m-%d')} → {df['game_date'].max().strftime('%Y-%m-%d')}")
    print(f"   Unique players: {df['personId'].nunique():,}")
    print(f"   Unique games: {df['gameId'].nunique():,}")
    
    # Season breakdown
    df['season'] = df['game_date'].apply(
        lambda d: '2024-25' if d < datetime(2025, 10, 1) else '2025-26'
    )
    print(f"\n   By season:")
    for season in ['2024-25', '2025-26']:
        season_df = df[df['season'] == season]
        if len(season_df) > 0:
            print(f"      {season}: {len(season_df):,} records")
            print(f"         Range: {season_df['game_date'].min().strftime('%Y-%m-%d')} → {season_df['game_date'].max().strftime('%Y-%m-%d')}")
    
    # Prop types
    if 'Prop_Type_Normalized' in df.columns:
        print(f"\n   By prop type:")
        for prop in ['PTS', 'REB', 'AST', '3PM']:
            count = len(df[df['Prop_Type_Normalized'] == prop])
            print(f"      {prop}: {count:,}")
    
    # Outcome distribution
    if 'outcome' in df.columns:
        over_count = (df['outcome'] == 1).sum()
        under_count = (df['outcome'] == 0).sum()
        total = over_count + under_count
        print(f"\n   Outcome distribution:")
        print(f"      OVER:  {over_count:,} ({100*over_count/total:.1f}%)")
        print(f"      UNDER: {under_count:,} ({100*under_count/total:.1f}%)")
    
    print("\n" + "="*70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Consolidated NBA Prop Data Engineering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_combined_dataset.py                    # Full pipeline
  python scripts/build_combined_dataset.py --force            # Force rebuild
  python scripts/build_combined_dataset.py --status           # Check existing data
  python scripts/build_combined_dataset.py --skip-kaggle      # Skip Kaggle update
  python scripts/build_combined_dataset.py --skip-odds        # Skip odds fetch
  python scripts/build_combined_dataset.py --incremental      # Incremental update (only yesterday)
        """
    )
    
    parser.add_argument('--target', type=str, default=None,
                        help='Target date for predictions (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--force', action='store_true',
                        help='Force complete rebuild even if data exists')
    parser.add_argument('--status', action='store_true',
                        help='Just print status of existing data, do not build')
    parser.add_argument('--skip-kaggle', action='store_true',
                        help='Skip Kaggle data update')
    parser.add_argument('--skip-odds', action='store_true',
                        help='Skip odds fetching')
    parser.add_argument('--min-games', type=int, default=10,
                        help='Min games required in current season (default: 10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for datasets')
    parser.add_argument('--incremental', action='store_true',
                        help='Incremental update mode: only process yesterday and append to training_data.parquet')
    
    args = parser.parse_args()
    
    # Status mode
    if args.status:
        print_status(args.output_dir)
        return 0
    
    # Full pipeline
    try:
        output_path = run_pipeline(
            target_date=args.target,
            force_rebuild=args.force,
            skip_kaggle=args.skip_kaggle,
            skip_odds=args.skip_odds,
            output_dir=args.output_dir,
            min_games_current_season=args.min_games,
            incremental=args.incremental
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print(f"   Output: {output_path}")
        print("\nNext steps:")
        print(f"   python scripts/train_with_tuning.py --data {output_path}")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
