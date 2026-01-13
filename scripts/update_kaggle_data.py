#!/usr/bin/env python3
"""
KAGGLE DATA UPDATE SCRIPT
=========================
Downloads the latest NBA data from Kaggle and overwrites local files.

Usage:
    python scripts/update_kaggle_data.py           # Update all data
    python scripts/update_kaggle_data.py --check   # Check freshness only

Dataset: eoinamoore/historical-nba-data-and-player-box-scores
"""
import os
import sys
import shutil
from datetime import datetime

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

from process.shared_config import KAGGLE_API_TOKEN, KAGGLE_DATA_DIR

# Kaggle dataset info
KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"

# Files we need from the dataset
REQUIRED_FILES = [
    "PlayerStatistics.csv",
    "TeamStatistics.csv", 
    "Games.csv",
    "Players.csv",
    "LeagueSchedule24_25.csv",
    "LeagueSchedule25_26.csv",
]


def setup_kaggle_auth():
    """
    Sets up Kaggle authentication using the API token.
    """
    # Set environment variable for kagglehub
    os.environ['KAGGLE_API_TOKEN'] = KAGGLE_API_TOKEN
    print(f"🔑 Kaggle API token configured")


def download_kaggle_dataset():
    """
    Downloads the latest NBA dataset from Kaggle using kagglehub.
    
    Returns:
        str: Path to downloaded dataset directory
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        import kagglehub
    
    print(f"\n📥 Downloading latest dataset from Kaggle...")
    print(f"   Dataset: {KAGGLE_DATASET}")
    
    # Download latest version
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    
    print(f"   ✅ Downloaded to: {path}")
    return path


def update_local_files(download_path: str):
    """
    Copies downloaded files to kaggle_data directory, overwriting existing.
    
    Args:
        download_path: Path where Kaggle downloaded the files
    """
    print(f"\n📁 Updating local files in: {KAGGLE_DATA_DIR}")
    
    # Ensure kaggle_data directory exists
    os.makedirs(KAGGLE_DATA_DIR, exist_ok=True)
    
    updated_files = []
    missing_files = []
    
    for filename in REQUIRED_FILES:
        src_path = os.path.join(download_path, filename)
        dst_path = os.path.join(KAGGLE_DATA_DIR, filename)
        
        if os.path.exists(src_path):
            # Get file sizes for comparison
            old_size = os.path.getsize(dst_path) if os.path.exists(dst_path) else 0
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            new_size = os.path.getsize(dst_path)
            
            size_change = new_size - old_size
            change_str = f"+{size_change:,}" if size_change >= 0 else f"{size_change:,}"
            
            print(f"   ✅ {filename}: {new_size:,} bytes ({change_str})")
            updated_files.append(filename)
        else:
            print(f"   ⚠️  {filename}: Not found in download")
            missing_files.append(filename)
    
    return updated_files, missing_files


def validate_data_freshness():
    """
    Validates that the data includes recent games.
    
    Returns:
        dict: Validation results including latest game date
    """
    import pandas as pd
    
    print(f"\n🔍 Validating data freshness...")
    
    player_stats_path = os.path.join(KAGGLE_DATA_DIR, "PlayerStatistics.csv")
    
    if not os.path.exists(player_stats_path):
        print("   ❌ PlayerStatistics.csv not found")
        return {"valid": False, "error": "File not found"}
    
    try:
        df = pd.read_csv(player_stats_path, low_memory=False)
        
        # Parse dates - handle mixed timezone formats
        df['game_date'] = pd.to_datetime(df['gameDateTimeEst'], utc=True, errors='coerce').dt.date
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['game_date'])
        
        latest_date = df['game_date'].max()
        earliest_date = df['game_date'].min()
        total_games = df['gameId'].nunique()
        total_players = df['personId'].nunique()
        
        # Check for 2025-26 season data
        cutoff_date = datetime(2025, 10, 1).date()
        season_25_26 = df[df['game_date'] >= cutoff_date]
        games_25_26 = season_25_26['gameId'].nunique() if len(season_25_26) > 0 else 0
        
        print(f"   📅 Date range: {earliest_date} → {latest_date}")
        print(f"   🏀 Total games: {total_games:,}")
        print(f"   👥 Total players: {total_players:,}")
        print(f"   📊 2025-26 season games: {games_25_26:,}")
        
        # Calculate days since last game
        today = datetime.now().date()
        days_old = (today - latest_date).days
        
        if days_old <= 1:
            print(f"   ✅ Data is up to date (latest: {latest_date})")
        elif days_old <= 3:
            print(f"   ⚠️  Data is {days_old} days old (latest: {latest_date})")
        else:
            print(f"   ❌ Data is {days_old} days old (latest: {latest_date})")
        
        return {
            "valid": True,
            "latest_date": str(latest_date),
            "earliest_date": str(earliest_date),
            "total_games": total_games,
            "total_players": total_players,
            "games_25_26": games_25_26,
            "days_old": days_old
        }
        
    except Exception as e:
        print(f"   ❌ Error validating data: {e}")
        return {"valid": False, "error": str(e)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update NBA data from Kaggle')
    parser.add_argument('--check', action='store_true',
                        help='Only check data freshness, do not download')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🏀 KAGGLE NBA DATA UPDATE")
    print(f"   Started: {datetime.now()}")
    print("="*60)
    
    if args.check:
        # Just validate existing data
        result = validate_data_freshness()
        return 0 if result.get("valid") else 1
    
    # Full update
    setup_kaggle_auth()
    
    try:
        download_path = download_kaggle_dataset()
        updated, missing = update_local_files(download_path)
        result = validate_data_freshness()
        
        print("\n" + "="*60)
        print("✅ UPDATE COMPLETE")
        print(f"   Files updated: {len(updated)}")
        print(f"   Files missing: {len(missing)}")
        if result.get("valid"):
            print(f"   Latest data: {result.get('latest_date')}")
            print(f"   2025-26 games: {result.get('games_25_26', 0)}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

