#!/usr/bin/env python3
"""
HISTORICAL ODDS FETCHER
=======================
Fetches ALL historical player prop odds for a season and stores locally.

Usage:
    python scripts/fetch_historical_odds.py                    # Fetch 2024-25 season
    python scripts/fetch_historical_odds.py --season 2023-24   # Specific season
    python scripts/fetch_historical_odds.py --force            # Force refresh

This creates: datasets/odds/historical_odds_SEASON.parquet

The Odds API historical endpoint works as:
1. Get event IDs from /historical/sports/{sport}/events
2. For each event, fetch odds from /historical/sports/{sport}/events/{eventId}/odds
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))

from process.shared_config import ODDS_API_KEY, ODDS_API_DELAY

# Paths
ODDS_DIR = os.path.join(PROJECT_ROOT, "datasets", "odds")
os.makedirs(ODDS_DIR, exist_ok=True)

# Season date ranges (training ranges - 10+ games played, regular season only)
# Import from shared_config for consistency
try:
    from process.shared_config import TRAINING_DATE_RANGES
    SEASON_RANGES = {
        "2024-25": (TRAINING_DATE_RANGES["2024-25"][0], TRAINING_DATE_RANGES["2024-25"][1] or "2025-04-13"),
        "2025-26": (TRAINING_DATE_RANGES["2025-26"][0], TRAINING_DATE_RANGES["2025-26"][1] or "2026-04-12"),
    }
except ImportError:
    # Fallback defaults
    SEASON_RANGES = {
        "2024-25": ("2024-11-10", "2025-04-13"),
        "2025-26": ("2025-11-13", "2026-04-12"),
    }

# Odds API endpoints
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def get_historical_odds_path(season: str) -> str:
    """Returns path to historical odds file for a season."""
    return os.path.join(ODDS_DIR, f"historical_odds_{season.replace('-', '_')}.parquet")


def load_local_odds(season: str) -> Optional[pd.DataFrame]:
    """Loads historical odds from local storage if available."""
    path = get_historical_odds_path(season)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"📂 Loaded {len(df)} odds records from {path}")
        return df
    return None


def get_events_for_date(date: str, sport: str = "basketball_nba") -> List[Dict]:
    """Gets list of events that START on a specific date (US time)."""
    url = f"{ODDS_API_BASE}/historical/sports/{sport}/events"
    
    # Query at end of day to catch all games
    params = {
        "apiKey": ODDS_API_KEY,
        "date": f"{date}T23:59:59Z",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            all_events = data.get("data", [])
            
            # Filter events: include if commence_time is on target date OR next day (for late night games)
            # NBA games in US happen between ~6pm-10pm local time
            # A 10pm EST game = 3am UTC next day
            # So we accept: target_date or target_date+1 (in UTC)
            target_dt = datetime.strptime(date, "%Y-%m-%d")
            next_day = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            
            filtered = []
            for event in all_events:
                commence = event.get("commence_time", "")
                if commence:
                    commence_date = commence[:10]
                    # Include if the game's UTC date is target OR target+1
                    # (covers both afternoon and late-night US games)
                    if commence_date == date or commence_date == next_day:
                        filtered.append(event)
            
            return filtered
        else:
            print(f"   ⚠️ Events API error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"   ⚠️ Events request failed: {e}")
        return []


def get_event_odds(event_id: str, date: str, sport: str = "basketball_nba") -> List[Dict]:
    """Gets player prop odds for a specific event."""
    url = f"{ODDS_API_BASE}/historical/sports/{sport}/events/{event_id}/odds"
    
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists,player_threes",
        "oddsFormat": "american",
        "date": f"{date}T12:00:00Z",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get("bookmakers", [])
        elif response.status_code == 422:
            # No odds available for this event
            return []
        else:
            return []
            
    except Exception as e:
        return []


def parse_event_odds(bookmakers: List[Dict], event: Dict, game_date: str, max_props_per_game: int = 15, preferred_bookmaker: str = "draftkings") -> List[Dict]:
    """
    Parses bookmaker odds into flat records.
    
    Uses a single source of truth: DraftKings bookmaker (or specified preferred_bookmaker).
    Only keeps "Over" side.
    Distributes evenly across prop types, then limits to max_props_per_game.
    
    Args:
        bookmakers: List of bookmaker dictionaries from API
        event: Event dictionary
        game_date: Game date string
        max_props_per_game: Maximum props to keep per game
        preferred_bookmaker: Bookmaker key to use as single source of truth (default: "draftkings")
    """
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    commence_time = event.get("commence_time", "")
    
    # Filter to preferred bookmaker (DraftKings)
    preferred_bookmaker_obj = None
    for bookmaker in bookmakers:
        bookmaker_key = bookmaker.get("key", "").lower()
        if bookmaker_key == preferred_bookmaker.lower():
            preferred_bookmaker_obj = bookmaker
            break
    
    if not preferred_bookmaker_obj:
        # If DraftKings not found, return empty list
        # Note: This is expected if DraftKings doesn't have odds for this event
        return []
    
    # Group by prop type: {prop_type: {(player) -> record}}
    by_type = {"PTS": {}, "REB": {}, "AST": {}, "3PM": {}}
    
    # Only process the preferred bookmaker
    bookmaker = preferred_bookmaker_obj
    for market in bookmaker.get("markets", []):
        market_key = market.get("key", "")
        
        prop_type_map = {
            "player_points": "PTS",
            "player_rebounds": "REB",
            "player_assists": "AST",
            "player_threes": "3PM",
        }
        prop_type = prop_type_map.get(market_key)
        if not prop_type:
            continue
        
        for outcome in market.get("outcomes", []):
            side = outcome.get("name", "")
            
            # Only keep "Over" side
            if side != "Over":
                continue
            
            player_name = outcome.get("description", "")
            point = outcome.get("point")
            price = outcome.get("price")
            
            if not player_name or point is None:
                continue
            
            if player_name not in by_type[prop_type]:
                by_type[prop_type][player_name] = {
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "player_name": player_name,
                    "prop_type": prop_type,
                    "line": point,
                    "side": side,
                    "odds": price,
                }
    
    # Distribute evenly across prop types
    # ~4 props per type if max_props_per_game=15 and 4 types
    props_per_type = max(1, max_props_per_game // 4)
    
    records = []
    for prop_type in ["PTS", "REB", "AST", "3PM"]:
        type_records = list(by_type[prop_type].values())[:props_per_type]
        records.extend(type_records)
    
    # If we have room, add more from any type
    remaining = max_props_per_game - len(records)
    if remaining > 0:
        all_remaining = []
        for prop_type in ["PTS", "REB", "AST", "3PM"]:
            type_records = list(by_type[prop_type].values())[props_per_type:]
            all_remaining.extend(type_records)
        records.extend(all_remaining[:remaining])
    
    return records[:max_props_per_game]


def fetch_season_odds(
    season: str, 
    force_refresh: bool = False, 
    max_dates: int = None, 
    max_props_per_game: int = 15,
    start_date_override: str = None,
    end_date_override: str = None
) -> pd.DataFrame:
    """
    Fetches all player prop odds for a season.
    
    For each date:
    1. Get event IDs
    2. For each event, fetch player prop odds
    3. Parse and store
    
    Supports resuming - if interrupted, will continue from where it left off.
    
    Args:
        season: Season string like "2024-25"
        force_refresh: If True, refetch even if local data exists
        max_dates: Limit number of dates (for testing)
        max_props_per_game: Max props to keep per game (default 15)
        start_date_override: Override start date (YYYY-MM-DD)
        end_date_override: Override end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with all odds for the season
    """
    local_path = get_historical_odds_path(season)
    
    # Check for existing data to resume from
    existing_df = None
    existing_dates = set()
    if os.path.exists(local_path) and not force_refresh:
        existing_df = pd.read_parquet(local_path)
        existing_dates = set(existing_df['game_date'].unique())
        print(f"📂 Found existing data: {len(existing_df)} records from {len(existing_dates)} dates")
        print(f"   Will resume and skip already-fetched dates")
    
    # Get date range - use overrides if provided, otherwise use season defaults
    if start_date_override and end_date_override:
        start_date = start_date_override
        end_date = end_date_override
    elif season in SEASON_RANGES:
        start_date, end_date = SEASON_RANGES[season]
    else:
        print(f"❌ Unknown season: {season}")
        return pd.DataFrame()
    
    # Cap end date to yesterday
    today = datetime.now().strftime("%Y-%m-%d")
    if end_date > today:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n📡 Fetching historical odds for {season}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   📌 Using DraftKings as single source of truth for all props")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    if max_dates:
        dates = dates[:max_dates]
    
    # Filter out already-fetched dates
    dates_to_fetch = [d for d in dates if d not in existing_dates]
    
    print(f"   Total dates in range: {len(dates)}")
    print(f"   Already fetched: {len(existing_dates)}")
    print(f"   Dates to fetch: {len(dates_to_fetch)}")
    
    if not dates_to_fetch:
        print("   ✅ All dates already fetched!")
        return existing_df
    
    print(f"   API requests needed: ~{len(dates_to_fetch) * 8} (1 per date + ~7 games per date)")
    
    all_records = []
    total_events = 0
    events_with_props = 0
    
    try:
        for i, date in enumerate(dates_to_fetch):
            print(f"   [{i+1}/{len(dates_to_fetch)}] {date}... ", end="", flush=True)
            
            # Get events for this date
            events = get_events_for_date(date)
            
            if not events:
                print("no games")
                continue
            
            date_records = []
            date_props = 0
            
            for event in events:
                event_id = event.get("id", "")
                if not event_id:
                    continue
                
                total_events += 1
                
                # Get odds for this event
                bookmakers = get_event_odds(event_id, date)
                
                if bookmakers:
                    records = parse_event_odds(bookmakers, event, date, max_props_per_game)
                    if records:
                        date_records.extend(records)
                        date_props += len(records)
                        events_with_props += 1
                    # Note: If no records, DraftKings likely doesn't have props for this event
                
                # Rate limiting between event calls
                time.sleep(ODDS_API_DELAY / 2)
            
            if date_records:
                all_records.extend(date_records)
                print(f"{len(events)} games, {date_props} props")
            else:
                print(f"{len(events)} games, no prop odds")
            
            # Rate limiting between dates
            time.sleep(ODDS_API_DELAY)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving progress...")
    
    print(f"\n📊 Fetch Summary:")
    print(f"   Total events: {total_events}")
    print(f"   Events with props: {events_with_props}")
    print(f"   New prop records: {len(all_records)}")
    
    # Merge with existing data
    if all_records:
        new_df = pd.DataFrame(all_records)
        
        if existing_df is not None and len(existing_df) > 0:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"   Merged with existing: {len(existing_df)} + {len(new_df)} = {len(df)}")
        else:
            df = new_df
        
        df.to_parquet(local_path, index=False)
        print(f"\n✅ Saved {len(df)} records to {local_path}")
        print(f"   📁 Historical odds file updated (will reuse this data to avoid re-fetching)")
        
        print(f"\n📊 Data Summary:")
        print(f"   Dates with data: {df['game_date'].nunique()}")
        print(f"   Unique players: {df['player_name'].nunique()}")
        print(f"   Prop types: {df['prop_type'].value_counts().to_dict()}")
        
        return df
    else:
        print("\n⚠️ No odds data fetched")
        print("   Historical player prop odds may require specific API subscription tier")
        return pd.DataFrame()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch and store historical NBA player prop odds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_historical_odds.py --season 2024-25          # Fetch full season
  python scripts/fetch_historical_odds.py --season 2025-26          # Fetch current season
  python scripts/fetch_historical_odds.py --start 2024-12-01 --end 2024-12-31  # Custom range
  python scripts/fetch_historical_odds.py --list                     # List local files
        """
    )
    parser.add_argument("--season", default="2024-25", help="Season (e.g., 2024-25)")
    parser.add_argument("--start", "--start-date", type=str, default=None, 
                        help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", "--end-date", type=str, default=None,
                        help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Force refresh even if cached")
    parser.add_argument("--list", action="store_true", help="List available local odds files")
    parser.add_argument("--test", type=int, default=None, help="Only fetch N dates (for testing)")
    parser.add_argument("--max-props", type=int, default=15, help="Max props per game (default: 15)")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📁 Local Odds Files:")
        if os.path.exists(ODDS_DIR):
            files = [f for f in os.listdir(ODDS_DIR) if f.endswith(".parquet")]
            if files:
                for f in files:
                    path = os.path.join(ODDS_DIR, f)
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    df = pd.read_parquet(path)
                    print(f"   {f}: {len(df)} records, {size_mb:.1f} MB")
                    print(f"      Date range: {df['game_date'].min()} → {df['game_date'].max()}")
            else:
                print("   No odds files found.")
        else:
            print("   No odds directory exists yet.")
        return
    
    fetch_season_odds(
        args.season, 
        force_refresh=args.force, 
        max_dates=args.test, 
        max_props_per_game=args.max_props,
        start_date_override=args.start,
        end_date_override=args.end
    )


if __name__ == "__main__":
    main()
