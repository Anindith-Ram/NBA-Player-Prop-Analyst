#!/usr/bin/env python3
"""
LIVE PREDICTION ENGINE
======================
Generates NBA player prop predictions for today's games.

This is the main entry point for daily predictions, combining:
1. Updated Kaggle data
2. Live odds from The Odds API
3. ML model predictions
4. LLM (Gemini) analysis wrapper
5. High-confidence filtering (7+)

Usage:
    python scripts/predict_today.py                      # Predict today's games
    python scripts/predict_today.py --date 2025-12-13    # Specific date
    python scripts/predict_today.py --confidence 8       # Higher confidence threshold
    python scripts/predict_today.py --update-data        # Update Kaggle data first
"""
import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))

import pandas as pd
import numpy as np
import requests

from process.shared_config import (
    ODDS_API_KEY, PREDICTIONS_DIR, DATASETS_DIR,
    TEAM_FULL_NAME_TO_ABBREV, wait_for_api, ODDS_API_DELAY
)


# ============================================================================
# ODDS API FUNCTIONS
# ============================================================================

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

def fetch_todays_player_props(target_date: str = None, max_props_per_game: int = 15) -> pd.DataFrame:
    """
    Fetches today's player prop odds from The Odds API.
    
    Args:
        target_date: Date string (YYYY-MM-DD). Defaults to today.
        
    Returns:
        DataFrame with player props and odds
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Fetching player props for {target_date}...")
    
    # Get events for today
    events = get_nba_events(target_date)
    if not events:
        print("   ⚠️ No NBA games found for this date")
        return pd.DataFrame()
    
    print(f"   Found {len(events)} games")
    print(f"   📌 Using DraftKings as single source of truth")
    print(f"   📌 Limiting to max {max_props_per_game} props per game")
    
    # Fetch player props for each event
    all_props = []
    
    for event in events:
        event_id = event.get('id')
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        commence_time = event.get('commence_time')
        
        props = fetch_event_props(event_id, max_props_per_game=max_props_per_game)
        
        if props:
            print(f"   📥 {away_team} @ {home_team}... {len(props)} props")
        else:
            print(f"   📥 {away_team} @ {home_team}... no props (DraftKings not available)")
        
        for prop in props:
            prop['event_id'] = event_id
            prop['home_team'] = home_team
            prop['away_team'] = away_team
            prop['commence_time'] = commence_time
            prop['game_date'] = target_date
            all_props.append(prop)
        
        wait_for_api('odds_api', ODDS_API_DELAY)
    
    if not all_props:
        print("   ⚠️ No player props available")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_props)
    print(f"   ✅ Fetched {len(df)} player props (max {max_props_per_game} per game)")
    
    return df


def get_nba_events(target_date: str) -> List[Dict]:
    """Gets NBA events for a specific date (games that haven't started yet)."""
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
    
    params = {
        "apiKey": ODDS_API_KEY,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            events = response.json()
            
            # Parse target date
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            now = datetime.now()
            filtered = []
            
            print(f"   API returned {len(events)} upcoming events")
            
            for event in events:
                commence = event.get('commence_time', '')
                home = event.get('home_team', '')
                away = event.get('away_team', '')
                
                if commence:
                    # Parse ISO timestamp (UTC)
                    event_dt_utc = datetime.fromisoformat(commence.replace('Z', '+00:00'))
                    
                    # Convert to local time (EST is UTC-5)
                    event_dt_local = event_dt_utc.replace(tzinfo=None) - timedelta(hours=5)
                    event_date_local = event_dt_local.date()
                    
                    # Only include games on target date that haven't started yet
                    is_target_date = event_date_local == target_dt.date()
                    is_future = event_dt_utc.replace(tzinfo=None) > now
                    
                    if is_target_date:
                        if is_future:
                            print(f"      ✅ {away} @ {home} - {event_dt_local.strftime('%I:%M %p')} EST")
                            filtered.append(event)
                        else:
                            print(f"      ⏭️  {away} @ {home} - Already started/finished")
            
            return filtered
        else:
            print(f"   ⚠️ Events API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"   ⚠️ Error fetching events: {e}")
        return []


def fetch_event_props(event_id: str, max_props_per_game: int = 15, preferred_bookmaker: str = "draftkings") -> List[Dict]:
    """
    Fetches player props for a specific event.
    
    Uses DraftKings as single source of truth and limits to max_props_per_game.
    Distributes props evenly across prop types (PTS, REB, AST, 3PM).
    
    Args:
        event_id: Event ID from Odds API
        max_props_per_game: Maximum props to return per game (default: 15)
        preferred_bookmaker: Bookmaker to use (default: "draftkings")
        
    Returns:
        List of prop dictionaries (max max_props_per_game)
    """
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
    
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists,player_threes",
        "oddsFormat": "american",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Find DraftKings bookmaker
            draftkings_bookmaker = None
            for bookmaker in data.get('bookmakers', []):
                book_key = bookmaker.get('key', '').lower()
                if book_key == preferred_bookmaker.lower():
                    draftkings_bookmaker = bookmaker
                    break
            
            if not draftkings_bookmaker:
                # DraftKings not available for this event
                return []
            
            # Collect props by type
            props_by_type = {'PTS': [], 'REB': [], 'AST': [], '3PM': []}
            
            for market in draftkings_bookmaker.get('markets', []):
                market_key = market.get('key')
                
                # Map market to prop type
                prop_type_map = {
                    'player_points': 'PTS',
                    'player_rebounds': 'REB',
                    'player_assists': 'AST',
                    'player_threes': '3PM',
                }
                prop_type = prop_type_map.get(market_key)
                if not prop_type:
                    continue
                
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description')
                    line = outcome.get('point')
                    price = outcome.get('price')
                    side = outcome.get('name')  # Over or Under
                    
                    if player_name and line is not None and side == 'Over':
                        # Only keep OVER side
                        props_by_type[prop_type].append({
                            'player_name': player_name,
                            'prop_type': prop_type,
                            'line': line,
                            'over_price': price,
                            'bookmaker': preferred_bookmaker,
                        })
            
            # Deduplicate by (player_name, prop_type, line) within each type
            unique_props_by_type = {}
            for prop_type, props in props_by_type.items():
                seen = set()
                unique = []
                for prop in props:
                    key = (prop['player_name'], prop['prop_type'], prop['line'])
                    if key not in seen:
                        seen.add(key)
                        unique.append(prop)
                unique_props_by_type[prop_type] = unique
            
            # Distribute evenly across prop types, then limit to max_props_per_game
            props_per_type = max(1, max_props_per_game // 4)  # ~4 props per type for 15 total
            
            selected_props = []
            for prop_type in ['PTS', 'REB', 'AST', '3PM']:
                type_props = unique_props_by_type.get(prop_type, [])[:props_per_type]
                selected_props.extend(type_props)
            
            # If we have room, add more from any type
            remaining = max_props_per_game - len(selected_props)
            if remaining > 0:
                all_remaining = []
                for prop_type in ['PTS', 'REB', 'AST', '3PM']:
                    type_props = unique_props_by_type.get(prop_type, [])[props_per_type:]
                    all_remaining.extend(type_props)
                selected_props.extend(all_remaining[:remaining])
            
            return selected_props[:max_props_per_game]
        else:
            return []
    except Exception as e:
        print(f"   ⚠️ Error fetching props for event {event_id}: {e}")
        return []


# ============================================================================
# EDGE CALCULATION
# ============================================================================

def american_odds_to_implied_prob(american_odds: float) -> float:
    """
    Converts American odds to implied probability.
    
    Args:
        american_odds: American odds (e.g., -110, +150)
        
    Returns:
        Implied probability (0.0 to 1.0)
    """
    if pd.isna(american_odds) or american_odds == 0:
        return 0.5  # Default to 50% if odds are missing
    
    if american_odds > 0:
        # Positive odds: probability = 100 / (odds + 100)
        return 100 / (american_odds + 100)
    else:
        # Negative odds: probability = abs(odds) / (abs(odds) + 100)
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_recent_performance_probability(
    historical_df: pd.DataFrame,
    player_name: str,
    prop_type: str,
    line: float,
    lookback_games: int = 10
) -> float:
    """
    Calculates the probability that recent performance exceeds the line.
    
    This is based on how often the player has exceeded the line in recent games.
    
    Args:
        historical_df: Historical player data
        player_name: Player name
        prop_type: Prop type (PTS, REB, AST, 3PM)
        line: Current line to beat
        lookback_games: Number of recent games to consider (default 10)
        
    Returns:
        Probability (0.0 to 1.0) that recent performance > line
    """
    # Map prop type to actual value column
    prop_to_actual = {
        'PTS': 'actual_value',  # Will need to check if this exists or use points
        'REB': 'actual_value',
        'AST': 'actual_value',
        '3PM': 'actual_value'
    }
    
    # Filter to player and prop type
    player_df = historical_df[
        (historical_df['FullName'] == player_name) &
        (historical_df['Prop_Type_Normalized'] == prop_type)
    ].copy()
    
    if len(player_df) == 0:
        return 0.5  # Default to 50% if no historical data
    
    # Get recent games (last N games)
    player_df = player_df.sort_values('game_date', ascending=False).head(lookback_games)
    
    if len(player_df) == 0:
        return 0.5
    
    # Map prop type to actual stat column in historical data
    prop_to_stat = {
        'PTS': 'points',
        'REB': 'reboundsTotal',
        'AST': 'assists',
        '3PM': 'threePointersMade'
    }
    
    stat_col = prop_to_stat.get(prop_type)
    
    # Try to use actual stat values first
    if stat_col and stat_col in player_df.columns:
        # Count how many times actual stat > line
        over_count = (player_df[stat_col] > line).sum()
    elif 'actual_value' in player_df.columns:
        # Fallback to actual_value if available
        over_count = (player_df['actual_value'] > line).sum()
    elif 'outcome' in player_df.columns:
        # Use outcome as proxy (1 = OVER, 0 = UNDER)
        # But we need to check if the line matches - for now use outcome
        over_count = player_df['outcome'].sum()
    else:
        # Final fallback: use recent average vs line
        prop_to_avg = {
            'PTS': 'L10_Avg_PTS',
            'REB': 'L10_Avg_REB',
            'AST': 'L10_Avg_AST',
            '3PM': 'L10_Avg_3PM'
        }
        avg_col = prop_to_avg.get(prop_type)
        if avg_col and avg_col in player_df.columns:
            recent_avg = player_df[avg_col].iloc[0] if len(player_df) > 0 else None
            if recent_avg is not None and not pd.isna(recent_avg):
                return 1.0 if recent_avg > line else 0.0
        return 0.5
    
    # Calculate probability
    probability = over_count / len(player_df) if len(player_df) > 0 else 0.5
    
    return probability


def calculate_edge_for_props(
    features_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    lookback_games: int = 10
) -> pd.DataFrame:
    """
    Calculates Edge for each prop: recent performance probability - market implied probability.
    
    Edge = P(recent performance > line) - P(market implied over)
    
    Interpretation:
    - Positive edge (Edge > 0): Recent performance suggests OVER more than market
    - Negative edge (Edge < 0): Recent performance suggests UNDER more than market
    - The more negative, the stronger the UNDER edge
    
    Args:
        features_df: Props with features (must have 'over_price', 'Line', 'FullName', 'Prop_Type_Normalized')
        historical_df: Historical player data for calculating recent performance
        lookback_games: Number of recent games to consider (default 10)
        
    Returns:
        DataFrame with 'Edge' column added
    """
    print("\n📊 Calculating Edge for props (OVER and UNDER)...")
    
    if len(features_df) == 0:
        return features_df
    
    edges = []
    edge_directions = []  # Track if edge suggests OVER or UNDER
    
    for idx, row in features_df.iterrows():
        player_name = row.get('FullName', '')
        prop_type = row.get('Prop_Type_Normalized', '')
        line = row.get('Line', 0)
        over_price = row.get('over_price', -110)
        
        # Calculate market implied probability for OVER
        market_implied_over_prob = american_odds_to_implied_prob(over_price)
        market_implied_under_prob = 1 - market_implied_over_prob
        
        # Calculate recent performance probability for OVER
        recent_perf_over_prob = calculate_recent_performance_probability(
            historical_df,
            player_name,
            prop_type,
            line,
            lookback_games
        )
        recent_perf_under_prob = 1 - recent_perf_over_prob
        
        # Edge = recent performance prob - market implied prob
        # Positive = OVER edge, Negative = UNDER edge
        edge = recent_perf_over_prob - market_implied_over_prob
        
        # Determine edge direction
        if edge > 0:
            edge_direction = 'OVER'
        elif edge < -0.05:  # Significant negative edge for UNDER
            edge_direction = 'UNDER'
        else:
            edge_direction = 'NEUTRAL'
        
        edges.append(edge)
        edge_directions.append(edge_direction)
    
    features_df = features_df.copy()
    features_df['Edge'] = edges
    features_df['Edge_Direction'] = edge_directions
    
    # Summary
    positive_edges = (features_df['Edge'] > 0).sum()
    negative_edges = (features_df['Edge'] < -0.05).sum()  # Significant UNDER edge
    neutral_edges = len(features_df) - positive_edges - negative_edges
    
    print(f"   ✅ Calculated Edge for {len(features_df)} props")
    print(f"   📈 Props with OVER edge (Edge > 0): {positive_edges} ({100*positive_edges/len(features_df):.1f}%)")
    print(f"   📉 Props with UNDER edge (Edge < -0.05): {negative_edges} ({100*negative_edges/len(features_df):.1f}%)")
    print(f"   ➖ Props with neutral edge: {neutral_edges} ({100*neutral_edges/len(features_df):.1f}%)")
    
    if positive_edges > 0:
        avg_positive_edge = features_df[features_df['Edge'] > 0]['Edge'].mean()
        print(f"   💰 Average OVER edge: {avg_positive_edge:.1%}")
    
    if negative_edges > 0:
        avg_negative_edge = features_df[features_df['Edge'] < -0.05]['Edge'].mean()
        print(f"   💰 Average UNDER edge: {avg_negative_edge:.1%}")
    
    return features_df


# ============================================================================
# FEATURE BUILDING
# ============================================================================

def build_features_for_props(props_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds ML features for today's props using historical data.
    
    Args:
        props_df: Today's props from Odds API
        historical_df: Historical player data with features
        
    Returns:
        DataFrame with features for each prop
    """
    print("\n🔧 Building features for today's props...")
    
    if len(props_df) == 0:
        return pd.DataFrame()
    
    # Get latest stats for each player from historical data
    latest_date = historical_df['game_date'].max()
    
    # Get list of players we have data for
    known_players = set(historical_df['FullName'].unique())
    
    # Group by player and get most recent features
    player_features = []
    skipped_players = set()
    
    for idx, prop in props_df.iterrows():
        player_name = prop['player_name']
        prop_type = prop['prop_type']
        
        # Find player in historical data
        player_df = historical_df[
            (historical_df['FullName'] == player_name) &
            (historical_df['Prop_Type_Normalized'] == prop_type)
        ]
        
        if len(player_df) == 0:
            # Try partial name match (last name)
            last_name = player_name.split()[-1]
            player_df = historical_df[
                (historical_df['FullName'].str.contains(last_name, case=False, na=False)) &
                (historical_df['Prop_Type_Normalized'] == prop_type)
            ]
        
        if len(player_df) > 0:
            # Get most recent row
            latest_row = player_df.sort_values('game_date').iloc[-1].to_dict()
            
            # Get today's game info
            home_team_full = prop.get('home_team', '')
            away_team_full = prop.get('away_team', '')
            
            # Convert to abbreviations
            home_abbrev = TEAM_FULL_NAME_TO_ABBREV.get(home_team_full, home_team_full[:3].upper())
            away_abbrev = TEAM_FULL_NAME_TO_ABBREV.get(away_team_full, away_team_full[:3].upper())
            
            # Determine player's team from historical data
            player_team = latest_row.get('TeamAbbrev', '')
            
            # Set correct opponent based on today's matchup
            if player_team == home_abbrev:
                opponent = away_abbrev
                is_home = True
            elif player_team == away_abbrev:
                opponent = home_abbrev
                is_home = False
            else:
                # Player might have been traded - use today's game info
                opponent = away_abbrev if player_team else home_abbrev
                is_home = player_team == home_abbrev
            
            # Update with today's prop info
            latest_row['Line'] = prop['line']
            latest_row['Prop_Type_Normalized'] = prop_type
            latest_row['FullName'] = player_name
            latest_row['over_price'] = prop.get('over_price', -110)
            latest_row['home_team'] = home_team_full
            latest_row['away_team'] = away_team_full
            latest_row['game_date'] = prop.get('game_date', '')
            latest_row['OpponentAbbrev'] = opponent
            latest_row['home'] = 1 if is_home else 0
            
            player_features.append(latest_row)
        else:
            skipped_players.add(player_name)
    
    if skipped_players:
        print(f"   ⚠️ Skipped {len(skipped_players)} players with no historical data:")
        for player in sorted(skipped_players)[:10]:  # Show first 10
            print(f"      - {player}")
        if len(skipped_players) > 10:
            print(f"      ... and {len(skipped_players) - 10} more")
    
    if not player_features:
        return pd.DataFrame()
    
    features_df = pd.DataFrame(player_features)
    print(f"   ✅ Built features for {len(features_df)} props ({len(props_df) - len(features_df)} skipped)")
    
    return features_df


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def run_ml_only_predictions(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs ML-only predictions (no LLM).
    
    Args:
        features_df: Props with ML features
        
    Returns:
        DataFrame with ML predictions
    """
    print(f"\n🤖 Running ML-only predictions...")
    
    from ml_pipeline.inference import MLPredictor
    
    predictor = MLPredictor()
    results = predictor.predict(features_df)
    
    # Calculate ML confidence (1-10 scale based on probability distance from 0.5)
    # Prob of 0.5 = confidence 5, Prob of 1.0 or 0.0 = confidence 10
    results['ml_confidence'] = results['ml_prob_over'].apply(
        lambda p: int(5 + 10 * abs(p - 0.5))
    ).clip(5, 10)
    
    print(f"\n   📊 ML Prediction Summary:")
    print(f"      Total props: {len(results)}")
    over_count = (results['ml_prediction'] == 'OVER').sum()
    under_count = (results['ml_prediction'] == 'UNDER').sum()
    print(f"      OVER predictions: {over_count}")
    print(f"      UNDER predictions: {under_count}")
    
    # Show confidence distribution
    for conf in range(10, 4, -1):
        count = (results['ml_confidence'] >= conf).sum()
        print(f"      Confidence >= {conf}: {count}")
    
    return results


def run_predictions(
    features_df: pd.DataFrame,
    confidence_threshold: int = 7,
    mode: str = 'premium',
    ml_only: bool = False,
    max_props: int = 10
) -> pd.DataFrame:
    """
    Runs predictions on today's props.
    
    Args:
        features_df: Props with ML features
        confidence_threshold: Minimum confidence for recommendations (default 7)
        mode: Prediction mode ('fast', 'balanced', 'premium')
        ml_only: Use ML only, skip LLM
        max_props: Max props for Gemini to analyze (default 10)
        
    Returns:
        DataFrame with predictions
    """
    if ml_only:
        results = run_ml_only_predictions(features_df)
        results['prediction'] = results['ml_prediction']
        results['confidence'] = results['ml_confidence']
        return results
    
    # Limit props for Gemini analysis
    if len(features_df) > max_props:
        print(f"\n📋 Limiting to top {max_props} props for Gemini analysis (from {len(features_df)} total)")
        # Run ML first to rank by confidence, then take top N
        ml_results = run_ml_only_predictions(features_df.copy())
        ml_results['ml_confidence'] = ml_results['ml_prob_over'].apply(
            lambda p: int(5 + 10 * abs(p - 0.5))
        ).clip(5, 10)
        # Sort by ML confidence and take top max_props
        top_indices = ml_results.nlargest(max_props, 'ml_confidence').index
        features_df = features_df.loc[top_indices]
        print(f"   Selected {len(features_df)} highest-confidence props for LLM analysis")
    
    print(f"\n🤖 Running predictions (mode={mode}, max_props={max_props})...")
    
    from ml_pipeline.hybrid_predictor import HybridPredictor
    
    predictor = HybridPredictor(mode=mode)
    results = predictor.predict(features_df)
    
    # Add combined recommendation
    results['recommendation'] = results.apply(
        lambda row: row['gemini_prediction'] if row['gemini_confidence'] >= confidence_threshold else None,
        axis=1
    )
    
    # Use gemini prediction/confidence as primary
    results['prediction'] = results['gemini_prediction']
    results['confidence'] = results['gemini_confidence']
    
    # Filter to high confidence only for final output
    high_conf = results[results['confidence'] >= confidence_threshold]
    
    print(f"\n   📊 Prediction Summary:")
    print(f"      Total props analyzed: {len(results)}")
    print(f"      High confidence (>={confidence_threshold}): {len(high_conf)}")
    
    if len(high_conf) > 0:
        over_count = (high_conf['prediction'] == 'OVER').sum()
        under_count = (high_conf['prediction'] == 'UNDER').sum()
        print(f"      OVER picks: {over_count}")
        print(f"      UNDER picks: {under_count}")
    
    predictor.print_stats()
    
    return results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def generate_betting_slate(results: pd.DataFrame, confidence_threshold: int = 7, target_date: str = None) -> str:
    """
    Generates a formatted betting slate from predictions.
    
    Args:
        results: Prediction results DataFrame
        confidence_threshold: Minimum confidence for picks
        target_date: Date for the slate
        
    Returns:
        Formatted string output
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Use 'confidence' column (works for both ML-only and hybrid)
    conf_col = 'confidence' if 'confidence' in results.columns else 'gemini_confidence'
    
    # Filter to high confidence
    high_conf = results[results[conf_col] >= confidence_threshold].copy()
    
    # Sort by confidence descending
    high_conf = high_conf.sort_values(conf_col, ascending=False)
    
    output = []
    output.append("="*70)
    output.append(f"NBA PROP PREDICTIONS - {target_date}")
    output.append("="*70)
    output.append(f"HIGH CONFIDENCE PICKS ({confidence_threshold}+ confidence)")
    output.append(f"Historical accuracy for {confidence_threshold}+ confidence: ~59%")
    output.append("-"*70)
    
    if len(high_conf) == 0:
        output.append("\n⚠️ No high-confidence picks for today.")
        output.append("Consider lowering the confidence threshold or waiting for more props.")
    else:
        for i, (idx, row) in enumerate(high_conf.iterrows(), 1):
            player = row.get('FullName', 'Unknown')
            prop_type = row.get('Prop_Type_Normalized', '')
            line = row.get('Line', 0)
            prediction = row.get('prediction', row.get('ml_prediction', ''))
            confidence = row.get('confidence', row.get('ml_confidence', 0))
            ml_pred = row.get('ml_prediction', '')
            ml_prob = row.get('ml_prob_over', 0.5)
            
            edge = row.get('Edge', 0)
            edge_direction = row.get('Edge_Direction', '')
            if 'Edge' in row:
                if edge > 0:
                    edge_str = f"Edge: {edge:+.1%} (OVER)"
                elif edge < -0.05:
                    edge_str = f"Edge: {edge:+.1%} (UNDER)"
                else:
                    edge_str = f"Edge: {edge:+.1%}"
            else:
                edge_str = ""
            
            output.append(f"\n{i}. {player} {prop_type} {prediction} {line}")
            output.append(f"   ML: {ml_pred} ({ml_prob:.0%}) | Confidence: {confidence}/10 | {edge_str}")
    
    output.append("\n" + "-"*70)
    output.append(f"Total high-confidence plays: {len(high_conf)}")
    
    # Calculate expected value
    if len(high_conf) > 0:
        # Assuming -110 odds and 59% win rate
        ev_per_pick = 0.59 * 0.91 - 0.41 * 1.0  # Win * profit - Loss * stake
        total_ev = ev_per_pick * len(high_conf)
        output.append(f"Expected edge per pick: {ev_per_pick*100:.1f}%")
    
    output.append("="*70)
    
    return "\n".join(output)


def save_predictions(results: pd.DataFrame, slate: str, target_date: str = None, confidence_threshold: int = 7):
    """Saves predictions to files."""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Determine confidence column
    conf_col = 'confidence' if 'confidence' in results.columns else 'ml_confidence'
    pred_col = 'prediction' if 'prediction' in results.columns else 'ml_prediction'
    
    # Create clean predictions CSV with key columns only
    clean_cols = ['FullName', 'Prop_Type_Normalized', 'Line', pred_col, conf_col, 
                  'ml_prediction', 'ml_prob_over', 'TeamAbbrev', 'OpponentAbbrev', 'Edge', 'Edge_Direction']
    clean_cols = [c for c in clean_cols if c in results.columns]
    
    clean_df = results[clean_cols].copy()
    col_names = ['Player', 'PropType', 'Line', 'Prediction', 'Confidence',
                 'ML_Prediction', 'ML_Prob_Over', 'Team', 'Opponent', 'Edge', 'Edge_Direction']
    clean_df.columns = col_names[:len(clean_cols)]
    
    # Filter to high confidence for the main output
    high_conf = results[results[conf_col] >= confidence_threshold]
    high_conf_clean = clean_df[results[conf_col] >= confidence_threshold]
    
    # Save clean high-confidence CSV
    clean_path = os.path.join(PREDICTIONS_DIR, f'picks_{target_date}.csv')
    high_conf_clean.to_csv(clean_path, index=False)
    print(f"\n📄 High-confidence picks saved to: {clean_path}")
    
    # Save all predictions CSV
    all_path = os.path.join(PREDICTIONS_DIR, f'all_predictions_{target_date}.csv')
    clean_df.to_csv(all_path, index=False)
    print(f"📄 All predictions saved to: {all_path}")
    
    # Save formatted slate
    slate_path = os.path.join(PREDICTIONS_DIR, f'betting_slate_{target_date}.txt')
    with open(slate_path, 'w') as f:
        f.write(slate)
    print(f"📄 Betting slate saved to: {slate_path}")
    
    # Save JSON for programmatic access
    json_path = os.path.join(PREDICTIONS_DIR, f'predictions_{target_date}.json')
    
    json_data = {
        'date': target_date,
        'generated_at': datetime.now().isoformat(),
        'total_props': len(results),
        'high_confidence_picks': len(high_conf),
        'confidence_threshold': confidence_threshold,
        'picks': []
    }
    
    for idx, row in high_conf.iterrows():
        json_data['picks'].append({
            'player': row.get('FullName', ''),
            'team': row.get('TeamAbbrev', ''),
            'opponent': row.get('OpponentAbbrev', ''),
            'prop_type': row.get('Prop_Type_Normalized', ''),
            'line': float(row.get('Line', 0)),
            'prediction': row.get(pred_col, row.get('ml_prediction', '')),
            'confidence': int(row.get(conf_col, 0)),
            'ml_prediction': row.get('ml_prediction', ''),
            'ml_probability': float(row.get('ml_prob_over', 0.5)),
            'edge': float(row.get('Edge', 0)) if 'Edge' in row else 0.0,
            'edge_direction': row.get('Edge_Direction', '') if 'Edge_Direction' in row else '',
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 JSON data saved to: {json_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NBA Player Prop Prediction Engine')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--confidence', type=int, default=7,
                        help='Minimum confidence threshold (default: 7)')
    parser.add_argument('--mode', choices=['fast', 'balanced', 'premium'],
                        default='premium',
                        help='Prediction mode (default: premium)')
    parser.add_argument('--ml-only', action='store_true',
                        help='Use ML predictions only (skip LLM/Gemini)')
    parser.add_argument('--max-props', type=int, default=10,
                        help='Max props for Gemini to analyze (default: 10)')
    parser.add_argument('--max-props-per-game', type=int, default=15,
                        help='Max props to fetch per game from Odds API (default: 15)')
    parser.add_argument('--update-data', action='store_true',
                        help='Update Kaggle data before predictions')
    parser.add_argument('--historical-data', type=str, default=None,
                        help='Path to historical data parquet')
    parser.add_argument('--skip-odds', action='store_true',
                        help='Skip odds fetching (use cached)')
    
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    mode_str = "ML-only" if args.ml_only else f"{args.mode} (max {args.max_props} props for Gemini)"
    
    print("\n" + "="*70)
    print("🏀 NBA PLAYER PROP PREDICTION ENGINE")
    print(f"   Target Date: {target_date}")
    print(f"   Mode: {mode_str}")
    print(f"   Confidence Threshold: {args.confidence}+")
    print(f"   Started: {datetime.now()}")
    print("="*70)
    
    # Step 1: Update Kaggle data if requested
    if args.update_data:
        print("\n📥 Updating Kaggle data...")
        from scripts.update_kaggle_data import main as update_kaggle
        update_kaggle()
    
    # Step 2: Fetch today's props
    if not args.skip_odds:
        props_df = fetch_todays_player_props(target_date, max_props_per_game=args.max_props_per_game)
        
        if len(props_df) == 0:
            print("\n❌ No props available for today. Exiting.")
            return 1
        
        # Cache the props
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        props_path = os.path.join(PREDICTIONS_DIR, f'props_{target_date}.csv')
        props_df.to_csv(props_path, index=False)
        print(f"   Cached props to: {props_path}")
    else:
        # Load cached props
        cached_path = os.path.join(PREDICTIONS_DIR, f'props_{target_date}.csv')
        if os.path.exists(cached_path):
            props_df = pd.read_csv(cached_path)
            print(f"\n📂 Loaded {len(props_df)} cached props")
        else:
            print("\n❌ No cached props found. Run without --skip-odds.")
            return 1
    
    # Step 3: Load historical data for features
    historical_path = args.historical_data
    if historical_path is None:
        # Try to find most recent dataset
        ml_training_dir = os.path.join(DATASETS_DIR, 'ml_training')
        candidates = [
            os.path.join(ml_training_dir, f'training_data_{target_date.replace("-", "")}.parquet'),
            os.path.join(ml_training_dir, 'training_data_20251213.parquet'),
            os.path.join(ml_training_dir, 'combined_2024_2026_for_' + target_date.replace('-', '') + '.parquet'),
            os.path.join(ml_training_dir, 'super_dataset_2024_25_regular_season.parquet'),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                historical_path = path
                break
    
    if historical_path is None or not os.path.exists(historical_path):
        print("\n❌ No historical data found. Run build_combined_dataset.py first.")
        return 1
    
    print(f"\n📂 Loading historical data from: {historical_path}")
    historical_df = pd.read_parquet(historical_path)
    print(f"   ✅ Loaded {len(historical_df):,} historical records")
    
    # Step 4: Build features
    features_df = build_features_for_props(props_df, historical_df)
    
    if len(features_df) == 0:
        print("\n❌ Could not build features for any props. Exiting.")
        return 1
    
    # Step 4.5: Calculate Edge and filter to props with significant edge (OVER or UNDER)
    print("\n" + "="*70)
    print("EDGE CALCULATION & FILTERING")
    print("="*70)
    features_df = calculate_edge_for_props(features_df, historical_df, lookback_games=10)
    
    # Filter to props with significant edge:
    # - Edge > 0: Recent performance suggests OVER more than market (OVER edge)
    # - Edge < -0.05: Recent performance suggests UNDER more than market (UNDER edge)
    # This threshold (-0.05) means recent performance is at least 5% less likely to go OVER than market suggests
    before_filter = len(features_df)
    under_edge_threshold = -0.05  # Configurable threshold for UNDER edge
    
    features_df_with_edge = features_df[
        (features_df['Edge'] > 0) | (features_df['Edge'] < under_edge_threshold)
    ].copy()
    after_filter = len(features_df_with_edge)
    
    over_edge_count = (features_df_with_edge['Edge'] > 0).sum()
    under_edge_count = (features_df_with_edge['Edge'] < under_edge_threshold).sum()
    
    print(f"\n   🔍 Filtering to props with significant edge...")
    print(f"   Before filter: {before_filter} props")
    print(f"   After filter: {after_filter} props")
    print(f"      - OVER edge (Edge > 0): {over_edge_count} props")
    print(f"      - UNDER edge (Edge < {under_edge_threshold:.0%}): {under_edge_count} props")
    print(f"   Removed: {before_filter - after_filter} props ({100*(before_filter - after_filter)/before_filter:.1f}%)")
    
    if after_filter == 0:
        print("\n⚠️  No props with significant edge found.")
        print("   This means recent performance doesn't significantly differ from market expectations.")
        print("   Consider:")
        print("   - Checking if historical data is up to date")
        print("   - Adjusting the lookback window")
        print("   - Reviewing market odds")
        return 1
    
    # IMPORTANT: Edge is calculated for filtering/display only, NOT as a model feature
    # The model only uses features from trainer.features (from training data)
    # Since Edge wasn't in training data, it won't be in trainer.features, so it's safe
    # But we keep Edge in the DataFrame for display/analysis purposes
    features_for_model = features_df_with_edge.copy()
    # Edge column will be preserved but NOT used by the model (model uses feature configs from training)
    
    # Step 5: Run predictions (on filtered props with significant edge - OVER or UNDER)
    print(f"\n   ✅ Passing {after_filter} props with significant edge to model")
    print(f"      ({over_edge_count} OVER edge, {under_edge_count} UNDER edge)")
    results = run_predictions(
        features_for_model, 
        args.confidence, 
        args.mode,
        ml_only=args.ml_only,
        max_props=args.max_props
    )
    
    # Add Edge back to results for display (it should already be there, but ensure it)
    if 'Edge' not in results.columns and 'Edge' in features_df_with_edge.columns:
        # Merge Edge back if it was dropped
        edge_map = features_df_with_edge.set_index(['FullName', 'Prop_Type_Normalized', 'Line'])['Edge'].to_dict()
        results['Edge'] = results.apply(
            lambda row: edge_map.get((row.get('FullName'), row.get('Prop_Type_Normalized'), row.get('Line')), 0),
            axis=1
        )
    
    # Step 6: Generate and save output
    slate = generate_betting_slate(results, args.confidence, target_date)
    print("\n" + slate)
    
    save_predictions(results, slate, target_date, args.confidence)
    
    print("\n" + "="*70)
    print("✅ PREDICTION GENERATION COMPLETE")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

