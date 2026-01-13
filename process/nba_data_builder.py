"""
NBA DATA BUILDER (nba_data_builder.py)
======================================
Self-Evolving NBA Neural System v3.0 - PySpark Edition

Purpose: Build training datasets from local CSV files for the Predict-Then-Reveal
evaluation loop. Generates TWO datasets:
1. Features Dataset - Everything Gemini needs to make predictions (NO actuals)
2. Actuals Dataset - Ground truth for comparison AFTER predictions are locked

Data Sources:
- PlayerStatistics.csv (from Kaggle)
- TeamStatistics.csv (from Kaggle)
- The Odds API (historical odds)

Key Philosophy:
- ZERO DATA LEAKAGE: Features dataset contains NO actual game outcomes
- The "reveal" happens AFTER Gemini makes predictions
- Self-reflexive learning from high-confidence misses
"""

import os
import sys
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, concat, concat_ws, when, coalesce, greatest,
    avg, sum as spark_sum, count, lag, lead, stddev,
    datediff, to_date, date_format, to_timestamp,
    row_number, rank, dense_rank,
    first, last, collect_list,
    udf, broadcast, expr,
    upper, regexp_replace
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, DoubleType, TimestampType, DateType
)

# Import shared configuration
from shared_config import (
    get_spark_session,
    stop_spark_session,
    ODDS_API_KEY, 
    PLAYER_STATS_CSV,
    TEAM_STATS_CSV,
    SEASON_DATE_RANGES,
    TEAM_FULL_NAME_TO_ABBREV,
    DATASETS_DIR, 
    wait_for_api,
    ODDS_API_DELAY
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Odds API Configuration
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
HISTORICAL_ODDS_URL = f"{ODDS_BASE_URL}/historical/sports/basketball_nba/events"


# ============================================================================
# MODULE 1: CSV DATA LOADING AND FILTERING (PySpark)
# ============================================================================

def load_player_statistics(spark: SparkSession) -> DataFrame:
    """
    Loads PlayerStatistics.csv into a PySpark DataFrame.
    
    Returns:
        DataFrame with all player statistics
    """
    print("📊 Loading PlayerStatistics.csv...")
    
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
    
    print(f"   ✅ Loaded {df.count():,} player-game records")
    return df


def load_team_statistics(spark: SparkSession) -> DataFrame:
    """
    Loads TeamStatistics.csv into a PySpark DataFrame.
    
    Returns:
        DataFrame with all team statistics
    """
    print("📊 Loading TeamStatistics.csv...")
    
    df = spark.read.csv(
        TEAM_STATS_CSV,
        header=True,
        inferSchema=True
    )
    
    # Parse timestamp and extract date
    df = df.withColumn(
        "game_date",
        to_date(to_timestamp(col("gameDateTimeEst"), "yyyy-MM-dd HH:mm:ss"))
    )
    
    print(f"   ✅ Loaded {df.count():,} team-game records")
    return df


def filter_by_season(df: DataFrame, season: str) -> DataFrame:
    """
    Filters a DataFrame to only include games from a specific NBA season.
    
    Args:
        df: DataFrame with game_date column
        season: Season string (e.g., "2023-24" or "2024-25")
    
    Returns:
        Filtered DataFrame
    """
    if season not in SEASON_DATE_RANGES:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_DATE_RANGES.keys())}")
    
    start_date, end_date = SEASON_DATE_RANGES[season]
    
    filtered = df.filter(
        (col("game_date") >= start_date) & 
        (col("game_date") <= end_date)
    )
    
    print(f"   📅 Filtered to {season} season ({start_date} to {end_date})")
    return filtered


def load_and_filter_seasons(
    spark: SparkSession, 
    seasons: List[str] = ["2023-24", "2024-25"]
) -> Tuple[DataFrame, DataFrame]:
    """
    Loads and filters both player and team statistics for specified seasons.
    
    Args:
        spark: SparkSession instance
        seasons: List of seasons to include
        
    Returns:
        Tuple of (player_df, team_df) filtered to specified seasons
    """
    print(f"\n{'='*60}")
    print(f"📥 LOADING DATA FOR SEASONS: {', '.join(seasons)}")
    print(f"{'='*60}")
    
    # Load full datasets
    player_df = load_player_statistics(spark)
    team_df = load_team_statistics(spark)
    
    # Filter to specified seasons
    player_filtered = None
    team_filtered = None
    
    for season in seasons:
        player_season = filter_by_season(player_df, season)
        team_season = filter_by_season(team_df, season)
        
        if player_filtered is None:
            player_filtered = player_season
            team_filtered = team_season
        else:
            player_filtered = player_filtered.union(player_season)
            team_filtered = team_filtered.union(team_season)
    
    player_count = player_filtered.count()
    team_count = team_filtered.count()
    
    print(f"\n✅ Final dataset sizes:")
    print(f"   Player records: {player_count:,}")
    print(f"   Team records: {team_count:,}")
    
    return player_filtered, team_filtered


# ============================================================================
# MODULE 2: TEAM NAME NORMALIZATION
# ============================================================================

def add_team_abbreviation(df: DataFrame, city_col: str, name_col: str, abbrev_col: str) -> DataFrame:
    """
    Adds a TeamAbbrev column by mapping City + Name to standard abbreviation.
    
    Args:
        df: DataFrame with team city and name columns
        city_col: Column name for team city (e.g., "playerteamCity")
        name_col: Column name for team name (e.g., "playerteamName")
        abbrev_col: Name for the new abbreviation column
        
    Returns:
        DataFrame with new abbreviation column
    """
    # Create full team name column
    df = df.withColumn(
        "_full_team_name",
        concat_ws(" ", col(city_col), col(name_col))
    )
    
    # Build CASE WHEN expression for mapping
    mapping_expr = None
    for full_name, abbrev in TEAM_FULL_NAME_TO_ABBREV.items():
        condition = col("_full_team_name") == full_name
        if mapping_expr is None:
            mapping_expr = when(condition, lit(abbrev))
        else:
            mapping_expr = mapping_expr.when(condition, lit(abbrev))
    
    # Add fallback for unknown teams
    mapping_expr = mapping_expr.otherwise(lit("UNK"))
    
    # Apply mapping and drop temp column
    df = df.withColumn(abbrev_col, mapping_expr)
    df = df.drop("_full_team_name")
    
    return df


def normalize_team_names(player_df: DataFrame, team_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Adds TeamAbbrev and OpponentAbbrev columns to both DataFrames.
    
    Args:
        player_df: Player statistics DataFrame
        team_df: Team statistics DataFrame
        
    Returns:
        Tuple of (player_df, team_df) with new abbreviation columns
    """
    print("\n🏀 Normalizing team names...")
    
    # Player DataFrame - has playerteamCity/playerteamName and opponentteamCity/opponentteamName
    player_df = add_team_abbreviation(
        player_df, "playerteamCity", "playerteamName", "TeamAbbrev"
    )
    player_df = add_team_abbreviation(
        player_df, "opponentteamCity", "opponentteamName", "OpponentAbbrev"
    )
    
    # Team DataFrame - has teamCity/teamName and opponentTeamCity/opponentTeamName
    team_df = add_team_abbreviation(
        team_df, "teamCity", "teamName", "TeamAbbrev"
    )
    team_df = add_team_abbreviation(
        team_df, "opponentTeamCity", "opponentTeamName", "OpponentAbbrev"
    )
    
    # Verify normalization
    unknown_player = player_df.filter(col("TeamAbbrev") == "UNK").count()
    unknown_team = team_df.filter(col("TeamAbbrev") == "UNK").count()
    
    if unknown_player > 0 or unknown_team > 0:
        print(f"   ⚠️ Found {unknown_player} player records and {unknown_team} team records with unknown team names")
    else:
        print("   ✅ All team names normalized successfully")
    
    return player_df, team_df


# ============================================================================
# MODULE 3: FEATURE ENGINEERING (PySpark Window Functions)
# ============================================================================

def add_team_context(team_df: DataFrame) -> DataFrame:
    """
    Adds team-level context features (pace, defensive rating, opponent allowed stats)
    using rolling windows to avoid leakage.
    """
    print("\n🏀 Engineering team context (pace/defense/opponent allowed)...")
    
    t = team_df.alias("t")
    o = team_df.alias("o")
    
    # Join with opponent row for the same game to access opponent stats
    joined = t.join(
        o,
        (col("t.gameId") == col("o.gameId")) & (col("t.teamId") == col("o.opponentTeamId")),
        "left"
    )
    
    team_poss = (
        col("t.fieldGoalsAttempted")
        + lit(0.44) * col("t.freeThrowsAttempted")
        - col("t.reboundsOffensive")
        + col("t.turnovers")
    )
    opp_poss = (
        col("o.fieldGoalsAttempted")
        + lit(0.44) * col("o.freeThrowsAttempted")
        - col("o.reboundsOffensive")
        + col("o.turnovers")
    )
    
    pace = lit(48.0) * ((team_poss + opp_poss) / lit(2.0)) / (col("t.numMinutes") / lit(5.0))
    def_rating = when(team_poss != 0, col("t.opponentScore") * lit(100.0) / team_poss)
    opp_def_rating = when(opp_poss != 0, col("t.teamScore") * lit(100.0) / opp_poss)
    
    enriched = joined.withColumn("Team_Pace_Game", pace) \
        .withColumn("Team_Def_Rating_Game", def_rating) \
        .withColumn("Allowed_PTS_Game", col("t.opponentScore")) \
        .withColumn("Allowed_REB_Game", col("o.reboundsTotal")) \
        .withColumn("Allowed_AST_Game", col("o.assists")) \
        .withColumn("Allowed_3PA_Game", col("o.threePointersAttempted")) \
        .withColumn("Opp_Def_Rating_Game", opp_def_rating)
    
    # Rolling windows (previous games only to avoid leakage)
    window_5 = (Window
        .partitionBy("t.teamId")
        .orderBy("t.gameDateTimeEst")
        .rowsBetween(-5, -1))
    window_10 = (Window
        .partitionBy("t.teamId")
        .orderBy("t.gameDateTimeEst")
        .rowsBetween(-10, -1))
    window_season = (Window
        .partitionBy("t.teamId")
        .orderBy("t.gameDateTimeEst")
        .rowsBetween(Window.unboundedPreceding, -1))
    
    metrics = [
        ("Team_Pace_Game", "Team_Pace"),
        ("Team_Def_Rating_Game", "Team_Def_Rating"),
        ("Allowed_PTS_Game", "Allowed_PTS"),
        ("Allowed_REB_Game", "Allowed_REB"),
        ("Allowed_AST_Game", "Allowed_AST"),
        ("Allowed_3PA_Game", "Allowed_3PA"),
        ("Opp_Def_Rating_Game", "Opp_Def_Rating")
    ]
    
    for base, prefix in metrics:
        enriched = enriched.withColumn(f"L5_{prefix}", avg(col(base)).over(window_5))
        enriched = enriched.withColumn(f"L10_{prefix}", avg(col(base)).over(window_10))
        enriched = enriched.withColumn(f"Season_{prefix}", avg(col(base)).over(window_season))
    
    # OPPONENT DEFENSIVE RANKINGS - How each team ranks in allowing stats
    # Lower rank = allows MORE of that stat (bad defense), Higher rank = allows LESS (good defense)
    rank_window_def = Window.partitionBy(col("t.game_date")).orderBy(col("Team_Def_Rating_Game").asc())
    rank_window_pts = Window.partitionBy(col("t.game_date")).orderBy(col("Season_Allowed_PTS").desc())
    rank_window_reb = Window.partitionBy(col("t.game_date")).orderBy(col("Season_Allowed_REB").desc())
    rank_window_ast = Window.partitionBy(col("t.game_date")).orderBy(col("Season_Allowed_AST").desc())
    rank_window_3pa = Window.partitionBy(col("t.game_date")).orderBy(col("Season_Allowed_3PA").desc())
    
    enriched = enriched.withColumn("Team_Def_Rank_Game", rank().over(rank_window_def))
    enriched = enriched.withColumn("Rank_Allows_PTS", rank().over(rank_window_pts))  # 1 = allows most PTS
    enriched = enriched.withColumn("Rank_Allows_REB", rank().over(rank_window_reb))  # 1 = allows most REB
    enriched = enriched.withColumn("Rank_Allows_AST", rank().over(rank_window_ast))  # 1 = allows most AST
    enriched = enriched.withColumn("Rank_Allows_3PA", rank().over(rank_window_3pa))  # 1 = allows most 3PA
    
    # Keep only team (t) columns plus engineered context to avoid duplicate column names (e.g., home)
    base_cols = [col(f"t.{c}").alias(c) for c in team_df.columns]
    context_cols = [
        "Team_Pace_Game", "Team_Def_Rating_Game", "Allowed_PTS_Game",
        "Allowed_REB_Game", "Allowed_AST_Game", "Allowed_3PA_Game",
        "Opp_Def_Rating_Game", "Team_Def_Rank_Game",
        "Rank_Allows_PTS", "Rank_Allows_REB", "Rank_Allows_AST", "Rank_Allows_3PA"
    ]
    for _, prefix in metrics:
        context_cols.extend([f"L5_{prefix}", f"L10_{prefix}", f"Season_{prefix}"])
    
    selected = base_cols + [col(c) for c in context_cols if c in enriched.columns]
    return enriched.select(*selected)


def add_player_advanced_features(player_df: DataFrame) -> DataFrame:
    """
    Adds player advanced features using ROLLING AVERAGES to prevent data leakage.
    
    CRITICAL FIX: All advanced stats must use rolling averages, NOT current game stats.
    Using current game stats (e.g., this game's TS%) would leak the outcome!
    
    Features (all calculated from PREVIOUS games):
    - Player_Poss: Season average possessions
    - Player_TS_Pct, Player_eFG_Pct: Season average efficiency
    - Player_3PA_Rate, Player_FT_Rate: Season average shot profile
    - Player_*_per100: Season average per-100-possession rates
    - Volatility (StdDev L5/L10) for key stats
    - Form deltas (L5 minus Season) for key stats
    """
    print("\n🏀 Engineering player advanced features (using rolling averages)...")
    
    # First, calculate per-game values (intermediate, NOT used as features)
    player_df = player_df.withColumn(
        "_poss_game",
        (col("fieldGoalsAttempted") + lit(0.44) * col("freeThrowsAttempted") + col("turnovers"))
    )
    player_df = player_df.withColumn(
        "_ts_pct_game",
        when((col("fieldGoalsAttempted") + lit(0.44) * col("freeThrowsAttempted")) > 0,
             col("points") / (lit(2) * (col("fieldGoalsAttempted") + lit(0.44) * col("freeThrowsAttempted"))))
    )
    player_df = player_df.withColumn(
        "_efg_pct_game",
        when(col("fieldGoalsAttempted") > 0,
             (col("fieldGoalsMade") + lit(0.5) * col("threePointersMade")) / col("fieldGoalsAttempted"))
    )
    player_df = player_df.withColumn(
        "_3pa_rate_game",
        when(col("fieldGoalsAttempted") > 0, col("threePointersAttempted") / col("fieldGoalsAttempted"))
    )
    player_df = player_df.withColumn(
        "_ft_rate_game",
        when(col("fieldGoalsAttempted") > 0, col("freeThrowsAttempted") / col("fieldGoalsAttempted"))
    )
    
    # Calculate per-game per-100 rates (intermediate)
    per100_games = [
        ("points", "_pts_per100_game"),
        ("reboundsTotal", "_reb_per100_game"),
        ("assists", "_ast_per100_game"),
        ("threePointersMade", "_3pm_per100_game")
    ]
    for stat, col_name in per100_games:
        player_df = player_df.withColumn(
            col_name,
            when(col("_poss_game") > 0, col(stat) / col("_poss_game") * lit(100))
        )
    
    # NOW calculate SEASON ROLLING AVERAGES using a window that EXCLUDES current game
    window_season = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(Window.unboundedPreceding, -1))  # All previous games, NOT current
    
    # Season average efficiency metrics (from PREVIOUS games only)
    player_df = player_df.withColumn("Player_Poss", avg(col("_poss_game")).over(window_season))
    player_df = player_df.withColumn("Player_TS_Pct", avg(col("_ts_pct_game")).over(window_season))
    player_df = player_df.withColumn("Player_eFG_Pct", avg(col("_efg_pct_game")).over(window_season))
    player_df = player_df.withColumn("Player_3PA_Rate", avg(col("_3pa_rate_game")).over(window_season))
    player_df = player_df.withColumn("Player_FT_Rate", avg(col("_ft_rate_game")).over(window_season))
    
    # Season average per-100 rates (from PREVIOUS games only)
    player_df = player_df.withColumn("Player_PTS_per100", avg(col("_pts_per100_game")).over(window_season))
    player_df = player_df.withColumn("Player_REB_per100", avg(col("_reb_per100_game")).over(window_season))
    player_df = player_df.withColumn("Player_AST_per100", avg(col("_ast_per100_game")).over(window_season))
    player_df = player_df.withColumn("Player_3PM_per100", avg(col("_3pm_per100_game")).over(window_season))
    
    # Drop intermediate per-game columns (these would leak if kept)
    player_df = player_df.drop(
        "_poss_game", "_ts_pct_game", "_efg_pct_game", "_3pa_rate_game", "_ft_rate_game",
        "_pts_per100_game", "_reb_per100_game", "_ast_per100_game", "_3pm_per100_game"
    )
    
    # Rolling volatility and form deltas (requires rolling windows)
    window_5 = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(-5, -1))
    
    window_10 = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(-10, -1))
    
    stats = [
        ("points", "PTS"),
        ("reboundsTotal", "REB"),
        ("assists", "AST"),
        ("threePointersMade", "3PM")
    ]
    
    for raw, short in stats:
        player_df = player_df.withColumn(f"L5_StdDev_{short}", stddev(col(raw)).over(window_5))
        player_df = player_df.withColumn(f"L10_StdDev_{short}", stddev(col(raw)).over(window_10))
        # Form delta: L5 - Season
        player_df = player_df.withColumn(
            f"Form_{short}",
            col(f"L5_Avg_{short}") - col(f"Season_Avg_{short}")
        )
    
    return player_df


def add_rolling_averages(player_df: DataFrame) -> DataFrame:
    """
    Calculates rolling averages for key stats using Window functions.
    
    CRITICAL: Uses rowsBetween(-5, -1) and (-10, -1) to look at PREVIOUS games only,
    preventing data leakage.
    
    Args:
        player_df: Player statistics DataFrame
        
    Returns:
        DataFrame with L5_Avg, L10_Avg, Season_Avg, and Home/Away splits
    """
    print("\n📈 Calculating rolling averages...")
    
    # Define window specs - ordered by game date, partitioned by player
    # rowsBetween(-5, -1) means "the 5 rows before this one" (excludes current)
    window_5 = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(-5, -1))
    
    window_10 = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(-10, -1))
    
    # Season average (all previous games)
    window_season = (Window
        .partitionBy("personId")
        .orderBy("gameDateTimeEst")
        .rowsBetween(Window.unboundedPreceding, -1))
    
    # HOME/AWAY SPLIT windows - partitioned by player AND home status
    window_5_home = (Window
        .partitionBy("personId", "home")
        .orderBy("gameDateTimeEst")
        .rowsBetween(-5, -1))
    
    window_season_home = (Window
        .partitionBy("personId", "home")
        .orderBy("gameDateTimeEst")
        .rowsBetween(Window.unboundedPreceding, -1))
    
    # Calculate rolling averages for each stat
    stats = [
        ("points", "PTS"),
        ("reboundsTotal", "REB"),
        ("assists", "AST"),
        ("threePointersMade", "3PM"),
        ("blocks", "BLK"),
        ("steals", "STL"),
        ("numMinutes", "MIN")
    ]
    
    for csv_col, stat_name in stats:
        # Overall rolling averages
        player_df = player_df.withColumn(
            f"L5_Avg_{stat_name}",
            avg(col(csv_col)).over(window_5)
        )
        player_df = player_df.withColumn(
            f"L10_Avg_{stat_name}",
            avg(col(csv_col)).over(window_10)
        )
        player_df = player_df.withColumn(
            f"Season_Avg_{stat_name}",
            avg(col(csv_col)).over(window_season)
        )
    
    # HOME/AWAY SPLITS for key stats (PTS, REB, AST, 3PM)
    home_away_stats = [("points", "PTS"), ("reboundsTotal", "REB"), ("assists", "AST"), ("threePointersMade", "3PM")]
    for csv_col, stat_name in home_away_stats:
        # L5 average in same venue type (home vs away)
        player_df = player_df.withColumn(
            f"L5_Avg_{stat_name}_SameVenue",
            avg(col(csv_col)).over(window_5_home)
        )
        # Season average in same venue type
        player_df = player_df.withColumn(
            f"Season_Avg_{stat_name}_SameVenue",
            avg(col(csv_col)).over(window_season_home)
        )
    
    # Also calculate standard deviation for consistency
    player_df = player_df.withColumn(
        "L10_StdDev_PTS",
        expr("stddev(points) OVER (PARTITION BY personId ORDER BY gameDateTimeEst ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)")
    )
    
    print("   ✅ Rolling averages calculated (L5, L10, Season, Home/Away splits)")
    return player_df


def add_rest_days(team_df: DataFrame) -> DataFrame:
    """
    Calculates days since last game for each team.
    
    Args:
        team_df: Team statistics DataFrame
        
    Returns:
        DataFrame with rest_days column
    """
    print("\n😴 Calculating rest days...")
    
    # Window partitioned by team, ordered by game date
    team_window = Window.partitionBy("teamId").orderBy("gameDateTimeEst")
    
    # Get previous game date
    team_df = team_df.withColumn(
        "prev_game_date",
        lag(col("game_date")).over(team_window)
    )
    
    # Calculate days difference
    team_df = team_df.withColumn(
        "rest_days",
        datediff(col("game_date"), col("prev_game_date"))
    )
    
    # First game of season has no previous - set to 7 (well rested)
    team_df = team_df.withColumn(
        "rest_days",
        coalesce(col("rest_days"), lit(7))
    )
    
    # Identify back-to-backs
    team_df = team_df.withColumn(
        "is_b2b",
        when(col("rest_days") == 1, lit(1)).otherwise(lit(0))
    )
    
    print("   ✅ Rest days calculated")
    return team_df


def add_composite_join_key(df: DataFrame) -> DataFrame:
    """
    Creates a composite join key for matching stats with odds.
    
    Format: YYYY-MM-DD_HOMEABBREV_AWAYABBREV
    
    Args:
        df: DataFrame with game_date, home, TeamAbbrev, OpponentAbbrev columns
        
    Returns:
        DataFrame with Join_Key column
    """
    # Determine home and away teams based on 'home' flag
    df = df.withColumn(
        "HomeTeamAbbrev",
        when(col("home") == 1, col("TeamAbbrev")).otherwise(col("OpponentAbbrev"))
    )
    df = df.withColumn(
        "AwayTeamAbbrev",
        when(col("home") == 0, col("TeamAbbrev")).otherwise(col("OpponentAbbrev"))
    )
    
    # Create composite key
    df = df.withColumn(
        "Join_Key",
        concat_ws("_",
            date_format(col("game_date"), "yyyy-MM-dd"),
            col("HomeTeamAbbrev"),
            col("AwayTeamAbbrev")
        )
    )
    
    return df


def engineer_features(player_df: DataFrame, team_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Applies all feature engineering transformations.
    
    Args:
        player_df: Player statistics DataFrame
        team_df: Team statistics DataFrame
        
    Returns:
        Tuple of (player_df, team_df) with engineered features
    """
    print(f"\n{'='*60}")
    print("🔧 FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    # Rolling averages for players
    player_df = add_rolling_averages(player_df)
    
    # Player advanced features (efficiency, per-100, volatility, form)
    player_df = add_player_advanced_features(player_df)
    
    # Rest days for teams
    team_df = add_rest_days(team_df)
    
    # Team context: pace, defensive rating, opponent allowed stats (rolling, no leakage)
    team_df = add_team_context(team_df)
    
    # Add join keys
    player_df = add_composite_join_key(player_df)
    team_df = add_composite_join_key(team_df)
    
    print("\n✅ Feature engineering complete")
    return player_df, team_df


# ============================================================================
# MODULE 4: HISTORICAL ODDS FETCHING
# ============================================================================

def fetch_historical_odds_for_date(
    target_date: str,
    allowed_join_keys: Optional[List[str]] = None,
    allowed_prop_types: Optional[List[str]] = None,
    max_props_per_date: Optional[int] = None,
) -> List[Dict]:
    """
    Fetches historical player prop odds from The Odds API for a specific date.
    
    OPTIMIZED APPROACH:
    1. Make ONE request to get all events for the date
    2. For each relevant event, make ONE request with ALL 4 prop markets combined
       (player_points,player_rebounds,player_assists,player_threes)
    
    This is efficient because:
    - Multiple markets in ONE request counts as 1 API call
    - We filter events by allowed_join_keys BEFORE making prop requests
    - We get all 4 prop types per game in a single call
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        allowed_join_keys: Optional list of Join_Keys to filter to (reduces API calls!)
        allowed_prop_types: Optional list of prop types to include
        max_props_per_date: Hard cap on raw props fetched for the date (pre-dedup)
        
    Returns:
        List of prop dictionaries with player, prop_type, line, over_odds
    """
    if not ODDS_API_KEY:
        return []
    
    # Historical API requires an ISO timestamp snapshot.
    # Use end-of-day to maximize availability of markets.
    snapshot = f"{target_date}T23:59:00Z"
    
    try:
        wait_for_api("odds_api", ODDS_API_DELAY)
        
        # Step 1: Get all events for this date (1 API request)
        events_url = f"{HISTORICAL_ODDS_URL}"
        events_params = {
            "apiKey": ODDS_API_KEY,
            "date": snapshot
        }
        
        events_response = requests.get(events_url, params=events_params, timeout=30)
        
        if events_response.status_code != 200:
            print(f"   ⚠️ Events fetch failed {events_response.status_code} for {target_date}")
            return []
        
        events_data = events_response.json()
        if isinstance(events_data, dict):
            events = events_data.get("data") or events_data.get("events") or []
            if not events and events_data.get("id"):
                events = [events_data]
        elif isinstance(events_data, list):
            events = events_data
        else:
            events = []
        
        if not events:
            print(f"   ⚠️ No events returned for {target_date}")
            return []
        
        # Step 2: Filter events to only those we need (reduces API calls!)
        relevant_events = []
        for event in events:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            home_abbrev = TEAM_FULL_NAME_TO_ABBREV.get(home_team, "UNK")
            away_abbrev = TEAM_FULL_NAME_TO_ABBREV.get(away_team, "UNK")
            join_key = f"{target_date}_{home_abbrev}_{away_abbrev}"
            
            # Only include events that match our allowed join keys
            if allowed_join_keys and join_key not in allowed_join_keys:
                continue
            
            event["_join_key"] = join_key
            event["_home_abbrev"] = home_abbrev
            event["_away_abbrev"] = away_abbrev
            relevant_events.append(event)
        
        if not relevant_events:
            return []
        
        # Dedup by (join_key, player, prop_type, line)
        dedup_props: Dict[Tuple[str, str, str, float], Dict] = {}
        
        # Step 3: Fetch player props for each RELEVANT event
        # Each call gets ALL 4 markets (pts, reb, ast, 3pm) in ONE request
        for event in relevant_events:
            if max_props_per_date and len(dedup_props) >= max_props_per_date:
                break
                
            event_id = event.get("id")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            join_key = event.get("_join_key")
            
            wait_for_api("odds_api", ODDS_API_DELAY)
            
            # OPTIMIZED: Request ALL 4 prop markets in ONE call
            odds_url = f"{ODDS_BASE_URL}/historical/sports/basketball_nba/events/{event_id}/odds"
            odds_params = {
                "apiKey": ODDS_API_KEY,
                "date": snapshot,
                "regions": "us",
                "markets": "player_points,player_rebounds,player_assists,player_threes",
                "oddsFormat": "american"
            }
            
            odds_response = requests.get(odds_url, params=odds_params, timeout=30)
            
            if odds_response.status_code != 200:
                continue

            odds_data = odds_response.json()
            
            # Handle response format
            event_data = odds_data.get("data") if isinstance(odds_data, dict) else None
            bookmakers = []
            if isinstance(event_data, dict):
                bookmakers = event_data.get("bookmakers", [])
            elif isinstance(odds_data, dict):
                bookmakers = odds_data.get("bookmakers", [])
            
            for bookmaker in bookmakers:
                if max_props_per_date and len(dedup_props) >= max_props_per_date:
                    break
                    
                for market in bookmaker.get("markets", []):
                    if max_props_per_date and len(dedup_props) >= max_props_per_date:
                        break
                        
                    prop_type = market.get("key", "").replace("player_", "").upper()
                    prop_type = prop_type.replace("THREES", "3PM")
                    
                    if allowed_prop_types and prop_type not in allowed_prop_types:
                        continue
                    
                    for outcome in market.get("outcomes", []):
                        if max_props_per_date and len(dedup_props) >= max_props_per_date:
                            break
                            
                        if outcome.get("name") == "Over":
                            player_name = outcome.get("description", "")
                            line = outcome.get("point")
                            over_odds = outcome.get("price")
                            
                            if player_name and line is not None:
                                try:
                                    line_val = float(line)
                                except (TypeError, ValueError):
                                    continue
                                
                                dedup_key = (join_key, player_name, prop_type, line_val)
                                existing = dedup_props.get(dedup_key)
                                
                                # Keep the best (most generous) Over odds if multiple books
                                if existing is None or (over_odds is not None and over_odds > existing.get("Over_Odds", -9999)):
                                    dedup_props[dedup_key] = {
                                        "Join_Key": join_key,
                                        "Player": player_name,
                                        "Prop_Type": prop_type,
                                        "Line": line_val,
                                        "Over_Odds": over_odds,
                                        "Home_Team": home_team,
                                        "Away_Team": away_team,
                                        "Date": target_date
                                    }
        
        all_props = list(dedup_props.values())
        return all_props
        
    except Exception as e:
        print(f"   ⚠️ Error fetching odds for {target_date}: {e}")
        return []


def fetch_historical_odds_batch(
    spark: SparkSession,
    dates: List[str],
    show_progress: bool = True,
    allowed_join_keys: Optional[List[str]] = None,
    allowed_players: Optional[Set[str]] = None,
    allowed_prop_types: Optional[List[str]] = None,
    max_per_player: int = 3,
    max_per_type: int = 8,
    global_cap: int = 50,
    max_props_per_date: Optional[int] = 50,
) -> DataFrame:
    """
    Fetches historical odds for a batch of dates and returns as DataFrame.
    
    OPTIMIZED:
    - 1 request to get events per date
    - 1 request per RELEVANT game (filtered by allowed_join_keys first)
    - All 4 prop markets (pts, reb, ast, 3pm) fetched in ONE request per game
    
    Args:
        spark: SparkSession instance
        dates: List of date strings (YYYY-MM-DD format)
        show_progress: Whether to show progress bar
        allowed_join_keys: Filter to only fetch odds for these games (reduces API calls!)
    
    Returns:
        DataFrame with odds data
    """
    print(f"\n{'='*60}")
    print(f"📜 FETCHING HISTORICAL ODDS ({len(dates)} dates)")
    print(f"   ⚡ OPTIMIZED: 1 request/date for events + 1 request/game for props")
    print(f"   ⚡ All 4 prop types (PTS, REB, AST, 3PM) in ONE request per game")
    print(f"{'='*60}")
    
    all_props = []
    total_raw = 0
    total_after_join_key = 0
    total_final = 0
    
    iterator = tqdm(dates, desc="Fetching Odds") if show_progress else dates
    
    for date in iterator:
        props = fetch_historical_odds_for_date(
            date,
            allowed_join_keys=allowed_join_keys,
            allowed_prop_types=allowed_prop_types,
            max_props_per_date=max_props_per_date,
        )
        raw_count = len(props)
        total_raw += raw_count

        # If restricting to known games, drop anything not matching join key list
        if allowed_join_keys:
            props = [p for p in props if p.get("Join_Key") in allowed_join_keys]
        after_join_key = len(props)
        total_after_join_key += after_join_key

        # Filter to allowed players (if provided)
        if allowed_players:
            props = [p for p in props if p.get("Player") in allowed_players]
        after_player_filter = len(props)

        # Apply caps to control volume
        player_counts: Dict[str, int] = {}
        prop_counts: Dict[str, int] = {}
        capped_props: List[Dict] = []
        for p in props:
            player = p.get("Player")
            ptype = p.get("Prop_Type")
            if max_per_player and player_counts.get(player, 0) >= max_per_player:
                continue
            if max_per_type and prop_counts.get(ptype, 0) >= max_per_type:
                continue
            capped_props.append(p)
            player_counts[player] = player_counts.get(player, 0) + 1
            prop_counts[ptype] = prop_counts.get(ptype, 0) + 1
            if global_cap and len(capped_props) >= global_cap:
                break

        final_count = len(capped_props)
        total_final += final_count
        print(
            f"   • {date}: raw={raw_count} after_join_key={after_join_key} "
            f"after_player={after_player_filter} capped={final_count}"
        )
        all_props.extend(capped_props)
    
    if not all_props:
        print("   ⚠️ No historical odds retrieved")
        # Return empty DataFrame with expected schema
        schema = StructType([
            StructField("Join_Key", StringType(), True),
            StructField("Player", StringType(), True),
            StructField("Prop_Type", StringType(), True),
            StructField("Line", DoubleType(), True),
            StructField("Over_Odds", IntegerType(), True),
            StructField("Home_Team", StringType(), True),
            StructField("Away_Team", StringType(), True),
            StructField("Date", StringType(), True)
        ])
        return spark.createDataFrame([], schema)
    
    # Efficiency stats
    print(f"\n   📊 API EFFICIENCY:")
    print(f"      Dates processed: {len(dates)}")
    print(f"      Est. API requests: {len(dates)} (events) + ~{len(dates) * 8} (props)")
    print(f"      Props retrieved: {len(all_props)} (after filters)")
    print(f"      All 4 prop types fetched per game in ONE request")
    print(
        f"   ✅ Retrieved {len(all_props)} prop lines "
        f"(raw {total_raw} → join_key {total_after_join_key} → capped {total_final})"
    )
    
    # Convert to DataFrame
    odds_df = spark.createDataFrame(all_props)
    
    return odds_df


# ============================================================================
# CLEANUP / COLUMN ORDERING HELPERS
# ============================================================================

def clean_and_reorder_features(df: DataFrame) -> DataFrame:
    """
    Drop redundant/useless columns, fill numeric nulls, and reorder columns
    for the model-facing preview dataset.
    """
    # Handle duplicate column names by first renaming ALL columns to unique positional names,
    # then selecting only the first occurrence of each original column name.
    original_names = df.columns
    # Create unique positional names to avoid any ambiguity
    temp_names = [f"__col_{i}__" for i in range(len(original_names))]
    df_renamed = df.toDF(*temp_names)
    
    # Now select only the first occurrence of each original column name
    seen = set()
    select_exprs = []
    for idx, orig_name in enumerate(original_names):
        if orig_name in seen:
            continue
        select_exprs.append(col(temp_names[idx]).alias(orig_name))
        seen.add(orig_name)
    df = df_renamed.select(*select_exprs)
    drop_cols = [
        "gameSubLabel", "gameLabel", "gameType", "seriesGameNumber",
        "playerteamCity", "playerteamName",
        "opponentteamCity", "opponentteamName",
        "HomeTeamAbbrev", "AwayTeamAbbrev",
        "Prop_Type", "_player_key"
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(*existing_drop)

    numeric_cols = [name for name, dtype in df.dtypes if dtype in ("double", "float", "int", "bigint", "smallint")]
    if numeric_cols:
        df = df.fillna(0, subset=numeric_cols)

    # LOGICAL COLUMN ORDER for Gemini model comprehension:
    # 1. GAME CONTEXT - What game/matchup are we analyzing?
    # 2. PLAYER IDENTITY - Who is the player?
    # 3. PROP INFO - What are we betting on?
    # 4. MODEL INSIGHTS - Quick interpretations for the model
    # 5. PLAYER RECENT FORM - Most important for predictions (L5, L10, Season)
    # 6. HOME/AWAY SPLITS - Venue-specific performance
    # 7. PLAYER VARIANCE & FORM - Consistency signals
    # 8. PLAYER ADVANCED STATS - Efficiency metrics  
    # 9. TEAM CONTEXT - Rest, pace, defensive environment
    # 10. OPPONENT CONTEXT - What they allow/how they defend
    # 11. ODDS & SCORING - Market info and model calculations
    desired_order = [
        # === 1. GAME CONTEXT ===
        "gameId", "game_date", "gameDateTimeEst",
        
        # === 2. MATCHUP ===
        "TeamAbbrev", "OpponentAbbrev", "Home_Team", "Away_Team", "home",
        
        # === 3. PLAYER IDENTITY ===
        "personId", "firstName", "lastName", "FullName",
        
        # === 4. PROP INFO (What we're betting on) ===
        "Prop_Type_Normalized", "Line",  # Only Line - no odds features (avoid model shortcut)
        
        # === 5. PLAYER RECENT PERFORMANCE (Key signals!) ===
        # L5 (Last 5 games - most recent form)
        "L5_Avg_PTS", "L5_Avg_REB", "L5_Avg_AST", "L5_Avg_3PM",
        "L5_Avg_BLK", "L5_Avg_STL", "L5_Avg_MIN",
        # L10 (Last 10 games - medium-term trend)
        "L10_Avg_PTS", "L10_Avg_REB", "L10_Avg_AST", "L10_Avg_3PM",
        "L10_Avg_BLK", "L10_Avg_STL", "L10_Avg_MIN",
        # Season (Overall baseline)
        "Season_Avg_PTS", "Season_Avg_REB", "Season_Avg_AST", "Season_Avg_3PM",
        "Season_Avg_BLK", "Season_Avg_STL", "Season_Avg_MIN",
        
        # === 7. HOME/AWAY SPLITS (Venue-specific performance) ===
        "L5_Avg_PTS_SameVenue", "L5_Avg_REB_SameVenue", "L5_Avg_AST_SameVenue", "L5_Avg_3PM_SameVenue",
        "Season_Avg_PTS_SameVenue", "Season_Avg_REB_SameVenue", "Season_Avg_AST_SameVenue", "Season_Avg_3PM_SameVenue",
        
        # === 8. PLAYER VARIANCE & FORM (Consistency signals) ===
        "L5_StdDev_PTS", "L5_StdDev_REB", "L5_StdDev_AST", "L5_StdDev_3PM",
        "L10_StdDev_PTS", "L10_StdDev_REB", "L10_StdDev_AST", "L10_StdDev_3PM",
        "Form_PTS", "Form_REB", "Form_AST", "Form_3PM",  # L5 - Season delta
        
        # === 9. PLAYER ADVANCED STATS (Efficiency) ===
        "Player_Poss", "Player_TS_Pct", "Player_eFG_Pct",
        "Player_3PA_Rate", "Player_FT_Rate",
        "Player_PTS_per100", "Player_REB_per100", "Player_AST_per100", "Player_3PM_per100",
        
        # === 10. TEAM CONTEXT ===
        "teamId", "rest_days", "is_b2b",  # Removed seasonWins/Losses (data leakage risk)
        "L5_Team_Pace", "L10_Team_Pace", "Season_Team_Pace",
        "L5_Team_Def_Rating", "L10_Team_Def_Rating", "Season_Team_Def_Rating",
        "L5_Allowed_PTS", "L10_Allowed_PTS", "Season_Allowed_PTS",
        "L5_Allowed_REB", "L10_Allowed_REB", "Season_Allowed_REB",
        "L5_Allowed_AST", "L10_Allowed_AST", "Season_Allowed_AST",
        "L5_Allowed_3PA", "L10_Allowed_3PA", "Season_Allowed_3PA",
        "L5_Opp_Def_Rating", "L10_Opp_Def_Rating", "Season_Opp_Def_Rating",
        "Team_Def_Rank_Game",
        "Rank_Allows_PTS", "Rank_Allows_REB", "Rank_Allows_AST", "Rank_Allows_3PA",
        
        # === 11. OPPONENT CONTEXT (What they allow) ===
        "Opp_rest_days", "Opp_is_b2b",  # Removed Opp_seasonWins/Losses (data leakage risk)
        "Opp_L5_Team_Pace", "Opp_L10_Team_Pace", "Opp_Season_Team_Pace",
        "Opp_L5_Team_Def_Rating", "Opp_L10_Team_Def_Rating", "Opp_Season_Team_Def_Rating",
        "Opp_L5_Allowed_PTS", "Opp_L10_Allowed_PTS", "Opp_Season_Allowed_PTS",
        "Opp_L5_Allowed_REB", "Opp_L10_Allowed_REB", "Opp_Season_Allowed_REB",
        "Opp_L5_Allowed_AST", "Opp_L10_Allowed_AST", "Opp_Season_Allowed_AST",
        "Opp_L5_Allowed_3PA", "Opp_L10_Allowed_3PA", "Opp_Season_Allowed_3PA",
        "Opp_L5_Opp_Def_Rating", "Opp_L10_Opp_Def_Rating", "Opp_Season_Opp_Def_Rating",
        "Opp_Team_Def_Rank_Game",
        "Opp_Rank_Allows_PTS", "Opp_Rank_Allows_REB", "Opp_Rank_Allows_AST", "Opp_Rank_Allows_3PA"
        
        # NOTE: Removed Section 12 (ODDS CALCULATIONS) entirely
        # Features removed: Edge, Z_Score, EV, Kelly, quality_score, Implied_Over_Pct, 
        # Decimal_Odds, Over_Odds, Model_Prob_Over, Projection_Mean, Projection_Std
        # These give the model the "easy way out" and prevent true generalization
    ]

    ordered = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    if ordered:
        df = df.select(*ordered, *remaining)
    return df


# ============================================================================
# MODULE 5: FINAL DATASET CONSTRUCTION
# ============================================================================

def build_features_dataset(
    player_df: DataFrame,
    team_df: DataFrame,
    odds_df: DataFrame = None
) -> DataFrame:
    """
    Builds the FEATURES dataset for Gemini predictions.
    
    CRITICAL: This dataset does NOT contain actual game outcomes.
    Only contains information available BEFORE the game.
    
    When odds are provided, creates ONE ROW PER PLAYER-PROP combination:
    - Player A + POINTS line
    - Player A + REBOUNDS line
    - Player A + ASSISTS line
    etc.
    
    Args:
        player_df: Player statistics with engineered features
        team_df: Team statistics with engineered features
        odds_df: Optional odds data (creates prop-level rows when provided)
        
    Returns:
        Features DataFrame (no actuals)
    """
    print("\n🔨 Building FEATURES dataset (no actuals)...")
    
    # Select feature columns from player_df (exclude actual stats)
    feature_cols = [
        # Identification
        "gameId", "game_date", "gameDateTimeEst", "Join_Key",
        "personId", "firstName", "lastName",
        "TeamAbbrev", "OpponentAbbrev", "home",

        # Engineered features (from PREVIOUS games only)
        "L5_Avg_PTS", "L5_Avg_REB", "L5_Avg_AST", "L5_Avg_3PM",
        "L5_Avg_BLK", "L5_Avg_STL", "L5_Avg_MIN",
        "L10_Avg_PTS", "L10_Avg_REB", "L10_Avg_AST", "L10_Avg_3PM",
        "L10_Avg_BLK", "L10_Avg_STL", "L10_Avg_MIN",
        "Season_Avg_PTS", "Season_Avg_REB", "Season_Avg_AST", "Season_Avg_3PM",
        "Season_Avg_BLK", "Season_Avg_STL", "Season_Avg_MIN",
        
        # HOME/AWAY SPLITS (performance in same venue type)
        "L5_Avg_PTS_SameVenue", "L5_Avg_REB_SameVenue", "L5_Avg_AST_SameVenue", "L5_Avg_3PM_SameVenue",
        "Season_Avg_PTS_SameVenue", "Season_Avg_REB_SameVenue", "Season_Avg_AST_SameVenue", "Season_Avg_3PM_SameVenue",
        
        "L10_StdDev_PTS",
        # Player advanced metrics
        "Player_Poss",
        "Player_TS_Pct", "Player_eFG_Pct",
        "Player_3PA_Rate", "Player_FT_Rate",
        "Player_PTS_per100", "Player_REB_per100", "Player_AST_per100", "Player_3PM_per100",
        "L5_StdDev_PTS", "L10_StdDev_PTS",
        "L5_StdDev_REB", "L10_StdDev_REB",
        "L5_StdDev_AST", "L10_StdDev_AST",
        "L5_StdDev_3PM", "L10_StdDev_3PM",
        "Form_PTS", "Form_REB", "Form_AST", "Form_3PM"
    ]
    
    # Only select columns that exist
    existing_cols = [c for c in feature_cols if c in player_df.columns]
    features_df = player_df.select(existing_cols)
    
    # Join with team stats for rest days and team/opponent context (rolling, no leakage)
    # NOTE: Removed seasonWins/seasonLosses - they can leak information about season progress
    team_cols = [
        "gameId", "teamId", "TeamAbbrev", "rest_days", "is_b2b",
        "L5_Team_Pace", "L10_Team_Pace", "Season_Team_Pace",
        "L5_Team_Def_Rating", "L10_Team_Def_Rating", "Season_Team_Def_Rating",
        "L5_Allowed_PTS", "L10_Allowed_PTS", "Season_Allowed_PTS",
        "L5_Allowed_REB", "L10_Allowed_REB", "Season_Allowed_REB",
        "L5_Allowed_AST", "L10_Allowed_AST", "Season_Allowed_AST",
        "L5_Allowed_3PA", "L10_Allowed_3PA", "Season_Allowed_3PA",
        "L5_Opp_Def_Rating", "L10_Opp_Def_Rating", "Season_Opp_Def_Rating",
        "Team_Def_Rank_Game",
        # Opponent defensive rankings (1 = allows most of that stat)
        "Rank_Allows_PTS", "Rank_Allows_REB", "Rank_Allows_AST", "Rank_Allows_3PA"
    ]
    existing_team_cols = [c for c in team_cols if c in team_df.columns]
    # Alias join keys to avoid ambiguity after join
    team_subset_raw = team_df.select(existing_team_cols).dropDuplicates(["gameId", "TeamAbbrev"])
    team_subset = team_subset_raw.select(
        *[col(c).alias(f"_team_{c}") if c in ("gameId", "TeamAbbrev") else col(c) for c in existing_team_cols]
    )
    
    features_df = features_df.join(
        team_subset,
        (features_df["gameId"] == team_subset["_team_gameId"]) &
        (features_df["TeamAbbrev"] == team_subset["_team_TeamAbbrev"]),
        "left"
    ).drop("_team_gameId", "_team_TeamAbbrev")
    
    # Opponent context (join on opponent abbrev)
    opp_cols = [c for c in team_cols if c != "teamId"]
    # Alias join keys; rename other columns to Opp_* for clarity
    opp_subset_raw = team_df.select(
        [col(c) for c in opp_cols if c in team_df.columns]
    ).dropDuplicates(["gameId", "TeamAbbrev"])
    opp_subset = opp_subset_raw.select(
        *[
            col(c).alias("_opp_gameId") if c == "gameId"
            else col(c).alias("_opp_TeamAbbrev") if c == "TeamAbbrev"
            else col(c).alias(f"Opp_{c}")
            for c in opp_cols if c in team_df.columns
        ]
    )
    
    features_df = features_df.join(
        opp_subset,
        (features_df["gameId"] == opp_subset["_opp_gameId"]) &
        (features_df["OpponentAbbrev"] == opp_subset["_opp_TeamAbbrev"]),
        "left"
    ).drop("_opp_gameId", "_opp_TeamAbbrev")
    
    # Join with odds if available - creates ONE ROW PER PLAYER-PROP
    if odds_df is not None and odds_df.count() > 0:
        print(f"   📊 Joining with {odds_df.count():,} prop lines from odds data...")
        
        # Create full player name in features for joining
        features_df = features_df.withColumn(
            "FullName",
            concat_ws(" ", col("firstName"), col("lastName"))
        )

        # Trim odds columns and defensively drop odds-side identifiers that can
        # collide (e.g., some feeds include a gameId). We only keep the fields
        # required for the join + scoring.
        odds_required_cols = [
            "Join_Key", "Player", "Prop_Type", "Line", "Over_Odds",
            "Home_Team", "Away_Team"
        ]
        odds_existing_cols = [c for c in odds_required_cols if c in odds_df.columns]
        odds_trimmed = odds_df.select(*[col(c) for c in odds_existing_cols])
        if "gameId" in odds_trimmed.columns:
            odds_trimmed = odds_trimmed.drop("gameId")
        
        # Join features with odds on Join_Key (game) + Player name
        # This creates one row per player-prop combination
        features_with_odds = features_df.join(
            odds_trimmed.select(
                col("Join_Key").alias("Odds_Join_Key"),
                col("Player").alias("Odds_Player"),
                col("Prop_Type"),
                col("Line"),
                col("Over_Odds"),
                col("Home_Team"),
                col("Away_Team")
            ),
            (features_df["Join_Key"] == col("Odds_Join_Key")) & 
            (features_df["FullName"] == col("Odds_Player")),
            "inner"  # Only keep players with odds
        ).drop("Odds_Join_Key", "Odds_Player")

        # Normalize prop type for downstream scoring
        features_with_odds = features_with_odds.withColumn(
            "Prop_Type_Normalized",
            when(col("Prop_Type").isin("POINTS", "PTS"), lit("PTS"))
            .when(col("Prop_Type").isin("REBOUNDS", "REB"), lit("REB"))
            .when(col("Prop_Type").isin("ASSISTS", "AST"), lit("AST"))
            .when(col("Prop_Type").isin("3PM", "THREES"), lit("3PM"))
            .otherwise(col("Prop_Type"))
        )
        
        # NOTE: Removed Implied_Over_Pct and Edge calculations
        # These market signal features give the model shortcuts and prevent generalization
        # Only Line is kept as the betting context
        
        # Drop Over_Odds column to prevent it from being used
        if "Over_Odds" in features_with_odds.columns:
            features_with_odds = features_with_odds.drop("Over_Odds")
        
        if features_with_odds.count() > 0:
                print(f"   ✅ Features dataset: {features_with_odds.count():,} player-prop records (with odds)")
                return clean_and_reorder_features(features_with_odds)
        else:
            print("   ⚠️ No matches between features and odds - returning base features")
    
    print(f"   ✅ Features dataset: {features_df.count():,} records (without odds)")
    return clean_and_reorder_features(features_df)


def build_actuals_dataset(player_df: DataFrame) -> DataFrame:
    """
    Builds the ACTUALS dataset for comparison AFTER predictions.
    
    This contains the ground truth that is HIDDEN from Gemini until
    after predictions are locked.
    
    Args:
        player_df: Player statistics DataFrame
        
    Returns:
        Actuals DataFrame with only identification and outcome columns
    """
    print("\n📊 Building ACTUALS dataset (ground truth)...")
    
    # Select only identification and actual stats
    actual_cols = [
        # Identification (for joining with predictions)
        "gameId", "game_date", "personId", "Join_Key",
        "firstName", "lastName",
        
        # ACTUAL outcomes (THE GROUND TRUTH)
        "win",
        "numMinutes",
        "points",
        "assists",
        "blocks",
        "steals",
        "reboundsTotal",
        "reboundsDefensive",
        "reboundsOffensive",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "fieldGoalsPercentage",
        "threePointersAttempted",
        "threePointersMade",
        "threePointersPercentage",
        "freeThrowsAttempted",
        "freeThrowsMade",
        "freeThrowsPercentage",
        "foulsPersonal",
        "turnovers",
        "plusMinusPoints"
    ]
    
    existing_cols = [c for c in actual_cols if c in player_df.columns]
    actuals_df = player_df.select(existing_cols)
    
    print(f"   ✅ Actuals dataset: {actuals_df.count():,} records")
    return actuals_df


def build_training_datasets(
    spark: SparkSession,
    seasons: List[str] = ["2024-25"],
    fetch_odds: bool = True,
    date_range: Tuple[str, str] = None
) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
    """
    Main entry point: Builds both FEATURES and ACTUALS datasets.
    
    Args:
        spark: SparkSession instance
        seasons: List of seasons to include
        fetch_odds: Whether to fetch historical odds
        date_range: Optional (start_date, end_date) tuple to filter data and odds
        
    Returns:
        Tuple of (features_df, actuals_df, odds_df)
    """
    print("\n" + "="*60)
    print("🏗️  NBA DATA BUILDER - PySpark Edition")
    print("="*60)
    print(f"Seasons: {', '.join(seasons)}")
    if date_range:
        print(f"Date Range: {date_range[0]} to {date_range[1]}")
    print(f"Fetch Odds: {fetch_odds}")
    
    # Step 1: Load and filter data (loads FULL season - DO NOT filter here!)
    player_df, team_df = load_and_filter_seasons(spark, seasons)
    
    # Step 2: Normalize team names (on full data)
    player_df, team_df = normalize_team_names(player_df, team_df)
    
    # Step 3: Engineer features ON FULL DATA (rolling averages need historical context!)
    # CRITICAL: Rolling averages (L5, L10, Season) must be calculated BEFORE date filtering
    # otherwise there's no historical data to compute the averages from.
    player_df, team_df = engineer_features(player_df, team_df)
    
    # Step 3.5: NOW filter to date range AFTER feature engineering
    if date_range:
        range_start, range_end = date_range
        player_df = player_df.filter(
            (col("game_date") >= range_start) & 
            (col("game_date") <= range_end)
        )
        team_df = team_df.filter(
            (col("game_date") >= range_start) & 
            (col("game_date") <= range_end)
        )
        record_count = player_df.count()
        print(f"   📅 Filtered to date range: {record_count:,} player records")
        
        if record_count == 0:
            print("   ⚠️ No games found in this date range!")
    
    # Prepare active players for single-day use (to filter odds and features)
    single_day = date_range and date_range[0] == date_range[1]
    active_players = None
    active_players_names = None
    if single_day:
        range_start = date_range[0]
        active_players = player_df.filter(
            (col("game_date") == range_start) & (col("numMinutes") > 0)
        ).select(
            "personId",
            "Join_Key",
            concat_ws(" ", col("firstName"), col("lastName")).alias("PlayerName")
        ).dropDuplicates()
        active_players_names = active_players.select("PlayerName", "Join_Key").dropDuplicates()

    # Step 4: Fetch historical odds (ONLY for dates in the filtered range)
    odds_df = None
    if fetch_odds and ODDS_API_KEY:
        # Get unique dates from player_df (already filtered by date_range)
        dates = [row.game_date.strftime("%Y-%m-%d") 
                 for row in player_df.select("game_date").distinct().collect()]
        dates = sorted(dates)
        
        print(f"\n📜 Fetching odds for {len(dates)} dates in range")
        
        if dates:
            allowed_join_keys = [
                row.Join_Key for row in player_df.select("Join_Key").distinct().collect() if row.Join_Key
            ]
            allowed_players_set = None
            if single_day and active_players_names is not None:
                allowed_players_set = set([row.PlayerName for row in active_players_names.collect()])
            odds_df = fetch_historical_odds_batch(
                spark,
                dates,
                allowed_join_keys=allowed_join_keys,
                allowed_players=allowed_players_set,
                allowed_prop_types=["POINTS", "REBOUNDS", "ASSISTS", "3PM"],
                max_per_player=3,
                max_per_type=8,
                global_cap=75,
            )

            # Reduce odds to players who actually played that day (single-day only)
            if single_day and active_players_names is not None and odds_df.count() > 0:
                pre_active_count = odds_df.count()
                odds_df_filtered = odds_df.join(
                    active_players_names,
                    (odds_df["Join_Key"] == active_players_names["Join_Key"]) &
                    (odds_df["Player"] == active_players_names["PlayerName"]),
                    "inner"
                ).drop(active_players_names["Join_Key"]).drop(active_players_names["PlayerName"])
                post_active_count = odds_df_filtered.count()
                print(f"   🧹 Active-player filter (single day): {pre_active_count} → {post_active_count}")

                if post_active_count < pre_active_count:
                    unmatched_sample = (
                        odds_df.join(
                            active_players_names,
                            (odds_df["Join_Key"] == active_players_names["Join_Key"]) &
                            (odds_df["Player"] == active_players_names["PlayerName"]),
                            "left_anti"
                        )
                        .select("Player")
                        .distinct()
                        .limit(10)
                        .collect()
                    )
                    sample_names = [row.Player for row in unmatched_sample]
                    if sample_names:
                        print(f"   ⚠️ Sample unmatched player labels from odds: {sample_names}")

                odds_df = odds_df_filtered
    
    # Step 5: Build final datasets
    features_df = build_features_dataset(player_df, team_df, odds_df)
    
    # Filter to players who actually played on the target date (when a single-day range is provided)
    if single_day and active_players is not None:
        features_df = features_df.join(active_players.select("personId", "Join_Key"), on=["personId", "Join_Key"], how="inner")
    actuals_df = build_actuals_dataset(player_df)
    
    print("\n" + "="*60)
    print("✅ DATA BUILD COMPLETE")
    print("="*60)
    print(f"Features dataset: {features_df.count():,} records (NO actuals)")
    print(f"Actuals dataset: {actuals_df.count():,} records (ground truth)")
    if odds_df:
        print(f"Odds dataset: {odds_df.count():,} prop lines")
    
    return features_df, actuals_df, odds_df


# ============================================================================
# MODULE 6: DATASET EXPORT
# ============================================================================

def save_datasets(
    features_df: DataFrame,
    actuals_df: DataFrame,
    odds_df: Optional[DataFrame] = None,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Saves datasets to CSV files.
    
    Args:
        features_df: Features DataFrame
        actuals_df: Actuals DataFrame
        odds_df: Optional odds DataFrame
        output_dir: Output directory (defaults to DATASETS_DIR)
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = output_dir or DATASETS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths = {}
    
    # Save features (for Gemini predictions)
    features_path = os.path.join(output_dir, f"features_{timestamp}.csv")
    features_df.toPandas().to_csv(features_path, index=False)
    paths["features"] = features_path
    print(f"📁 Saved features to: {features_path}")
    
    # Save actuals (ground truth - HIDDEN until reveal)
    actuals_path = os.path.join(output_dir, f"actuals_{timestamp}.csv")
    actuals_df.toPandas().to_csv(actuals_path, index=False)
    paths["actuals"] = actuals_path
    print(f"📁 Saved actuals to: {actuals_path}")
        
    # Save odds if available
    if odds_df is not None and odds_df.count() > 0:
        odds_path = os.path.join(output_dir, f"odds_{timestamp}.csv")
        odds_df.toPandas().to_csv(odds_path, index=False)
        paths["odds"] = odds_path
        print(f"📁 Saved odds to: {odds_path}")
    
    return paths


# ============================================================================
# MODULE 6.5: SUPER DATASET BUILDER
# ============================================================================

def load_local_odds(spark: SparkSession, season: str = "2024-25") -> Optional[DataFrame]:
    """
    Loads historical odds from local parquet file (fetched from Odds API).
    
    Transforms Odds API format to pipeline expected format:
    - game_date, player_name, prop_type, line, odds -> Join_Key, Player, Prop_Type, Line, Over_Odds
    
    Args:
        spark: SparkSession
        season: Season string (e.g., "2024-25")
        
    Returns:
        DataFrame with odds, or None if file doesn't exist
    """
    # NBA team name to abbreviation mapping
    TEAM_ABBREV = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "LA Clippers": "LAC",
        "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
        "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS"
    }
    
    odds_dir = os.path.join(DATASETS_DIR, "odds")
    season_file = season.replace("-", "_")
    odds_path = os.path.join(odds_dir, f"historical_odds_{season_file}.parquet")
    
    if not os.path.exists(odds_path):
        print(f"   ⚠️ No local odds file found for {season}")
        print(f"   Run: python scripts/fetch_historical_odds.py --season {season}")
        return None
    
    print(f"📂 Loading local odds from: {odds_path}")
    raw_odds = spark.read.parquet(odds_path)
    raw_count = raw_odds.count()
    print(f"   ✅ Loaded {raw_count:,} raw odds records")
    
    # Filter to just "Over" lines (we need one row per player-prop, not Over AND Under)
    if "side" in raw_odds.columns:
        raw_odds = raw_odds.filter(col("side") == "Over")
    
    # Create team abbreviation mapping expression
    home_abbrev_expr = col("home_team")
    away_abbrev_expr = col("away_team")
    for full_name, abbrev in TEAM_ABBREV.items():
        home_abbrev_expr = when(col("home_team") == full_name, lit(abbrev)).otherwise(home_abbrev_expr)
        away_abbrev_expr = when(col("away_team") == full_name, lit(abbrev)).otherwise(away_abbrev_expr)
    
    raw_odds = raw_odds.withColumn("HomeAbbrev", home_abbrev_expr)
    raw_odds = raw_odds.withColumn("AwayAbbrev", away_abbrev_expr)
    
    # Transform Odds API format -> Pipeline format
    # Create Join_Key: YYYY-MM-DD_HOME_AWAY format (matching player data format)
    odds_df = raw_odds.withColumn(
        "Join_Key",
        concat_ws("_",
            col("game_date"),
            col("HomeAbbrev"),
            col("AwayAbbrev")
        )
    )
    
    # Rename columns to match expected format
    odds_df = odds_df.select(
        col("Join_Key"),
        col("player_name").alias("Player"),
        upper(col("prop_type")).alias("Prop_Type"),
        col("line").alias("Line"),
        col("odds").alias("Over_Odds"),
        col("HomeAbbrev").alias("Home_Team"),
        col("AwayAbbrev").alias("Away_Team"),
        col("game_date")
    )
    
    # Aggregate by player-prop to get single line if multiple bookmakers
    from pyspark.sql.functions import first, avg as spark_avg
    odds_df = odds_df.groupBy("Join_Key", "Player", "Prop_Type", "Home_Team", "Away_Team", "game_date").agg(
        first("Line").alias("Line"),
        spark_avg("Over_Odds").cast("int").alias("Over_Odds")
    )
    
    final_count = odds_df.count()
    print(f"   ✅ Transformed to {final_count:,} unique player-prop records")
    
    return odds_df


def build_super_dataset(
    spark: SparkSession,
    season: str = "2024-25",
    use_local_odds: bool = True,
    output_path: str = None
) -> DataFrame:
    """
    Creates a clean, unified training dataset for the entire season.
    
    This is the main entry point for generating production-ready ML training data.
    
    Pipeline:
    1. Load Kaggle data (PlayerStatistics, TeamStatistics)
    2. Engineer features (rolling averages, rest days, etc.)
    3. Load local odds (from parquet, no API calls)
    4. Join features with odds (creates player-prop rows)
    5. Join with actuals for outcome labels
    6. Clean up columns (remove redundant/technical artifacts)
    7. Save to parquet
    
    Args:
        spark: SparkSession
        season: Season string (e.g., "2024-25")
        use_local_odds: Load odds from local parquet (True) or skip odds (False)
        output_path: Output path for parquet file
        
    Returns:
        Clean DataFrame ready for ML training
    """
    print("\n" + "="*60)
    print("🏆 SUPER DATASET BUILDER")
    print("="*60)
    print(f"Season: {season}")
    print(f"Use Local Odds: {use_local_odds}")
    
    # Step 1: Load and filter data
    player_df, team_df = load_and_filter_seasons(spark, [season])
    
    # Step 2: Normalize team names
    player_df, team_df = normalize_team_names(player_df, team_df)
    
    # Step 3: Engineer features (on FULL data for proper rolling windows)
    player_df, team_df = engineer_features(player_df, team_df)
    
    # Step 4: Load local odds
    odds_df = None
    if use_local_odds:
        odds_df = load_local_odds(spark, season)
    
    # Step 5: Build features dataset (joins player + team + odds)
    features_df = build_features_dataset(player_df, team_df, odds_df)
    
    # Step 6: Build actuals dataset
    actuals_df = build_actuals_dataset(player_df)
    
    # Step 7: Join features with actuals for outcome labels
    print("\n🔗 Joining features with actuals for outcome labels...")
    
    # Create FullName in actuals for joining
    actuals_with_name = actuals_df.withColumn(
        "FullName_Actual",
        concat_ws(" ", col("firstName"), col("lastName"))
    ).select(
        col("gameId").alias("actual_gameId"),
        col("personId").alias("actual_personId"),
        col("game_date").alias("actual_game_date"),
        "FullName_Actual",
        "points", "reboundsTotal", "assists", "threePointersMade"
    )
    
    # Join on personId and game_date
    super_df = features_df.join(
        actuals_with_name,
        (features_df["personId"] == actuals_with_name["actual_personId"]) &
        (features_df["game_date"] == actuals_with_name["actual_game_date"]),
        "left"
    ).drop("actual_gameId", "actual_personId", "actual_game_date", "FullName_Actual")
    
    # Step 8: Calculate outcome label based on prop type
    super_df = super_df.withColumn(
        "actual_value",
        when(col("Prop_Type_Normalized") == "PTS", col("points"))
        .when(col("Prop_Type_Normalized") == "REB", col("reboundsTotal"))
        .when(col("Prop_Type_Normalized") == "AST", col("assists"))
        .when(col("Prop_Type_Normalized") == "3PM", col("threePointersMade"))
        .otherwise(lit(None))
    )
    
    super_df = super_df.withColumn(
        "outcome",
        when(col("actual_value") > col("Line"), lit(1)).otherwise(lit(0))
    )
    
    # Step 9: COLUMN CLEANUP - Remove redundant/technical columns
    print("\n🧹 Cleaning up columns...")
    
    # Columns to remove
    remove_cols = [
        # Redundant names (keep FullName only)
        "firstName", "lastName",
        # Technical artifacts
        "Join_Key", "player_key", "b",
        # Redundant team columns (use TeamAbbrev/OpponentAbbrev + home flag)
        "Home_Team", "Away_Team", "HomeTeamAbbrev", "AwayTeamAbbrev",
        # Game metadata noise
        "gameSubLabel", "gameLabel", "gameType", "seriesGameNumber",
        "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName",
        # Prop_Type (keep Prop_Type_Normalized)
        "Prop_Type",
    ]
    
    existing_remove = [c for c in remove_cols if c in super_df.columns]
    if existing_remove:
        super_df = super_df.drop(*existing_remove)
        print(f"   Removed {len(existing_remove)} redundant columns: {existing_remove[:5]}{'...' if len(existing_remove) > 5 else ''}")
    
    # Ensure FullName exists
    if "FullName" not in super_df.columns:
        # This shouldn't happen, but handle gracefully
        super_df = super_df.withColumn("FullName", lit("Unknown"))
    
    # Step 10: Add encoded categorical features for ML
    super_df = add_player_tiering(super_df)
    super_df = add_consistency_score(super_df)
    
    # Encode categoricals
    tier_encoding = {"STAR": 4, "STARTER": 3, "ROTATION": 2, "BENCH": 1}
    reliability_encoding = {"SNIPER": 3, "STANDARD": 2, "VOLATILE": 1}
    
    tier_expr = None
    for tier, val in tier_encoding.items():
        if tier_expr is None:
            tier_expr = when(col("Player_Tier") == tier, lit(val))
        else:
            tier_expr = tier_expr.when(col("Player_Tier") == tier, lit(val))
    tier_expr = tier_expr.otherwise(lit(2))
    super_df = super_df.withColumn("Player_Tier_encoded", tier_expr)
    
    rel_expr = None
    for rel, val in reliability_encoding.items():
        if rel_expr is None:
            rel_expr = when(col("Reliability_Tag") == rel, lit(val))
        else:
            rel_expr = rel_expr.when(col("Reliability_Tag") == rel, lit(val))
    rel_expr = rel_expr.otherwise(lit(2))
    super_df = super_df.withColumn("Reliability_Tag_encoded", rel_expr)
    
    # Final count
    final_count = super_df.count()
    final_cols = len(super_df.columns)
    
    print(f"\n✅ Super Dataset built: {final_count:,} records, {final_cols} columns")
    
    # Show column categories
    print("\n📊 Column breakdown:")
    id_cols = [c for c in super_df.columns if c in ["personId", "FullName", "gameId", "game_date", "gameDateTimeEst"]]
    context_cols = [c for c in super_df.columns if c in ["home", "is_b2b", "rest_days", "TeamAbbrev", "OpponentAbbrev", "teamId"]]
    market_cols = [c for c in super_df.columns if c in ["Line", "Over_Odds", "Prop_Type_Normalized", "Implied_Over_Pct"]]
    outcome_cols = [c for c in super_df.columns if c in ["points", "reboundsTotal", "assists", "threePointersMade", "actual_value", "outcome"]]
    print(f"   Identifiers: {len(id_cols)}")
    print(f"   Context: {len(context_cols)}")
    print(f"   Market: {len(market_cols)}")
    print(f"   Outcomes: {len(outcome_cols)}")
    print(f"   Features: {final_cols - len(id_cols) - len(context_cols) - len(market_cols) - len(outcome_cols)}")
    
    # Step 11: Save to parquet
    if output_path is None:
        season_file = season.replace("-", "_")
        output_dir = os.path.join(DATASETS_DIR, "ml_training")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"super_dataset_{season_file}.parquet")
    
    print(f"\n💾 Saving to: {output_path}")
    
    # Convert to pandas and save
    pdf = super_df.toPandas()
    pdf.to_parquet(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Saved: {file_size_mb:.1f} MB")
    
    # Show outcome distribution
    over_count = (pdf["outcome"] == 1).sum()
    under_count = (pdf["outcome"] == 0).sum()
    total = len(pdf)
    print(f"\n📈 Outcome distribution:")
    print(f"   OVER:  {over_count:,} ({100*over_count/total:.1f}%)")
    print(f"   UNDER: {under_count:,} ({100*under_count/total:.1f}%)")
    
    print("\n" + "="*60)
    print("🏆 SUPER DATASET COMPLETE")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"Records: {final_count:,}")
    print(f"Columns: {final_cols}")
    print(f"Size: {file_size_mb:.1f} MB")
    print("\nNext steps:")
    print("  python scripts/train_with_tuning.py --data " + output_path)
    
    return super_df


# ============================================================================
# MODULE 7: UTILITY FUNCTIONS FOR EVALUATION LOOP
# ============================================================================

def get_features_for_date(
    features_df: DataFrame,
    target_date: str
) -> DataFrame:
    """
    Gets features for a specific game date (for prediction).
    
    Args:
        features_df: Full features DataFrame
        target_date: Date string (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame for that date
    """
    return features_df.filter(
        date_format(col("game_date"), "yyyy-MM-dd") == target_date
    )


def get_actuals_for_date(
    actuals_df: DataFrame,
    target_date: str
) -> DataFrame:
    """
    Gets actuals for a specific game date (for reveal after prediction).
    
    Args:
        actuals_df: Full actuals DataFrame
        target_date: Date string (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame for that date
    """
    return actuals_df.filter(
        date_format(col("game_date"), "yyyy-MM-dd") == target_date
    )


def compare_predictions_to_actuals(
    predictions: List[Dict],
    actuals_df: DataFrame
) -> List[Dict]:
    """
    Compares Gemini's predictions to actual results.
    
    Args:
        predictions: List of prediction dicts with {personId, prop_type, prediction, confidence}
        actuals_df: Actuals DataFrame for the same date
        
    Returns:
        List of comparison results with hit/miss status
    """
    # Convert actuals to dict for fast lookup
    actuals_dict = {}
    for row in actuals_df.collect():
        key = row["personId"]
        actuals_dict[key] = {
            "points": row["points"],
            "reboundsTotal": row["reboundsTotal"],
            "assists": row["assists"],
            "threePointersMade": row["threePointersMade"]
        }
    
    prop_to_actual_col = {
        "PTS": "points",
        "POINTS": "points",
        "REB": "reboundsTotal",
        "REBOUNDS": "reboundsTotal",
        "AST": "assists",
        "ASSISTS": "assists",
        "3PM": "threePointersMade",
        "THREES": "threePointersMade"
    }
    
    results = []
    for pred in predictions:
        player_id = pred.get("personId")
        prop_type = pred.get("prop_type", "").upper()
        predicted_over = pred.get("prediction", "").upper() == "OVER"
        line = pred.get("line", 0)
        confidence = pred.get("confidence", 0)
        
        actual_col = prop_to_actual_col.get(prop_type)
        if not actual_col or player_id not in actuals_dict:
            continue
        
        actual_value = actuals_dict[player_id].get(actual_col, 0)
        actual_over = actual_value > line
        
        hit = (predicted_over == actual_over)
        
        results.append({
            **pred,
            "actual_value": actual_value,
            "actual_over": actual_over,
            "hit": hit,
            "is_high_confidence_miss": (confidence >= 75 and not hit)
        })
    
    return results


# ============================================================================
# MODULE 8: BACKWARD COMPATIBILITY - LIVE PIPELINE FUNCTIONS
# ============================================================================
# These functions maintain backward compatibility with the existing
# nba_ai_pipeline.py for LIVE (production) predictions.

import pandas as pd
from datetime import timezone
from zoneinfo import ZoneInfo


def fetch_live_odds() -> pd.DataFrame:
    """
    Fetches upcoming games and player props from The Odds API.
    Used for PRODUCTION mode - gets current odds for today's games.
    
    Returns:
        DataFrame with player props including lines and implied probabilities.
    """
    if not ODDS_API_KEY:
        print("WARNING: ODDS_API_KEY not found.")
        return pd.DataFrame()

    print("🔴 LIVE MODE: Fetching current odds from The Odds API...")
    
    # Get upcoming games
    games_response = requests.get(
        f"{ODDS_BASE_URL}/sports/basketball_nba/events",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
    )
    
    if games_response.status_code != 200:
        print(f"Error fetching games: {games_response.text}")
        return pd.DataFrame()
        
    games = games_response.json()
    
    # Filter to today's games only (Eastern time)
    eastern = ZoneInfo('America/New_York')
    today_local = datetime.now(eastern).date()
    
    todays_games = []
    for g in games:
        game_time_utc = datetime.strptime(g['commence_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        game_time_local = game_time_utc.astimezone(eastern)
        if game_time_local.date() == today_local:
            todays_games.append(g)
    
    game_ids = [g['id'] for g in todays_games]
    print(f"📅 Found {len(games)} total games, {len(game_ids)} scheduled for today ({today_local})")
    
    # Fetch spreads and totals for blowout risk assessment
    game_lines = _fetch_game_lines()
    
    # Fetch player props
    all_props = []
    seen_props = set()
    market_types = ["player_points", "player_rebounds", "player_assists", "player_threes"]
    
    print(f"Fetching props for {len(game_ids)} games...")
    for game_id in tqdm(game_ids, desc="Fetching Odds"):
        for market_type in market_types:
            props = _fetch_market_props(game_id, market_type, game_lines, seen_props)
            all_props.extend(props)
    
    df = pd.DataFrame(all_props)
    print(f"✅ Fetched {len(all_props)} unique props")
    return df


def _fetch_game_lines() -> Dict:
    """Fetches spreads and totals for all games."""
    game_lines = {}
    
    spreads_response = requests.get(
        f"{ODDS_BASE_URL}/sports/basketball_nba/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "decimal"
        }
    )
    
    if spreads_response.status_code == 200:
        for game in spreads_response.json():
            game_key = f"{game['home_team']}|{game['away_team']}"
            game_lines[game_key] = {'spread': None, 'total': None}
            
            for book in game.get('bookmakers', [])[:1]:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == game['home_team']:
                                game_lines[game_key]['spread'] = abs(outcome.get('point', 0))
                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                game_lines[game_key]['total'] = outcome.get('point', 0)

    return game_lines
    
    
def _fetch_market_props(game_id: str, market_type: str, game_lines: Dict, seen_props: set) -> List[Dict]:
    """Fetches props for a specific market type."""
    props = []
    
    response = requests.get(
        f"{ODDS_BASE_URL}/sports/basketball_nba/events/{game_id}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": market_type,
            "oddsFormat": "decimal"
        }
    )
    
    if response.status_code != 200:
        return props
    
    data = response.json()
    game_info = {
        "home_team": data.get('home_team', 'TBD'),
        "away_team": data.get('away_team', 'TBD')
    }
    
    for bookmaker in data.get('bookmakers', []):
        for market_data in bookmaker.get('markets', []):
            for outcome in market_data.get('outcomes', []):
                player_name = outcome.get('description') 
                bet_type = outcome.get('name')
                
                if not player_name or bet_type != 'Over':
                    continue
                
                prop_type = market_data.get('key', '').replace("player_", "").upper().replace("THREES", "FG3M")
                prop_key = f"{player_name}|{prop_type}|{outcome.get('point')}"
                
                if prop_key in seen_props:
                    continue
                seen_props.add(prop_key)
                
                game_key = f"{game_info['home_team']}|{game_info['away_team']}"
                
                row = {
                    "Player": player_name,
                    "Team": "TBD",
                    "Opponent": "TBD",
                    "Location": "TBD",
                    "Prop": prop_type,
                    "Line": outcome.get('point'),
                    "Over_Odds": outcome.get('price'),
                    "Implied_Over": (1 / outcome.get('price')) * 100 if outcome.get('price') else 0,
                    "Bookmaker": bookmaker.get('title', 'Unknown'),
                    "_Home_Team": game_info['home_team'],
                    "_Away_Team": game_info['away_team'],
                    "Vegas_Spread": game_lines.get(game_key, {}).get('spread'),
                    "Game_Total": game_lines.get(game_key, {}).get('total'),
                    "Line_Source": "LIVE_ODDS"
                }
                props.append(row)
    
    return props




# ============================================================================
# PLAYER TIERING AND CONSISTENCY SCORING
# ============================================================================

def add_player_tiering(df: DataFrame) -> DataFrame:
    """
    Adds Player_Tier column using dynamic percentiles calculated from the day's slate.
    
    Tiers are based on Season_Avg_PTS and Season_Avg_MIN percentiles:
    - STAR: Top 20% in both PTS and MIN (elite producers with guaranteed volume)
    - STARTER: Top 40% in minutes (but not STAR)
    - ROTATION: Top 60% in minutes (role players with some opportunity)
    - BENCH: Bottom 40% in minutes (unreliable volume)
    
    Args:
        df: DataFrame with Season_Avg_PTS and Season_Avg_MIN columns
        
    Returns:
        DataFrame with Player_Tier column added
    """
    # Step A: Calculate percentiles for Season_Avg_PTS and Season_Avg_MIN
    # Using approx_percentile for efficiency on large datasets
    percentiles = df.select(
        expr("percentile_approx(Season_Avg_PTS, 0.8)").alias("p80_pts"),
        expr("percentile_approx(Season_Avg_MIN, 0.8)").alias("p80_min"),
        expr("percentile_approx(Season_Avg_MIN, 0.6)").alias("p60_min"),
        expr("percentile_approx(Season_Avg_MIN, 0.4)").alias("p40_min")
    ).collect()[0]
    
    p80_pts = float(percentiles["p80_pts"]) if percentiles["p80_pts"] else 15.0
    p80_min = float(percentiles["p80_min"]) if percentiles["p80_min"] else 30.0
    p60_min = float(percentiles["p60_min"]) if percentiles["p60_min"] else 25.0
    p40_min = float(percentiles["p40_min"]) if percentiles["p40_min"] else 20.0
    
    print(f"   📊 Tier thresholds: STAR(PTS>{p80_pts:.1f} & MIN>{p80_min:.1f}), "
          f"STARTER(MIN>{p60_min:.1f}), ROTATION(MIN>{p40_min:.1f})")
    
    # Step B: Assign tiers based on percentile thresholds
    df = df.withColumn("Player_Tier",
        when((col("Season_Avg_PTS") > lit(p80_pts)) & (col("Season_Avg_MIN") > lit(p80_min)), lit("STAR"))
        .when(col("Season_Avg_MIN") > lit(p60_min), lit("STARTER"))
        .when(col("Season_Avg_MIN") > lit(p40_min), lit("ROTATION"))
        .otherwise(lit("BENCH"))
    )
    
    return df


def add_consistency_score(df: DataFrame) -> DataFrame:
    """
    Adds Reliability_Tag column based on player consistency.
    
    Uses CV (Coefficient of Variation) = StdDev / Mean internally to measure
    how volatile a player is relative to their output.
    Lower CV = more consistent, higher CV = more boom-or-bust.
    
    NOTE: CV_Score is computed internally but NOT kept as a feature.
    This prevents the model from using CV as a shortcut.
    
    Reliability Tags (based on internal CV calculation):
    - SNIPER (CV < 0.25): Highly consistent, trust projections
    - STANDARD (CV 0.25-0.45): Normal variance
    - VOLATILE (CV > 0.45): Boom-or-bust, lottery ticket
    
    Args:
        df: DataFrame with StdDev and Avg columns for each stat type
        
    Returns:
        DataFrame with Reliability_Tag column (no CV_Score)
    """
    # Calculate CV internally (temporary column, will be dropped)
    df = df.withColumn("_cv_internal",
        when(col("Prop_Type_Normalized") == "PTS",
            when(col("Season_Avg_PTS") > 0, 
                 coalesce(col("L5_StdDev_PTS"), col("L10_StdDev_PTS"), lit(5.0)) / col("Season_Avg_PTS")
            ).otherwise(lit(1.0)))
        .when(col("Prop_Type_Normalized") == "REB",
            when(col("Season_Avg_REB") > 0,
                 coalesce(col("L5_StdDev_REB"), col("L10_StdDev_REB"), lit(2.5)) / col("Season_Avg_REB")
            ).otherwise(lit(1.0)))
        .when(col("Prop_Type_Normalized") == "AST",
            when(col("Season_Avg_AST") > 0,
                 coalesce(col("L5_StdDev_AST"), col("L10_StdDev_AST"), lit(2.0)) / col("Season_Avg_AST")
            ).otherwise(lit(1.0)))
        .when(col("Prop_Type_Normalized") == "3PM",
            when(col("Season_Avg_3PM") > 0,
                 coalesce(col("L5_StdDev_3PM"), col("L10_StdDev_3PM"), lit(1.0)) / col("Season_Avg_3PM")
            ).otherwise(lit(1.0)))
        .otherwise(lit(1.0))
    )
    
    # Assign reliability tags based on CV thresholds
    df = df.withColumn("Reliability_Tag",
        when(col("_cv_internal") < 0.25, lit("SNIPER"))      # High safety, consistent
        .when(col("_cv_internal") < 0.45, lit("STANDARD"))   # Normal variance
        .otherwise(lit("VOLATILE"))                          # Boom-or-bust
    )
    
    # Drop internal CV column - not exposed as a feature
    df = df.drop("_cv_internal")
    
    return df


def apply_quality_gates(df: DataFrame, strict: bool = False) -> DataFrame:
    """
    Apply quality gate filtering rules.
    
    Args:
        df: DataFrame with Player_Tier, Reliability_Tag columns
        strict: If True, apply stricter filters (for live betting). 
                If False (default), only apply minimal filters (for training data).
        
    Returns:
        Filtered DataFrame
    """
    initial_count = df.count()
    
    if strict:
        # STRICT MODE: For live betting, be more selective
        # Rule 1: Exclude BENCH players (unreliable playing time)
        df = df.filter(col("Player_Tier") != "BENCH")
        after_bench = df.count()
        
        # Rule 2: Exclude VOLATILE players (too inconsistent)
        df = df.filter(col("Reliability_Tag") != "VOLATILE")
        after_volatile = df.count()
        
        print(f"   🚦 Quality gates (STRICT): {initial_count} → {after_bench} (no BENCH) → "
              f"{after_volatile} (no VOLATILE)")
    else:
        # RELAXED MODE: For training data, maximize samples
        # Only filter out truly invalid data (no restrictions on tier/reliability)
        # This gives us more training data to learn from
        print(f"   🚦 Quality gates (RELAXED): {initial_count} → {initial_count} (no tier/reliability filters)")
    
    return df


def sample_training_data(df: DataFrame, total_props: int = 30) -> DataFrame:
    """
    Simple proportional sampling for training data.
    
    NO quality filtering - just samples evenly across games and prop types.
    This ensures unbiased training data that represents all games.
    
    Strategy:
    1. Distribute props across all games proportionally
    2. Within each game, balance across prop types (PTS, REB, AST, 3PM)
    3. No filtering based on quality scores, tiers, or reliability
    
    Args:
        df: Features DataFrame with player props
        total_props: Target total props to sample (default 30)
        
    Returns:
        DataFrame with ~total_props samples, distributed across all games
    """
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, col, rand, lit
    
    raw_count = df.count()
    print(f"\n   📊 TRAINING DATA SAMPLING (No Quality Filters)")
    print(f"      Raw input: {raw_count} props")
    
    # Identify game column
    game_col = None
    if "Join_Key" in df.columns:
        game_col = "Join_Key"
    elif "gameId" in df.columns:
        game_col = "gameId"
    
    if game_col is None:
        print(f"      ⚠️ No game column found, taking random sample")
        return df.orderBy(rand()).limit(total_props)
    
    # Count unique games
    num_games = df.select(game_col).distinct().count()
    print(f"      Games found: {num_games}")
    
    if num_games == 0:
        return df.limit(total_props)
    
    # Calculate props per game (rounded up to ensure coverage)
    props_per_game = max(4, (total_props + num_games - 1) // num_games)  # Ceiling division
    props_per_game_per_type = max(1, props_per_game // 4)
    
    print(f"      Target: {props_per_game} props/game, {props_per_game_per_type} per type/game")
    
    # Ensure Prop_Type_Normalized exists
    if "Prop_Type_Normalized" not in df.columns:
        if "Prop_Type" in df.columns:
            df = df.withColumn("Prop_Type_Normalized", col("Prop_Type"))
        else:
            df = df.withColumn("Prop_Type_Normalized", lit("PTS"))
    
    # Rank props within each game + prop type combination
    # Use random ordering for unbiased selection
    window_game_prop = Window.partitionBy(game_col, "Prop_Type_Normalized").orderBy(rand())
    df = df.withColumn("_sample_rank", row_number().over(window_game_prop))
    
    # Take top N from each game+prop combination
    sampled_df = df.filter(col("_sample_rank") <= props_per_game_per_type).drop("_sample_rank")
    
    final_count = sampled_df.count()
    
    # Show distribution
    print(f"      Sampled: {final_count} props")
    
    # Prop type distribution
    prop_dist = sampled_df.groupBy("Prop_Type_Normalized").count().collect()
    prop_str = ", ".join([f"{r['Prop_Type_Normalized']}:{r['count']}" for r in prop_dist])
    print(f"      Prop types: {prop_str}")
    
    # Game distribution
    game_dist = sampled_df.groupBy(game_col).count().collect()
    games_covered = len(game_dist)
    props_per_game_actual = [r['count'] for r in game_dist]
    avg_per_game = sum(props_per_game_actual) / len(props_per_game_actual) if props_per_game_actual else 0
    print(f"      Games covered: {games_covered}/{num_games} (avg {avg_per_game:.1f} props/game)")
    
    return sampled_df


def apply_quality_scoring_to_df(df: DataFrame, initial_limit: int = 100, final_limit: int = None, 
                                  min_final: int = 8, max_final: int = 50, games_multiplier: float = 10.0,
                                  strict_mode: bool = False) -> DataFrame:
    """
    Apply player tiering to a features DataFrame and return props (NO FILTERING).
    
    This function adds player classification (tier, reliability) but does NOT 
    create market signal features that could give the model shortcuts.
    
    REMOVED from output (to prevent model shortcuts):
    - Z_Score, EV, Kelly, quality_score
    - Implied_Over_Pct, Decimal_Odds, Over_Odds
    - Edge, Model_Prob_Over
    - CV_Score
    
    KEPT for model features:
    - Line: The betting line (needed as context)
    - Player_Tier, Reliability_Tag (and encoded versions)
    
    Args:
        df: Features DataFrame with player props and odds
        final_limit: Max props per date (default 50)
        strict_mode: Ignored (kept for API compatibility)
        
    Returns:
        DataFrame with player tiering, limited to final_limit props
    """
    # ================================================================
    # FUNNEL TRACKING - Stage 0: Raw Input
    # ================================================================
    raw_count = df.count()
    print(f"\n   📊 PROP SELECTION FUNNEL (CLEAN FEATURES ONLY):")
    print(f"      Stage 0 - Raw input: {raw_count} props")
    
    # Prop_Type_Normalized should already exist; if not, create it from Prop_Type
    if "Prop_Type_Normalized" not in df.columns and "Prop_Type" in df.columns:
        df = df.withColumn("Prop_Type_Normalized", col("Prop_Type"))
    elif "Prop_Type_Normalized" not in df.columns:
        df = df.withColumn("Prop_Type_Normalized", lit("PTS"))
    
    # Player key for deduplication
    df = df.withColumn(
        "player_key",
        coalesce(
            col("personId").cast(StringType()),
            col("FullName"),
            concat_ws(" ", col("firstName"), col("lastName"))
        )
    )
    
    scored_count = df.count()
    print(f"      Stage 1 - After setup: {scored_count} props")
    
    # ================================================================
    # PLAYER TIERING (kept for features, not filtering)
    # ================================================================
    df = add_player_tiering(df)
    df = add_consistency_score(df)
    
    # Cache for performance
    df = df.cache()
    tiered_count = df.count()
    
    # Show distribution
    tier_counts = df.groupBy("Player_Tier").count().collect()
    tier_str = ", ".join([f"{r['Player_Tier']}:{r['count']}" for r in tier_counts])
    print(f"      Stage 2 - After tiering: {tiered_count} props")
    print(f"         👤 Tiers: {tier_str}")
    
    # ================================================================
    # NO QUALITY FILTERING - We keep ALL props
    # ================================================================
    print(f"      Stage 3 - NO quality filters applied (keeping all props)")
    
    # ================================================================
    # DEDUPLICATION - One prop per player+prop_type combo
    # ================================================================
    # This is necessary to avoid duplicate bets on same player/prop
    window_player_prop = Window.partitionBy("player_key", "Prop_Type_Normalized").orderBy(
        col("Over_Odds").asc_nulls_last()  # Prefer better odds (more negative = better for OVER)
    )
    df = df.withColumn("player_prop_rank", row_number().over(window_player_prop)).filter(col("player_prop_rank") == 1).drop("player_prop_rank")
    deduped_count = df.count()
    print(f"      Stage 4 - After deduplication: {deduped_count} unique player-prop combos")
    
    # ================================================================
    # BALANCED PROP TYPE AND GAME SELECTION
    # ================================================================
    # Select props distributed across BOTH prop types AND games
    # This ensures training data represents all games, not just one
    
    if final_limit is None:
        if "Join_Key" in df.columns:
            num_games = df.select("Join_Key").distinct().count()
        else:
            num_games = max(3, deduped_count // 10)
        final_limit = int(max(min_final, min(max_final, num_games * games_multiplier)))
    
    # Count unique games
    if "Join_Key" in df.columns:
        num_games = df.select("Join_Key").distinct().count()
    elif "gameId" in df.columns:
        num_games = df.select("gameId").distinct().count()
    else:
        num_games = 1
    
    print(f"      📅 Target limit: {final_limit} props across {num_games} games")
    
    # Strategy: Select props per game first, then balance by prop type
    # This ensures ALL games are represented in the training data
    
    # Calculate props per game (with minimum of 8 per game to get good coverage)
    props_per_game = max(8, final_limit // max(1, num_games))
    props_per_game_per_type = max(2, props_per_game // 4)  # ~2 per type per game minimum
    
    print(f"      🎯 Target: ~{props_per_game} props/game, ~{props_per_game_per_type} per type/game")
    
    # Use window function to rank within each game+prop_type combination
    from pyspark.sql.window import Window
    
    # Determine game column
    game_col = "Join_Key" if "Join_Key" in df.columns else "gameId" if "gameId" in df.columns else None
    
    if game_col:
        # Rank props within each game + prop type by quality
        window_game_prop = Window.partitionBy(game_col, "Prop_Type_Normalized").orderBy(
            col("Over_Odds").asc_nulls_last(),  # Prefer better odds
            col("Line").asc_nulls_last(),       # Then lower lines (more likely to hit)
            col("personId")                      # Deterministic tiebreaker
        )
        
        df = df.withColumn("_rank_in_game_prop", row_number().over(window_game_prop))
        
        # Take top N from each game+prop combination
        selected_dfs = []
        for prop_type in ["PTS", "REB", "AST", "3PM"]:
            prop_df = df.filter(
                (col("Prop_Type_Normalized") == prop_type) & 
                (col("_rank_in_game_prop") <= props_per_game_per_type)
            )
            if prop_df.count() > 0:
                selected_dfs.append(prop_df)
        
        # Union all selections
        if selected_dfs:
            from functools import reduce
            top_df = reduce(lambda a, b: a.unionAll(b), selected_dfs)
            top_df = top_df.drop("_rank_in_game_prop")
        else:
            top_df = df.drop("_rank_in_game_prop").limit(final_limit)
        
        # If we're still under the limit, we can add more
        current_count = top_df.count()
        if current_count < final_limit:
            print(f"      ℹ️ Selected {current_count} props (under limit of {final_limit})")
    else:
        # Fallback: No game column, use old logic
        props_per_type = max(1, final_limit // 4)
        selected_dfs = []
        for prop_type in ["PTS", "REB", "AST", "3PM"]:
            prop_df = df.filter(col("Prop_Type_Normalized") == prop_type)
            prop_count = prop_df.count()
            if prop_count > 0:
                take_n = min(props_per_type, prop_count)
                selected = prop_df.orderBy(
                    col("Over_Odds").asc_nulls_last(),
                    col("personId")
                ).limit(take_n)
                selected_dfs.append(selected)
        
        if selected_dfs:
            from functools import reduce
            top_df = reduce(lambda a, b: a.unionAll(b), selected_dfs)
        else:
            top_df = df.limit(final_limit)
    
    final_count = top_df.count()
    
    # Show prop type distribution
    final_prop_types = top_df.groupBy("Prop_Type_Normalized").count().collect()
    prop_type_str = ", ".join([f"{r['Prop_Type_Normalized']}:{r['count']}" for r in final_prop_types])
    print(f"      Stage 5 - Final selection: {final_count} props")
    print(f"         🏀 Prop types: {prop_type_str}")
    
    # Show game distribution (critical for training data quality)
    game_col = "Join_Key" if "Join_Key" in top_df.columns else "gameId" if "gameId" in top_df.columns else None
    if game_col:
        final_games = top_df.select(game_col).distinct().count()
        games_dist = top_df.groupBy(game_col).count().collect()
        props_per_game_str = ", ".join([f"{r['count']}" for r in sorted(games_dist, key=lambda x: -x['count'])[:5]])
        print(f"         🎮 Games covered: {final_games} (props per game: {props_per_game_str}{'...' if len(games_dist) > 5 else ''})")
    
    # Show outcome distribution (if we have actuals - won't during generation)
    if "outcome" in top_df.columns:
        over_count = top_df.filter(col("outcome") == 1).count()
        under_count = final_count - over_count
        print(f"         📊 Outcomes: OVER:{over_count}, UNDER:{under_count} ({100*over_count/final_count:.1f}% OVER)")
    
    # ================================================================
    # FUNNEL SUMMARY
    # ================================================================
    print(f"\n   📈 FUNNEL SUMMARY: {raw_count} → {scored_count} → {tiered_count} → {deduped_count} → {final_count}")
    retention_rate = (final_count / raw_count * 100) if raw_count > 0 else 0
    print(f"   📊 Retention rate: {retention_rate:.1f}%")
    
    # Cleanup
    try:
        df.unpersist()
    except:
        pass
    
    return top_df

if __name__ == "__main__":
    import argparse
    from datetime import datetime as dt, timedelta
    
    parser = argparse.ArgumentParser(
        description="NBA Data Builder - Generate training data from Kaggle CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate training data for a single date
  python nba_data_builder.py training --date 2024-12-15

  # Generate training data for a date range
  python nba_data_builder.py training --start 2024-12-01 --end 2024-12-31

  # Generate training data for the last 7 days
  python nba_data_builder.py training --days 7

  # Preview top props for a date (no saving)
  python nba_data_builder.py preview --date 2024-12-15 --max_props 10

  # Generate features only (no odds fetching)
  python nba_data_builder.py features --date 2024-12-15 --no-odds

  # List available dates in the dataset
  python nba_data_builder.py list-dates

NOTE: Training data is saved to datasets/training_inputs/ for ML training.
      Run 'python ml_pipeline/data_preprocessor.py' to create labeled dataset.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========================================
    # training command - Generate ML training data
    # ========================================
    train_parser = subparsers.add_parser(
        "training", 
        help="Generate training data CSVs for ML",
        description="Generate training data with features and odds for ML training"
    )
    train_parser.add_argument(
        "--date", "-d",
        type=str,
        help="Single date (YYYY-MM-DD)"
    )
    train_parser.add_argument(
        "--start", "-s",
        type=str,
        help="Start date for range (YYYY-MM-DD)"
    )
    train_parser.add_argument(
        "--end", "-e",
        type=str,
        help="End date for range (YYYY-MM-DD)"
    )
    train_parser.add_argument(
        "--days",
        type=int,
        help="Generate for last N days (alternative to --start/--end)"
    )
    train_parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to use (default: 2024-25)"
    )
    train_parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip fetching odds (faster, but less data)"
    )
    train_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: datasets/training_inputs)"
    )
    train_parser.add_argument(
        "--max-props",
        type=int,
        default=50,
        help="Max props per date (default: 50)"
    )
    
    # ========================================
    # super-dataset command - Build unified training dataset
    # ========================================
    super_parser = subparsers.add_parser(
        "super-dataset",
        help="Build unified super dataset for ML training (parquet)",
        description="Creates a single clean parquet file with all training data for the season"
    )
    super_parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to build (default: 2024-25)"
    )
    super_parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip loading local odds"
    )
    super_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for parquet file"
    )
    
    # ========================================
    # features command - Generate features only
    # ========================================
    features_parser = subparsers.add_parser(
        "features",
        help="Generate feature dataset (without full training pipeline)"
    )
    features_parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Target date (YYYY-MM-DD)"
    )
    features_parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip fetching odds"
    )
    features_parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory"
    )
    
    # ========================================
    # list-dates command - Show available dates
    # ========================================
    list_parser = subparsers.add_parser(
        "list-dates",
        help="List available dates in the dataset"
    )
    list_parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to check (default: 2024-25)"
    )
    
    args = parser.parse_args()
    
    # ========================================
    # Command handlers
    # ========================================
    
    if args.command == "training":
        # Determine date range
        if args.date:
            start_date = args.date
            end_date = args.date
        elif args.start and args.end:
            start_date = args.start
            end_date = args.end
        elif args.days:
            end_date = (dt.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (dt.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        else:
            parser.error("Must specify --date, --start/--end, or --days")
        
        output_dir = args.output_dir or os.path.join(DATASETS_DIR, "training_inputs")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🏀 NBA DATA BUILDER - Training Data Generation")
        print("="*60)
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Season: {args.season}")
        print(f"Fetch Odds: {not args.no_odds}")
        print(f"Output: {output_dir}")
        print("="*60)
        
        spark = get_spark_session()
        
        try:
            # Build datasets
            features_df, actuals_df, odds_df = build_training_datasets(
                spark,
                seasons=[args.season],
                fetch_odds=not args.no_odds,
                date_range=(start_date, end_date)
            )
            
            if features_df.count() == 0:
                print("⚠️ No data found for this date range!")
                sys.exit(1)
            
            # Use simple proportional sampling - NO derivative features
            scored_df = sample_training_data(features_df, total_props=args.max_props)
            
            if scored_df.count() == 0:
                print("⚠️ No props sampled!")
                sys.exit(1)
            
            # Save training data
            pdf = scored_df.toPandas()
            
            # Clean up columns
            drop_cols = [
                "gameSubLabel", "gameLabel", "gameType", "seriesGameNumber",
                "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName",
                "HomeTeamAbbrev", "AwayTeamAbbrev", "Prop_Type", "_player_key", "player_key"
            ]
            pdf = pdf.drop(columns=[c for c in drop_cols if c in pdf.columns], errors="ignore")
            
            # Save
            ts = dt.now().strftime("%Y%m%d_%H%M%S")
            fname = f"training_data_{start_date}_{ts}.csv"
            fpath = os.path.join(output_dir, fname)
            pdf.to_csv(fpath, index=False)
            
            print("\n" + "="*60)
            print("✅ TRAINING DATA GENERATED")
            print("="*60)
            print(f"Saved: {fpath}")
            print(f"Records: {len(pdf)}")
            print(f"Columns: {len(pdf.columns)}")
            print("\nNext steps:")
            print("  1. python ml_pipeline/data_preprocessor.py  # Create labeled dataset")
            print("  2. python scripts/train_with_tuning.py     # Train models with tuning")
            print("="*60)
            
        finally:
            stop_spark_session()
    
    elif args.command == "super-dataset":
        spark = get_spark_session()
        
        try:
            build_super_dataset(
                spark,
                season=args.season,
                use_local_odds=not args.no_odds,
                output_path=args.output
            )
        finally:
            stop_spark_session()
    
    elif args.command == "features":
        output_dir = args.output_dir or DATASETS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        spark = get_spark_session()
        
        try:
            features_df, actuals_df, odds_df = build_training_datasets(
                spark,
                seasons=["2024-25"],
                fetch_odds=not args.no_odds,
                date_range=(args.date, args.date)
            )
            
            # Save
            paths = save_datasets(features_df, actuals_df, odds_df, output_dir)
            print(f"\n✅ Features saved to: {paths.get('features', 'N/A')}")
            
        finally:
            stop_spark_session()
    
    elif args.command == "list-dates":
        spark = get_spark_session()
        
        try:
            df = spark.read.csv(PLAYER_STATS_CSV, header=True, inferSchema=True)
            df = df.withColumn(
                "game_date",
                to_date(to_timestamp(col("gameDateTimeEst"), "yyyy-MM-dd HH:mm:ss"))
            )
            
            # Filter by season
            if args.season in SEASON_DATE_RANGES:
                start, end = SEASON_DATE_RANGES[args.season]
                df = df.filter((col("game_date") >= start) & (col("game_date") <= end))
            
            # Get unique dates
            dates = df.select("game_date").distinct().orderBy("game_date").collect()
            
            print(f"\n📅 Available dates in {args.season}:")
            print("-" * 40)
            for i, row in enumerate(dates):
                date_str = row.game_date.strftime("%Y-%m-%d")
                print(f"  {date_str}", end="")
                if (i + 1) % 5 == 0:
                    print()
            print(f"\n\nTotal: {len(dates)} dates")
            
        finally:
            stop_spark_session()
    
    else:
        parser.print_help()
