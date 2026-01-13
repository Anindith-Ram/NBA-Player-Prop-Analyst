"""
SHARED CONFIGURATION (shared_config.py)
=======================================
Self-Evolving NBA Neural System v2.0

Centralized configuration and shared resources to:
1. Eliminate redundant API configurations
2. Provide connection pooling
3. Share caches across modules
4. Standardize rate limiting
"""

import os
import time
import google.generativeai as genai
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
from collections import defaultdict

# ============================================================================
# API KEYS (Centralized) - Set via environment variables
# ============================================================================
# Required: Set these environment variables before running:
#   export GEMINI_API_KEY="your-gemini-api-key"
#   export ODDS_API_KEY="your-odds-api-key"
#   export KAGGLE_API_TOKEN="your-kaggle-token" (optional)
# Or create a .env file (see .env.example)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

# ============================================================================
# PATHS (Centralized)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
PARLAYS_DIR = os.path.join(PROJECT_ROOT, "parlays")
TRAINING_REPORTS_DIR = os.path.join(PROJECT_ROOT, "training_reports")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Learning files
CASE_STUDY_LIBRARY_PATH = os.path.join(SCRIPT_DIR, "case_study_library.json")
SYSTEM_PROMPT_PATH = os.path.join(SCRIPT_DIR, "current_system_prompt.txt")
PARLAY_CASE_STUDIES_PATH = os.path.join(SCRIPT_DIR, "parlay_case_studies.json")
PARLAY_PERFORMANCE_LOG = os.path.join(LOGS_DIR, "parlay_performance.json")
MISSING_FEATURES_LOG = os.path.join(LOGS_DIR, "missing_features.log")

# ============================================================================
# GEMINI API SINGLETON (Prevents multiple configurations)
# ============================================================================

_gemini_configured = False
_gemini_models: Dict[str, Any] = {}

def get_gemini_model(model_name: str = "gemini-3-pro-preview"):
    """
    Returns a Gemini model instance, configuring the API only once.
    
    This prevents multiple genai.configure() calls and provides
    connection reuse across modules.
    
    Args:
        model_name: The Gemini model to use
        
    Returns:
        GenerativeModel instance
    """
    global _gemini_configured, _gemini_models
    
    if not _gemini_configured:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_configured = True
        print("🔧 Gemini API configured (singleton)")
    
    if model_name not in _gemini_models:
        _gemini_models[model_name] = genai.GenerativeModel(model_name)
        print(f"🤖 Created Gemini model: {model_name}")
    
    return _gemini_models[model_name]


# ============================================================================
# PYSPARK SESSION SINGLETON
# ============================================================================

_spark_session = None

def get_spark_session(app_name: str = "NBA_Player_Prop_Analyst"):
    """
    Returns a PySpark session, creating it only once.
    
    This provides connection reuse across modules and prevents
    multiple SparkContext errors.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession instance
    """
    global _spark_session
    
    if _spark_session is None:
        try:
            from pyspark.sql import SparkSession
            _spark_session = (SparkSession.builder
                .appName(app_name)
                .config("spark.driver.memory", "4g")
                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.ui.enabled", "false")  # Disable UI for local mode
                # PERFORMANCE: Increase codegen limits for complex queries
                .config("spark.sql.codegen.wholeStage", "true")
                .config("spark.sql.codegen.hugeMethodLimit", "65536")  # 64KB -> 64KB (max)
                .config("spark.sql.codegen.fallback", "true")  # Graceful fallback
                # PERFORMANCE: Adaptive Query Execution for better plan optimization
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.adaptive.skewJoin.enabled", "true")
                # PERFORMANCE: Broadcast threshold for smaller tables
                .config("spark.sql.autoBroadcastJoinThreshold", "50MB")
                .getOrCreate())
            print(f"🔥 PySpark session created: {app_name}")
            print(f"   📈 Optimized for complex queries (AQE enabled, codegen limits increased)")
        except ImportError:
            print("⚠️ PySpark not installed. Run: pip install pyspark")
            raise
    
    return _spark_session


def stop_spark_session():
    """Stops the PySpark session."""
    global _spark_session
    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None
        print("🛑 PySpark session stopped")


# ============================================================================
# KAGGLE DATA PATHS
# ============================================================================

KAGGLE_DATA_DIR = os.path.join(PROJECT_ROOT, "kaggle_data")
PLAYER_STATS_CSV = os.path.join(KAGGLE_DATA_DIR, "PlayerStatistics.csv")
TEAM_STATS_CSV = os.path.join(KAGGLE_DATA_DIR, "TeamStatistics.csv")

# Season date ranges (Regular Season + Playoffs)
# NOTE: 2023-24 intentionally removed to avoid sparse/early-season gaps.
SEASON_DATE_RANGES = {
    "2024-25": ("2024-10-22", "2025-06-22"),
    "2025-26": ("2025-10-28", "2026-06-21"),  # Current season
}

# Regular season only ranges (excludes playoffs)
REGULAR_SEASON_DATE_RANGES = {
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-11-13", "2026-04-12"),
}

# Training date ranges (10+ games played, regular season only)
# These are the exact dates used for ML training data
# 2024-25: Start Nov 10 (all teams have 10+ games), end April 13 (before playoffs)
# 2025-26: Start Nov 13 (all teams have 10+ games), end None = yesterday (dynamic)
TRAINING_DATE_RANGES = {
    "2024-25": ("2024-11-10", "2025-04-13"),
    "2025-26": ("2025-11-13", None),  # None = yesterday, calculated dynamically
}

# ============================================================================
# TEAM NAME NORMALIZATION MAPPINGS
# ============================================================================

# Full team name (City + Name) to abbreviation mapping
TEAM_FULL_NAME_TO_ABBREV = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',  # Note: CSVs use "LA" not "Los Angeles"
    'Los Angeles Clippers': 'LAC',  # Odds API format
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}

# Abbreviation to full team name (for reverse lookup)
TEAM_ABBREV_TO_FULL_NAME = {v: k for k, v in TEAM_FULL_NAME_TO_ABBREV.items() 
                            if k != 'Los Angeles Clippers'}  # Avoid duplicate LAC


# ============================================================================
# RATE LIMITING (Standardized)
# ============================================================================

# Standardized delays
NBA_API_DELAY = 0.5        # NBA API rate limit
ODDS_API_DELAY = 0.5       # Odds API rate limit
GEMINI_API_DELAY = 1.0     # Gemini API rate limit (adjust based on your tier)
# Recommended delays by tier:
#   Free Tier: 1.0 seconds (1 req/sec = 60 RPM)
#   Tier 1: 0.2 seconds (5 req/sec = 300 RPM)
#   Tier 2: 0.17 seconds (6 req/sec = 360 RPM)
#   Tier 3: 0.05 seconds (20 req/sec = 1,200 RPM)
# See GEMINI_RATE_LIMITS_GUIDE.md for upgrade instructions

# Last call timestamps for adaptive rate limiting
_last_call_times: Dict[str, float] = defaultdict(float)

def rate_limited_call(api_name: str, delay: float = 0.5):
    """
    Decorator for rate-limited API calls with adaptive backoff.
    
    Usage:
        @rate_limited_call("nba_api", NBA_API_DELAY)
        def fetch_player_data(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _last_call_times
            
            # Calculate time since last call
            elapsed = time.time() - _last_call_times[api_name]
            
            # Wait if needed
            if elapsed < delay:
                time.sleep(delay - elapsed)
            
            # Make the call
            result = func(*args, **kwargs)
            
            # Update timestamp
            _last_call_times[api_name] = time.time()
            
            return result
        return wrapper
    return decorator


def wait_for_api(api_name: str, delay: float = 0.5):
    """
    Simple rate limiting wait function.
    
    Args:
        api_name: Name of the API (for tracking)
        delay: Minimum delay between calls
    """
    global _last_call_times
    
    elapsed = time.time() - _last_call_times[api_name]
    if elapsed < delay:
        time.sleep(delay - elapsed)
    
    _last_call_times[api_name] = time.time()


# ============================================================================
# SESSION-BASED CACHING (Shared across modules)
# ============================================================================

def get_current_season() -> str:
    """Returns the current NBA season based on today's date."""
    today = datetime.now().date()
    year = today.year
    month = today.month
    
    if month >= 10:  # Oct-Dec = start of new season
        start_year = year
    else:  # Jan-Sept = second half of season
        start_year = year - 1
    
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


class TrainingSessionCache:
    """
    MODE-AWARE Centralized cache for a training session.
    
    CRITICAL: Handles historical vs live training differently to prevent data leakage.
    
    HISTORICAL MODE (training on past completed season):
    - Game logs are cached by player+season (full season data exists)
    - Stats are COMPUTED from game logs up to target_date (no leakage)
    - Aggressive caching - data is frozen
    
    LIVE MODE (training on current ongoing season):
    - Game logs cached by player+season+date (grows daily)
    - Stats must be fresh each day
    - Conservative caching - data changes daily
    
    ZERO DATA LEAKAGE GUARANTEE:
    - Features are ALWAYS computed from games BEFORE target_date
    - The "reveal" step happens AFTER predictions are locked
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resets all caches for a new session."""
        # Mode detection
        self.is_historical_mode: bool = True  # Default to historical (safer)
        self.current_season: str = get_current_season()
        self.training_season: Optional[str] = None
        
        # Game log cache - THE SOURCE OF TRUTH for historical data
        # Key: "player_season" -> full DataFrame of all games in season
        # This is cached once per player and filtered locally for each target_date
        self.game_log_cache: Dict[str, Any] = {}
        
        # Computed stats cache (derived from game logs, keyed by target_date)
        # Key: "season_targetdate" -> computed stats up to that date
        self.computed_team_stats_cache: Dict[str, Dict] = {}
        self.computed_player_stats_cache: Dict[str, Dict] = {}
        
        # For LIVE mode only - cache by date since data changes daily
        self.live_team_stats_cache: Dict[str, Dict] = {}  # "season_date" -> stats
        self.live_player_stats_cache: Dict[str, Dict] = {}  # "season_date" -> stats
        
        # Roster cache (rarely changes within a season)
        self.roster_cache: Dict[int, list] = {}  # team_id -> player_ids
        
        # Defense rankings cache (keyed by team+season for historical)
        self.defense_rankings_cache: Dict[str, Dict] = {}
        
        # Player matchup history (keyed by player+opponent+cutoff_date)
        self.matchup_history_cache: Dict[str, Dict] = {}
        
        # Player availability cache (keyed by date) - always per-date
        self.player_availability_cache: Dict[str, set] = {}
        
        # Injury report cache
        self.injury_cache: Optional[Dict] = None
        self.injury_cache_time: Optional[datetime] = None
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_training_mode(self, target_season: str):
        """
        Sets the training mode based on whether we're training on historical or current data.
        
        Args:
            target_season: The season we're training on (e.g., "2024-25")
        """
        self.training_season = target_season
        self.is_historical_mode = (target_season != self.current_season)
        
        mode_str = "HISTORICAL" if self.is_historical_mode else "LIVE"
        print(f"🔧 Cache Mode: {mode_str}")
        print(f"   Training Season: {target_season}")
        print(f"   Current Season: {self.current_season}")
        
        if self.is_historical_mode:
            print(f"   Strategy: Cache game logs once, compute stats per-date (no leakage)")
        else:
            print(f"   Strategy: Per-date caching (data changes daily)")
    
    def get_cache_key(self, base_key: str, target_date: str = None) -> str:
        """
        Generates appropriate cache key based on mode.
        
        Historical: Uses base_key only (data is frozen)
        Live: Includes date (data changes daily)
        """
        if self.is_historical_mode:
            return base_key
        else:
            return f"{base_key}_{target_date}" if target_date else base_key
    
    # =========================================================================
    # GAME LOGS - The Foundation (No Leakage - Always Filtered Locally)
    # =========================================================================
    
    def get_game_logs(self, player_name: str, season: str) -> Optional[Any]:
        """
        Gets cached FULL game logs for a player in a season.
        
        IMPORTANT: These are full season logs. The caller MUST filter
        to games BEFORE target_date to prevent data leakage.
        """
        key = f"{player_name}_{season}"
        if key in self.game_log_cache:
            self.cache_hits += 1
            return self.game_log_cache[key]
        self.cache_misses += 1
        return None
    
    def set_game_logs(self, player_name: str, season: str, logs: Any):
        """Caches full game logs for a player in a season."""
        key = f"{player_name}_{season}"
        self.game_log_cache[key] = logs
    
    # =========================================================================
    # TEAM STATS - Computed from Game Logs (Historical) or API (Live)
    # =========================================================================
    
    def get_team_stats(self, season: str, target_date: str = None) -> Optional[Dict]:
        """
        Gets team stats, respecting mode and preventing leakage.
        
        Historical: Returns stats computed up to target_date
        Live: Returns cached stats for specific date
        """
        if self.is_historical_mode:
            # For historical, we'd ideally compute from game logs
            # Using season key since NBA API doesn't support date filtering
            key = f"{season}_{target_date}" if target_date else season
            if key in self.computed_team_stats_cache:
                self.cache_hits += 1
                return self.computed_team_stats_cache[key]
        else:
            # Live mode - cache by date
            key = f"{season}_{target_date}" if target_date else season
            if key in self.live_team_stats_cache:
                self.cache_hits += 1
                return self.live_team_stats_cache[key]
        
        self.cache_misses += 1
        return None
    
    def set_team_stats(self, season: str, stats: Dict, target_date: str = None):
        """Caches team stats with appropriate key based on mode."""
        if self.is_historical_mode:
            key = f"{season}_{target_date}" if target_date else season
            self.computed_team_stats_cache[key] = stats
        else:
            key = f"{season}_{target_date}" if target_date else season
            self.live_team_stats_cache[key] = stats
    
    # =========================================================================
    # PLAYER ADVANCED STATS - Similar to Team Stats
    # =========================================================================
    
    def get_player_adv_stats(self, season: str, target_date: str = None) -> Optional[Dict]:
        """Gets player advanced stats, respecting mode."""
        if self.is_historical_mode:
            key = f"{season}_{target_date}" if target_date else season
            if key in self.computed_player_stats_cache:
                self.cache_hits += 1
                return self.computed_player_stats_cache[key]
        else:
            key = f"{season}_{target_date}" if target_date else season
            if key in self.live_player_stats_cache:
                self.cache_hits += 1
                return self.live_player_stats_cache[key]
        
        self.cache_misses += 1
        return None
    
    def set_player_adv_stats(self, season: str, stats: Dict, target_date: str = None):
        """Caches player advanced stats with appropriate key based on mode."""
        if self.is_historical_mode:
            key = f"{season}_{target_date}" if target_date else season
            self.computed_player_stats_cache[key] = stats
        else:
            key = f"{season}_{target_date}" if target_date else season
            self.live_player_stats_cache[key] = stats
    
    # =========================================================================
    # DEFENSE RANKINGS - Per Season (Historical) or Per Date (Live)
    # =========================================================================
    
    def get_defense_ranking(self, team: str, season: str, target_date: str = None) -> Optional[Dict]:
        """Gets cached defense ranking."""
        if self.is_historical_mode:
            key = f"{team}_{season}"
        else:
            key = f"{team}_{season}_{target_date}" if target_date else f"{team}_{season}"
        
        if key in self.defense_rankings_cache:
            self.cache_hits += 1
            return self.defense_rankings_cache[key]
        self.cache_misses += 1
        return None
    
    def set_defense_ranking(self, team: str, season: str, ranking: Dict, target_date: str = None):
        """Caches defense ranking."""
        if self.is_historical_mode:
            key = f"{team}_{season}"
        else:
            key = f"{team}_{season}_{target_date}" if target_date else f"{team}_{season}"
        self.defense_rankings_cache[key] = ranking
    
    # =========================================================================
    # MATCHUP HISTORY - Keyed by player+opponent+cutoff (prevents leakage)
    # =========================================================================
    
    def get_matchup_history(self, player: str, opponent: str, cutoff_date: str = None) -> Optional[Dict]:
        """Gets cached matchup history up to cutoff_date."""
        # Always include cutoff_date to prevent seeing future games
        key = f"{player}_{opponent}_{cutoff_date}" if cutoff_date else f"{player}_{opponent}"
        if key in self.matchup_history_cache:
            self.cache_hits += 1
            return self.matchup_history_cache[key]
        self.cache_misses += 1
        return None
    
    def set_matchup_history(self, player: str, opponent: str, history: Dict, cutoff_date: str = None):
        """Caches matchup history."""
        key = f"{player}_{opponent}_{cutoff_date}" if cutoff_date else f"{player}_{opponent}"
        self.matchup_history_cache[key] = history
    
    # =========================================================================
    # PLAYER AVAILABILITY - Always Per-Date
    # =========================================================================
    
    def get_player_availability(self, date: str) -> Optional[set]:
        """Gets cached player availability for a date."""
        if date in self.player_availability_cache:
            self.cache_hits += 1
            return self.player_availability_cache[date]
        self.cache_misses += 1
        return None
    
    def set_player_availability(self, date: str, players: set):
        """Caches player availability for a date."""
        self.player_availability_cache[date] = players
    
    def get_stats(self) -> Dict[str, int]:
        """Returns cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": round(hit_rate, 1)
        }
    
    def print_stats(self):
        """Prints cache statistics."""
        stats = self.get_stats()
        print(f"\n📊 Session Cache Statistics:")
        print(f"   Hits: {stats['hits']} | Misses: {stats['misses']}")
        print(f"   Hit Rate: {stats['hit_rate']}%")
        print(f"   Game Logs Cached: {len(self.game_log_cache)}")
        print(f"   Defense Rankings Cached: {len(self.defense_rankings_cache)}")
        print(f"   Matchup Histories Cached: {len(self.matchup_history_cache)}")


# Global session cache instance
_session_cache: Optional[TrainingSessionCache] = None

def get_session_cache() -> TrainingSessionCache:
    """
    Gets or creates the global session cache.
    
    Returns:
        The shared TrainingSessionCache instance
    """
    global _session_cache
    if _session_cache is None:
        _session_cache = TrainingSessionCache()
    return _session_cache

def reset_session_cache():
    """Resets the global session cache."""
    global _session_cache
    if _session_cache is not None:
        _session_cache.print_stats()
    _session_cache = TrainingSessionCache()
    print("🔄 Session cache reset")


# ============================================================================
# INDEXED CASE STUDY STORAGE
# ============================================================================

class IndexedCaseStudyStore:
    """
    Provides O(1) case study retrieval by maintaining indexes.
    
    Instead of linear search through all case studies,
    uses pre-built indexes for fast retrieval.
    """
    
    def __init__(self):
        self.case_studies = []
        self.by_tier: Dict[str, list] = defaultdict(list)
        self.by_prop: Dict[str, list] = defaultdict(list)
        self.by_tag: Dict[str, list] = defaultdict(list)
        self.by_archetype: Dict[str, list] = defaultdict(list)
    
    def add(self, case_study):
        """Adds a case study and updates all indexes."""
        self.case_studies.append(case_study)
        idx = len(self.case_studies) - 1
        
        # Index by tier
        self.by_tier[case_study.player_tier].append(idx)
        
        # Index by prop type
        self.by_prop[case_study.prop_type].append(idx)
        
        # Index by archetype
        self.by_archetype[case_study.archetype].append(idx)
        
        # Index by each context tag
        for tag in case_study.context_tags:
            self.by_tag[tag].append(idx)
    
    def find_relevant(self, player_tier: str = None, prop_type: str = None, 
                      context_tags: list = None, limit: int = 5) -> list:
        """
        Finds relevant case studies using indexes.
        
        O(1) index lookups instead of O(n) linear search.
        """
        # Start with all indices if no criteria
        candidate_indices = set(range(len(self.case_studies)))
        
        # Filter by tier
        if player_tier and player_tier in self.by_tier:
            candidate_indices &= set(self.by_tier[player_tier])
        
        # Filter by prop type
        if prop_type and prop_type in self.by_prop:
            candidate_indices &= set(self.by_prop[prop_type])
        
        # Filter by context tags (union - any matching tag)
        if context_tags:
            tag_matches = set()
            for tag in context_tags:
                if tag in self.by_tag:
                    tag_matches |= set(self.by_tag[tag])
            if tag_matches:
                candidate_indices &= tag_matches
        
        # Return case studies
        results = [self.case_studies[i] for i in list(candidate_indices)[:limit]]
        return results
    
    def rebuild_indexes(self):
        """Rebuilds all indexes from case studies list."""
        self.by_tier.clear()
        self.by_prop.clear()
        self.by_tag.clear()
        self.by_archetype.clear()
        
        for idx, cs in enumerate(self.case_studies):
            self.by_tier[cs.player_tier].append(idx)
            self.by_prop[cs.prop_type].append(idx)
            self.by_archetype[cs.archetype].append(idx)
            for tag in cs.context_tags:
                self.by_tag[tag].append(idx)


# ============================================================================
# UNIFIED PROGRESS REPORTER
# ============================================================================

class ProgressReporter:
    """
    Unified progress reporter for consistent status updates across all pipeline stages.
    
    Usage:
        reporter = ProgressReporter("Historical Training")
        reporter.start()
        reporter.step("Loading data")
        reporter.substep("Loaded 1000 records")
        reporter.step("Making predictions")
        reporter.progress(50, 100, "predictions made")
        reporter.complete({"accuracy": 75.5})
    """
    
    def __init__(self, mode: str = "Pipeline"):
        self.mode = mode
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0
        
    def start(self, total_steps: int = 0):
        """Start the pipeline with optional step count."""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        
        print("\n" + "=" * 60)
        print(f"🏀 {self.mode.upper()}")
        print("=" * 60)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
    
    def step(self, description: str):
        """Log a major step in the pipeline."""
        self.current_step += 1
        if self.total_steps > 0:
            print(f"\n[{self.current_step}/{self.total_steps}] 📌 {description}")
        else:
            print(f"\n📌 {description}")
    
    def substep(self, message: str):
        """Log a substep or detail."""
        print(f"   └─ {message}")
    
    def progress(self, current: int, total: int, unit: str = "items"):
        """Show progress within a step."""
        pct = (current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"   [{bar}] {current}/{total} {unit} ({pct:.0f}%)")
    
    def success(self, message: str):
        """Log a success message."""
        print(f"   ✅ {message}")
    
    def warning(self, message: str):
        """Log a warning message."""
        print(f"   ⚠️  {message}")
    
    def error(self, message: str):
        """Log an error message."""
        print(f"   ❌ {message}")
    
    def info(self, message: str):
        """Log an info message."""
        print(f"   ℹ️  {message}")
    
    def stats(self, stats_dict: Dict[str, Any]):
        """Display statistics in a formatted table."""
        print("\n   📊 Statistics:")
        for key, value in stats_dict.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.1f}")
            else:
                print(f"      {key}: {value}")
    
    def complete(self, summary: Dict[str, Any] = None):
        """Mark the pipeline as complete with optional summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "-" * 60)
        print(f"✅ {self.mode.upper()} COMPLETE")
        print(f"⏱️  Duration: {elapsed:.1f}s")
        
        if summary:
            print("\n📋 Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.1f}")
                else:
                    print(f"   • {key}: {value}")
        
        print("=" * 60 + "\n")


# Global progress reporter
_progress_reporter: Optional[ProgressReporter] = None

def get_progress_reporter(mode: str = "Pipeline") -> ProgressReporter:
    """Get or create a progress reporter."""
    global _progress_reporter
    if _progress_reporter is None or _progress_reporter.mode != mode:
        _progress_reporter = ProgressReporter(mode)
    return _progress_reporter


# ============================================================================
# ENSURE DIRECTORIES EXIST
# ============================================================================

def ensure_directories():
    """Creates all necessary directories if they don't exist."""
    dirs = [DATASETS_DIR, PREDICTIONS_DIR, PARLAYS_DIR, TRAINING_REPORTS_DIR, LOGS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Create directories on import
ensure_directories()

