"""
FEATURE CONFIGURATION
Defines which features are used for each prop type model.
Based on domain expertise and feature importance analysis.
"""

# Core features used by ALL prop models
CORE_FEATURES = [
    # Context
    'home', 'is_b2b', 'rest_days',
    
    # Market signals (your existing quality scoring)
    'Edge', 'Z_Score', 'EV', 'Kelly', 'quality_score',
    'Implied_Over_Pct', 'Decimal_Odds',
    
    # Player classification
    'Player_Tier_encoded',      # STAR=3, STARTER=2, ROTATION=1, BENCH=0
    'Reliability_Tag_encoded',  # SNIPER=2, STANDARD=1, VOLATILE=0
    'CV_Score',
    
    # Minutes (affects all props)
    'L5_Avg_MIN', 'L10_Avg_MIN', 'Season_Avg_MIN',
    
    # Opponent rest/B2B (affects defensive intensity)
    'Opp_is_b2b', 'Opp_rest_days',
]

# PTS-specific features (27 total)
PTS_FEATURES = CORE_FEATURES + [
    # Recent form
    'L5_Avg_PTS', 'L10_Avg_PTS', 'Season_Avg_PTS',
    'L5_Avg_PTS_SameVenue', 'Season_Avg_PTS_SameVenue',
    'Form_PTS',  # Trend indicator
    
    # Variance (critical for prop betting)
    'L5_StdDev_PTS', 'L10_StdDev_PTS',
    
    # Efficiency metrics
    'Player_TS_Pct', 'Player_eFG_Pct', 'Player_PTS_per100',
    
    # Matchup defense
    'Opp_Rank_Allows_PTS', 'Opp_L5_Allowed_PTS', 'Opp_L10_Allowed_PTS',
    'Opp_Season_Allowed_PTS',
    'Opp_L5_Team_Def_Rating', 'Opp_L10_Team_Def_Rating',
    'Opp_Season_Team_Def_Rating',
    
    # Pace (more possessions = more points)
    'L5_Team_Pace', 'L10_Team_Pace',
    'Opp_L5_Team_Pace', 'Opp_L10_Team_Pace',
    
    # Line analysis
    'Line',  # The actual line to beat
]

# REB-specific features (25 total)
REB_FEATURES = CORE_FEATURES + [
    # Recent form
    'L5_Avg_REB', 'L10_Avg_REB', 'Season_Avg_REB',
    'L5_Avg_REB_SameVenue', 'Season_Avg_REB_SameVenue',
    'Form_REB',
    
    # Variance
    'L5_StdDev_REB', 'L10_StdDev_REB',
    
    # Advanced
    'Player_REB_per100',
    
    # Matchup defense
    'Opp_Rank_Allows_REB', 'Opp_L5_Allowed_REB', 'Opp_L10_Allowed_REB',
    'Opp_Season_Allowed_REB',
    
    # Pace
    'L5_Team_Pace', 'L10_Team_Pace',
    'Opp_L5_Team_Pace', 'Opp_L10_Team_Pace',
    
    # Line
    'Line',
]

# AST-specific features (25 total)
AST_FEATURES = CORE_FEATURES + [
    # Recent form
    'L5_Avg_AST', 'L10_Avg_AST', 'Season_Avg_AST',
    'L5_Avg_AST_SameVenue', 'Season_Avg_AST_SameVenue',
    'Form_AST',
    
    # Variance
    'L5_StdDev_AST', 'L10_StdDev_AST',
    
    # Advanced
    'Player_AST_per100',
    
    # Matchup defense
    'Opp_Rank_Allows_AST', 'Opp_L5_Allowed_AST', 'Opp_L10_Allowed_AST',
    'Opp_Season_Allowed_AST',
    
    # Pace
    'L5_Team_Pace', 'L10_Team_Pace',
    'Opp_L5_Team_Pace', 'Opp_L10_Team_Pace',
    
    # Line
    'Line',
]

# 3PM-specific features (25 total)
THREE_PM_FEATURES = CORE_FEATURES + [
    # Recent form
    'L5_Avg_3PM', 'L10_Avg_3PM', 'Season_Avg_3PM',
    'L5_Avg_3PM_SameVenue', 'Season_Avg_3PM_SameVenue',
    'Form_3PM',
    
    # Variance (3PM is inherently volatile)
    'L5_StdDev_3PM', 'L10_StdDev_3PM',
    
    # Shooting profile
    'Player_3PA_Rate', 'Player_3PM_per100',
    
    # Matchup defense
    'Opp_Rank_Allows_3PA', 'Opp_L5_Allowed_3PA', 'Opp_L10_Allowed_3PA',
    'Opp_Season_Allowed_3PA',
    
    # Pace
    'L5_Team_Pace', 'L10_Team_Pace',
    'Opp_L5_Team_Pace', 'Opp_L10_Team_Pace',
    
    # Line
    'Line',
]

# Mapping for easy access
PROP_FEATURES = {
    'PTS': PTS_FEATURES,
    'REB': REB_FEATURES,
    'AST': AST_FEATURES,
    '3PM': THREE_PM_FEATURES,
}

# Prop type to actual column mapping (for labeling)
PROP_TO_ACTUAL = {
    'PTS': 'points',
    'REB': 'reboundsTotal',
    'AST': 'assists',
    '3PM': 'threePointersMade'
}

# Categorical encoding maps
TIER_ENCODING = {'STAR': 3, 'STARTER': 2, 'ROTATION': 1, 'BENCH': 0}
RELIABILITY_ENCODING = {'SNIPER': 2, 'STANDARD': 1, 'VOLATILE': 0}
