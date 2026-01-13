"""
HYBRID ML + LLM PREDICTOR
=========================
Phase 1 Implementation: ML-Primary Architecture with Gemini for edge cases.

Routes predictions based on:
1. Prop type performance (AST/3PM strong, PTS/REB weak)
2. ML confidence level
3. Edge case flags (B2B, volatility, etc.)

Modes:
- FAST: ML only (instant)
- BALANCED: ML + Gemini for weak props and edge cases
- PREMIUM: ML + Gemini for ALL props
"""
import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'config'))  # For config.shared_config
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))

from ml_pipeline.inference import get_ml_predictor

# Import Gemini integration
try:
    from config.shared_config import get_gemini_model, wait_for_api, GEMINI_API_DELAY
    HAS_GEMINI = True
except ImportError:
    # Fallback to process/shared_config if config doesn't exist
    try:
        from config.shared_config import get_gemini_model, wait_for_api, GEMINI_API_DELAY
        HAS_GEMINI = True
    except ImportError:
        HAS_GEMINI = False
        print("⚠️ Gemini not available. Running ML-only mode.")


class PredictionSource(Enum):
    """Tracks which system made the final prediction."""
    ML_DIRECT = "ml_direct"           # ML confidence high, used directly
    ML_VALIDATED = "ml_validated"     # ML + LLM agreed
    LLM_OVERRIDE = "llm_override"     # LLM disagreed and won
    LLM_PRIMARY = "llm_primary"       # LLM was primary decision maker


class PredictionMode(Enum):
    """Operational modes for different use cases."""
    FAST = "fast"           # ML only, <1 sec
    BALANCED = "balanced"   # ML + selective LLM, ~20 sec
    PREMIUM = "premium"     # ML + LLM for all, ~60 sec


@dataclass
class HybridConfig:
    """Configuration for hybrid prediction routing."""
    # Confidence thresholds
    ml_high_confidence: int = 7      # Use ML directly above this
    ml_low_confidence: int = 5       # Always use LLM below this
    
    # Prop-specific trust levels (based on model performance)
    # AST: 55.2%, 3PM: 52.6% - these work
    # PTS: 51.2%, REB: 50.6% - these need LLM help
    strong_props: Tuple[str, ...] = ("AST", "3PM")
    weak_props: Tuple[str, ...] = ("PTS", "REB")
    
    # Edge case flags (always send to LLM regardless of confidence)
    check_b2b: bool = True
    check_volatile: bool = True
    spread_threshold: float = 12.0   # |Spread| > this triggers LLM
    
    # Disagreement resolution
    disagreement_rule: str = "higher_confidence"  # or "always_llm", "always_ml"


class HybridPredictor:
    """
    Main orchestrator for ML + LLM hybrid predictions.
    
    Usage:
        predictor = HybridPredictor(mode='balanced')
        results = predictor.predict(features_df)
    """
    
    def __init__(self, mode: str = 'balanced', config: HybridConfig = None, blind_mode: bool = False):
        """
        Initialize hybrid predictor.
        
        Args:
            mode: 'fast', 'balanced', or 'premium'
            config: Optional custom configuration
            blind_mode: If True, Gemini does NOT see ML predictions (independent analysis)
        """
        self.mode = PredictionMode(mode)
        self.config = config or HybridConfig()
        self.blind_mode = blind_mode
        
        if blind_mode:
            print("🔒 BLIND MODE: Gemini will make INDEPENDENT predictions (no ML hints)")
        self.ml_predictor = get_ml_predictor(verbose=False)
        
        # Initialize Gemini if available
        # Using gemini-2.5-pro-preview for maximum reasoning quality
        self.gemini_model = None
        if HAS_GEMINI:
            try:
                self.gemini_model = get_gemini_model('gemini-3-pro-preview')
                print("✅ Gemini 3 Pro initialized for hybrid predictions")
            except Exception as e:
                print(f"⚠️ Gemini 3 Pro failed, trying flash: {e}")
                try:
                    self.gemini_model = get_gemini_model('gemini-2.0-flash')
                    print("✅ Gemini 2.0 Flash initialized (fallback)")
                except Exception as e2:
                    print(f"⚠️ Gemini initialization failed: {e2}")
        
        # Stats tracking
        self.stats = {
            'ml_direct': 0,
            'ml_validated': 0,
            'llm_override': 0,
            'llm_primary': 0,
            'total': 0
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using hybrid ML + LLM approach.
        
        Args:
            df: Features DataFrame with props to predict
            
        Returns:
            DataFrame with predictions, confidences, sources, and reasoning
        """
        if self.mode == PredictionMode.FAST:
            return self._predict_fast(df)
        elif self.mode == PredictionMode.BALANCED:
            return self._predict_balanced(df)
        elif self.mode == PredictionMode.PREMIUM:
            return self._predict_premium(df)
    
    def _predict_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast mode: ML only, no LLM calls.
        Best for: Live betting, screening, real-time monitoring.
        """
        # Get ML predictions
        results = self.ml_predictor.predict(df)
        
        # Add source tracking
        results['prediction_source'] = PredictionSource.ML_DIRECT.value
        results['reasoning'] = "ML prediction (fast mode)"
        
        self.stats['ml_direct'] += len(results)
        self.stats['total'] += len(results)
        
        return results
    
    def _predict_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balanced mode: ML primary, LLM for edge cases and weak props.
        Best for: Daily predictions, standard workflow.
        """
        # Get ML predictions first
        results = self.ml_predictor.predict(df)
        
        # Determine which props need LLM review
        needs_llm = self._identify_llm_candidates(results)
        
        # Add routing decision columns
        results['needs_llm'] = needs_llm
        results['prediction_source'] = np.where(
            needs_llm,
            PredictionSource.LLM_PRIMARY.value,
            PredictionSource.ML_DIRECT.value
        )
        
        # For props that need LLM, we'll mark them for Gemini processing
        # The actual Gemini call would be handled by nba_ai_pipeline.py
        results['reasoning'] = np.where(
            needs_llm,
            "Needs LLM review: " + self._get_llm_reason(results),
            "ML prediction (high confidence or strong prop type)"
        )
        
        # Update stats
        ml_direct_count = (~needs_llm).sum()
        llm_count = needs_llm.sum()
        self.stats['ml_direct'] += ml_direct_count
        self.stats['llm_primary'] += llm_count
        self.stats['total'] += len(results)
        
        return results
    
    def _predict_premium(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Premium mode: ML + LLM for all props.
        Best for: High-stakes analysis, parlay construction.
        
        Sends EACH prop to Gemini INDIVIDUALLY for maximum reasoning quality.
        This allows Gemini to focus its full attention on one prop at a time.
        """
        # Get ML predictions first (for context)
        results = self.ml_predictor.predict(df)
        
        if self.gemini_model is None:
            print("⚠️ Gemini not available, falling back to ML-only")
            results['prediction_source'] = PredictionSource.ML_DIRECT.value
            results['reasoning'] = "Gemini unavailable, ML prediction used"
            self.stats['ml_direct'] += len(results)
            self.stats['total'] += len(results)
            return results
        
        print(f"\n🤖 Premium Mode: Analyzing {len(results)} props INDIVIDUALLY with Gemini...")
        print("   (Full attention on each prop for maximum reasoning quality)")
        
        # Process each prop individually for maximum reasoning quality
        gemini_predictions = []
        gemini_confidences = []
        gemini_reasonings = []
        gemini_edge_factors = []
        agrees_with_ml_list = []
        
        for i, (idx, row) in enumerate(results.iterrows()):
            player_name = row.get('FullName', 'Unknown')
            prop_type = row.get('Prop_Type_Normalized', 'PTS')
            line = row.get('Line', 0)
            
            print(f"   [{i+1}/{len(results)}] Analyzing {player_name} {prop_type} {line}...", end=" ")
            
            gemini_result = self._analyze_single_prop(row)
            
            if gemini_result:
                gemini_predictions.append(gemini_result.get('prediction', row.get('ml_prediction', 'UNKNOWN')))
                gemini_confidences.append(gemini_result.get('confidence', 5))
                gemini_reasonings.append(gemini_result.get('reasoning', ''))
                gemini_edge_factors.append(gemini_result.get('edge_factors', []))
                agrees = gemini_result.get('agrees_with_ml', True)
                agrees_with_ml_list.append(agrees)
                
                pred = gemini_result.get('prediction', '?')
                conf = gemini_result.get('confidence', 0)
                agree_str = "✓" if agrees else "✗"
                print(f"{pred} (conf={conf}) {agree_str}")
            else:
                # Fallback to ML
                gemini_predictions.append(row.get('ml_prediction', 'UNKNOWN'))
                gemini_confidences.append(row.get('ml_confidence', 5))
                gemini_reasonings.append('Gemini quota exceeded, using ML prediction')
                gemini_edge_factors.append([])
                agrees_with_ml_list.append(True)
                
                # Show quota warning only once
                if i == 0:
                    print("⚠️ Gemini API quota exceeded - using ML predictions")
                    print("   Check your Gemini API plan/billing at: https://ai.dev/usage?tab=rate-limit")
                else:
                    print("⚠️ Using ML")
        
        # Add results to DataFrame
        results['gemini_prediction'] = gemini_predictions
        results['gemini_confidence'] = gemini_confidences
        results['reasoning'] = gemini_reasonings
        results['edge_factors'] = gemini_edge_factors
        results['agrees_with_ml'] = agrees_with_ml_list
        
        # Final prediction: Use Gemini's prediction in premium mode
        results['final_prediction'] = results['gemini_prediction']
        results['final_confidence'] = results['gemini_confidence']
        results['prediction_source'] = PredictionSource.LLM_PRIMARY.value
        
        # Summary
        disagreements = sum(1 for a in agrees_with_ml_list if not a)
        if disagreements > 0:
            print(f"\n   📊 Gemini disagreed with ML on {disagreements}/{len(results)} props")
        
        self.stats['llm_primary'] += len(results)
        self.stats['total'] += len(results)
        
        return results
    
    def _analyze_single_prop(self, row: pd.Series, max_retries: int = 2) -> Optional[Dict]:
        """
        Analyze a SINGLE prop with Gemini for maximum reasoning quality.
        
        This gives Gemini full attention on one prop at a time,
        allowing for deeper analysis and better predictions.
        
        Args:
            row: Single prop data row
            max_retries: Number of retry attempts on failure
            
        Returns:
            Dict with prediction, confidence, reasoning, edge_factors, agrees_with_ml
        """
        if self.gemini_model is None:
            return None
        
        # Build focused single-prop prompt
        prompt = self._build_single_prop_prompt(row)
        
        for attempt in range(max_retries):
            try:
                wait_for_api("gemini", GEMINI_API_DELAY)
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1
                    },
                    request_options={"timeout": 60}
                )
                
                response_text = response.text
                
                # Extract JSON if wrapped in code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                result = json.loads(response_text)
                
                # Handle if result is a list (sometimes Gemini wraps in array)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                # Validate required fields
                if isinstance(result, dict) and 'prediction' in result:
                    return result
                else:
                    # If response doesn't have expected structure, retry
                    if attempt < max_retries - 1:
                        continue
                    return None
                
            except Exception as e:
                # Check for quota/quota exhaustion errors
                error_msg = str(e).lower()
                if 'resourceexhausted' in type(e).__name__.lower() or 'quota' in error_msg or '429' in error_msg:
                    # Quota exceeded - don't retry, just return None
                    return None
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue
                return None
        
        return None
    
    def _build_single_prop_prompt(self, row: pd.Series) -> str:
        """
        Build a COMPREHENSIVE prompt for analyzing a SINGLE prop.
        
        This prompt includes ALL the same signal density as the ML model:
        - Minutes data (critical for opportunity)
        - Team pace metrics (affects volume)
        - Complete opponent defense stats (L5/L10/Season)
        - Team defensive ratings
        """
        player_name = row.get('FullName', 'Unknown')
        prop_type = row.get('Prop_Type_Normalized', 'PTS')
        line = row.get('Line', 0)
        
        # ========== RECENT PERFORMANCE ==========
        l5_avg = row.get(f'L5_Avg_{prop_type}', 0) or 0
        l10_avg = row.get(f'L10_Avg_{prop_type}', 0) or 0
        season_avg = row.get(f'Season_Avg_{prop_type}', 0) or 0
        
        # Variance/consistency
        l5_std = row.get(f'L5_StdDev_{prop_type}', 0) or 0
        l10_std = row.get(f'L10_StdDev_{prop_type}', 0) or 0
        form = row.get(f'Form_{prop_type}', 0) or 0
        
        # Venue splits
        venue_avg = row.get(f'L5_Avg_{prop_type}_SameVenue', 0) or 0
        season_venue_avg = row.get(f'Season_Avg_{prop_type}_SameVenue', 0) or 0
        
        # ========== MINUTES DATA (CRITICAL) ==========
        l5_min = row.get('L5_Avg_MIN', 0) or 0
        l10_min = row.get('L10_Avg_MIN', 0) or 0
        season_min = row.get('Season_Avg_MIN', 0) or 0
        min_trend = l5_min - season_min  # Positive = increased role
        
        # ========== PACE/OPPORTUNITY ==========
        team_pace_l5 = row.get('L5_Team_Pace', 100) or 100
        team_pace_l10 = row.get('L10_Team_Pace', 100) or 100
        opp_pace_l5 = row.get('Opp_L5_Team_Pace', 100) or 100
        opp_pace_l10 = row.get('Opp_L10_Team_Pace', 100) or 100
        projected_pace = (team_pace_l5 + opp_pace_l5) / 2
        pace_vs_avg = projected_pace - 100  # Positive = faster than average
        
        # ========== OPPONENT DEFENSE (FULL) ==========
        opp_allows_l5 = row.get(f'Opp_L5_Allowed_{prop_type}', 0) or 0
        opp_allows_l10 = row.get(f'Opp_L10_Allowed_{prop_type}', 0) or 0
        opp_allows_season = row.get(f'Opp_Season_Allowed_{prop_type}', 0) or 0
        opp_def_rank = row.get(f'Opp_Rank_Allows_{prop_type}', 15) or 15
        
        # Opponent team defensive rating
        opp_def_rating_l5 = row.get('Opp_L5_Team_Def_Rating', 110) or 110
        opp_def_rating_l10 = row.get('Opp_L10_Team_Def_Rating', 110) or 110
        opp_def_rating_season = row.get('Opp_Season_Team_Def_Rating', 110) or 110
        
        # Defensive rating interpretation
        if opp_def_rating_l5 < 108:
            def_quality = "Elite Defense"
        elif opp_def_rating_l5 < 112:
            def_quality = "Good Defense"
        elif opp_def_rating_l5 < 115:
            def_quality = "Average Defense"
        else:
            def_quality = "Weak Defense"
        
        # ========== ADVANCED STATS ==========
        if prop_type == 'PTS':
            per100 = row.get('Player_PTS_per100', 0) or 0
            ts_pct = row.get('Player_TS_Pct', 0) or 0
            efg_pct = row.get('Player_eFG_Pct', 0) or 0
            advanced_str = f"Per-100: {per100:.1f}, TS%: {ts_pct:.1%}, eFG%: {efg_pct:.1%}"
        elif prop_type == 'REB':
            per100 = row.get('Player_REB_per100', 0) or 0
            advanced_str = f"Per-100: {per100:.1f}"
        elif prop_type == 'AST':
            per100 = row.get('Player_AST_per100', 0) or 0
            advanced_str = f"Per-100: {per100:.1f}"
        elif prop_type == '3PM':
            per100 = row.get('Player_3PM_per100', 0) or 0
            rate = row.get('Player_3PA_Rate', 0) or 0
            advanced_str = f"Per-100: {per100:.1f}, 3PA Rate: {rate:.1%}"
        else:
            advanced_str = "N/A"
        
        # ========== LINE ANALYSIS ==========
        # How does the line compare to averages?
        line_vs_l5 = l5_avg - line
        line_vs_l10 = l10_avg - line
        line_vs_season = season_avg - line
        
        if line_vs_l5 > 2:
            line_assessment = "LINE IS LOW - player averaging well above"
        elif line_vs_l5 < -2:
            line_assessment = "LINE IS HIGH - player averaging below"
        else:
            line_assessment = "LINE IS FAIR - close to recent average"
        
        # ========== ML PREDICTION CONTEXT ==========
        ml_pred = row.get('ml_prediction', 'UNKNOWN')
        ml_conf = row.get('ml_confidence', 5)
        ml_prob = row.get('ml_prob_over', 0.5)
        
        if self.blind_mode:
            ml_section = """
═══════════════════════════════════════════════════════════════
ML MODEL PREDICTION
═══════════════════════════════════════════════════════════════

[BLIND MODE: Make your own independent prediction]
You do NOT have access to ML model output. Use only the data above.
"""
        else:
            ml_section = f"""
═══════════════════════════════════════════════════════════════
ML MODEL PREDICTION
═══════════════════════════════════════════════════════════════

Prediction: {ml_pred}
Confidence: {ml_conf}/10
Probability Over: {ml_prob:.1%}
"""
        
        # ========== CONTEXT ==========
        is_b2b = row.get('is_b2b', 0)
        opp_is_b2b = row.get('Opp_is_b2b', 0)
        is_home = row.get('home', 0)
        rest_days = row.get('rest_days', 0)
        opp_rest_days = row.get('Opp_rest_days', 0)
        opponent = row.get('OpponentAbbrev', 'UNK')
        team = row.get('TeamAbbrev', 'UNK')
        
        # Player tier
        player_tier = row.get('Player_Tier', 'ROTATION')
        reliability = row.get('Reliability_Tag', 'STANDARD')
        
        # ========== BUILD PROMPT ==========
        if self.blind_mode:
            analysis_questions = """Analyze this prop deeply. Consider:
1. Is the line set fairly given recent performance?
2. Does the matchup favor OVER or UNDER?
3. Any fatigue/rest factors?
4. Is the player running hot or cold?
5. What is your conviction level?"""
        else:
            analysis_questions = """Analyze this prop deeply. Consider:
1. Is the line set fairly given recent performance?
2. Does the matchup favor OVER or UNDER?
3. Any fatigue/rest factors?
4. Is the player running hot or cold?
5. Does the ML model have the right read, or is there an edge it might miss?"""
        
        prompt = f"""You are an elite NBA prop betting analyst with access to comprehensive data. Your role is to synthesize all available information into a confident, well-reasoned betting prediction.

═══════════════════════════════════════════════════════════════
YOUR ROLE & METHODOLOGY
═══════════════════════════════════════════════════════════════

You are analyzing a SINGLE player prop with your full attention. Your goal is to determine whether the player will go OVER or UNDER the line, with appropriate confidence.

HOW TO USE THE DATA PROVIDED:

1. RECENT PERFORMANCE (L5/L10/Season):
   • Compare the line to recent averages - if L5 avg is significantly above/below the line, that's a key signal
   • Use standard deviation to assess consistency - low std = more reliable, high std = volatile
   • Form trend (L5 vs Season) shows if player is improving or declining
   • Same venue splits matter - some players perform differently at home vs away

2. MINUTES CONTEXT (Critical):
   • Minutes = opportunity. More minutes = more chances to hit the prop
   • If minutes are trending UP, player's role is expanding (favor OVER)
   • If minutes are trending DOWN, role is shrinking (favor UNDER)
   • Compare L5 minutes to season - significant changes indicate role shifts

3. PACE & OPPORTUNITY:
   • Faster pace = more possessions = more opportunities for stats
   • Projected game pace above 100 = favorable for OVER
   • Projected game pace below 100 = fewer opportunities, consider UNDER
   • Both teams' pace matters - average them for projected game pace

4. OPPONENT DEFENSE:
   • Opponent allows (L5/L10/Season) shows recent defensive performance
   • Rank #1-10 = soft defense (favor OVER), Rank #20-30 = tough defense (favor UNDER)
   • Defensive rating < 108 = elite defense (harder to score), > 115 = weak defense (easier to score)
   • Recent defensive trends (L5) may be more relevant than season averages

5. LINE ANALYSIS:
   • Line vs L5 avg: If line is 2+ points below L5 avg = LINE IS LOW (favor OVER)
   • Line vs L5 avg: If line is 2+ points above L5 avg = LINE IS HIGH (favor UNDER)
   • Fair lines (within 2 points) require deeper matchup analysis

6. SITUATIONAL FACTORS:
   • B2B games = fatigue risk, especially for older players (favor UNDER)
   • Rest days: More rest = better performance (favor OVER)
   • Player tier: STAR players are more reliable, ROTATION players are volatile
   • Reliability tag: SNIPER = consistent (trust trends), VOLATILE = unpredictable (need larger edges)

7. ML MODEL CONTEXT (if provided):
   • The ML model has analyzed the same data using statistical patterns
   • Consider whether you agree or disagree - if you disagree, explain why
   • Your reasoning may catch edge cases the model misses (injuries, role changes, matchup nuances)

HOW TO MAKE A CONFIDENT PREDICTION:

1. SYNTHESIS: Weigh all factors together, not in isolation
   • If multiple signals point the same direction, confidence should be higher
   • If signals conflict, confidence should be lower (5-6/10)
   • Strong signals (line 3+ points off, elite matchup, role expansion) = higher confidence (8-10/10)

2. EDGE IDENTIFICATION:
   • Look for situations where the line doesn't reflect recent performance or matchup
   • Favor OVER when: line is low, matchup is soft, minutes trending up, pace is fast
   • Favor UNDER when: line is high, matchup is tough, fatigue present, minutes trending down

3. CONFIDENCE CALIBRATION:
   • 9-10/10: Multiple strong signals align, clear edge, reliable player
   • 7-8/10: Good signals align, some edge, reasonable confidence
   • 5-6/10: Mixed signals, fair line, moderate confidence
   • 3-4/10: Weak signals, conflicting data, low confidence (avoid betting)
   • 1-2/10: No clear edge, avoid

4. REASONING REQUIREMENTS:
   • Your reasoning must explain WHY you're making this prediction
   • Reference specific data points (e.g., "L5 avg is 24.5 vs line of 22.5, and opponent ranks #28 in defense")
   • Identify the key edge factors that drive your decision
   • Acknowledge any risks or counter-arguments

5. LINE VALUE ASSESSMENT:
   • LOW: Line is significantly below recent performance (value on OVER)
   • HIGH: Line is significantly above recent performance (value on UNDER)
   • FAIR: Line matches recent performance (requires matchup/pace analysis)

Remember: You're making a betting decision. Be decisive when signals are clear, but acknowledge uncertainty when data is mixed. Quality over quantity - a well-reasoned 7/10 confidence pick is better than a forced 9/10.

═══════════════════════════════════════════════════════════════
PROP TO ANALYZE
═══════════════════════════════════════════════════════════════

Player: {player_name} ({team})
Prop: {prop_type} OVER/UNDER {line}
Opponent: {opponent}
Venue: {'HOME' if is_home else 'AWAY'}

═══════════════════════════════════════════════════════════════
RECENT PERFORMANCE
═══════════════════════════════════════════════════════════════

{prop_type} Averages:
  • Last 5 games:  {l5_avg:.1f} (std: {l5_std:.1f})
  • Last 10 games: {l10_avg:.1f} (std: {l10_std:.1f})
  • Season:        {season_avg:.1f}
  • Same venue L5: {venue_avg:.1f}

Form Trend: {'+' if form > 0 else ''}{form:.1f} (L5 vs Season)
Advanced: {advanced_str}

═══════════════════════════════════════════════════════════════
MINUTES CONTEXT (Critical for opportunity)
═══════════════════════════════════════════════════════════════

  • L5 Avg MIN:  {l5_min:.1f}
  • L10 Avg MIN: {l10_min:.1f}
  • Season MIN:  {season_min:.1f}
  • Minutes Trend: {'+' if min_trend > 0 else ''}{min_trend:.1f} ({'increased' if min_trend > 0 else 'decreased'} role)

═══════════════════════════════════════════════════════════════
PACE & OPPORTUNITY
═══════════════════════════════════════════════════════════════

  • {team} Pace (L5): {team_pace_l5:.1f} | (L10): {team_pace_l10:.1f}
  • {opponent} Pace (L5): {opp_pace_l5:.1f} | (L10): {opp_pace_l10:.1f}
  • Projected Game Pace: {projected_pace:.1f} ({'+' if pace_vs_avg > 0 else ''}{pace_vs_avg:.1f} vs league avg)
  • Impact: {'Faster pace = more possessions = more opportunity' if pace_vs_avg > 0 else 'Slower pace = fewer possessions = less opportunity' if pace_vs_avg < -2 else 'Average pace'}

═══════════════════════════════════════════════════════════════
OPPONENT DEFENSE vs {prop_type}
═══════════════════════════════════════════════════════════════

{opponent} allows to opponents:
  • L5:     {opp_allows_l5:.1f} {prop_type}
  • L10:    {opp_allows_l10:.1f} {prop_type}
  • Season: {opp_allows_season:.1f} {prop_type}
  • Rank: #{int(opp_def_rank)} (1 = allows most)

{opponent} Defensive Rating:
  • L5: {opp_def_rating_l5:.1f} | L10: {opp_def_rating_l10:.1f} | Season: {opp_def_rating_season:.1f}
  • Assessment: {def_quality} (lower = better defense)

═══════════════════════════════════════════════════════════════
LINE ANALYSIS
═══════════════════════════════════════════════════════════════

Line: {line}
  • vs L5 avg:     {'+' if line_vs_l5 > 0 else ''}{line_vs_l5:.1f}
  • vs L10 avg:    {'+' if line_vs_l10 > 0 else ''}{line_vs_l10:.1f}
  • vs Season avg: {'+' if line_vs_season > 0 else ''}{line_vs_season:.1f}
  
Assessment: {line_assessment}

═══════════════════════════════════════════════════════════════
SITUATIONAL FACTORS
═══════════════════════════════════════════════════════════════

Player Classification: {player_tier} / {reliability}
Player Rest: {rest_days} days (B2B: {'YES ⚠️ FATIGUE RISK' if is_b2b else 'No'})
Opponent Rest: {opp_rest_days} days (B2B: {'YES - defense may be tired' if opp_is_b2b else 'No'})
{ml_section}
═══════════════════════════════════════════════════════════════
YOUR ANALYSIS
═══════════════════════════════════════════════════════════════

{analysis_questions}

Respond with your expert analysis in JSON format:
{{
    "prediction": "OVER" or "UNDER",
    "confidence": 1-10,
    "reasoning": "Your detailed analysis (2-3 sentences)",
    "edge_factors": ["Key factor 1", "Key factor 2"],
    "agrees_with_ml": true or false,
    "line_value": "FAIR" or "HIGH" or "LOW"
}}

Think carefully. This is a high-stakes decision."""
        
        return prompt
    
    def _identify_llm_candidates(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine which props should be sent to LLM for review.
        
        Returns:
            Boolean Series where True = needs LLM
        """
        needs_llm = pd.Series(False, index=df.index)
        
        # Rule 1: Weak prop types always need LLM
        if 'Prop_Type_Normalized' in df.columns:
            weak_mask = df['Prop_Type_Normalized'].isin(self.config.weak_props)
            needs_llm = needs_llm | weak_mask
        
        # Rule 2: Low ML confidence needs LLM
        if 'ml_confidence' in df.columns:
            low_conf_mask = df['ml_confidence'] < self.config.ml_low_confidence
            needs_llm = needs_llm | low_conf_mask
        
        # Rule 3: Edge case flags
        if self.config.check_b2b and 'is_b2b' in df.columns:
            b2b_mask = df['is_b2b'] == 1
            needs_llm = needs_llm | b2b_mask
        
        if self.config.check_volatile and 'Reliability_Tag' in df.columns:
            volatile_mask = df['Reliability_Tag'] == 'VOLATILE'
            needs_llm = needs_llm | volatile_mask
        
        # Rule 4: High confidence on strong props = skip LLM
        if 'ml_confidence' in df.columns and 'Prop_Type_Normalized' in df.columns:
            strong_high_conf = (
                df['Prop_Type_Normalized'].isin(self.config.strong_props) &
                (df['ml_confidence'] >= self.config.ml_high_confidence)
            )
            # These DON'T need LLM even if other flags triggered
            # But we'll still respect volatility and B2B flags
            needs_llm = needs_llm & ~(strong_high_conf & 
                                       ~df.get('is_b2b', pd.Series(False)).astype(bool))
        
        return needs_llm
    
    def _get_llm_reason(self, df: pd.DataFrame) -> pd.Series:
        """Generate reason strings for why LLM is needed."""
        reasons = []
        
        for idx, row in df.iterrows():
            reason_parts = []
            
            prop_type = row.get('Prop_Type_Normalized', '')
            if prop_type in self.config.weak_props:
                reason_parts.append(f"{prop_type} model is weak")
            
            conf = row.get('ml_confidence', 5)
            if conf < self.config.ml_low_confidence:
                reason_parts.append(f"low ML confidence ({conf})")
            
            if row.get('is_b2b', 0) == 1:
                reason_parts.append("B2B game")
            
            if row.get('Reliability_Tag', '') == 'VOLATILE':
                reason_parts.append("volatile player")
            
            reasons.append("; ".join(reason_parts) if reason_parts else "edge case")
        
        return pd.Series(reasons, index=df.index)
    
    def get_stats(self) -> Dict:
        """Get prediction source statistics."""
        total = max(self.stats['total'], 1)
        return {
            'total_predictions': self.stats['total'],
            'ml_direct_pct': self.stats['ml_direct'] / total * 100,
            'ml_validated_pct': self.stats['ml_validated'] / total * 100,
            'llm_override_pct': self.stats['llm_override'] / total * 100,
            'llm_primary_pct': self.stats['llm_primary'] / total * 100,
        }
    
    def print_stats(self):
        """Print prediction routing statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("HYBRID PREDICTOR STATS")
        print("="*50)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"ML Direct:    {stats['ml_direct_pct']:.1f}%")
        print(f"ML Validated: {stats['ml_validated_pct']:.1f}%")
        print(f"LLM Override: {stats['llm_override_pct']:.1f}%")
        print(f"LLM Primary:  {stats['llm_primary_pct']:.1f}%")
        print("="*50)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_hybrid_predictor(mode: str = 'balanced') -> HybridPredictor:
    """Factory function to create a hybrid predictor."""
    return HybridPredictor(mode=mode)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hybrid predictor')
    parser.add_argument('--mode', choices=['fast', 'balanced', 'premium'], 
                        default='balanced', help='Prediction mode')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to features data (CSV or parquet)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Take sample for testing
    df = df.head(100)
    print(f"Testing with {len(df)} samples...")
    
    # Run hybrid predictor
    predictor = HybridPredictor(mode=args.mode)
    results = predictor.predict(df)
    
    # Show results
    print("\nSample predictions:")
    display_cols = ['Prop_Type_Normalized', 'ml_prediction', 'ml_confidence', 
                    'prediction_source', 'needs_llm']
    display_cols = [c for c in display_cols if c in results.columns]
    print(results[display_cols].head(20).to_string())
    
    # Show stats
    predictor.print_stats()

