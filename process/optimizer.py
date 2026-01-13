"""
THE STRATEGIST (optimizer.py)
=============================
Self-Evolving NBA Neural System v2.0 - Parlay Optimizer with Learning

Purpose: Constructs optimal betting slips based on AI predictions and correlations.

SELF-LEARNING FEATURES:
1. Parlay Case Studies: Learns from which parlay strategies work/fail
2. Correlation Learning: Tracks which correlation patterns are profitable
3. Size Optimization: Learns optimal parlay sizes based on historical performance
4. Theme Analysis: Identifies which betting themes perform best

Integration:
- Reads predictions from the production pipeline
- Learns from past parlay outcomes
- Applies learned patterns to future parlay construction

OPTIMIZATIONS (v2.1):
- Centralized Gemini configuration
- Shared configuration across modules
"""

import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from tqdm import tqdm
import time

# Import shared configuration
from shared_config import (
    get_gemini_model,
    PREDICTIONS_DIR,
    PARLAYS_DIR,
    TRAINING_REPORTS_DIR,
    PARLAY_CASE_STUDIES_PATH,
    PARLAY_PERFORMANCE_LOG,
    wait_for_api,
    GEMINI_API_DELAY
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ============================================================================
# PARLAY LEARNING DATA STRUCTURES
# ============================================================================

@dataclass
class ParlayLeg:
    """A single leg of a parlay."""
    player: str
    team: str
    prop: str
    prediction: str
    line: float
    confidence: int
    actual_outcome: Optional[str] = None
    hit: Optional[bool] = None


@dataclass
class ParlayResult:
    """Complete result of a parlay for learning."""
    id: str
    date: str
    timestamp: str
    parlay_size: int
    theme: str
    correlation_type: str
    risk_level: str
    legs: List[ParlayLeg]
    legs_hit: int
    total_legs: int
    outcome: str  # "WIN", "LOSS", "PARTIAL"
    average_confidence: float
    

@dataclass
class ParlayCaseStudy:
    """A learning from a parlay failure or success."""
    id: str
    date: str
    parlay_size: int
    theme: str
    correlation_pattern: str
    outcome: str
    legs_hit: int
    total_legs: int
    insight: str  # What we learned
    correction: str  # How to adjust future parlays
    context: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PARLAY LEARNING SYSTEM
# ============================================================================

class ParlayLearningSystem:
    """
    Learns from parlay outcomes to improve future parlay construction.
    
    Tracks:
    - Win rates by parlay size (2-pick, 3-pick, etc.)
    - Win rates by theme (star stack, pace stack, etc.)
    - Correlation pattern performance
    - Risk level outcomes
    """
    
    def __init__(self, case_studies_path: str = PARLAY_CASE_STUDIES_PATH):
        self.case_studies_path = case_studies_path
        self.case_studies: List[ParlayCaseStudy] = []
        self.performance_history: List[ParlayResult] = []
        self._load()
        self._setup_gemini()
    
    def _load(self):
        """Loads parlay case studies and performance history."""
        # Load case studies
        if os.path.exists(self.case_studies_path):
            try:
                with open(self.case_studies_path, 'r') as f:
                    data = json.load(f)
                    if data and isinstance(data, list) and len(data) > 0:
                        self.case_studies = [ParlayCaseStudy(**cs) for cs in data]
                    else:
                        self.case_studies = []
                print(f"🎰 Loaded {len(self.case_studies)} parlay case studies")
            except Exception as e:
                print(f"Warning: Error loading parlay case studies: {e}")
                self.case_studies = []
        else:
            self.case_studies = []
            print(f"🎰 Loaded {len(self.case_studies)} parlay case studies (file doesn't exist yet)")
        
        # Load performance history
        if os.path.exists(PARLAY_PERFORMANCE_LOG):
            try:
                with open(PARLAY_PERFORMANCE_LOG, 'r') as f:
                    data = json.load(f)
                    # Convert to ParlayResult objects (simplified)
                    self.performance_history = data if isinstance(data, list) else []
            except:
                self.performance_history = []
    
    def _setup_gemini(self):
        """
        Configures Gemini for parlay analysis.
        
        OPTIMIZED: Uses shared config singleton to prevent multiple API configurations.
        """
        try:
            # Using gemini-2.5-pro-preview for best reasoning capabilities
            self.model = get_gemini_model('gemini-3-pro-preview')
        except ValueError:
            self.model = None
    
    def save(self):
        """Persists case studies to file."""
        os.makedirs(os.path.dirname(self.case_studies_path), exist_ok=True)
        data = [asdict(cs) for cs in self.case_studies]
        with open(self.case_studies_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.case_studies)} parlay case studies")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculates performance statistics from history."""
        if not self.performance_history:
            return {}
        
        # By size
        size_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
        # By theme
        theme_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
        # By risk level
        risk_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
        
        for result in self.performance_history:
            size = result.get('parlay_size', 0)
            theme = result.get('theme', 'Unknown')
            risk = result.get('risk_level', 'medium')
            outcome = result.get('outcome', 'LOSS')
            
            size_stats[size]["total"] += 1
            theme_stats[theme]["total"] += 1
            risk_stats[risk]["total"] += 1
            
            if outcome == "WIN":
                size_stats[size]["wins"] += 1
                theme_stats[theme]["wins"] += 1
                risk_stats[risk]["wins"] += 1
            else:
                size_stats[size]["losses"] += 1
                theme_stats[theme]["losses"] += 1
                risk_stats[risk]["losses"] += 1
        
        # Calculate win rates
        for stats in [size_stats, theme_stats, risk_stats]:
            for key in stats:
                total = stats[key]["total"]
                wins = stats[key]["wins"]
                stats[key]["win_rate"] = (wins / total * 100) if total > 0 else 0
        
        return {
            "by_size": dict(size_stats),
            "by_theme": dict(theme_stats),
            "by_risk": dict(risk_stats),
            "total_parlays": len(self.performance_history),
            "overall_win_rate": sum(1 for r in self.performance_history if r.get('outcome') == 'WIN') / len(self.performance_history) * 100 if self.performance_history else 0
        }
    
    def get_relevant_insights(self, parlay_size: int, theme: str, risk_level: str) -> List[str]:
        """
        RAG-style retrieval of relevant parlay learnings.
        """
        insights = []
        
        for cs in self.case_studies:
            # Match by size
            if cs.parlay_size == parlay_size:
                insights.append(f"Size {parlay_size}: {cs.correction}")
            # Match by theme
            if cs.theme.lower() in theme.lower() or theme.lower() in cs.theme.lower():
                insights.append(f"Theme '{cs.theme}': {cs.correction}")
        
        # Also add general stats-based insights
        stats = self.get_performance_stats()
        
        if stats.get('by_size', {}).get(parlay_size, {}).get('win_rate', 50) < 30:
            insights.append(f"⚠️ {parlay_size}-leg parlays have low win rate - consider smaller size")
        
        return insights[:5]  # Limit to 5 insights
    
    def log_parlay_result(self, result: Dict[str, Any]):
        """Logs a parlay result for learning."""
        os.makedirs(os.path.dirname(PARLAY_PERFORMANCE_LOG), exist_ok=True)
        
        # Load existing
        existing = []
        if os.path.exists(PARLAY_PERFORMANCE_LOG):
            try:
                with open(PARLAY_PERFORMANCE_LOG, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.append(result)
        
        with open(PARLAY_PERFORMANCE_LOG, 'w') as f:
            json.dump(existing, f, indent=2)
        
        self.performance_history = existing
    
    def generate_case_study(self, failed_parlay: Dict[str, Any]) -> Optional[ParlayCaseStudy]:
        """Generates a case study from a failed parlay."""
        if not self.model:
            return None
        
        prompt = f"""
You are analyzing a failed sports betting parlay to extract learnings.

### THE FAILED PARLAY:
- Size: {failed_parlay.get('parlay_size', 'Unknown')} legs
- Theme: {failed_parlay.get('theme', 'Unknown')}
- Risk Level: {failed_parlay.get('risk_level', 'Unknown')}
- Legs Hit: {failed_parlay.get('legs_hit', 0)}/{failed_parlay.get('total_legs', 0)}
- Average Confidence: {failed_parlay.get('average_confidence', 0):.1f}/10

### INDIVIDUAL LEGS:
{json.dumps(failed_parlay.get('legs', []), indent=2)}

### TASK:
Analyze why this parlay failed and provide a Case Study in JSON format:

{{
    "insight": "<What went wrong - 1 sentence>",
    "correction": "<Specific rule to apply for future parlays, e.g., 'Avoid combining 3+ role players in the same parlay'>",
    "correlation_pattern": "<What correlation pattern was problematic, if any>"
}}

Return ONLY the JSON.
"""
        
        try:
            time.sleep(1.0)
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            result = json.loads(response.text)
            
            case_study = ParlayCaseStudy(
                id=f"pcs_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                date=datetime.now().isoformat(),
                parlay_size=failed_parlay.get('parlay_size', 0),
                theme=failed_parlay.get('theme', 'Unknown'),
                correlation_pattern=result.get('correlation_pattern', ''),
                outcome="LOSS",
                legs_hit=failed_parlay.get('legs_hit', 0),
                total_legs=failed_parlay.get('total_legs', 0),
                insight=result.get('insight', ''),
                correction=result.get('correction', ''),
                context={"legs": failed_parlay.get('legs', [])}
            )
            
            self.case_studies.append(case_study)
            self.save()
            
            return case_study
            
        except Exception as e:
            print(f"Error generating parlay case study: {e}")
            return None


# ============================================================================
# CORRELATION ENGINE
# ============================================================================

class CorrelationEngine:
    """
    Analyzes correlations between picks to optimize parlay construction.
    
    Now enhanced with learning from past parlay outcomes.
    """
    
    def __init__(self, learning_system: ParlayLearningSystem = None):
        self.learning_system = learning_system or ParlayLearningSystem()
    
    def check_correlation(self, pick1: pd.Series, pick2: pd.Series) -> Tuple[str, str]:
        """
        Checks the correlation between two picks.
        
        Returns:
            Tuple of (correlation_type, reason)
        """
        # Same team cannibalization check
        if pick1.get('Team') == pick2.get('Team'):
            if pick1.get('Prop') == pick2.get('Prop'):
                if pick1.get('AI_Prediction') == 'OVER' and pick2.get('AI_Prediction') == 'OVER':
                    return ('negative', 'Same team, same stat - volume cannibalization')
        
        # Same team synergy check (different stats)
        if pick1.get('Team') == pick2.get('Team'):
            if pick1.get('Prop') != pick2.get('Prop'):
                props = {pick1.get('Prop'), pick2.get('Prop')}
                if props == {'POINTS', 'ASSISTS'}:
                    return ('positive', 'Playmaker synergy - scoring and facilitating')
        
        # Opposing teams in same game
        if pick1.get('Opponent') == pick2.get('Team') or pick1.get('Team') == pick2.get('Opponent'):
            spread1 = abs(float(pick1.get('Vegas_Spread', 0) or 0))
            spread2 = abs(float(pick2.get('Vegas_Spread', 0) or 0))
            
            if spread1 > 12 or spread2 > 12:
                return ('negative', 'Same game with blowout risk - sitting risk')
        
        # High game total synergy
        total1 = float(pick1.get('Game_Total', 220) or 220)
        total2 = float(pick2.get('Game_Total', 220) or 220)
        
        if total1 > 235 and total2 > 235:
            if pick1.get('AI_Prediction') == 'OVER' and pick2.get('AI_Prediction') == 'OVER':
                return ('positive', 'Both in high-scoring environments')
        
        return ('neutral', 'Independent outcomes')
    
    def is_safe_combination(self, picks: List[pd.Series]) -> Tuple[bool, List[str]]:
        """Checks if a combination of picks is safe."""
        warnings = []
        
        for i, pick1 in enumerate(picks):
            for j, pick2 in enumerate(picks):
                if i >= j:
                    continue
                
                corr_type, reason = self.check_correlation(pick1, pick2)
                
                if corr_type == 'negative':
                    warnings.append(f"⚠️ {pick1.get('Player')} vs {pick2.get('Player')}: {reason}")
        
        return (len(warnings) == 0, warnings)


# ============================================================================
# OPTIMIZER CLASS
# ============================================================================

class Optimizer:
    """
    Constructs optimal betting slips with self-learning capabilities.
    """
    
    def __init__(self, predictions_df: pd.DataFrame):
        self.df = predictions_df.copy()
        self.df['AI_Confidence'] = pd.to_numeric(self.df['AI_Confidence'], errors='coerce').fillna(0)
        self.df['Line'] = pd.to_numeric(self.df['Line'], errors='coerce')
        
        self.learning_system = ParlayLearningSystem()
        self.correlation_engine = CorrelationEngine(self.learning_system)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._setup_gemini()
    
    def _setup_gemini(self):
        """
        Configures the Gemini API client.
        
        OPTIMIZED: Uses shared config singleton to prevent multiple API configurations.
        """
        try:
            # Using gemini-3-pro-preview for best reasoning capabilities
            self.model = get_gemini_model('gemini-3-pro-preview')
        except ValueError:
            self.model = None
    
    def filter_top_picks(self, min_confidence: int = 8) -> pd.DataFrame:
        """Returns high-confidence picks."""
        mask = (self.df['AI_Confidence'] >= min_confidence) & \
               (self.df['AI_Prediction'].isin(['OVER', 'UNDER']))
        return self.df[mask].copy()
    
    def generate_power_plays(self, top_picks_df: pd.DataFrame, slip_size: int = 2) -> List[List[pd.Series]]:
        """
        Strategy A: Aggressive Power Plays (2-Pick entries).
        Now enhanced with learning insights.
        """
        print(f"\n⚡ Generating {slip_size}-Pick Power Plays...")
        
        # Get learning insights for this size
        insights = self.learning_system.get_relevant_insights(slip_size, "power_play", "low")
        if insights:
            print(f"   📚 Applying {len(insights)} learned insights")
        
        slips = []
        sorted_picks = top_picks_df.sort_values(by='AI_Confidence', ascending=False)
        used_indices = set()
        
        for i, row1 in sorted_picks.iterrows():
            if i in used_indices:
                continue
            
            current_slip = [row1]
            used_indices.add(i)
            
            for j, row2 in sorted_picks.iterrows():
                if j in used_indices:
                    continue
                
                corr_type, _ = self.correlation_engine.check_correlation(row1, row2)
                
                if corr_type != 'negative':
                    current_slip.append(row2)
                    used_indices.add(j)
                    if len(current_slip) == slip_size:
                        break
            
            if len(current_slip) == slip_size:
                slips.append(current_slip)
        
        print(f"   Created {len(slips)} power plays")
        return slips
    
    def generate_flex_plays(self, top_picks_df: pd.DataFrame, slip_size: int = 5) -> List[List[pd.Series]]:
        """
        Strategy B: Safety Flex Plays (5 or 6 Pick entries).
        """
        print(f"\n🎲 Generating {slip_size}-Pick Flex Plays...")
        
        # Check historical performance for this size
        stats = self.learning_system.get_performance_stats()
        size_stats = stats.get('by_size', {}).get(slip_size, {})
        if size_stats.get('win_rate', 50) < 20:
            print(f"   ⚠️ Warning: {slip_size}-leg parlays have {size_stats.get('win_rate', 0):.1f}% historical win rate")
        
        slips = []
        available_picks = top_picks_df.sort_values(by='AI_Confidence', ascending=False)
        
        if len(available_picks) < slip_size:
            print(f"   Not enough picks for {slip_size}-pick Flex Play.")
            return []
        
        current_slip = []
        
        for i, row in available_picks.iterrows():
            is_compatible, warnings = self.correlation_engine.is_safe_combination(
                current_slip + [row]
            )
            
            if is_compatible:
                current_slip.append(row)
            
            if len(current_slip) == slip_size:
                slips.append(current_slip)
                current_slip = []
        
        print(f"   Created {len(slips)} flex plays")
        return slips
    
    def _format_picks_for_ai(self, top_picks_df: pd.DataFrame) -> str:
        """Formats picks for AI consumption."""
        lines = []
        for idx, (_, row) in enumerate(top_picks_df.iterrows(), 1):
            opponent = row.get('Opponent', 'UNK')
            opp_rank = row.get('Opp_Rank', 'N/A')
            game_total = row.get('Game_Total', 'N/A')
            spread = row.get('Vegas_Spread', 'N/A')
            usg = row.get('USG_PCT', 0)
            usg_str = f"{float(usg)*100:.1f}%" if usg else "N/A"
            
            lines.append(
                f"{idx}. {row['Player']} ({row['Team']}) - {row['Prop']} {row['AI_Prediction']} {row['Line']} | "
                f"Conf: {int(row['AI_Confidence'])}/10 | vs {opponent} (Def Rank: {opp_rank}) | "
                f"Total: {game_total} | Spread: {spread} | USG: {usg_str}"
            )
        return "\n".join(lines)
    
    def generate_ai_parlays(self, top_picks_df: pd.DataFrame, max_retries: int = 3) -> Dict:
        """
        Uses AI to create parlays, now enhanced with learning insights.
        """
        if self.model is None:
            print("⚠️ Gemini API not configured.")
            return {"parlays": [], "error": "API not configured"}
        
        if len(top_picks_df) < 2:
            return {"parlays": [], "error": "Not enough picks"}
        
        picks_summary = self._format_picks_for_ai(top_picks_df)
        num_picks = len(top_picks_df)
        
        # Get performance stats for context
        stats = self.learning_system.get_performance_stats()
        
        # Get relevant case studies
        insights = []
        for cs in self.learning_system.case_studies[-10:]:
            insights.append(f"- {cs.theme}: {cs.correction}")
        
        insights_text = "\n".join(insights) if insights else "No historical insights yet."
        
        # Build stats summary
        stats_text = ""
        if stats:
            stats_text = f"""
### HISTORICAL PERFORMANCE:
- Overall Win Rate: {stats.get('overall_win_rate', 0):.1f}%
- Best Size: {max(stats.get('by_size', {}).items(), key=lambda x: x[1].get('win_rate', 0))[0] if stats.get('by_size') else 'N/A'} legs
- Total Parlays Tracked: {stats.get('total_parlays', 0)}
"""
        
        prompt = f"""You are an expert sports bettor creating optimal parlay strategies.
You learn from past mistakes and apply historical insights.

### PICKS AVAILABLE ({num_picks} total):
{picks_summary}

{stats_text}

### LEARNED INSIGHTS (Apply these!):
{insights_text}

### YOUR TASK:
Create 3-4 optimal parlay combinations. Each should have a clear strategic theme.

### CORRELATION RULES:

**POSITIVE (GOOD to combine):**
- Same team synergy: Primary scorer + playmaker
- High-pace environment: Multiple overs in games with Total > 230
- Soft defense stacking: Multiple overs against Def Rank > 20

**NEGATIVE (AVOID):**
- Same team, same stat, both OVER: Volume cannibalization
- Same game blowout risk (spread > 12): Stars may sit
- Opposing players in lopsided games

### PARLAY SIZE GUIDELINES:
- Power Play (2 picks): Highest confidence, safest
- Standard (3-4 picks): Mixed themes
- Long Shot (5+ picks): Only if all positively correlated

### OUTPUT FORMAT (JSON ONLY):
{{
  "parlays": [
    {{
      "name": "Short descriptive name",
      "size": 2,
      "theme": "One sentence strategic theme",
      "pick_numbers": [1, 3],
      "reasoning": "2-3 sentences explaining correlation and strategy",
      "risk_level": "low|medium|high"
    }}
  ]
}}
"""
        
        for attempt in range(max_retries):
            try:
                print(f"\n🧠 Generating AI-optimized parlays (with learning)...")
                wait_for_api("gemini", GEMINI_API_DELAY)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                    request_options={"timeout": 60}
                )
                
                result = json.loads(response.text)
                
                # Enrich parlays
                enriched_parlays = []
                for parlay in result.get('parlays', []):
                    pick_numbers = parlay.get('pick_numbers', [])
                    picks_data = []
                    
                    for num in pick_numbers:
                        if 1 <= num <= len(top_picks_df):
                            row = top_picks_df.iloc[num - 1]
                            picks_data.append({
                                'player': row['Player'],
                                'team': row['Team'],
                                'prop': row['Prop'],
                                'prediction': row['AI_Prediction'],
                                'line': row['Line'],
                                'confidence': int(row['AI_Confidence'])
                            })
                    
                    if len(picks_data) >= 2:
                        parlay['picks'] = picks_data
                        parlay['avg_confidence'] = sum(p['confidence'] for p in picks_data) / len(picks_data)
                        enriched_parlays.append(parlay)
                
                print(f"✅ Generated {len(enriched_parlays)} AI-optimized parlays")
                return {"parlays": enriched_parlays}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 2)
                    continue
                else:
                    print(f"❌ AI parlay generation failed: {e}")
                    return {"parlays": [], "error": str(e)}
        
        return {"parlays": [], "error": "Max retries exceeded"}
    
    def generate_markdown_report(self, top_picks: pd.DataFrame, ai_parlays: Dict = None,
                                  power_slips: List = None, flex_slips: List = None) -> str:
        """Generate a timestamped markdown report."""
        
        lines = []
        lines.append(f"# 🏀 NBA Betting Slips")
        lines.append(f"")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Timestamp:** `{self.timestamp}`")
        lines.append("")
        
        # Performance stats if available
        stats = self.learning_system.get_performance_stats()
        if stats.get('total_parlays', 0) > 0:
            lines.append(f"### 📈 Historical Performance")
            lines.append(f"- Overall Win Rate: {stats.get('overall_win_rate', 0):.1f}%")
            lines.append(f"- Parlays Tracked: {stats.get('total_parlays', 0)}")
            lines.append("")
        
        lines.append(f"---")
        lines.append("")
        
        # Summary
        ai_parlay_list = ai_parlays.get('parlays', []) if ai_parlays else []
        lines.append(f"## 🎯 Summary")
        lines.append(f"- **High-Confidence Picks:** {len(top_picks)}")
        lines.append(f"- **AI-Optimized Parlays:** {len(ai_parlay_list)}")
        if power_slips:
            lines.append(f"- **Backup Power Plays (2-Pick):** {len(power_slips)}")
        lines.append("")
        
        # All Picks
        lines.append(f"---")
        lines.append("")
        lines.append(f"## 📋 ALL HIGH-CONFIDENCE PICKS")
        lines.append("")
        
        for i, (_, p) in enumerate(top_picks.sort_values('AI_Confidence', ascending=False).iterrows()):
            lines.append(f"**{i+1}. {p['Player']}** ({p['Team']}) - {p['Prop']} **{p['AI_Prediction']}** {p['Line']} | Conf: **{int(p['AI_Confidence'])}/10**")
            reason = str(p.get('AI_Reason', ''))
            if len(reason) > 100:
                reason = reason[:100] + "..."
            lines.append(f"   > {reason}")
            lines.append("")
        
        # AI Parlays
        if ai_parlay_list:
            lines.append(f"---")
            lines.append("")
            lines.append(f"## 🧠 AI-OPTIMIZED PARLAYS")
            lines.append("")
            lines.append(f"*Parlays created with correlation analysis and historical learning.*")
            lines.append("")
            
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            
            for i, parlay in enumerate(ai_parlay_list, 1):
                name = parlay.get('name', f'Parlay #{i}')
                theme = parlay.get('theme', '')
                reasoning = parlay.get('reasoning', '')
                risk = parlay.get('risk_level', 'medium')
                avg_conf = parlay.get('avg_confidence', 0)
                picks = parlay.get('picks', [])
                
                lines.append(f"### {i}. {name} {risk_emoji.get(risk, '🟡')}")
                lines.append(f"**Theme:** {theme}")
                lines.append(f"**Risk:** {risk.capitalize()} | **Avg Confidence:** {avg_conf:.1f}/10 | **Legs:** {len(picks)}")
                lines.append("")
                
                lines.append("| Player | Team | Prop | Pick | Line | Conf |")
                lines.append("|--------|------|------|------|------|------|")
                for pick in picks:
                    lines.append(f"| {pick['player']} | {pick['team']} | {pick['prop']} | **{pick['prediction']}** | {pick['line']} | {pick['confidence']}/10 |")
                lines.append("")
                
                lines.append(f"> **AI Reasoning:** {reasoning}")
                lines.append("")
        
        # Backup Power Plays
        if power_slips:
            lines.append(f"---")
            lines.append("")
            lines.append(f"## ⚡ BACKUP: POWER PLAYS (2-Pick)")
            lines.append("")
            
            for i, slip in enumerate(power_slips):
                avg_conf = sum(p['AI_Confidence'] for p in slip) / len(slip)
                lines.append(f"### Slip #{i+1}")
                lines.append(f"**Avg Confidence: {avg_conf:.1f}/10**")
                lines.append("")
                for p in slip:
                    lines.append(f"- **{p['Player']}** ({p['Team']}): {p['Prop']} **{p['AI_Prediction']}** {p['Line']} (Conf: {int(p['AI_Confidence'])})")
                lines.append("")
        
        # Flex Plays
        if flex_slips:
            lines.append(f"---")
            lines.append("")
            lines.append(f"## 🎲 FLEX PLAYS (5-Pick)")
            lines.append("")
            
            for i, slip in enumerate(flex_slips):
                avg_conf = sum(p['AI_Confidence'] for p in slip) / len(slip)
                lines.append(f"### Slip #{i+1}")
                lines.append(f"**Avg Confidence: {avg_conf:.1f}/10**")
                lines.append("")
                for p in slip:
                    lines.append(f"- **{p['Player']}** ({p['Team']}): {p['Prop']} **{p['AI_Prediction']}** {p['Line']} (Conf: {int(p['AI_Confidence'])})")
                lines.append("")
        
        lines.append(f"---")
        lines.append("")
        lines.append(f"*Generated by Self-Evolving NBA Neural System v2.0*")
        lines.append(f"*Timestamp: {self.timestamp}*")
        lines.append(f"*Good luck! 🍀*")
        
        return "\n".join(lines)


# ============================================================================
# PARLAY EVALUATOR
# ============================================================================

class ParlayEvaluator:
    """
    Evaluates past parlays against actual results to enable learning.
    """
    
    def __init__(self):
        self.learning_system = ParlayLearningSystem()
    
    def evaluate_parlay_file(self, parlay_file: str, actuals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates parlays from a betting slips file against actual results.
        
        Args:
            parlay_file: Path to betting slips markdown file
            actuals: Dictionary of actual results {player_name: stats}
        
        Returns:
            Evaluation results
        """
        # This would parse the markdown file and evaluate each parlay
        # For now, returns placeholder
        print(f"📊 Evaluating parlays from: {parlay_file}")
        
        # TODO: Implement markdown parsing and evaluation
        return {"evaluated": 0, "wins": 0, "losses": 0}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_optimizer(predictions_path: str = None, min_confidence: int = 8) -> str:
    """
    Runs the optimizer on a predictions file.
    """
    os.makedirs(PARLAYS_DIR, exist_ok=True)
    os.makedirs(TRAINING_REPORTS_DIR, exist_ok=True)
    
    # Find predictions file
    if predictions_path is None:
        files = [f for f in os.listdir(PREDICTIONS_DIR) 
                 if f.startswith('nba_predictions_') and f.endswith('.csv')]
        
        if not files:
            print("❌ No prediction files found in predictions/")
            return ""
        
        files.sort(reverse=True)
        predictions_path = os.path.join(PREDICTIONS_DIR, files[0])
        print(f"📂 Using most recent: {files[0]}")
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    print(f"📊 Loaded {len(df)} props from predictions")
    
    # Create optimizer
    opt = Optimizer(df)
    
    # Show learning stats
    stats = opt.learning_system.get_performance_stats()
    if stats.get('total_parlays', 0) > 0:
        print(f"\n📈 Historical Performance:")
        print(f"   Overall Win Rate: {stats.get('overall_win_rate', 0):.1f}%")
        print(f"   Total Parlays: {stats.get('total_parlays', 0)}")
    
    # Get top picks
    top_picks = opt.filter_top_picks(min_confidence=min_confidence)
    print(f"\n🎯 Found {len(top_picks)} HIGH-CONFIDENCE picks (>= {min_confidence}/10)")
    
    if len(top_picks) == 0:
        print("No high-confidence picks found.")
        return ""
    
    # Show all top picks
    print("\n=== ALL HIGH-CONFIDENCE PICKS ===")
    for i, (_, p) in enumerate(top_picks.sort_values('AI_Confidence', ascending=False).iterrows()):
        print(f"{i+1}. {p['Player']} ({p['Team']}) - {p['Prop']} {p['AI_Prediction']} {p['Line']} | Conf: {p['AI_Confidence']}/10")
    
    # Generate AI parlays
    ai_parlays = opt.generate_ai_parlays(top_picks)
    
    if ai_parlays.get('parlays'):
        print(f"\n=== AI-OPTIMIZED PARLAYS ===")
        for i, parlay in enumerate(ai_parlays['parlays'], 1):
            print(f"\n{i}. {parlay.get('name', 'Parlay')} ({parlay.get('risk_level', 'medium')} risk)")
            print(f"   Theme: {parlay.get('theme', '')}")
            for pick in parlay.get('picks', []):
                print(f"   - {pick['player']}: {pick['prop']} {pick['prediction']} {pick['line']}")
    
    # Generate backup power plays
    power_slips = opt.generate_power_plays(top_picks, slip_size=2)
    
    # Generate flex plays if enough picks
    flex_slips = []
    if len(top_picks) >= 5:
        flex_slips = opt.generate_flex_plays(top_picks, slip_size=5)
    
    # Generate markdown report
    report = opt.generate_markdown_report(
        top_picks,
        ai_parlays=ai_parlays,
        power_slips=power_slips,
        flex_slips=flex_slips
    )
    
    # Determine output filename with timestamp
    md_filename = f"betting_slips_{opt.timestamp}.md"
    md_path = os.path.join(PARLAYS_DIR, md_filename)
    
    with open(md_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Betting slips saved to: parlays/{md_filename}")
    return md_path


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🎰 THE STRATEGIST - Parlay Optimizer v2.0")
    print("   Now with Self-Learning Capabilities!")
    print("="*60)
    
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else None
    min_conf = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    run_optimizer(predictions_path, min_conf)
