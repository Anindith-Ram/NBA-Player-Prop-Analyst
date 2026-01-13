"""
THE NEURAL CORE (nba_ai_pipeline.py)
====================================
Self-Evolving NBA Neural System v3.0 - Unified Pipeline

This file combines:
- THE GYMNASIUM (Reflexive Trainer): Practices on historical data to refine intuition
- THE LIVE ENGINE (Production Pipeline): Real-time predictions with evolved intelligence
- HISTORICAL TRAINING (PySpark): Scalable backtesting with Predict-Then-Reveal architecture

TRAINING MODES:
1. HISTORICAL TRAINING: Backtest on past data to refine reasoning BEFORE live betting
   - Uses PySpark for scalable data processing
   - Model predicts WITHOUT seeing outcomes (temporal blindness)
   - Learns from mistakes and evolves its thinking
   
2. LIVE TRAINING: Evaluate yesterday's predictions against actual results
   - Takes existing predictions and compares to actuals
   - Generates case studies from failures
   - Daily feedback loop for continuous improvement

Key Features:
1. REFLEXIVE INTELLIGENCE: No static rules - builds Case Studies from experience
2. SELF-EVOLVING PROMPTS: Rewrites its own System Prompt based on learning
3. RAG INJECTION: Retrieves relevant past mistakes when analyzing similar players
4. ZERO DATA LEAKAGE: Strict temporal blindness during training
5. COMPREHENSIVE REPORTING: Full visibility into what the model learns
"""

import os
import json
import re
import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from tqdm import tqdm
from pyspark.sql.functions import col

# Import shared configuration
from shared_config import (
    get_gemini_model,
    get_session_cache,
    reset_session_cache,
    get_spark_session,
    stop_spark_session,
    IndexedCaseStudyStore,
    CASE_STUDY_LIBRARY_PATH,
    SYSTEM_PROMPT_PATH,
    DATASETS_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
    PARLAYS_DIR,
    TRAINING_REPORTS_DIR,
    MISSING_FEATURES_LOG,
    wait_for_api,
    GEMINI_API_DELAY
)

# Import the Time Machine (Data Builder)
from nba_data_builder import (
    generate_features, 
    get_actual_results_for_players,
    fetch_live_odds,
    apply_smart_filter,
    run_live_pipeline,
    # PySpark functions for historical training
    build_training_datasets,
    get_features_for_date,
    get_actuals_for_date,
    compare_predictions_to_actuals,
    apply_quality_scoring_to_df
)

# Import the Training Reporter
from training_reporter import (
    TrainingReporter,
    PredictionResult,
    CaseStudySummary
)

# Import ML Predictor (optional - graceful degradation if not available)
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_pipeline'))
    from ml_pipeline.inference import get_ml_predictor, add_ml_predictions
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False
    print("⚠️ ML Predictor not available. Running without ML signals.")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESS_DIR = SCRIPT_DIR

# Additional local paths
PROMPT_EVOLUTION_LOG_PATH = os.path.join(LOGS_DIR, "prompt_evolution.log")
MISSING_FEATURES_LOG_PATH = MISSING_FEATURES_LOG

# Legacy brain file (for migration)
LEGACY_BRAIN_PATH = os.path.join(PROCESS_DIR, "analyst_brain.json")

# Training configuration
EVOLUTION_THRESHOLD = 100  # Rewrite system prompt after this many predictions
MAX_CASE_STUDIES = 200    # Maximum case studies to retain
HIGH_CONFIDENCE_THRESHOLD = 75  # Minimum confidence % for case study generation
DEFAULT_PROP_TYPES = ["PTS", "REB", "AST", "3PM"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CaseStudy:
    """A single learning experience from a past mistake."""
    id: str
    date: str
    archetype: str          # e.g., "High Usage Star vs Slow Pace Team"
    player_tier: str        # STAR, STARTER, ROTATION, BENCH
    prop_type: str          # POINTS, REBOUNDS, ASSISTS, FG3M
    context_tags: List[str] # e.g., ["b2b", "elite_defense", "blowout"]
    mistake: str            # What went wrong
    correction_vector: str  # The learned correction
    original_confidence: int
    actual_outcome: str     # OVER or UNDER
    predicted_outcome: str
    player_name: str        # For reference
    embedding_key: str      # For similarity search
    game_date: str = ""     # Date of the game (YYYY-MM-DD) - prevents data leakage
    # Severity metrics - proportional learning
    miss_margin: float = 0.0        # Actual - Line (signed, negative = missed under for OVER bet)
    miss_margin_pct: float = 0.0    # |miss_margin| / Line * 100
    severity: str = "UNKNOWN"       # MINOR (<1 unit), MODERATE (1-2), MAJOR (>2)
    reliability_tag: str = "UNKNOWN" # SNIPER, STANDARD, VOLATILE


@dataclass
class TrainingResult:
    """Results from a single training prediction."""
    player: str
    prop: str
    line: float
    prediction: str
    confidence: int
    actual_value: float
    actual_outcome: str
    is_correct: bool
    player_tier: str
    context: Dict[str, Any]
    reason: str
    game_date: str = ""  # Date of the game (YYYY-MM-DD)


# ============================================================================
# SEVERITY CALCULATION
# ============================================================================

def calculate_miss_severity(actual: float, line: float, prediction: str) -> Tuple[float, float, str]:
    """
    Calculate how badly the model missed a prediction.
    
    Severity is proportional to how wrong the prediction was:
    - MINOR: Miss by <1 unit OR <10% of line (variance, not model error)
    - MODERATE: Miss by 1-2 units OR 10-25% of line
    - MAJOR: Miss by >2 units AND >25% of line (significant model error)
    
    Args:
        actual: Actual stat value achieved
        line: The betting line
        prediction: "OVER" or "UNDER"
        
    Returns:
        Tuple of (miss_margin, miss_margin_pct, severity)
        - miss_margin: Signed value showing direction of miss
        - miss_margin_pct: Absolute percentage of line
        - severity: "MINOR", "MODERATE", or "MAJOR"
    """
    # Calculate miss margin based on prediction direction
    if prediction.upper() == "OVER":
        # For OVER bets, we wanted actual > line
        # Negative margin means we missed under the line
        miss_margin = actual - line
    else:  # UNDER
        # For UNDER bets, we wanted actual < line
        # Negative margin means we missed over the line
        miss_margin = line - actual
    
    # Calculate percentage of line
    miss_margin_pct = (abs(miss_margin) / line * 100) if line > 0 else 0
    
    # Determine severity
    abs_margin = abs(miss_margin)
    
    if abs_margin < 1.0 or miss_margin_pct < 10:
        # Close call - variance, not model error
        severity = "MINOR"
    elif abs_margin < 2.0 or miss_margin_pct < 25:
        # Moderate miss - one factor was likely misjudged
        severity = "MODERATE"
    else:
        # Major miss - fundamental error in analysis
        severity = "MAJOR"
    
    return miss_margin, miss_margin_pct, severity


# ============================================================================
# CASE STUDY LIBRARY
# ============================================================================

class CaseStudyLibrary:
    """
    Manages the collection of Case Studies - the system's experiential memory.
    
    OPTIMIZED: Uses indexed storage for O(1) retrieval instead of O(n) linear search.
    """
    
    def __init__(self, library_path: str = CASE_STUDY_LIBRARY_PATH):
        self.library_path = library_path
        self.case_studies: List[CaseStudy] = []
        
        # OPTIMIZATION: Indexed storage for fast retrieval
        self._index_by_tier: Dict[str, List[int]] = {}
        self._index_by_prop: Dict[str, List[int]] = {}
        self._index_by_tag: Dict[str, List[int]] = {}
        
        self._load()
    
    def _load(self):
        """Loads case studies from JSON file and builds indexes."""
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, 'r') as f:
                    data = json.load(f)
                    if data and isinstance(data, list) and len(data) > 0:
                        self.case_studies = [CaseStudy(**cs) for cs in data]
                        self._rebuild_indexes()
                        print(f"📚 Loaded {len(self.case_studies)} case studies from memory")
                        # Show breakdown by archetype
                        archetypes = {}
                        for cs in self.case_studies:
                            archetypes[cs.archetype] = archetypes.get(cs.archetype, 0) + 1
                        top_archetypes = sorted(archetypes.items(), key=lambda x: x[1], reverse=True)[:5]
                        if top_archetypes:
                            print(f"   Top patterns: {', '.join([f'{arch}({count})' for arch, count in top_archetypes])}")
                    else:
                        self.case_studies = []
                        print(f"📚 Case study library is empty")
            except Exception as e:
                print(f"⚠️ Warning: Error loading case studies: {e}")
                self.case_studies = []
        else:
            self.case_studies = []
            print(f"📚 Starting with fresh case study library")
    
    def _rebuild_indexes(self):
        """Rebuilds all indexes from the case studies list."""
        self._index_by_tier.clear()
        self._index_by_prop.clear()
        self._index_by_tag.clear()
        
        for idx, cs in enumerate(self.case_studies):
            # Index by tier
            if cs.player_tier not in self._index_by_tier:
                self._index_by_tier[cs.player_tier] = []
            self._index_by_tier[cs.player_tier].append(idx)
            
            # Index by prop type
            if cs.prop_type not in self._index_by_prop:
                self._index_by_prop[cs.prop_type] = []
            self._index_by_prop[cs.prop_type].append(idx)
            
            # Index by each context tag
            for tag in cs.context_tags:
                if tag not in self._index_by_tag:
                    self._index_by_tag[tag] = []
                self._index_by_tag[tag].append(idx)
    
    def save(self):
        """Persists case studies to JSON file."""
        os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
        data = [asdict(cs) for cs in self.case_studies]
        with open(self.library_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.case_studies)} case studies")
    
    def add_case_study(self, case_study: CaseStudy):
        """Adds a new case study, updates indexes, and prunes if necessary."""
        self.case_studies.append(case_study)
        idx = len(self.case_studies) - 1
        
        # Update indexes
        if case_study.player_tier not in self._index_by_tier:
            self._index_by_tier[case_study.player_tier] = []
        self._index_by_tier[case_study.player_tier].append(idx)
        
        if case_study.prop_type not in self._index_by_prop:
            self._index_by_prop[case_study.prop_type] = []
        self._index_by_prop[case_study.prop_type].append(idx)
        
        for tag in case_study.context_tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = []
            self._index_by_tag[tag].append(idx)
        
        # Prune if over limit
        if len(self.case_studies) > MAX_CASE_STUDIES:
            self.case_studies = sorted(
                self.case_studies, 
                key=lambda x: x.date, 
                reverse=True
            )[:MAX_CASE_STUDIES]
            self._rebuild_indexes()
    
    def get_archetype_summary(self) -> Dict[str, int]:
        """Returns count of case studies by archetype for analysis."""
        return {archetype: sum(1 for cs in self.case_studies if cs.archetype == archetype) 
                for archetype in set(cs.archetype for cs in self.case_studies)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the case study library."""
        return {
            "total_case_studies": len(self.case_studies),
            "unique_tiers": len(self._index_by_tier),
            "unique_props": len(self._index_by_prop),
            "unique_tags": len(self._index_by_tag),
            "archetypes": self.get_archetype_summary()
        }


# ============================================================================
# CONFIDENCE CALIBRATOR
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrates prediction confidence based on historical accuracy.
    
    Tracks confidence vs actual accuracy and adjusts future predictions
    to better reflect true probability.
    """
    
    def __init__(self, calibration_file: str = None):
        self.calibration_file = calibration_file or os.path.join(
            os.path.dirname(CASE_STUDY_LIBRARY_PATH), 
            "confidence_calibration.json"
        )
        self.calibration_data = self._load_calibration()
    
    def _load_calibration(self) -> Dict[str, Any]:
        """Loads historical confidence calibration data."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
            except:
                return self._default_calibration()
        return self._default_calibration()
    
    def _default_calibration(self) -> Dict[str, Any]:
        """Returns default calibration structure."""
        return {
            "confidence_buckets": {str(i): {"total": 0, "correct": 0, "accuracy": 0.0} for i in range(1, 11)},
            "tier_calibration": {tier: {"total": 0, "correct": 0, "accuracy": 0.0} 
                                for tier in ["STAR", "SECONDARY", "ROTATION", "ROLE"]},
            "prop_calibration": {prop: {"total": 0, "correct": 0, "accuracy": 0.0} 
                                for prop in ["POINTS", "REBOUNDS", "ASSISTS", "FG3M"]},
            "last_updated": None
        }
    
    def record_prediction(self, confidence: int, is_correct: bool, 
                         player_tier: str = None, prop_type: str = None):
        """Records a prediction outcome for calibration."""
        conf_str = str(confidence)
        
        # Update confidence bucket
        if conf_str in self.calibration_data["confidence_buckets"]:
            bucket = self.calibration_data["confidence_buckets"][conf_str]
            bucket["total"] += 1
            if is_correct:
                bucket["correct"] += 1
            bucket["accuracy"] = (bucket["correct"] / bucket["total"] * 100) if bucket["total"] > 0 else 0.0
        
        # Update tier calibration
        if player_tier and player_tier in self.calibration_data["tier_calibration"]:
            tier_data = self.calibration_data["tier_calibration"][player_tier]
            tier_data["total"] += 1
            if is_correct:
                tier_data["correct"] += 1
            tier_data["accuracy"] = (tier_data["correct"] / tier_data["total"] * 100) if tier_data["total"] > 0 else 0.0
        
        # Update prop calibration
        if prop_type and prop_type in self.calibration_data["prop_calibration"]:
            prop_data = self.calibration_data["prop_calibration"][prop_type]
            prop_data["total"] += 1
            if is_correct:
                prop_data["correct"] += 1
            prop_data["accuracy"] = (prop_data["correct"] / prop_data["total"] * 100) if prop_data["total"] > 0 else 0.0
        
        self.calibration_data["last_updated"] = datetime.now().isoformat()
        self._save_calibration()
    
    def calibrate_confidence(self, raw_confidence: int, 
                            player_tier: str = None, prop_type: str = None) -> int:
        """Adjusts confidence based on historical accuracy."""
        conf_str = str(raw_confidence)
        
        if conf_str in self.calibration_data["confidence_buckets"]:
            bucket = self.calibration_data["confidence_buckets"][conf_str]
            if bucket["total"] >= 10:
                actual_accuracy = bucket["accuracy"]
                expected_accuracy = 50 + (raw_confidence - 5) * 10
                
                if actual_accuracy < expected_accuracy - 10:
                    adjustment = -1
                else:
                    adjustment = 0
                
                calibrated = max(1, min(10, raw_confidence + adjustment))
                
                # Apply tier-specific adjustment
                if player_tier and player_tier in self.calibration_data["tier_calibration"]:
                    tier_data = self.calibration_data["tier_calibration"][player_tier]
                    if tier_data["total"] >= 10 and tier_data["accuracy"] < 50:
                            calibrated = max(1, calibrated - 1)
                
                # Apply prop-specific adjustment
                if prop_type and prop_type in self.calibration_data["prop_calibration"]:
                    prop_data = self.calibration_data["prop_calibration"][prop_type]
                    if prop_data["total"] >= 10 and prop_data["accuracy"] < 50:
                            calibrated = max(1, calibrated - 1)
                
                return calibrated
        
        return raw_confidence
    
    def _save_calibration(self):
        """Saves calibration data to file."""
        try:
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save calibration data: {e}")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Returns summary of calibration performance."""
        total_all = sum(d["total"] for d in self.calibration_data["confidence_buckets"].values())
        correct_all = sum(d["correct"] for d in self.calibration_data["confidence_buckets"].values())
        
        return {
            "overall_accuracy": (correct_all / total_all * 100) if total_all > 0 else 0.0,
            "total_predictions": total_all
        }


# ============================================================================
# SYSTEM PROMPT EVOLUTION
# ============================================================================

class EvolvingSystemPrompt:
    """
    Manages the self-evolving system prompt.
    
    The prompt evolves through:
    1. Accumulating case studies
    2. Meta-learning sessions every N predictions
    3. Automatic rewriting based on patterns in failures
    """
    
    DEFAULT_PROMPT = """
### SYSTEM ROLE & OBJECTIVE
You are the **"Reflexive Analyst,"** an AI sports betting analyst that learns from experience.

### CORE PHILOSOPHY
- Every prediction is informed by past experience
- Similar contexts often produce similar outcomes
- Confidence should reflect both data quality AND historical accuracy

### PLAYER TIER CLASSIFICATION
**STAR (USG% > 22% or PIE > 0.12):** High volume, consistent touches - trust small edges
**SECONDARY (USG% 18-22%):** Key contributors but not primary options
**ROTATION (USG% 15-18%):** Situational contributors
**ROLE PLAYER (USG% < 18%):** Volatile, require large edges (>3.5 pts)
        
        ### OUTPUT FORMAT (JSON ONLY)
Return a valid JSON object. No markdown outside JSON.
{
            "prediction": "OVER" or "UNDER",
            "confidence": <integer 1-10>,
    "risk_factor": "<primary risk if any>",
    "reason": "<1-2 sentence synthesis>"
}
        """

    def __init__(self, prompt_path: str = SYSTEM_PROMPT_PATH):
        self.prompt_path = prompt_path
        self.current_prompt = self._load()
        self.prediction_count = 0
    
    def _load(self) -> str:
        """Loads the current system prompt from file."""
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, 'r') as f:
                prompt = f.read()
                if prompt.strip():
                    print(f"🧠 Loaded evolved system prompt ({len(prompt)} chars)")
                    return prompt
        
        self.save(self.DEFAULT_PROMPT)
        return self.DEFAULT_PROMPT
    
    def save(self, prompt: str = None):
        """Saves the current system prompt."""
        if prompt:
            self.current_prompt = prompt
        
        os.makedirs(os.path.dirname(self.prompt_path), exist_ok=True)
        with open(self.prompt_path, 'w') as f:
            f.write(self.current_prompt)
    
    def get_prompt(self) -> str:
        """Returns the current system prompt."""
        return self.current_prompt
    
    def evolve(self, case_studies: List[CaseStudy], model: Any) -> List[str]:
        """Meta-learning: Rewrites the system prompt based on accumulated case studies."""
        if len(case_studies) < 10:
            print("Not enough case studies for evolution yet.")
            return []
        
        # Group case studies by archetype
        archetype_counts = {}
        archetype_examples = {}
        
        for cs in case_studies[-50:]:
            key = cs.archetype
            archetype_counts[key] = archetype_counts.get(key, 0) + 1
            if key not in archetype_examples:
                archetype_examples[key] = cs
        
        recurring = [(k, v) for k, v in archetype_counts.items() if v >= 3]
        recurring.sort(key=lambda x: x[1], reverse=True)
        
        if not recurring:
            print("No recurring patterns found for evolution.")
            return []
        
        patterns_text = "\n".join([
            f"- {arch}: {count} occurrences. Correction: '{archetype_examples[arch].correction_vector}'"
            for arch, count in recurring[:10]
        ])
        
        evolution_prompt = f"""
You are a meta-learning AI that improves its own instructions.

### CURRENT SYSTEM PROMPT:
{self.current_prompt[:2000]}...

### RECURRING MISTAKE PATTERNS:
{patterns_text}

### TASK:
Rewrite the system prompt to incorporate these learnings.
Output ONLY the new system prompt text. No explanations.
"""
        
        try:
            print("\n🧬 EVOLVING SYSTEM PROMPT...")
            response = model.generate_content(evolution_prompt)
            new_prompt = response.text.strip()
            
            if len(new_prompt) > 500:
                self._log_evolution(self.current_prompt, new_prompt, recurring)
                self.current_prompt = new_prompt
                self.save()
                print(f"✅ System prompt evolved! Incorporated {len(recurring)} patterns")
                return [f"{arch} ({count})" for arch, count in recurring]
            else:
                print("Evolution produced invalid output, keeping current prompt.")
                return []
                
        except Exception as e:
            print(f"Evolution failed: {e}")
            return []
    
    def _log_evolution(self, old_prompt: str, new_prompt: str, patterns: List[Tuple[str, int]]):
        """Logs prompt evolution for audit trail."""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "patterns_addressed": [{"archetype": p[0], "count": p[1]} for p in patterns],
            "old_length": len(old_prompt),
            "new_length": len(new_prompt),
        }
        
        with open(PROMPT_EVOLUTION_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")


# ============================================================================
# THE REFLEXIVE PREDICTION ENGINE
# ============================================================================

class ReflexivePredictionEngine:
    """
    The core prediction engine with reflexive learning capabilities.
    
    Modes:
    - TRAINING: Predicts on historical data, learns from mistakes
    - PRODUCTION: Predicts on live data with evolved intelligence
    """
    
    def __init__(self):
        self._setup_gemini()
        self.case_library = CaseStudyLibrary()
        self.system_prompt = EvolvingSystemPrompt()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.prediction_logs = []
        self.training_results: List[TrainingResult] = []
    
    def _setup_gemini(self):
        """Configures the Gemini API client using centralized configuration."""
        self.model = get_gemini_model('gemini-3-pro-preview')
        self.reflection_model = get_gemini_model('gemini-3-pro-preview')
    
    def _save_training_data_csv(self, props: List[Dict], target_date: str):
        """
        Saves the COMPLETE data sent to the model as a CSV file.
        Includes ALL features from the data builder (100+ columns).
        
        Creates: datasets/training_inputs/training_data_{target_date}_{timestamp}.csv
        """
        import pandas as pd
        from datetime import datetime
        
        # Create directory if it doesn't exist
        output_dir = Path(__file__).parent.parent / "datasets" / "training_inputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{target_date}_{timestamp}.csv"
        filepath = output_dir / filename
        
        # Convert to DataFrame with ALL columns
        if not props:
            print(f"   ⚠️ No props to save for {target_date}")
            return
        
        # Build DataFrame with ALL available columns (100+ features)
        df_export = pd.DataFrame(props)
        
        # Add metadata columns at the front (only if they don't already exist)
        if 'generation_timestamp' not in df_export.columns:
            df_export.insert(0, 'generation_timestamp', timestamp)
        else:
            df_export['generation_timestamp'] = timestamp
        
        if 'game_date' not in df_export.columns:
            df_export.insert(1, 'game_date', target_date)
        # else: game_date already exists from data builder, keep it
        
        # Reorder columns: metadata first, then key identifiers, then everything else
        priority_cols = [
            'generation_timestamp', 'game_date',
            # Player identification
            'firstName', 'lastName', 'personId', 'teamTricode', 'player_key',
            # Prop details
            'Prop_Type_Normalized', 'Line', 'Over_Odds', 'Under_Odds',
            # Quality metrics
            'quality_score', 'EV', 'Kelly', 'Z_Score', 'Edge',
            # Classification
            'Player_Tier', 'Reliability_Tag', 'CV_Score',
        ]
        
        # Get priority columns that exist, then all other columns
        existing_priority = [c for c in priority_cols if c in df_export.columns]
        other_cols = [c for c in df_export.columns if c not in priority_cols]
        ordered_cols = existing_priority + sorted(other_cols)
        
        df_export = df_export[ordered_cols]
        
        # Save to CSV
        df_export.to_csv(filepath, index=False)
        total_cols = len(df_export.columns)
        print(f"   💾 Saved training data: {filename} ({len(props)} props, {total_cols} features)")
    
    def _get_player_tier(self, usg: float, pie: float) -> str:
        """Classifies player into tier."""
        if usg > 0.25 or pie > 0.15:
            return "STAR"
        elif usg > 0.22 or pie > 0.12:
            return "SECONDARY"
        elif usg > 0.18 or pie > 0.08:
            return "ROTATION"
        else:
            return "ROLE"
    
    def _get_context_tags(self, row: pd.Series) -> List[str]:
        """Extracts context tags for case study matching."""
        tags = []
        
        if row.get('Is_B2B', 0) == 1:
            tags.append("b2b")
        
        opp_rank = float(row.get('Opp_Rank', 15) or 15)
        if opp_rank <= 5:
            tags.append("elite_defense")
        elif opp_rank <= 10:
            tags.append("good_defense")
        elif opp_rank >= 25:
            tags.append("weak_defense")
        
        pace = float(row.get('Opp_Pace', 100) or 100)
        if pace > 102:
            tags.append("fast_pace")
        elif pace < 97:
            tags.append("slow_pace")
        
        spread = abs(float(row.get('Vegas_Spread', 0) or 0))
        if spread > 14:
            tags.append("blowout_risk")
        elif spread > 10:
            tags.append("moderate_spread")
        
        total = float(row.get('Game_Total', 220) or 220)
        if total > 235:
            tags.append("high_total")
        elif total < 215:
            tags.append("low_total")
        
        edge = float(row.get('Diff_L5', 0) or 0)
        if abs(edge) > 5:
            tags.append("large_edge")
        elif abs(edge) < 2:
            tags.append("small_edge")
        
        return tags
    
    def _construct_prompt(self, player_data: pd.Series) -> str:
        """Builds the Hybrid Reflexive analysis prompt."""
        row = player_data
        
        player_name = row.get('Player', 'Unknown')
        opponent = row.get('Opponent', 'Unknown')
        location = row.get('Location', 'Unknown')
        prop = row.get('Prop', 'POINTS')
        line = float(row.get('Line', 0) or 0)
        
        usg = float(row.get('USG_PCT', 0) or 0)
        pie = float(row.get('PIE', 0) or 0)
        player_tier = self._get_player_tier(usg, pie)
        
        l5_avg = float(row.get('L5_Avg', 0) or 0)
        season_avg = float(row.get('Season_Avg', 0) or 0)
        edge = float(row.get('Diff_L5', 0) or 0)
        hit_rate = float(row.get('Hit_Rate', 0) or 0)
        opp_rank = float(row.get('Opp_Rank', 15) or 15)
        opp_pace = float(row.get('Opp_Pace', 100) or 100)
        spread = float(row.get('Vegas_Spread', 0) or 0)
        game_total = float(row.get('Game_Total', 220) or 220)
        is_b2b = int(row.get('Is_B2B', 0) or 0)
        
        # Load ALL Case Studies
        all_case_studies = self.case_library.case_studies
        
        lessons_lines = []
        if all_case_studies:
            lessons_lines.append("**PAST EXPERIENCES (Case Studies):**")
            for cs in all_case_studies[:50]:  # Limit to most recent 50
                lessons_lines.append(f"  • **{cs.archetype}**: {cs.correction_vector}")
        else:
            lessons_lines.append("  No past experiences found.")
        
        lessons_text = "\n".join(lessons_lines)
        
        prompt = f"""
### SYSTEM ROLE
You are the **"Reflexive Analyst,"** an elite AI sports betting analyst.

### PLAYER CONTEXT
- Player: **{player_name}** vs **{opponent}** ({location})
- Player Tier: {player_tier} (USG: {usg*100:.1f}%)
- Prop: {prop} | Line: {line}

### STATISTICAL DATA
- L5 Avg: {l5_avg:.1f} | Season Avg: {season_avg:.1f}
- Edge (L5 - Line): {edge:+.1f} points
- Historical Hit Rate: {hit_rate:.1f}%
- Opponent Defense Rank: {opp_rank:.0f}/30
- Opponent Pace: {opp_pace:.1f}
- Vegas Spread: {spread:+.1f} | Game Total: {game_total:.1f}
- Back-to-Back: {'YES' if is_b2b == 1 else 'NO'}

### REFLEXIVE LAYER (Past Mistakes)
{lessons_text}

### OUTPUT FORMAT (JSON ONLY)
{{
    "archetype": "<player situation type>",
    "prediction": "OVER" or "UNDER",
    "confidence": <integer 1-10>,
    "reason": "<synthesis of data + experience>"
}}
"""
        return prompt

    def analyze_prop(self, player_row: pd.Series, max_retries: int = 3) -> Dict[str, Any]:
        """Analyzes a single prop with reflexive intelligence."""
        prompt = self._construct_prompt(player_row)
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                wait_for_api("gemini", GEMINI_API_DELAY)
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1
                    },
                    request_options={"timeout": 60}
                )
                
                inference_time = (time.time() - start_time) * 1000
                result = json.loads(response.text)
                
                # Calibrate confidence
                raw_confidence = result.get('confidence', 5)
                usg = float(player_row.get('USG_PCT', 0) or 0)
                pie = float(player_row.get('PIE', 0) or 0)
                player_tier = self._get_player_tier(usg, pie)
                prop_type = player_row.get('Prop', 'POINTS')
                
                calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
                    raw_confidence=raw_confidence,
                    player_tier=player_tier,
                    prop_type=prop_type
                )
                
                result['confidence'] = calibrated_confidence
                if raw_confidence != calibrated_confidence:
                    result['original_confidence'] = raw_confidence
                
                # Log prediction
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "player": player_row.get('Player'),
                    "prop": prop_type,
                    "line": player_row.get('Line'),
                    "prediction": result.get('prediction'),
                    "confidence": calibrated_confidence,
                    "inference_time_ms": round(inference_time, 1)
                }
                self.prediction_logs.append(log_entry)
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ['504', 'deadline', 'timeout', '503', '429'])
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"⚠️ Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error analyzing {player_row.get('Player')}: {e}")
                    return {"prediction": "ERROR", "confidence": 0, "reason": str(e)}
        
        return {"prediction": "ERROR", "confidence": 0, "reason": "Max retries exceeded"}
    
    def run_batch_analysis(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """Runs analysis on a batch of props."""
        print(f"\n🎯 Analyzing {len(df)} props with Reflexive Intelligence...")
        print(f"   📚 Case Studies: {len(self.case_library.case_studies)} available")
        
        results = []
        self.prediction_logs = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="AI Analysis"):
            analysis = self.analyze_prop(row)
            
            result_row = row.to_dict()
            result_row['AI_Prediction'] = analysis.get('prediction')
            result_row['AI_Confidence'] = analysis.get('confidence')
            result_row['AI_Archetype'] = analysis.get('archetype', '')
            result_row['AI_Reason'] = analysis.get('reason', '')
            
            results.append(result_row)
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"💾 Analysis saved to {output_path}")
        
        return result_df

    def save_logs(self, log_path: str):
        """Saves prediction logs to JSON."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        existing_logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    existing_logs = json.load(f)
            except:
                pass
        
        all_logs = existing_logs + self.prediction_logs
        
        with open(log_path, 'w') as f:
            json.dump(all_logs, f, indent=2)
        
        print(f"📊 Saved {len(self.prediction_logs)} prediction logs")


# ============================================================================
# TRAINED DATES TRACKER (PREVENTS DATA LEAKAGE)
# ============================================================================

TRAINED_DATES_PATH = os.path.join(PROCESS_DIR, "trained_dates.json")

def load_trained_dates() -> Dict[str, List[str]]:
    """Loads the record of dates that have been trained on."""
    if os.path.exists(TRAINED_DATES_PATH):
        try:
            with open(TRAINED_DATES_PATH, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
    return {"trained_dates": [], "training_sessions": []}

def save_trained_dates(data: Dict[str, List[str]]):
    """Saves the record of trained dates."""
    with open(TRAINED_DATES_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def mark_dates_as_trained(dates: List[str], session_info: Dict = None):
    """Marks dates as trained to prevent re-training."""
    data = load_trained_dates()
    
    # Add new dates
    for date in dates:
        if date not in data["trained_dates"]:
            data["trained_dates"].append(date)
    
    # Log session info
    if session_info:
        session_info["timestamp"] = datetime.now().isoformat()
        data["training_sessions"].append(session_info)
    
    save_trained_dates(data)
    print(f"📝 Marked {len(dates)} dates as trained (total: {len(data['trained_dates'])})")

def get_untrained_dates(requested_dates: List[str]) -> List[str]:
    """Returns only dates that haven't been trained on yet."""
    data = load_trained_dates()
    trained = set(data["trained_dates"])
    untrained = [d for d in requested_dates if d not in trained]
    
    if len(untrained) < len(requested_dates):
        already_trained = len(requested_dates) - len(untrained)
        print(f"⚠️ Skipping {already_trained} dates already trained on")
    
    return untrained


# ============================================================================
# PYSPARK HISTORICAL TRAINER (SCALABLE TRAINING)
# ============================================================================

class HistoricalTrainer:
    """
    PySpark-based historical trainer for scalable backtesting.
    
    Uses the Predict-Then-Reveal architecture:
    1. Load features WITHOUT actuals (includes odds/lines)
    2. Gemini makes predictions (temporally blind)
    3. Reveal actuals AFTER predictions locked
    4. Compare predictions vs reality
    5. Learn from high-confidence mistakes → Case Studies
    
    Features:
    - Tracks trained dates to prevent data leakage
    - Supports date range training
    - Stores case studies for self-reflexive learning
    - Smart filter: Only top 10 props per date (by edge magnitude)
    """
    
    MAX_PROPS_PER_DATE = 10  # Smart filter limit
    
    def __init__(self):
        self.engine = ReflexivePredictionEngine()
        self.total_predictions = 0
        self.total_hits = 0
        self.case_studies_created = 0
        self.trained_dates_this_session = []
    
    def _smart_filter_props(self, features_list: List[Dict], max_props: int = None) -> List[Dict]:
        """
        Smart filter: Select the top props based on edge magnitude.
        
        Prioritizes:
        1. Props with largest absolute edge (|avg - Line|)
        2. Props with complete data
        3. Balanced mix of prop types
        
        Args:
            features_list: List of feature dicts
            max_props: Maximum props to return (default: MAX_PROPS_PER_DATE)
            
        Returns:
            Filtered list of top props
        """
        max_props = max_props or self.MAX_PROPS_PER_DATE
        
        # PROP TYPE MAPPING: Map from odds API names to stat column names
        PROP_TO_STAT = {
            'PTS': 'PTS', 'POINTS': 'PTS',
            'REB': 'REB', 'REBOUNDS': 'REB',
            'AST': 'AST', 'ASSISTS': 'AST',
            '3PM': '3PM', 'THREES': '3PM'
        }
        
        # Calculate edge for each prop and keep only the best entry per (player, prop_type)
        best_by_player_prop = {}
        for f in features_list:
            line = f.get('Line')
            raw_prop_type = f.get('Prop_Type', 'PTS').upper()
            stat_key = PROP_TO_STAT.get(raw_prop_type, 'PTS')
            
            # Skip if no line
            if line is None:
                continue
            
            try:
                line = float(line)
            except (ValueError, TypeError):
                continue
            
            # Get the correct averages for THIS prop type
            l5_avg = f.get(f'L5_Avg_{stat_key}')
            l10_avg = f.get(f'L10_Avg_{stat_key}')
            season_avg = f.get(f'Season_Avg_{stat_key}')
            
            # Edge with fallback: prefer L5, else L10, else Season
            edge_source = l5_avg
            if edge_source is None:
                edge_source = l10_avg
            if edge_source is None:
                edge_source = season_avg
            
            edge = None
            if edge_source is not None:
                try:
                    edge = float(edge_source) - line
                except (ValueError, TypeError):
                    edge = None
            
            # Quality score components
            quality_score = 0.0
            
            # Edge strength
            if edge is not None:
                quality_score += abs(edge) * 2.0
            
            # Data completeness
            if l5_avg is not None:
                quality_score += 2.0
            if l10_avg is not None:
                quality_score += 1.0
            if season_avg is not None:
                quality_score += 0.5
            
            # Odds quality: prefer available odds; slight bonus if line has odds and not extreme
            over_odds = f.get('Over_Odds')
            if over_odds is not None:
                quality_score += 0.5
                try:
                    oo = float(over_odds)
                    quality_score += max(0.0, 1.5 - abs(oo) / 150.0)
                except Exception:
                    pass
            
            # Rest / schedule info
            if f.get('rest_days') is not None:
                quality_score += 0.3
            if f.get('is_b2b') is not None:
                quality_score += 0.2
            
            # Team/opponent context completeness
            context_keys = [
                "L5_Team_Pace", "L5_Team_Def_Rating",
                "L5_Allowed_PTS", "L5_Allowed_REB", "L5_Allowed_AST", "L5_Allowed_3PA",
                "Opp_L5_Team_Pace", "Opp_L5_Team_Def_Rating",
                "Opp_L5_Allowed_PTS", "Opp_L5_Allowed_REB", "Opp_L5_Allowed_AST", "Opp_L5_Allowed_3PA"
            ]
            context_present = sum(1 for k in context_keys if f.get(k) is not None)
            quality_score += 0.05 * context_present  # small additive bonus
            
            # Store normalized prop type
            f['Prop_Type_Normalized'] = stat_key
            
            # Track player to avoid one-player dominance
            player_key = f"{f.get('firstName', '')}_{f.get('lastName', '')}"

            candidate = {
                **f,
                'Edge': edge if edge is not None else 0.0,
                'quality_score': quality_score,
                '_player_key': player_key
            }
            
            dedup_key = (player_key, stat_key)
            existing = best_by_player_prop.get(dedup_key)
            if existing is None:
                best_by_player_prop[dedup_key] = candidate
            else:
                # Prefer higher quality; tie-break on larger abs edge, then better Over odds (less negative)
                better = candidate
                if candidate['quality_score'] < existing['quality_score']:
                    better = existing
                elif candidate['quality_score'] == existing['quality_score']:
                    if abs(candidate['Edge']) < abs(existing['Edge']):
                        better = existing
                    else:
                        try:
                            cand_odds = float(candidate.get('Over_Odds', 0))
                            exist_odds = float(existing.get('Over_Odds', 0))
                            if abs(cand_odds) > abs(exist_odds):
                                better = existing
                        except Exception:
                            better = candidate
                best_by_player_prop[dedup_key] = better
        
        scored_props = list(best_by_player_prop.values())
        # Deterministic ordering: quality desc, abs(edge) desc, line asc, name asc
        scored_props.sort(
            key=lambda x: (
                -x['quality_score'],
                -abs(x.get('Edge', 0.0)),
                float(x.get('Line', 0) or 0),
                x.get('lastName', ''), x.get('firstName', '')
            )
        )
        
        # Select top props with balanced mix
        selected = []
        prop_type_counts = {}
        player_counts = {}
        max_per_type = max(3, max_props // 3)  # At least 3 per type
        max_per_player = 2  # Prevent single player from dominating
        
        for prop in scored_props:
            if len(selected) >= max_props:
                break
            
            prop_type = prop.get('Prop_Type_Normalized', 'PTS')
            count = prop_type_counts.get(prop_type, 0)
            player_key = prop.get('_player_key')
            player_count = player_counts.get(player_key, 0)
            
            if count >= max_per_type:
                continue
            if player_count >= max_per_player:
                continue
            
            selected.append(prop)
            prop_type_counts[prop_type] = count + 1
            player_counts[player_key] = player_count + 1
        
        return selected
    
    def preview_dataset(self, features_list: List[Dict], date: str, show_all: bool = False):
        """
        Displays a preview of what data the model will see.
        
        Args:
            features_list: List of feature dicts
            date: The date being evaluated
            show_all: If True, shows all props; otherwise shows sample
        """
        print(f"\n" + "="*70)
        print(f"📋 DATASET PREVIEW FOR {date}")
        print("="*70)
        
        if not features_list:
            print("   ⚠️ No data available")
            return
        
        # Show sample (first 5) or all
        sample = features_list if show_all else features_list[:5]
        
        print(f"\n📊 Total props available: {len(features_list)}")
        print(f"   Showing: {'All' if show_all else f'First {len(sample)}'}")
        
        # Group by prop type
        prop_counts = {}
        for f in features_list:
            pt = f.get('Prop_Type', 'UNKNOWN')
            prop_counts[pt] = prop_counts.get(pt, 0) + 1
        
        print(f"\n📈 Props by type:")
        for pt, count in sorted(prop_counts.items()):
            print(f"   {pt}: {count}")
        
        # Dataframe-style view of exactly what will be sent to Gemini
        try:
            df = pd.DataFrame(features_list)
            display_cols = [
                'game_date', 'firstName', 'lastName', 'TeamAbbrev', 'OpponentAbbrev',
                'home', 'is_b2b', 'rest_days',
                'Prop_Type_Normalized', 'Prop_Type', 'Line', 'Over_Odds', 'Implied_Over_Pct', 'Edge',
                'L5_Avg_PTS', 'L10_Avg_PTS', 'Season_Avg_PTS',
                'L5_Avg_REB', 'L10_Avg_REB', 'Season_Avg_REB',
                'L5_Avg_AST', 'L10_Avg_AST', 'Season_Avg_AST',
                'L5_Avg_3PM', 'L10_Avg_3PM', 'Season_Avg_3PM',
                'L5_Team_Pace', 'L10_Team_Pace', 'Season_Team_Pace',
                'L5_Team_Def_Rating', 'L10_Team_Def_Rating', 'Season_Team_Def_Rating',
                'L5_Allowed_PTS', 'L10_Allowed_PTS', 'Season_Allowed_PTS',
                'L5_Allowed_REB', 'L10_Allowed_REB', 'Season_Allowed_REB',
                'L5_Allowed_AST', 'L10_Allowed_AST', 'Season_Allowed_AST',
                'L5_Allowed_3PA', 'L10_Allowed_3PA', 'Season_Allowed_3PA',
                'Opp_L5_Team_Pace', 'Opp_L10_Team_Pace', 'Opp_Season_Team_Pace',
                'Opp_L5_Team_Def_Rating', 'Opp_L10_Team_Def_Rating', 'Opp_Season_Team_Def_Rating',
                'Opp_L5_Allowed_PTS', 'Opp_L10_Allowed_PTS', 'Opp_Season_Allowed_PTS',
                'Opp_L5_Allowed_REB', 'Opp_L10_Allowed_REB', 'Opp_Season_Allowed_REB',
                'Opp_L5_Allowed_AST', 'Opp_L10_Allowed_AST', 'Opp_Season_Allowed_AST',
                'Opp_L5_Allowed_3PA', 'Opp_L10_Allowed_3PA', 'Opp_Season_Allowed_3PA',
                'Opp_rest_days', 'Opp_is_b2b'
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            if display_cols:
                print(f"\n🗂️  Dataframe preview (first 10 rows actually sent to Gemini):")
                print(df[display_cols].head(10).to_string(index=False))
        except Exception as e:
            print(f"\n   ⚠️ Could not render dataframe preview: {e}")
        
        print(f"\n" + "-"*70)
        print("SAMPLE PLAYER PROPS:")
        print("-"*70)
        
        for i, f in enumerate(sample, 1):
            player_name = f"{f.get('firstName', '')} {f.get('lastName', '')}".strip() or "Unknown"
            team = f.get('TeamAbbrev', 'UNK')
            opponent = f.get('OpponentAbbrev', 'UNK')
            home = "Home" if f.get('home') == 1 else "Away"
            prop_type = f.get('Prop_Type', 'PTS')
            line = f.get('Line', 'N/A')
            
            # Get averages
            l5 = f.get(f'L5_Avg_{prop_type}') or f.get('L5_Avg_PTS', 'N/A')
            l10 = f.get(f'L10_Avg_{prop_type}') or f.get('L10_Avg_PTS', 'N/A')
            season = f.get(f'Season_Avg_{prop_type}') or f.get('Season_Avg_PTS', 'N/A')
            
            # Calculate edge
            edge = f.get('Edge')
            if edge is None and l5 != 'N/A' and line != 'N/A':
                try:
                    edge = float(l5) - float(line)
                except:
                    edge = None
            
            edge_str = f"{edge:+.1f}" if isinstance(edge, (int, float)) else "N/A"
            
            over_odds = f.get('Over_Odds', 'N/A')
            rest = f.get('rest_days', 'N/A')
            b2b = "YES" if f.get('is_b2b') else "No"
            
            print(f"\n{i}. {player_name} ({team} {home} vs {opponent})")
            print(f"   Prop: {prop_type} | Line: {line}")
            print(f"   L5: {l5} | L10: {l10} | Season: {season}")
            print(f"   Edge: {edge_str} | Odds: {over_odds}")
            print(f"   Rest: {rest} days | B2B: {b2b}")
        
        print("\n" + "-"*70)
        
        # Show available columns
        if features_list:
            cols = list(features_list[0].keys())
            print(f"\n📋 Available columns ({len(cols)}):")
            # Show important columns
            important = ['firstName', 'lastName', 'TeamAbbrev', 'OpponentAbbrev', 
                        'Prop_Type', 'Line', 'Over_Odds', 'Edge',
                        'L5_Avg_PTS', 'L10_Avg_PTS', 'Season_Avg_PTS',
                        'rest_days', 'is_b2b', 'home']
            present = [c for c in important if c in cols]
            missing = [c for c in important if c not in cols]
            
            print(f"   ✅ Present: {', '.join(present[:10])}...")
            if missing:
                print(f"   ❌ Missing: {', '.join(missing)}")
    
    def _build_prediction_prompt(self, features: List[Dict], prop_type: str = "ALL") -> str:
        """
        Builds a prompt for Gemini to make predictions.
        
        Args:
            features: List of feature dicts (already filtered)
            prop_type: "ALL" or specific type (for reference only)
        """
        system_prompt = ""
        if os.path.exists(SYSTEM_PROMPT_PATH):
            with open(SYSTEM_PROMPT_PATH, 'r') as f:
                system_prompt = f.read()
        
        # Include case studies for self-reflexive learning
        # This includes case studies from EARLIER in the current training session
        case_studies_text = ""
        available_case_studies = self.engine.case_library.case_studies
        if available_case_studies:
            # Use most recent case studies (includes those from current session)
            recent_studies = available_case_studies[-30:]  # Last 30 for recency
            case_studies_text = f"\n### PAST MISTAKES TO AVOID ({len(recent_studies)} lessons from experience):\n"
            for cs in recent_studies:
                # Include game date to show temporal context
                date_context = f" [from {cs.game_date}]" if cs.game_date else ""
                case_studies_text += f"• **{cs.archetype}**{date_context}: {cs.correction_vector}\n"
        
        prompt = f"""{system_prompt}

{case_studies_text}

You are analyzing NBA player props. For EACH prop below, predict OVER or UNDER.

=== PLAYER PROPS TO ANALYZE ===
"""
        # PROP TYPE MAPPING for averages
        PROP_TO_STAT = {
            'PTS': 'PTS', 'POINTS': 'PTS',
            'REB': 'REB', 'REBOUNDS': 'REB',
            'AST': 'AST', 'ASSISTS': 'AST',
            '3PM': '3PM', 'THREES': '3PM'
        }
        
        for i, p in enumerate(features, 1):
            player_name = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
            team = p.get('TeamAbbrev', 'UNK')
            opponent = p.get('OpponentAbbrev', 'UNK')
            home = "Home" if p.get('home') == 1 else "Away"
            
            # Get prop type for THIS specific prop
            this_prop = p.get('Prop_Type_Normalized') or p.get('Prop_Type', 'PTS')
            this_prop = this_prop.upper()
            stat_key = PROP_TO_STAT.get(this_prop, 'PTS')
            
            # Get the prop-specific data
            line = p.get('Line')
            over_odds = p.get('Over_Odds', 'N/A')
            edge = p.get('Edge')
            
            # Skip props without lines
            if line is None:
                continue
            
            # Get averages for THIS SPECIFIC PROP TYPE
            l5_avg = p.get(f'L5_Avg_{stat_key}')
            l10_avg = p.get(f'L10_Avg_{stat_key}')
            season_avg = p.get(f'Season_Avg_{stat_key}')
            
            # Calculate edge if not provided
            if edge is None and l5_avg is not None:
                try:
                    edge = float(l5_avg) - float(line)
                except:
                    edge = None
            
            rest_days = p.get('rest_days')
            is_b2b = "YES ⚠️" if p.get('is_b2b') else "No"
            
            # Team context (pace/defense/opponent allowed)
            def safe_fmt(val):
                try:
                    return f"{float(val):.1f}"
                except Exception:
                    return "N/A"
            
            team_pace = safe_fmt(p.get("L5_Team_Pace") or p.get("L10_Team_Pace") or p.get("Season_Team_Pace"))
            opp_pace = safe_fmt(p.get("Opp_L5_Team_Pace") or p.get("Opp_L10_Team_Pace") or p.get("Opp_Season_Team_Pace"))
            team_def = safe_fmt(p.get("L5_Team_Def_Rating") or p.get("L10_Team_Def_Rating") or p.get("Season_Team_Def_Rating"))
            opp_def = safe_fmt(p.get("Opp_L5_Team_Def_Rating") or p.get("Opp_L10_Team_Def_Rating") or p.get("Opp_Season_Team_Def_Rating"))
            
            allowed_map = {
                "PTS": ("Opp_L5_Allowed_PTS", "Opp_L10_Allowed_PTS", "Opp_Season_Allowed_PTS"),
                "REB": ("Opp_L5_Allowed_REB", "Opp_L10_Allowed_REB", "Opp_Season_Allowed_REB"),
                "AST": ("Opp_L5_Allowed_AST", "Opp_L10_Allowed_AST", "Opp_Season_Allowed_AST"),
                "3PM": ("Opp_L5_Allowed_3PA", "Opp_L10_Allowed_3PA", "Opp_Season_Allowed_3PA")
            }
            allowed_keys = allowed_map.get(stat_key, ("Opp_L5_Allowed_PTS", "Opp_L10_Allowed_PTS", "Opp_Season_Allowed_PTS"))
            allowed_vals = [p.get(k) for k in allowed_keys]
            opp_allowed = safe_fmt(next((v for v in allowed_vals if v is not None), None))
            
            # Format values safely
            line_str = f"{float(line):.1f}" if line is not None else "N/A"
            l5_str = f"{float(l5_avg):.1f}" if l5_avg is not None else "N/A"
            l10_str = f"{float(l10_avg):.1f}" if l10_avg is not None else "N/A"
            season_str = f"{float(season_avg):.1f}" if season_avg is not None else "N/A"
            edge_str = f"{float(edge):+.1f}" if edge is not None else "N/A"
            rest_str = f"{rest_days}" if rest_days is not None else "N/A"
            
            prompt += f"""
{i}. **{player_name}** ({team} {home} vs {opponent})
   PROP: {this_prop} | LINE: {line_str}
   L5 Avg: {l5_str} | L10 Avg: {l10_str} | Season: {season_str}
   Edge (L5 - Line): {edge_str}
   Odds: {over_odds} | Rest: {rest_str} days | B2B: {is_b2b}
   Team pace L5: {team_pace} | Opp pace L5: {opp_pace}
   Team def rtg L5: {team_def} | Opp def rtg L5: {opp_def}
   Opp allowed {stat_key} (L5): {opp_allowed}
"""
            # Add ML signal if available
            ml_prob = p.get('ml_prob_over')
            ml_pred = p.get('ml_prediction')
            ml_conf = p.get('ml_confidence')
            ml_edge = p.get('ml_edge')
            
            if ml_prob is not None and ml_pred is not None:
                ml_prob_str = f"{float(ml_prob):.1%}"
                ml_edge_str = f"{float(ml_edge):+.1%}" if ml_edge is not None else "N/A"
                prompt += f"""   **ML MODEL SIGNAL**: P(OVER)={ml_prob_str} | Pred={ml_pred} | Conf={ml_conf}/10 | Edge={ml_edge_str}
"""
        
        prompt += f"""

### OUTPUT FORMAT (JSON ONLY - NO MARKDOWN FENCES):
{{
    "predictions": [
        {{
            "player_name": "First Last",
            "prop_type": "PTS/REB/AST/3PM",
            "line": 24.5,
            "prediction": "OVER",
            "confidence": 75,
            "reasoning": "Brief explanation"
        }}
    ]
}}

IMPORTANT:
- You MUST make a prediction for EACH of the {len(features)} props above
- Use the exact player names and prop types shown
- Confidence: 0-100 (higher = more confident)
- If Edge is positive, lean OVER. If negative, lean UNDER.
- If ML MODEL SIGNAL is shown, use it as a STRONG reference point:
  * ML P(OVER) > 55% suggests OVER, < 45% suggests UNDER
  * You may override ML if you identify news/injury context or unusual factors
  * When ML confidence is high (7+/10), require strong reasoning to disagree
"""
        return prompt
        
    def _get_gemini_predictions(self, features: List[Dict], prop_type: str = "ALL") -> List[Dict]:
        """
        Gets predictions from Gemini for a batch of player props.
        
        Args:
            features: List of feature dicts (already filtered by smart filter)
            prop_type: "ALL" to predict all props, or specific type
        
        Returns:
            List of prediction dicts with personId for matching to actuals
        """
        if not features:
            return []
        
        print(f"   📤 Sending {len(features)} props to Gemini...")
        
        prompt = self._build_prediction_prompt(features, prop_type)
        
        wait_for_api("gemini", GEMINI_API_DELAY)
        
        try:
            response = self.engine.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text)
            predictions = result.get("predictions", [])
            
            # Enrich predictions with original feature data for comparison
            for pred in predictions:
                # Find matching feature to get personId and line
                player_name = pred.get("player_name", "").lower().strip()
                pred_prop = pred.get("prop_type", "").upper()
                
                for f in features:
                    full_name = f"{f.get('firstName', '')} {f.get('lastName', '')}".lower().strip()
                    feature_prop = f.get('Prop_Type_Normalized', f.get('Prop_Type', '')).upper()
                    
                    # Match by name AND prop type
                    name_match = player_name in full_name or full_name in player_name
                    prop_match = pred_prop == feature_prop or pred_prop in feature_prop or feature_prop in pred_prop
                    
                    if name_match and prop_match:
                        pred["line"] = f.get("Line", 0)
                        pred["personId"] = f.get("personId")  # FIXED: Use 'personId' not 'person_id'
                        pred["game_date"] = str(f.get("game_date", ""))
                        break
                
                # DEBUG: Show enrichment
                if pred.get("personId"):
                    print(f"   ✅ Matched: {pred.get('player_name')} ({pred.get('prop_type')}) → personId={pred.get('personId')}")
                else:
                    print(f"   ⚠️ No match: {pred.get('player_name')} ({pred.get('prop_type')})")
            
            print(f"\n   ✅ Received {len(predictions)} predictions from Gemini")
            return predictions
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   Raw text: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"   ⚠️ Gemini error: {e}")
            return []
    
    def _generate_case_study_from_miss(self, miss: Dict) -> Optional[Dict]:
        """
        Generates a case study from a miss with severity-proportional reflection.
        
        - MINOR misses: Log only, no detailed reflection (variance, not error)
        - MODERATE misses: Brief case study
        - MAJOR misses: Full detailed case study with root cause analysis
        """
        severity = miss.get('severity', 'UNKNOWN')
        miss_margin = miss.get('miss_margin', 0.0)
        miss_margin_pct = miss.get('miss_margin_pct', 0.0)
        player_tier = miss.get('player_tier', 'UNKNOWN')
        reliability_tag = miss.get('reliability_tag', 'UNKNOWN')
        
        # MINOR misses: Don't overlearn from variance
        if severity == "MINOR":
            print(f"      ℹ️ Skipping reflection for MINOR miss (margin: {miss_margin:.1f}, {miss_margin_pct:.1f}%)")
            return {
                "archetype": "Variance Miss",
                "mistake": f"Close call - missed by {abs(miss_margin):.1f} ({miss_margin_pct:.1f}%) - within normal variance",
                "correction": "No adjustment needed - this was natural variance, not a model error",
                "severity": severity,
                "generated_at": datetime.now().isoformat(),
                "source": miss
            }
        
        # Build severity-aware prompt
        severity_guidance = ""
        if severity == "MODERATE":
            severity_guidance = """
SEVERITY: MODERATE (Miss by 1-2 units or 10-25% of line)
- Identify ONE likely contributing factor
- Create a focused, specific correction rule
- Don't over-generalize from this single miss
"""
        else:  # MAJOR
            severity_guidance = """
SEVERITY: MAJOR (Miss by >2 units AND >25% of line)
- This is a significant model error requiring full root cause analysis
- Identify multiple potential factors that led to this miss
- Create specific, actionable correction rules
- Consider if this pattern should flag similar props in the future
"""
        
        prompt = f"""You made a wrong prediction. Analyze this miss proportionally to its severity.

### MISS SEVERITY CONTEXT:
{severity_guidance}
- Miss Margin: {miss_margin:.1f} (predicted {"OVER" if miss_margin < 0 else "UNDER"}, player went the other way)
- Miss Percentage: {miss_margin_pct:.1f}% of the line

### PLAYER CONTEXT:
- Player: {miss.get('player_name', 'Unknown')}
- Player Tier: {player_tier} (STAR/STARTER/ROTATION influence)
- Reliability: {reliability_tag} (SNIPER=consistent, STANDARD=normal, VOLATILE=unpredictable)
- Prop Type: {miss.get('prop_type', 'PTS')}

### THE MISS:
- Predicted: {miss.get('prediction')} (Confidence: {miss.get('confidence')}%)
- Line: {miss.get('line', 0)}
- Actual: {miss.get('actual_value')}
- Reasoning: {miss.get('reasoning', 'N/A')}

### OUTPUT (JSON):
{{
    "archetype": "<pattern type - be specific to tier/reliability combo>",
    "mistake": "<what went wrong - proportional to severity>",
    "correction": "<rule for next time - specific and actionable>"
}}
"""
        
        wait_for_api("gemini", GEMINI_API_DELAY)
        
        try:
            response = self.engine.model.generate_content(prompt)
            response_text = response.text
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text)
            result["generated_at"] = datetime.now().isoformat()
            result["source"] = miss
            result["severity"] = severity
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Case study generation error: {e}")
            return None
    
    def run_evaluation_for_date(
        self,
        features_df,
        actuals_df,
        target_date: str,
        prop_types: List[str] = None,
        initial_props: int = 75,
        final_props: int = None  # None = dynamic based on slate size (1.5 props/game, min 8, max 15)
    ) -> Dict:
        """
        Runs Predict-Then-Reveal evaluation for a single date.
        
        CRITICAL: NO DATA LEAKAGE - Only uses data from BEFORE target_date.
        Features are pre-computed using rolling windows that exclude current game.
        
        Flow:
        1. DATA COLLECTION: Get features for date (NO actuals visible)
        2. QUALITY SCORING: Apply EV/Kelly/Z-Score to rank props (75 → dynamic 8-15)
        3. PREDICTION PHASE: Model analyzes each prop with tqdm progress
        4. REVEAL PHASE: After ALL predictions locked, show actuals
        5. COMPARISON: Show how model did on each prop
        6. REFLECTION: Generate case studies from high-confidence misses
        
        Args:
            features_df: Pre-built features DataFrame (already filtered to date range)
            actuals_df: Actuals DataFrame (ground truth, hidden until reveal)
            target_date: The date being evaluated (YYYY-MM-DD)
            prop_types: Prop types to include (default: all)
            initial_props: First-pass limit for quality scoring (default: 75)
            final_props: Final props to predict (default: None = dynamic 1.5 props/game, min 8, max 15)
        """
        prop_types = prop_types or DEFAULT_PROP_TYPES
        
        print(f"\n{'='*70}")
        print(f"📅 TRAINING DATE: {target_date}")
        print(f"{'='*70}")
        
        # ================================================================
        # PHASE 1: DATA COLLECTION (NO ACTUALS VISIBLE)
        # ================================================================
        print(f"\n🔍 PHASE 1: DATA COLLECTION")
        print(f"   Loading features for {target_date}...")
        
        date_features_df = get_features_for_date(features_df, target_date)
        raw_count = date_features_df.count()
        
        if raw_count == 0:
            print(f"   ⚠️ No games found for {target_date}")
            return {"date": target_date, "error": "No games found"}
        
        # Check for odds data
        has_odds = date_features_df.filter(col("Line").isNotNull()).count() > 0
        if not has_odds:
            print(f"   ⚠️ No odds data for {target_date}")
            return {"date": target_date, "error": "No odds data", "total_props": raw_count}
        
        print(f"   ✅ Found {raw_count} player-prop combinations with odds")
        
        # ================================================================
        # PHASE 2: QUALITY SCORING (Apply EV/Kelly/Z-Score ranking)
        # ================================================================
        print(f"\n📊 PHASE 2: QUALITY SCORING")
        print(f"   Applying EV/Kelly/Z-Score ranking...")
        final_desc = "dynamic (8-15 based on slate)" if final_props is None else str(final_props)
        print(f"   Raw: {raw_count} props | Initial cap: {initial_props} | Final: {final_desc}")
        
        # Apply quality scoring using the same logic as preview
        scored_df = apply_quality_scoring_to_df(date_features_df, initial_limit=initial_props, final_limit=final_props)
        scored_count = scored_df.count()
        
        if scored_count == 0:
            print(f"   ⚠️ No props passed quality scoring")
            return {"date": target_date, "error": "No valid props after scoring"}
        
        # Convert to list for processing
        top_props = [row.asDict() for row in scored_df.collect()]
        
        # ================================================================
        # SAVE TRAINING DATA TO CSV (What the model actually sees)
        # ================================================================
        self.engine._save_training_data_csv(top_props, target_date)
        
        print(f"   ✅ Selected top {len(top_props)} props by quality score")
        
        # Show the selected props
        print(f"\n   📋 SELECTED PROPS FOR ANALYSIS:")
        print(f"   {'─'*65}")
        for i, prop in enumerate(top_props, 1):
            player = f"{prop.get('firstName', '')} {prop.get('lastName', '')}".strip()
            prop_type = prop.get('Prop_Type_Normalized', 'PTS')
            line = prop.get('Line', 0)
            edge = prop.get('Edge', 0)
            quality = prop.get('quality_score', 0)
            ev = prop.get('EV', 0)
            kelly = prop.get('Kelly', 0)
            
            edge_str = f"{edge:+.1f}" if isinstance(edge, (int, float)) else "N/A"
            print(f"   {i:2d}. {player:<25} | {prop_type:<4} | Line: {line:>5.1f} | Edge: {edge_str:>6} | QS: {quality:.2f}")
        print(f"   {'─'*65}")
        
        # Preview sent to Gemini (small sample with key fields)
        preview_rows = min(6, len(top_props))
        print(f"\n   👀 PREVIEW OF DATA SENT TO GEMINI (first {preview_rows} of {len(top_props)}):")
        print(f"   {'Player':<22} | {'Prop':<4} | {'Line':>5} | {'EV':>6} | {'Kelly':>6} | {'QS':>6} | {'Tier':<8} | {'Rel':<9}")
        print(f"   {'─'*80}")
        for prop in top_props[:preview_rows]:
            player = f"{prop.get('firstName', '')} {prop.get('lastName', '')}".strip()
            prop_type = prop.get('Prop_Type_Normalized', 'PTS')
            line = prop.get('Line', 0)
            ev = prop.get('EV', 0)
            kelly = prop.get('Kelly', 0)
            quality = prop.get('quality_score', 0)
            tier = prop.get('Player_Tier', 'UNK')
            rel = prop.get('Reliability_Tag', 'UNK')
            print(f"   {player:<22} | {prop_type:<4} | {line:>5.1f} | {ev:>6.3f} | {kelly:>6.3f} | {quality:>6.2f} | {tier:<8} | {rel:<9}")
        print(f"   {'─'*80}")
        
        # ================================================================
        # PHASE 3: PREDICTION (Model analyzes each prop - NO ACTUALS)
        # ================================================================
        print(f"\n🎯 PHASE 3: PREDICTION (Model is blind to outcomes)")
        print(f"   Analyzing {len(top_props)} props...")
        
        # Add ML predictions to features (if available)
        if HAS_ML_PREDICTOR:
            try:
                print(f"   🤖 Adding ML model signals to features...")
                top_props = add_ml_predictions(top_props)
                ml_over = sum(1 for p in top_props if p.get('ml_prediction') == 'OVER')
                ml_under = sum(1 for p in top_props if p.get('ml_prediction') == 'UNDER')
                print(f"   ✅ ML predictions: {ml_over} OVER, {ml_under} UNDER")
            except Exception as e:
                print(f"   ⚠️ ML prediction error: {e} (continuing without ML signals)")
        
        # Show case study availability (including from earlier dates in this session)
        total_case_studies = len(self.engine.case_library.case_studies)
        session_case_studies = self.case_studies_created  # From current training session
        pre_session_studies = total_case_studies - session_case_studies
        if total_case_studies > 0:
            print(f"   📚 Case studies available: {total_case_studies} total "
                  f"({pre_session_studies} pre-session + {session_case_studies} from this session)")
        else:
            print(f"   📚 No case studies yet (model will learn from mistakes)")
        
        all_predictions = []
        
        # Show progress for each prop being analyzed
        for prop in tqdm(top_props, desc="   Analyzing props", unit="prop"):
            player = f"{prop.get('firstName', '')} {prop.get('lastName', '')}".strip()
            prop_type = prop.get('Prop_Type_Normalized', 'PTS')
            line = prop.get('Line', 0)
            
            # Update tqdm description with current player
            tqdm.write(f"      → {player} | {prop_type} | Line: {line}")
        
        # Get batch predictions from Gemini
        predictions = self._get_gemini_predictions(top_props, "ALL")
        
        # Match predictions back to props and enrich with personId and tier data
        for pred in predictions:
            player_name = pred.get("player_name", "").lower().strip()
            pred_prop = pred.get("prop_type", "").upper()
            
            for prop in top_props:
                full_name = f"{prop.get('firstName', '')} {prop.get('lastName', '')}".lower().strip()
                feature_prop = prop.get('Prop_Type_Normalized', '').upper()
                
                name_match = player_name in full_name or full_name in player_name
                prop_match = pred_prop == feature_prop
                
                if name_match and prop_match:
                    pred["personId"] = prop.get("personId")
                    pred["line"] = prop.get("Line", 0)
                    pred["game_date"] = target_date
                    pred["edge"] = prop.get("Edge", 0)
                    pred["quality_score"] = prop.get("quality_score", 0)
                    # Add tier and reliability data for severity-weighted learning
                    pred["player_tier"] = prop.get("Player_Tier", "UNKNOWN")
                    pred["reliability_tag"] = prop.get("Reliability_Tag", "UNKNOWN")
                    break
            
            all_predictions.append(pred)
        
        if not all_predictions:
            print(f"   ⚠️ No predictions made")
            return {"date": target_date, "error": "No predictions made", "total_predictions": 0}
        
        # Show predictions summary
        print(f"\n   ✅ PREDICTIONS LOCKED ({len(all_predictions)} total):")
        print(f"   {'─'*65}")
        for i, pred in enumerate(all_predictions, 1):
            conf = pred.get('confidence', 0)
            conf_bar = '█' * (conf // 10) + '░' * (10 - conf // 10)
            print(f"   {i:2d}. {pred.get('player_name', 'Unknown'):<25} | {pred.get('prop_type', '?'):<4} | "
                  f"{pred.get('prediction', '?'):<5} | Conf: [{conf_bar}] {conf}%")
        print(f"   {'─'*65}")
        
        # ================================================================
        # PHASE 4: REVEAL (Show what actually happened)
        # ================================================================
        print(f"\n🎭 PHASE 4: REVEAL")
        print(f"   Revealing actual game results for {target_date}...")
        
        date_actuals_df = get_actuals_for_date(actuals_df, target_date)
        actuals_count = date_actuals_df.count()
        
        print(f"   📊 Loaded {actuals_count} player box scores")
        
        # ================================================================
        # PHASE 5: COMPARISON (How did the model do?)
        # ================================================================
        print(f"\n📈 PHASE 5: RESULTS COMPARISON")
        
        comparisons = compare_predictions_to_actuals(all_predictions, date_actuals_df)
        
        total = len(comparisons)
        hits = sum(1 for c in comparisons if c.get("hit", False))
        misses = total - hits
        accuracy = (hits / total * 100) if total > 0 else 0
        
        high_conf_misses = [c for c in comparisons if c.get("is_high_confidence_miss", False)]
        
        # Show detailed comparison
        print(f"\n   {'─'*75}")
        print(f"   {'PLAYER':<25} | {'PROP':<4} | {'PRED':<5} | {'LINE':>5} | {'ACTUAL':>6} | {'RESULT':<8}")
        print(f"   {'─'*75}")
        
        for comp in comparisons:
            result_icon = "✅ HIT" if comp.get("hit") else "❌ MISS"
            actual = comp.get('actual_value', 0)
            line = comp.get('line', 0)
            over_under = "OVER" if actual > line else "UNDER"
            
            print(f"   {comp.get('player_name', 'Unknown'):<25} | {comp.get('prop_type', '?'):<4} | "
                  f"{comp.get('prediction', '?'):<5} | {line:>5.1f} | {actual:>6.1f} | {result_icon}")
        
        print(f"   {'─'*75}")
        print(f"\n   📊 ACCURACY: {hits}/{total} ({accuracy:.1f}%)")
        print(f"   ✅ Hits: {hits} | ❌ Misses: {misses}")
        
        if high_conf_misses:
            print(f"   ⚠️ High-confidence misses (≥75%): {len(high_conf_misses)}")
        
        self.total_predictions += total
        self.total_hits += hits
        
        # ================================================================
        # PHASE 6: REFLECTION (Learn from mistakes - severity-proportional)
        # ================================================================
        new_case_studies = []
        
        if high_conf_misses:
            print(f"\n🔬 PHASE 6: REFLECTION (Analyzing {len(high_conf_misses)} high-confidence misses)")
            
            # Calculate severity for each miss
            severity_counts = {"MINOR": 0, "MODERATE": 0, "MAJOR": 0}
            
            for miss in high_conf_misses[:5]:  # Limit to 5 per day
                player = miss.get('player_name', 'Unknown')
                actual = miss.get('actual_value', 0)
                line = miss.get('line', 0)
                prediction = miss.get('prediction', 'OVER')
                
                # Calculate severity
                miss_margin, miss_margin_pct, severity = calculate_miss_severity(actual, line, prediction)
                
                # Add severity data to miss dict for case study generation
                miss['miss_margin'] = miss_margin
                miss['miss_margin_pct'] = miss_margin_pct
                miss['severity'] = severity
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                severity_icon = {"MINOR": "📉", "MODERATE": "⚠️", "MAJOR": "🔴"}
                print(f"   {severity_icon.get(severity, '?')} {player}: {severity} miss "
                      f"(margin: {miss_margin:+.1f}, {miss_margin_pct:.1f}%)")
                
                # Generate case study (will be proportional to severity)
                case_study = self._generate_case_study_from_miss(miss)
                if case_study:
                    cs = CaseStudy(
                        id=f"cs_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(new_case_studies)}",
                        date=datetime.now().isoformat(),
                        archetype=case_study.get('archetype', 'Unknown'),
                        player_tier=miss.get('player_tier', 'UNKNOWN'),
                        prop_type=miss.get('prop_type', 'PTS'),
                        context_tags=[],
                        mistake=case_study.get('mistake', ''),
                        correction_vector=case_study.get('correction', ''),
                        original_confidence=miss.get('confidence', 0),
                        actual_outcome="MISS",
                        predicted_outcome=miss.get('prediction', ''),
                        player_name=miss.get('player_name', ''),
                        embedding_key=f"pyspark_{target_date}",
                        game_date=target_date,
                        # New severity fields
                        miss_margin=miss_margin,
                        miss_margin_pct=miss_margin_pct,
                        severity=severity,
                        reliability_tag=miss.get('reliability_tag', 'UNKNOWN')
                    )
                    self.engine.case_library.add_case_study(cs)
                    new_case_studies.append(case_study)
                    self.case_studies_created += 1
                    
                    # Only show detailed lesson for MODERATE/MAJOR
                    if severity != "MINOR":
                        print(f"      📝 Lesson: {case_study.get('archetype', 'Unknown')}")
                        print(f"         Correction: {case_study.get('correction', 'N/A')[:80]}...")
            
            # Summary of severity distribution
            print(f"\n   📊 Severity breakdown: {severity_counts}")
            
            if new_case_studies:
                self.engine.case_library.save()
                meaningful_lessons = len([c for c in new_case_studies if c.get('severity', 'UNKNOWN') != 'MINOR'])
                total_now = len(self.engine.case_library.case_studies)
                print(f"   📚 Added {len(new_case_studies)} case studies ({meaningful_lessons} actionable)")
                print(f"   💡 Total case studies now: {total_now} (will be used for subsequent dates in this session)")
        else:
            print(f"\n✨ No high-confidence misses - no reflection needed")
        
        print(f"\n{'='*70}")
        print(f"📅 COMPLETED: {target_date} | Accuracy: {accuracy:.1f}% ({hits}/{total})")
        print(f"{'='*70}")
        
        return {
            "date": target_date,
            "total_predictions": total,
            "hits": hits,
            "accuracy": accuracy,
            "high_confidence_misses": len(high_conf_misses),
            "new_case_studies": len(new_case_studies)
        }
    
    def run_historical_training(
        self,
        start_date: str = None,
        end_date: str = None,
        seasons: List[str] = None,
        prop_types: List[str] = None,
        fetch_odds: bool = True
    ) -> Dict:
        """
        Runs historical training over a date range using PySpark.
        
        Args:
            start_date: Start date (YYYY-MM-DD) - if provided, trains on date range
            end_date: End date (YYYY-MM-DD) - if provided, trains on date range
            seasons: Seasons to load data from (default: ["2024-25"])
            prop_types: Prop types to train on (default: PTS, REB, AST, 3PM)
            fetch_odds: Whether to fetch historical odds (REQUIRED for proper training)
            
        If no date range provided:
            - Automatically starts from earliest untrained date
            - Trains for 1 week (7 days) by default
            
        Returns:
            Training results dictionary
        """
        seasons = seasons or ["2024-25"]
        prop_types = prop_types or DEFAULT_PROP_TYPES
        
        # Default training length: 1 week (7 days)
        DEFAULT_TRAINING_DAYS = 7
        
        spark = get_spark_session()
        
        try:
            # If no date range specified, find earliest untrained date
            if not start_date or not end_date:
                # First, get ALL dates from the CSVs (quick scan)
                from pyspark.sql import functions as F
                from shared_config import PLAYER_STATS_CSV, SEASON_DATE_RANGES
                
                temp_df = spark.read.csv(PLAYER_STATS_CSV, header=True, inferSchema=True)
                temp_df = temp_df.withColumn("game_date", F.to_date(F.col("gameDateTimeEst")))
                
                # Filter to valid seasons
                all_season_dates = []
                for season, (s_start, s_end) in SEASON_DATE_RANGES.items():
                    if season in seasons:
                        season_dates = temp_df.filter(
                            (F.col("game_date") >= s_start) & (F.col("game_date") <= s_end)
                        ).select("game_date").distinct().collect()
                        all_season_dates.extend([r.game_date.strftime("%Y-%m-%d") for r in season_dates])
                
                all_season_dates = sorted(set(all_season_dates))
                
                # Find untrained dates
                untrained = get_untrained_dates(all_season_dates)
                
                if not untrained:
                    print("\n⚠️ All available dates have been trained on!")
                    print("   Use 'python nba_ai_pipeline.py reset' to allow retraining.")
                    return {"error": "All dates already trained", "total_dates": len(all_season_dates)}
                
                # Start from earliest untrained date, train for 1 week
                start_date = min(untrained)
                # Calculate end date (7 days from start)
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = start_dt + timedelta(days=DEFAULT_TRAINING_DAYS - 1)
                end_date = end_dt.strftime("%Y-%m-%d")
                
                print(f"\n📅 Auto-selected date range: {start_date} to {end_date}")
                print(f"   (Earliest untrained date + {DEFAULT_TRAINING_DAYS} days)")
            
            print("\n" + "="*60)
            print("🏋️ HISTORICAL TRAINING - Self-Reflexive Learning")
            print("="*60)
            print(f"Seasons: {', '.join(seasons)}")
            print(f"Date Range: {start_date} to {end_date}")
            print(f"Prop types: {prop_types}")
            print(f"Fetch odds: {fetch_odds}")
            
            # Build datasets - WITH date range filtering (only fetches odds for these dates)
            features_df, actuals_df, odds_df = build_training_datasets(
                spark, seasons, fetch_odds=fetch_odds,
                date_range=(start_date, end_date)
            )
            
            # Get unique dates from the filtered data
            all_dates = [row.game_date.strftime("%Y-%m-%d") 
                         for row in features_df.select("game_date").distinct().collect()]
            all_dates.sort()
            
            if not all_dates:
                print(f"\n⚠️ No games found in date range {start_date} to {end_date}")
                return {"error": "No games in range", "start": start_date, "end": end_date}
            
            print(f"\n📅 Found {len(all_dates)} game dates with data")
            
            # Filter out already-trained dates
            untrained_dates = get_untrained_dates(all_dates)
            
            if not untrained_dates:
                print("\n⚠️ All dates in this range have already been trained on!")
                print("   Use a different date range or 'python nba_ai_pipeline.py reset' to retrain.")
                return {"error": "All dates already trained", "dates_requested": len(all_dates)}
            
            print(f"📅 {len(untrained_dates)} untrained dates to process")
            
            # Use all untrained dates in range (no random sampling)
            sample_dates = sorted(untrained_dates)
            
            print(f"\n📅 Selected {len(sample_dates)} dates for training")
            
            # Run evaluation for each date
            all_results = []
            
            for date in tqdm(sample_dates, desc="Training"):
                result = self.run_evaluation_for_date(
                    features_df, actuals_df, date, prop_types
                )
                all_results.append(result)
            
            # Calculate overall metrics
            overall_accuracy = (self.total_hits / self.total_predictions * 100) if self.total_predictions > 0 else 0
            
            # Generate report
            report = {
                "training_completed_at": datetime.now().isoformat(),
                "seasons": seasons,
                "dates_evaluated": len(sample_dates),
                "total_predictions": self.total_predictions,
                "total_hits": self.total_hits,
                "overall_accuracy": overall_accuracy,
                "new_case_studies": self.case_studies_created,
                "date_results": all_results
            }
            
            # Save report
            report_path = os.path.join(
                TRAINING_REPORTS_DIR, 
                f"historical_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(TRAINING_REPORTS_DIR, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Mark dates as trained to prevent re-training
            mark_dates_as_trained(sample_dates, {
                "start_date": start_date,
                "end_date": end_date,
                "dates_trained": len(sample_dates),
                "accuracy": overall_accuracy,
                "case_studies_created": self.case_studies_created
            })
            
            print("\n" + "="*60)
            print("✅ HISTORICAL TRAINING COMPLETE")
            print("="*60)
            print(f"Dates: {len(sample_dates)} | Predictions: {self.total_predictions}")
            print(f"Accuracy: {overall_accuracy:.1f}% | Case Studies: {self.case_studies_created}")
            print(f"Report: {report_path}")
            print(f"📝 Dates marked as trained (won't be retrained)")
            
            return report
            
        finally:
            stop_spark_session()


# ============================================================================
# LIVE EVALUATION
# ============================================================================

class LiveEvaluator:
    """Evaluates existing predictions against actual results."""
    
    def __init__(self):
        self.engine = ReflexivePredictionEngine()
        self.reporter = TrainingReporter(mode="live")
    
    def _generate_case_study(self, result: TrainingResult) -> Optional[Tuple[CaseStudy, CaseStudySummary]]:
        """Generates a case study from a high-confidence miss."""
        prompt = f"""
You are a sports betting analyst reviewing a failed prediction.

### THE MISTAKE:
- Player: {result.player} ({result.player_tier})
- Prop: {result.prop} | Line: {result.line}
- Prediction: {result.prediction} (Confidence: {result.confidence}/10)
- Actual: {result.actual_outcome} ({result.actual_value})
- Reasoning: {result.reason}

### CONTEXT:
{json.dumps(result.context, indent=2, default=str)}

### OUTPUT (JSON ONLY):
{{
    "archetype": "<pattern name>",
    "mistake": "<what went wrong>",
    "correction_vector": "<rule for similar situations>"
}}
"""
        
        try:
            wait_for_api("gemini", GEMINI_API_DELAY)
            response = self.engine.reflection_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            reflection = json.loads(response.text)
            
            case_study = CaseStudy(
                id=f"cs_{datetime.now().strftime('%Y%m%d%H%M%S')}_{result.player[:3]}",
                date=datetime.now().isoformat(),
                archetype=reflection.get('archetype', 'Unknown'),
                player_tier=result.player_tier,
                prop_type=result.prop,
                context_tags=result.context.get('tags', []),
                mistake=reflection.get('mistake', ''),
                correction_vector=reflection.get('correction_vector', ''),
                original_confidence=result.confidence,
                actual_outcome=result.actual_outcome,
                predicted_outcome=result.prediction,
                player_name=result.player,
                embedding_key=f"{result.player_tier}_{result.prop}",
                game_date=result.game_date
            )
            
            summary = CaseStudySummary(
                archetype=reflection.get('archetype', 'Unknown'),
                player=result.player,
                prop=result.prop,
                mistake=reflection.get('mistake', ''),
                correction=reflection.get('correction_vector', ''),
                confidence_before=result.confidence,
                actual_outcome=result.actual_outcome,
                game_date=result.game_date
            )
            
            return case_study, summary
            
        except Exception as e:
            print(f"Error generating case study: {e}")
            return None
    
    def evaluate_predictions_file(self, predictions_path: str) -> Dict[str, Any]:
        """Evaluates a predictions CSV file against actual results."""
        print(f"\n{'='*60}")
        print(f"📊 LIVE EVALUATION")
        print(f"   File: {predictions_path}")
        print(f"{'='*60}")
        
        df = pd.read_csv(predictions_path)
        
        # Extract date from filename
        filename = os.path.basename(predictions_path)
        if '_predictions_' in filename:
            date_part = filename.split('_predictions_')[1].split('_')[0]
            target_date = datetime.strptime(date_part, "%Y%m%d").date()
        else:
            target_date = (datetime.now() - timedelta(days=1)).date()
        
        print(f"   Date: {target_date}")
        
        # Fetch actual results
        player_names = df['Player'].unique().tolist()
        actuals = get_actual_results_for_players(target_date, player_names)
        
        # Evaluate
        prop_map = {'POINTS': 'PTS', 'REBOUNDS': 'REB', 'ASSISTS': 'AST', 'FG3M': 'FG3M'}
        mistakes = []
        
        for idx, row in df.iterrows():
            player = row['Player']
            
            if player not in actuals:
                continue
            
            actual_stats = actuals[player]
            prop = row['Prop']
            line = float(row['Line'])
            prediction = row.get('AI_Prediction')
            confidence = int(row.get('AI_Confidence', 0) or 0)
            
            if pd.isna(prediction) or prediction == 'ERROR':
                continue
            
            stat_col = prop_map.get(prop)
            if not stat_col:
                continue
            
            try:
                actual_value = float(actual_stats[stat_col])
            except:
                continue
            
            actual_outcome = "OVER" if actual_value > line else "UNDER"
            is_correct = (actual_outcome == prediction)
            
            usg = float(row.get('USG_PCT', 0) or 0)
            pie = float(row.get('PIE', 0) or 0)
            player_tier = self.engine._get_player_tier(usg, pie)
            context_tags = self.engine._get_context_tags(row)
            
            self.engine.confidence_calibrator.record_prediction(
                confidence=confidence,
                is_correct=is_correct,
                player_tier=player_tier,
                prop_type=prop
            )
            
            self.reporter.log_prediction(PredictionResult(
                player=player,
                team=row.get('Team', ''),
                prop=prop,
                line=line,
                prediction=prediction,
                confidence=confidence,
                actual_value=actual_value,
                actual_outcome=actual_outcome,
                is_correct=is_correct,
                player_tier=player_tier,
                context_tags=context_tags,
                reason=row.get('AI_Reason', ''),
                risk_factor=row.get('AI_Risk_Factor', ''),
                game_date=str(target_date)
            ))
            
            if not is_correct and confidence >= 7:
                result = TrainingResult(
                    player=player,
                    prop=prop,
                    line=line,
                    prediction=prediction,
                    confidence=confidence,
                    actual_value=actual_value,
                    actual_outcome=actual_outcome,
                    is_correct=is_correct,
                    player_tier=player_tier,
                    context={"tags": context_tags},
                    reason=row.get('AI_Reason', ''),
                    game_date=str(target_date)
                )
                mistakes.append(result)
        
        # Generate case studies from high-confidence mistakes
        print(f"\n🔬 Generating case studies for {len(mistakes)} mistakes...")
        
        for mistake in tqdm(mistakes, desc="Creating Case Studies"):
            case_result = self._generate_case_study(mistake)
            if case_result:
                case_study, summary = case_result
                self.engine.case_library.add_case_study(case_study)
                self.reporter.log_case_study(summary)
        
        self.engine.case_library.save()
        
        report_path = self.reporter.generate_report({
            "start": str(target_date),
            "end": str(target_date)
        })
        
        self.reporter.print_summary()
        
        return {"report_path": report_path}


# ============================================================================
# THE LIVE ENGINE - PRODUCTION PIPELINE
# ============================================================================

class ProductionPipeline:
    """The Live Engine: Real-time predictions using evolved intelligence."""
    
    def __init__(self):
        self.engine = ReflexivePredictionEngine()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run(self, max_props: int = 20) -> Tuple[pd.DataFrame, str]:
        """Runs the full production pipeline."""
        print(f"\n{'='*60}")
        print(f"🔴 PRODUCTION MODE: Live Analysis")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Case Studies: {len(self.engine.case_library.case_studies)}")
        print(f"{'='*60}")
        
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        features_df, data_filepath = run_live_pipeline(max_props)
        
        if features_df.empty:
            print("No data to analyze.")
            return pd.DataFrame(), ""
        
        output_filename = f"nba_predictions_{self.timestamp}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, output_filename)
        
        predictions_df = self.engine.run_batch_analysis(features_df, output_path)
        
        log_path = os.path.join(LOGS_DIR, "prediction_logs.json")
        self.engine.save_logs(log_path)
        
        print(f"\n✅ Production complete! Output: {output_filename}")
        
        return predictions_df, output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_historical_training(
    start_date: str = None, 
    end_date: str = None, 
    fetch_odds: bool = True
) -> Dict:
    """
    Run HISTORICAL training mode using PySpark.
    
    Trains the model on past games where outcomes are known.
    Uses the Predict-Then-Reveal architecture to prevent data leakage.
    
    Args:
        start_date: Start date (YYYY-MM-DD) - trains on this date range
        end_date: End date (YYYY-MM-DD) - trains on this date range
        fetch_odds: Fetch historical odds (REQUIRED for proper training)
        
    If no date range provided:
        - Starts from the earliest UNTRAINED date
        - Trains for 1 week (7 days) by default
        
    Returns:
        Training results dictionary
        
    Example:
        # Train on specific date range
        run_historical_training("2024-01-15", "2024-01-31")
        
        # Auto-select next untrained week
        run_historical_training()
    """
    trainer = HistoricalTrainer()
    return trainer.run_historical_training(
        start_date=start_date,
        end_date=end_date,
        seasons=["2024-25"],
        prop_types=["PTS", "REB", "AST", "3PM"],
        fetch_odds=fetch_odds
    )


def run_live_evaluation(predictions_path: str = None):
    """Run LIVE evaluation mode."""
    evaluator = LiveEvaluator()
    
    if predictions_path is None:
        files = [f for f in os.listdir(PREDICTIONS_DIR) 
                 if f.startswith('nba_predictions_') and f.endswith('.csv')]
        files.sort(reverse=True)
        
        if files:
            predictions_path = os.path.join(PREDICTIONS_DIR, files[0])
            print(f"📂 Using most recent: {files[0]}")
        else:
            print("❌ No prediction files found.")
            return {}
    
    return evaluator.evaluate_predictions_file(predictions_path)


def run_production_mode(max_props: int = 20):
    """Run the system in production mode."""
    pipeline = ProductionPipeline()
    return pipeline.run(max_props)


def show_trained_dates():
    """Shows which dates have been trained on."""
    data = load_trained_dates()
    
    print("\n" + "="*60)
    print("📅 TRAINED DATES HISTORY")
    print("="*60)
    
    trained = data.get("trained_dates", [])
    sessions = data.get("training_sessions", [])
    
    if not trained:
        print("No dates trained yet.")
        return
    
    print(f"\nTotal dates trained: {len(trained)}")
    
    # Group by month
    from collections import defaultdict
    by_month = defaultdict(list)
    for d in sorted(trained):
        month = d[:7]  # YYYY-MM
        by_month[month].append(d)
    
    print("\nBy month:")
    for month, dates in sorted(by_month.items()):
        print(f"  {month}: {len(dates)} dates")
    
    if sessions:
        print(f"\nTraining sessions: {len(sessions)}")
        for s in sessions[-5:]:  # Show last 5 sessions
            print(f"  - {s.get('timestamp', 'Unknown')[:10]}: {s.get('dates_trained', 0)} dates, {s.get('accuracy', 0):.1f}% accuracy")


def reset_trained_dates():
    """Resets the trained dates tracking (allows retraining)."""
    if os.path.exists(TRAINED_DATES_PATH):
        os.remove(TRAINED_DATES_PATH)
        print("✅ Trained dates history cleared. You can now retrain on any date.")
    else:
        print("No trained dates file to clear.")


def preview_training_data(target_date: str = None, max_props: int = 10):
    """
    Preview what the training dataset looks like for a specific date.
    
    Shows:
    - Raw data from CSVs
    - Engineered features
    - Odds/lines joined
    - Smart filter results
    - What Gemini would receive
    
    Args:
        target_date: Date to preview (YYYY-MM-DD). Defaults to earliest in 2023-24 season.
        max_props: Number of props to show after smart filter
    """
    print("\n" + "="*70)
    print("📋 TRAINING DATA PREVIEW")
    print("="*70)
    
    target_date = target_date or "2023-10-24"  # First day of 2023-24 season
    print(f"Target Date: {target_date}")
    print(f"Max Props: {max_props}")
    
    spark = get_spark_session()
    
    try:
        # Build datasets for just this one date
        features_df, actuals_df, odds_df = build_training_datasets(
            spark, 
            seasons=["2024-25"],
            fetch_odds=True,
            date_range=(target_date, target_date)
        )
        
        # Get features for this date
        date_features_df = get_features_for_date(features_df, target_date)
        
        if date_features_df.count() == 0:
            print(f"\n⚠️ No games found for {target_date}")
            return
        
        features_list = [row.asDict() for row in date_features_df.collect()]
        
        print(f"\n📊 RAW DATA STATS:")
        print(f"   Total player-prop records: {len(features_list)}")
        
        # Show columns available
        if features_list:
            cols = list(features_list[0].keys())
            print(f"   Total columns: {len(cols)}")
            
            # Group by category
            id_cols = [c for c in cols if any(x in c.lower() for x in ['id', 'name', 'team', 'date'])]
            stat_cols = [c for c in cols if any(x in c.lower() for x in ['avg', 'l5', 'l10', 'season'])]
            odds_cols = [c for c in cols if any(x in c.lower() for x in ['line', 'odds', 'prop', 'edge'])]
            
            print(f"\n   📝 ID columns ({len(id_cols)}): {', '.join(id_cols[:5])}...")
            print(f"   📈 Stat columns ({len(stat_cols)}): {', '.join(stat_cols[:5])}...")
            print(f"   💰 Odds columns ({len(odds_cols)}): {', '.join(odds_cols[:5])}...")
        
        # Apply smart filter
        trainer = HistoricalTrainer()
        filtered = trainer._smart_filter_props(features_list, max_props)
        
        print(f"\n🎯 AFTER SMART FILTER: {len(features_list)} → {len(filtered)} props")
        
        # Show the filtered props in detail
        trainer.preview_dataset(filtered, target_date, show_all=True)
        
        # Show what prompt would look like
        print("\n" + "="*70)
        print("📝 SAMPLE GEMINI PROMPT (first 3 props):")
        print("="*70)
        
        sample_props = filtered[:3]
        if sample_props:
            for p in sample_props:
                player_name = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                team = p.get('TeamAbbrev', 'UNK')
                opponent = p.get('OpponentAbbrev', 'UNK')
                home = "Home" if p.get('home') == 1 else "Away"
                prop_type = p.get('Prop_Type', 'PTS')
                line = p.get('Line')
                edge = p.get('Edge')
                
                l5 = p.get(f'L5_Avg_{prop_type}') or p.get('L5_Avg_PTS')
                
                print(f"\n**{player_name}** ({team} {home} vs {opponent})")
                print(f"- PROP: {prop_type} | LINE: {line}")
                print(f"- L5 Avg: {l5}")
                print(f"- Edge: {edge:+.1f}" if edge else "- Edge: N/A")
                print("---")
        
        # Show actuals (ground truth) for context
        print("\n" + "="*70)
        print("📊 ACTUALS (Ground Truth - Hidden from Model):")
        print("="*70)
        
        date_actuals_df = get_actuals_for_date(actuals_df, target_date)
        actuals_count = date_actuals_df.count()
        print(f"   Total player-games with outcomes: {actuals_count}")
        
        if actuals_count > 0:
            actuals_sample = date_actuals_df.limit(5).collect()
            print("\n   Sample actuals:")
            for a in actuals_sample:
                a_dict = a.asDict()
                name = f"{a_dict.get('firstName', '')} {a_dict.get('lastName', '')}"
                pts = a_dict.get('points', 'N/A')
                reb = a_dict.get('reboundsTotal', 'N/A')
                ast = a_dict.get('assists', 'N/A')
                print(f"   - {name}: PTS={pts}, REB={reb}, AST={ast}")
        
    finally:
        stop_spark_session()
    
    print("\n" + "="*70)
    print("✅ PREVIEW COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import sys
    import re
    from datetime import datetime as _dt
    
    def validate_date(date_str: str, arg_name: str) -> str:
        """Validate date format is YYYY-MM-DD and is a real date."""
        if not date_str:
            return date_str
        # Check format with regex first
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            print(f"\n❌ ERROR: Invalid {arg_name} format: '{date_str}'")
            print(f"   Expected format: YYYY-MM-DD (e.g., 2024-12-15)")
            print(f"   Got: '{date_str}' (length={len(date_str)})")
            sys.exit(1)
        # Try to parse as actual date
        try:
            _dt.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            print(f"\n❌ ERROR: Invalid {arg_name}: '{date_str}'")
            print(f"   {e}")
            sys.exit(1)
        return date_str
    
    print("\n" + "="*60)
    print("🧠 SELF-EVOLVING NBA NEURAL SYSTEM v3.0")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "train" or mode == "historical":
            # Historical training with date range
            # Usage: python nba_ai_pipeline.py train 2024-01-15 2024-01-31
            # Or: python nba_ai_pipeline.py train (auto-selects next untrained week)
            start = sys.argv[2] if len(sys.argv) > 2 else None
            end = sys.argv[3] if len(sys.argv) > 3 else None
            
            # Validate date formats before proceeding
            start = validate_date(start, "start_date")
            end = validate_date(end, "end_date")
            
            if start and end:
                print(f"\n🏋️ HISTORICAL TRAINING: {start} to {end}")
            else:
                print("\n🏋️ HISTORICAL TRAINING: Auto-selecting next untrained week")
            
            run_historical_training(start, end)
            
        elif mode == "evaluate" or mode == "live":
            # Evaluate yesterday's predictions
            print("\n📊 LIVE EVALUATION MODE")
            pred_file = sys.argv[2] if len(sys.argv) > 2 else None
            run_live_evaluation(pred_file)
            
        elif mode == "production" or mode == "predict":
            # Make predictions for today
            print("\n🔴 PRODUCTION MODE")
            max_props = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            run_production_mode(max_props)
            
        elif mode == "status" or mode == "trained":
            # Show which dates have been trained
            show_trained_dates()
            
        elif mode == "reset":
            # Reset trained dates tracking
            reset_trained_dates()
            
        elif mode == "preview":
            # Preview training data for a specific date
            date = sys.argv[2] if len(sys.argv) > 2 else None
            max_props = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            preview_training_data(date, max_props)
            
        else:
            print(f"Unknown mode: {mode}")
            print("\n" + "="*60)
            print("USAGE:")
            print("="*60)
            print("\n📋 PREVIEW (see data before training):")
            print("  python nba_ai_pipeline.py preview 2023-10-24           # Preview specific date")
            print("  python nba_ai_pipeline.py preview 2023-10-24 20        # Preview with 20 props")
            print("\n📚 TRAINING:")
            print("  python nba_ai_pipeline.py train 2024-01-15 2024-01-31  # Train on date range")
            print("  python nba_ai_pipeline.py train                         # Auto-select next untrained week")
            print("\n📊 EVALUATION:")
            print("  python nba_ai_pipeline.py evaluate                      # Evaluate yesterday")
            print("  python nba_ai_pipeline.py evaluate predictions.csv      # Evaluate specific file")
            print("\n🔴 PRODUCTION:")
            print("  python nba_ai_pipeline.py production                    # Today's predictions")
            print("  python nba_ai_pipeline.py production 30                 # Max 30 props")
            print("\n📅 UTILITIES:")
            print("  python nba_ai_pipeline.py status                        # Show trained dates")
            print("  python nba_ai_pipeline.py reset                         # Clear trained dates")
    else:
        print("\nNo mode specified. Running PRODUCTION mode...")
        run_production_mode()
