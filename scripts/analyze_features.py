#!/usr/bin/env python3
"""
FEATURE IMPORTANCE ANALYSIS
===========================
Analyzes which features actually have predictive signal vs noise.

Usage:
    python scripts/analyze_features.py

This will:
1. Load trained XGBoost models
2. Extract feature importances (gain, weight, cover)
3. Identify signal vs noise features
4. Generate recommendations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))

# Set library path for XGBoost
os.environ['DYLD_LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

from ml_pipeline.model_trainer import PropModelTrainer

# Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml_models", "prop_models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "training_reports")


def load_model(prop_type: str) -> PropModelTrainer:
    """Loads a trained model for a prop type."""
    path = os.path.join(MODELS_DIR, f"{prop_type.lower()}_model.pkl")
    if os.path.exists(path):
        return PropModelTrainer.load(path)
    return None


def get_feature_importance(model: PropModelTrainer, importance_type: str = "gain") -> Dict[str, float]:
    """
    Extracts feature importance from XGBoost model.
    
    importance_type:
        - 'gain': Average gain of splits using this feature (BEST for understanding)
        - 'weight': Number of times feature is used for splitting
        - 'cover': Average coverage of splits using this feature
    """
    if model is None or model.model is None:
        return {}
    
    try:
        booster = model.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        
        # Normalize to percentages
        total = sum(importance.values()) if importance else 1
        return {k: v / total * 100 for k, v in importance.items()}
    except Exception as e:
        print(f"   Error extracting importance: {e}")
        return {}


def categorize_feature(feature_name: str) -> str:
    """Categorizes a feature into a logical group."""
    name = feature_name.lower()
    
    if any(x in name for x in ['l5_avg', 'l10_avg', 'season_avg', 'l5_stddev', 'l10_stddev']):
        return "Player Recent Form"
    elif any(x in name for x in ['opp_l5', 'opp_l10', 'opp_season', 'opp_rank', 'rank_allows']):
        return "Opponent Defense"
    elif any(x in name for x in ['pace', 'def_rating', 'off_rating']):
        return "Pace & Efficiency"
    elif any(x in name for x in ['rest_days', 'is_b2b', 'b2b']):
        return "Rest & Schedule"
    elif any(x in name for x in ['home', 'away', 'venue']):
        return "Home/Away"
    elif any(x in name for x in ['line', 'odds', 'ev', 'kelly', 'z_score', 'edge', 'quality', 'implied']):
        return "Market/Odds"
    elif any(x in name for x in ['tier', 'reliability', 'cv']):
        return "Player Classification"
    elif any(x in name for x in ['usg', 'pie', 'ts_', 'efg', 'per_100']):
        return "Advanced Stats"
    else:
        return "Other"


def analyze_feature_importance(verbose: bool = True) -> Dict:
    """
    Comprehensive feature importance analysis across all models.
    """
    results = {
        "models": {},
        "aggregated": {},
        "category_summary": {},
        "recommendations": []
    }
    
    print("\n" + "="*70)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    all_importances = []
    
    for prop_type in ["PTS", "REB", "AST", "3PM"]:
        model = load_model(prop_type)
        
        if model is None:
            print(f"\n⚠️ {prop_type} model not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"📊 {prop_type} MODEL")
        print(f"{'='*50}")
        
        # Get importances using different metrics
        gain_importance = get_feature_importance(model, "gain")
        weight_importance = get_feature_importance(model, "weight")
        
        if not gain_importance:
            print("   No feature importance available")
            continue
        
        # Sort by gain (most meaningful)
        sorted_features = sorted(gain_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Store results
        results["models"][prop_type] = {
            "features": model.features if hasattr(model, 'features') else [],
            "importance_gain": gain_importance,
            "importance_weight": weight_importance,
        }
        
        # Display top features
        print(f"\n🏆 TOP 15 FEATURES (by gain):")
        print(f"{'Rank':<5} {'Feature':<40} {'Gain %':<10} {'Category':<20}")
        print("-" * 75)
        
        for i, (feature, importance) in enumerate(sorted_features[:15], 1):
            category = categorize_feature(feature)
            print(f"{i:<5} {feature:<40} {importance:>6.1f}%    {category:<20}")
            
            all_importances.append({
                "prop_type": prop_type,
                "feature": feature,
                "importance": importance,
                "category": category,
                "rank": i
            })
        
        # Bottom features (potential noise)
        print(f"\n🔇 BOTTOM 10 FEATURES (potential noise):")
        for feature, importance in sorted_features[-10:]:
            if importance < 1.0:  # Less than 1% contribution
                print(f"   {feature}: {importance:.2f}%")
    
    # Aggregate across models
    print("\n" + "="*70)
    print("  AGGREGATED ANALYSIS (Across All Models)")
    print("="*70)
    
    if all_importances:
        df = pd.DataFrame(all_importances)
        
        # Average importance by feature
        avg_by_feature = df.groupby("feature").agg({
            "importance": "mean",
            "prop_type": "count"
        }).rename(columns={"prop_type": "models_used"})
        avg_by_feature = avg_by_feature.sort_values("importance", ascending=False)
        
        print(f"\n🌟 MOST IMPORTANT FEATURES (average across models):")
        print(f"{'Feature':<45} {'Avg Importance':<15} {'In N Models':<12}")
        print("-" * 72)
        
        for feature, row in avg_by_feature.head(20).iterrows():
            print(f"{feature:<45} {row['importance']:>10.1f}%      {int(row['models_used']):<12}")
        
        results["aggregated"]["top_features"] = avg_by_feature.head(20).to_dict()
        
        # Category analysis
        print(f"\n📁 IMPORTANCE BY CATEGORY:")
        category_importance = df.groupby("category")["importance"].sum().sort_values(ascending=False)
        total_importance = category_importance.sum()
        
        for category, importance in category_importance.items():
            pct = importance / total_importance * 100
            bar = "█" * int(pct / 2)
            print(f"   {category:<25} {pct:>5.1f}% {bar}")
        
        results["category_summary"] = category_importance.to_dict()
        
        # Identify noise features
        noise_features = avg_by_feature[avg_by_feature["importance"] < 0.5].index.tolist()
        
        print(f"\n🗑️ POTENTIAL NOISE FEATURES (<0.5% average importance):")
        for f in noise_features[:15]:
            print(f"   • {f}")
        
        if len(noise_features) > 15:
            print(f"   ... and {len(noise_features) - 15} more")
        
        results["noise_features"] = noise_features
    
    # Generate recommendations
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    # Check for market feature dominance
    if all_importances:
        df = pd.DataFrame(all_importances)
        market_features = df[df["category"] == "Market/Odds"]["importance"].sum()
        total = df["importance"].sum()
        market_pct = market_features / total * 100 if total > 0 else 0
        
        if market_pct > 40:
            rec = f"⚠️ MARKET FEATURES DOMINATE ({market_pct:.1f}%): Model may be learning odds patterns, not player performance"
            recommendations.append(rec)
            print(f"\n{rec}")
        
        # Check for player form dominance
        form_features = df[df["category"] == "Player Recent Form"]["importance"].sum()
        form_pct = form_features / total * 100 if total > 0 else 0
        
        if form_pct > 50:
            rec = f"ℹ️ PLAYER FORM DOMINATES ({form_pct:.1f}%): Model relies heavily on recent averages (expected behavior)"
            recommendations.append(rec)
            print(f"\n{rec}")
        
        # Check for opponent features
        opp_features = df[df["category"] == "Opponent Defense"]["importance"].sum()
        opp_pct = opp_features / total * 100 if total > 0 else 0
        
        if opp_pct < 10:
            rec = f"⚠️ OPPONENT FEATURES UNDERUSED ({opp_pct:.1f}%): Consider adding more matchup-specific features"
            recommendations.append(rec)
            print(f"\n{rec}")
        
        # Check for schedule features
        schedule_features = df[df["category"] == "Rest & Schedule"]["importance"].sum()
        schedule_pct = schedule_features / total * 100 if total > 0 else 0
        
        if schedule_pct < 5:
            rec = f"⚠️ SCHEDULE FEATURES UNDERUSED ({schedule_pct:.1f}%): rest_days and is_b2b should matter more"
            recommendations.append(rec)
            print(f"\n{rec}")
    
    # Final recommendations
    print("\n📋 ACTION ITEMS:")
    print("   1. Remove noise features (<0.5% importance) to reduce overfitting")
    print("   2. If market features dominate, consider training WITHOUT odds-derived features")
    print("   3. Focus on features that vary across games (matchup, rest, pace)")
    print("   4. Be wary of player-specific features that cause memorization")
    
    results["recommendations"] = recommendations
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, f"feature_importance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Report saved: {report_path}")
    
    return results


def analyze_feature_correlations():
    """
    Analyzes correlations between features and outcomes.
    """
    print("\n" + "="*70)
    print("  FEATURE-OUTCOME CORRELATION ANALYSIS")
    print("="*70)
    
    # Load labeled data
    training_dir = os.path.join(PROJECT_ROOT, "datasets", "training_inputs")
    
    all_data = []
    for f in os.listdir(training_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(training_dir, f))
            all_data.append(df)
    
    if not all_data:
        print("No training data found")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nLoaded {len(df)} samples")
    
    # Add outcome column if we have actuals
    # For now, calculate correlations with numeric columns
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Focus on key features
    key_features = [
        "L5_Avg_PTS", "L10_Avg_PTS", "Season_Avg_PTS",
        "L5_Avg_REB", "L5_Avg_AST", "L5_Avg_3PM",
        "Line", "Edge", "quality_score", "EV", "Kelly",
        "rest_days", "is_b2b", "home",
        "Opp_L5_Team_Def_Rating", "L5_Team_Pace",
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        print(f"\n📊 Feature Statistics:")
        print(f"{'Feature':<30} {'Mean':<12} {'Std':<12} {'Missing %':<12}")
        print("-" * 66)
        
        for feature in available_features[:15]:
            mean = df[feature].mean()
            std = df[feature].std()
            missing = df[feature].isna().sum() / len(df) * 100
            print(f"{feature:<30} {mean:>10.2f}   {std:>10.2f}   {missing:>8.1f}%")


if __name__ == "__main__":
    results = analyze_feature_importance()
    analyze_feature_correlations()

