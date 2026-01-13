"""
TRAINING REPORTER (training_reporter.py)
========================================
Self-Evolving NBA Neural System v2.0

Purpose: Generates comprehensive, timestamped training reports that show:
- Model performance metrics
- Case studies created (what the model learned)
- Data wishlist (what additional data the model wants)
- Prompt evolution history
- Step-by-step visibility into the training process

All reports are timestamped for organization and historical tracking.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TRAINING_REPORTS_DIR = os.path.join(PROJECT_ROOT, "training_reports")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MISSING_FEATURES_LOG = os.path.join(LOGS_DIR, "missing_features.log")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PredictionResult:
    """A single prediction result for tracking."""
    player: str
    team: str
    prop: str
    line: float
    prediction: str
    confidence: int
    actual_value: float
    actual_outcome: str
    is_correct: bool
    player_tier: str
    context_tags: List[str]
    reason: str
    risk_factor: str = ""
    game_date: str = ""  # Date of the game (YYYY-MM-DD)


@dataclass
class CaseStudySummary:
    """Summary of a case study for reporting."""
    archetype: str
    player: str
    prop: str
    mistake: str
    correction: str
    confidence_before: int
    actual_outcome: str
    game_date: str = ""  # Date of the game


@dataclass  
class TrainingSession:
    """Complete data for a training session."""
    session_id: str
    timestamp: str
    mode: str  # "historical" or "live"
    date_range: Dict[str, str]  # {"start": "2025-11-20", "end": "2025-11-27"}
    
    # Performance metrics
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # Breakdown by tier
    tier_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Breakdown by prop type
    prop_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Case studies created
    case_studies: List[CaseStudySummary] = field(default_factory=list)
    
    # Data wishlist
    data_wishlist: List[Dict[str, Any]] = field(default_factory=list)
    
    # Individual predictions for detailed view
    predictions: List[PredictionResult] = field(default_factory=list)
    
    # Prompt evolution (if occurred)
    prompt_evolved: bool = False
    evolution_patterns: List[str] = field(default_factory=list)
    
    # Parlay optimizer training stats (NEW)
    parlay_stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TRAINING REPORTER CLASS
# ============================================================================

class TrainingReporter:
    """
    Generates comprehensive training reports with full visibility into:
    - What the model predicted
    - Where it succeeded/failed
    - What it learned (case studies)
    - What data it wishes it had
    - How its thinking evolved
    """
    
    def __init__(self, mode: str = "historical"):
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"training_{self.timestamp}"
        
        # Initialize tracking
        self.predictions: List[PredictionResult] = []
        self.case_studies: List[CaseStudySummary] = []
        self.data_requests: List[Dict[str, Any]] = []
        self.evolution_patterns: List[str] = []
        self.parlay_training_stats: Dict[str, Any] = {}  # Parlay optimizer training stats
        
        # Ensure directories exist
        os.makedirs(TRAINING_REPORTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    def log_prediction(self, result: PredictionResult):
        """Logs a single prediction result."""
        self.predictions.append(result)
        
        # Real-time console output
        status = "✅" if result.is_correct else "❌"
        print(f"  {status} {result.player} {result.prop}: "
              f"Pred={result.prediction} (Conf:{result.confidence}) | "
              f"Actual={result.actual_outcome} ({result.actual_value})")
    
    def log_case_study(self, case_study: CaseStudySummary):
        """Logs a case study created from a mistake."""
        self.case_studies.append(case_study)
        
        # Real-time console output
        print(f"\n  📝 CASE STUDY: {case_study.archetype}")
        print(f"     Mistake: {case_study.mistake}")
        print(f"     Correction: {case_study.correction}")
    
    def log_data_request(self, feature: str, context: str, impact: str):
        """Logs a data feature the model wishes it had."""
        request = {
            "timestamp": datetime.now().isoformat(),
            "feature": feature,
            "context": context,
            "impact": impact
        }
        self.data_requests.append(request)
        
        # Also append to persistent log
        with open(MISSING_FEATURES_LOG, 'a') as f:
            f.write(json.dumps(request) + "\n")
        
        print(f"  💡 DATA REQUEST: {feature}")
        print(f"     Context: {context}")
    
    def log_evolution(self, patterns: List[str]):
        """Logs when the system prompt evolved."""
        self.evolution_patterns = patterns
        
        print(f"\n  🧬 PROMPT EVOLUTION TRIGGERED")
        print(f"     Patterns incorporated: {len(patterns)}")
        for p in patterns[:3]:
            print(f"     - {p}")
    
    def log_parlay_training(self, parlay_stats: Dict[str, Any]):
        """Logs parlay optimizer training results (accumulates across days)."""
        # Accumulate stats across training days
        if not self.parlay_training_stats:
            self.parlay_training_stats = {
                'parlays_generated': 0,
                'parlays_evaluated': 0,
                'wins': 0,
                'losses': 0,
                'case_studies_created': 0
            }
        
        self.parlay_training_stats['parlays_generated'] += parlay_stats.get('parlays_generated', 0)
        self.parlay_training_stats['parlays_evaluated'] += parlay_stats.get('parlays_evaluated', 0)
        self.parlay_training_stats['wins'] += parlay_stats.get('wins', 0)
        self.parlay_training_stats['losses'] += parlay_stats.get('losses', 0)
        self.parlay_training_stats['case_studies_created'] += parlay_stats.get('case_studies_created', 0)
        
        if parlay_stats.get('parlays_evaluated', 0) > 0:
            wins = parlay_stats.get('wins', 0)
            losses = parlay_stats.get('losses', 0)
            case_studies = parlay_stats.get('case_studies_created', 0)
            print(f"\n  🎰 PARLAY TRAINING: {wins}W / {losses}L")
            print(f"     Case Studies Created: {case_studies}")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculates comprehensive metrics from predictions."""
        if not self.predictions:
            return {}
        
        total = len(self.predictions)
        correct = sum(1 for p in self.predictions if p.is_correct)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Breakdown by tier
        tier_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})
        for p in self.predictions:
            tier_breakdown[p.player_tier]["total"] += 1
            if p.is_correct:
                tier_breakdown[p.player_tier]["correct"] += 1
        
        for tier in tier_breakdown:
            t = tier_breakdown[tier]
            t["accuracy"] = (t["correct"] / t["total"] * 100) if t["total"] > 0 else 0
        
        # Breakdown by prop type
        prop_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})
        for p in self.predictions:
            prop_breakdown[p.prop]["total"] += 1
            if p.is_correct:
                prop_breakdown[p.prop]["correct"] += 1
        
        for prop in prop_breakdown:
            pr = prop_breakdown[prop]
            pr["accuracy"] = (pr["correct"] / pr["total"] * 100) if pr["total"] > 0 else 0
        
        # Breakdown by context tag
        tag_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})
        for p in self.predictions:
            for tag in p.context_tags:
                tag_breakdown[tag]["total"] += 1
                if p.is_correct:
                    tag_breakdown[tag]["correct"] += 1
        
        for tag in tag_breakdown:
            tg = tag_breakdown[tag]
            tg["accuracy"] = (tg["correct"] / tg["total"] * 100) if tg["total"] > 0 else 0
        
        # High confidence performance
        high_conf = [p for p in self.predictions if p.confidence >= 8]
        high_conf_correct = sum(1 for p in high_conf if p.is_correct)
        high_conf_accuracy = (high_conf_correct / len(high_conf) * 100) if high_conf else 0
        
        # Breakdown by date (DAILY PERFORMANCE)
        daily_breakdown = defaultdict(lambda: {"total": 0, "correct": 0, "case_studies": 0})
        for p in self.predictions:
            date_key = p.game_date if p.game_date else "Unknown"
            daily_breakdown[date_key]["total"] += 1
            if p.is_correct:
                daily_breakdown[date_key]["correct"] += 1
        
        # Count case studies per date
        for cs in self.case_studies:
            date_key = cs.game_date if cs.game_date else "Unknown"
            if date_key in daily_breakdown:
                daily_breakdown[date_key]["case_studies"] += 1
        
        for date_key in daily_breakdown:
            d = daily_breakdown[date_key]
            d["accuracy"] = (d["correct"] / d["total"] * 100) if d["total"] > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "tier_breakdown": dict(tier_breakdown),
            "prop_breakdown": dict(prop_breakdown),
            "tag_breakdown": dict(tag_breakdown),
            "daily_breakdown": dict(daily_breakdown),
            "high_confidence": {
                "total": len(high_conf),
                "correct": high_conf_correct,
                "accuracy": high_conf_accuracy
            }
        }
    
    def _aggregate_data_wishlist(self) -> List[Dict[str, Any]]:
        """Aggregates data requests by feature, counting occurrences."""
        feature_counts = defaultdict(lambda: {"count": 0, "contexts": []})
        
        for req in self.data_requests:
            feature = req["feature"]
            feature_counts[feature]["count"] += 1
            feature_counts[feature]["contexts"].append(req["context"])
        
        # Also read from persistent log
        if os.path.exists(MISSING_FEATURES_LOG):
            try:
                with open(MISSING_FEATURES_LOG, 'r') as f:
                    for line in f:
                        if line.strip():
                            req = json.loads(line)
                            feature = req.get("feature", "Unknown")
                            feature_counts[feature]["count"] += 1
            except:
                pass
        
        # Sort by count
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {"feature": f, "count": d["count"], "contexts": d["contexts"][:3]}
            for f, d in sorted_features[:10]
        ]
    
    def generate_report(self, date_range: Dict[str, str] = None) -> str:
        """
        Generates a comprehensive markdown training report.
        
        Returns:
            Path to the generated report file.
        """
        metrics = self._calculate_metrics()
        wishlist = self._aggregate_data_wishlist()
        
        # Build report
        lines = []
        
        # Header
        lines.append(f"# 🏋️ Training Report")
        lines.append(f"")
        lines.append(f"**Session ID:** `{self.session_id}`")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Mode:** {self.mode.upper()} Training")
        if date_range:
            lines.append(f"**Date Range:** {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Performance Summary
        lines.append("## 📈 Performance Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Predictions | {metrics.get('total', 0)} |")
        lines.append(f"| Correct | {metrics.get('correct', 0)} |")
        lines.append(f"| **Overall Accuracy** | **{metrics.get('accuracy', 0):.1f}%** |")
        lines.append(f"| High-Confidence (8+) Accuracy | {metrics.get('high_confidence', {}).get('accuracy', 0):.1f}% ({metrics.get('high_confidence', {}).get('correct', 0)}/{metrics.get('high_confidence', {}).get('total', 0)}) |")
        lines.append(f"| Case Studies Created | {len(self.case_studies)} |")
        lines.append("")
        
        # Daily Performance Breakdown (NEW!)
        if metrics.get('daily_breakdown'):
            lines.append("### 📅 Daily Performance")
            lines.append("")
            lines.append("*See how the model performed each day and whether it improved over time:*")
            lines.append("")
            lines.append("| Date | Predictions | Correct | Accuracy | Case Studies | Trend |")
            lines.append("|------|-------------|---------|----------|--------------|-------|")
            
            # Sort dates chronologically
            sorted_dates = sorted(
                [(k, v) for k, v in metrics['daily_breakdown'].items() if k != "Unknown"],
                key=lambda x: x[0]
            )
            
            prev_accuracy = None
            for date_str, data in sorted_dates:
                accuracy = data['accuracy']
                
                # Determine trend
                if prev_accuracy is None:
                    trend = "🆕 Start"
                elif accuracy > prev_accuracy + 5:
                    trend = "📈 Improving"
                elif accuracy < prev_accuracy - 5:
                    trend = "📉 Declining"
                else:
                    trend = "➡️ Stable"
                
                lines.append(f"| {date_str} | {data['total']} | {data['correct']} | {data['accuracy']:.1f}% | {data.get('case_studies', 0)} | {trend} |")
                prev_accuracy = accuracy
            
            lines.append("")
        
        # Accuracy by Tier
        if metrics.get('tier_breakdown'):
            lines.append("### 📊 Accuracy by Player Tier")
            lines.append("")
            lines.append("| Tier | Predictions | Correct | Accuracy |")
            lines.append("|------|-------------|---------|----------|")
            for tier, data in sorted(metrics['tier_breakdown'].items()):
                lines.append(f"| {tier} | {data['total']} | {data['correct']} | {data['accuracy']:.1f}% |")
            lines.append("")
        
        # Accuracy by Prop Type
        if metrics.get('prop_breakdown'):
            lines.append("### 🎯 Accuracy by Prop Type")
            lines.append("")
            lines.append("| Prop | Predictions | Correct | Accuracy |")
            lines.append("|------|-------------|---------|----------|")
            for prop, data in sorted(metrics['prop_breakdown'].items()):
                lines.append(f"| {prop} | {data['total']} | {data['correct']} | {data['accuracy']:.1f}% |")
            lines.append("")
        
        # Accuracy by Context Tag
        if metrics.get('tag_breakdown'):
            lines.append("### 🏷️ Accuracy by Context Tag")
            lines.append("")
            lines.append("| Context | Predictions | Accuracy | Insight |")
            lines.append("|---------|-------------|----------|---------|")
            for tag, data in sorted(metrics['tag_breakdown'].items(), key=lambda x: x[1]['total'], reverse=True):
                insight = "⚠️ Struggling" if data['accuracy'] < 50 else "✅ Strong" if data['accuracy'] > 60 else "📊 Average"
                lines.append(f"| {tag} | {data['total']} | {data['accuracy']:.1f}% | {insight} |")
            lines.append("")
        
        # Case Studies (What I Learned)
        lines.append("---")
        lines.append("")
        lines.append("## 🧠 CASE STUDIES GENERATED (What I Learned)")
        lines.append("")
        
        if self.case_studies:
            for i, cs in enumerate(self.case_studies, 1):
                lines.append(f"### {i}. \"{cs.archetype}\"")
                lines.append(f"- **Player:** {cs.player} ({cs.prop})")
                lines.append(f"- **Confidence Before:** {cs.confidence_before}/10")
                lines.append(f"- **Actual Outcome:** {cs.actual_outcome}")
                lines.append(f"- **Mistake:** {cs.mistake}")
                lines.append(f"- **🔧 Correction:** {cs.correction}")
                lines.append("")
        else:
            lines.append("*No case studies generated this session (no high-confidence mistakes).*")
            lines.append("")
        
        # Data Wishlist
        lines.append("---")
        lines.append("")
        lines.append("## 💡 DATA WISHLIST (What I Wish I Had)")
        lines.append("")
        lines.append("*These are features the model identified as potentially helpful for better predictions:*")
        lines.append("")
        
        if wishlist:
            for i, item in enumerate(wishlist, 1):
                lines.append(f"### {i}. {item['feature']}")
                lines.append(f"- **Requested:** {item['count']} time(s)")
                if item.get('contexts'):
                    lines.append(f"- **Example contexts:** {', '.join(item['contexts'][:3])}")
                lines.append("")
        else:
            lines.append("*No specific data requests logged this session.*")
            lines.append("")
        
        # Prompt Evolution
        if self.evolution_patterns:
            lines.append("---")
            lines.append("")
            lines.append("## 🧬 PROMPT EVOLUTION (How My Thinking Changed)")
            lines.append("")
            lines.append("*The system prompt was rewritten to incorporate these learned patterns:*")
            lines.append("")
            for pattern in self.evolution_patterns:
                lines.append(f"- {pattern}")
            lines.append("")
        
        # Parlay Optimizer Training Stats
        if self.parlay_training_stats and self.parlay_training_stats.get('parlays_evaluated', 0) > 0:
            lines.append("---")
            lines.append("")
            lines.append("## 🎰 PARLAY OPTIMIZER TRAINING")
            lines.append("")
            lines.append("*The parlay optimizer was trained alongside predictions:*")
            lines.append("")
            
            ps = self.parlay_training_stats
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Parlays Generated | {ps.get('parlays_generated', 0)} |")
            lines.append(f"| Parlays Evaluated | {ps.get('parlays_evaluated', 0)} |")
            lines.append(f"| Wins | {ps.get('wins', 0)} |")
            lines.append(f"| Losses | {ps.get('losses', 0)} |")
            wins = ps.get('wins', 0)
            total = ps.get('parlays_evaluated', 1)
            win_rate = (wins / total * 100) if total > 0 else 0
            lines.append(f"| **Win Rate** | **{win_rate:.1f}%** |")
            lines.append(f"| Parlay Case Studies Created | {ps.get('case_studies_created', 0)} |")
            lines.append("")
        
        # Detailed Predictions (Optional - can be long)
        lines.append("---")
        lines.append("")
        lines.append("## 📋 DETAILED PREDICTIONS")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand all predictions</summary>")
        lines.append("")
        lines.append("| # | Player | Prop | Line | Prediction | Conf | Actual | Result |")
        lines.append("|---|--------|------|------|------------|------|--------|--------|")
        
        for i, p in enumerate(self.predictions, 1):
            result = "✅" if p.is_correct else "❌"
            lines.append(f"| {i} | {p.player} | {p.prop} | {p.line} | {p.prediction} | {p.confidence}/10 | {p.actual_value} ({p.actual_outcome}) | {result} |")
        
        lines.append("")
        lines.append("</details>")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated by Self-Evolving NBA Neural System v2.0*")
        lines.append(f"*Timestamp: {self.timestamp}*")
        
        # Save report
        report_content = "\n".join(lines)
        
        # Determine filename based on mode
        if self.mode == "historical":
            filename = f"historical_training_{self.timestamp}.md"
        else:
            filename = f"live_training_{self.timestamp}.md"
        
        filepath = os.path.join(TRAINING_REPORTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        print(f"\n📄 Training report saved: training_reports/{filename}")
        
        return filepath
    
    def print_summary(self):
        """Prints a quick summary to console."""
        metrics = self._calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 TRAINING SESSION SUMMARY")
        print("="*60)
        print(f"  Mode: {self.mode.upper()}")
        print(f"  Total Predictions: {metrics.get('total', 0)}")
        print(f"  Correct: {metrics.get('correct', 0)}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.1f}%")
        print(f"  Case Studies Created: {len(self.case_studies)}")
        print(f"  Data Requests Logged: {len(self.data_requests)}")
        print("="*60)


# ============================================================================
# STANDALONE UTILITIES
# ============================================================================

def generate_data_wishlist_report() -> str:
    """
    Generates a standalone report of all data feature requests.
    Useful for understanding what data to collect next.
    """
    os.makedirs(TRAINING_REPORTS_DIR, exist_ok=True)
    
    feature_counts = defaultdict(lambda: {"count": 0, "contexts": [], "first_seen": None})
    
    if os.path.exists(MISSING_FEATURES_LOG):
        with open(MISSING_FEATURES_LOG, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        req = json.loads(line)
                        feature = req.get("feature", "Unknown")
                        feature_counts[feature]["count"] += 1
                        feature_counts[feature]["contexts"].append(req.get("context", ""))
                        if not feature_counts[feature]["first_seen"]:
                            feature_counts[feature]["first_seen"] = req.get("timestamp", "")
                    except:
                        continue
    
    # Build report
    lines = []
    lines.append("# 💡 Data Wishlist Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("*This report aggregates all data features the model has requested.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    sorted_features = sorted(
        feature_counts.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    if sorted_features:
        lines.append("## 📋 Requested Features (by frequency)")
        lines.append("")
        lines.append("| Rank | Feature | Requests | Priority |")
        lines.append("|------|---------|----------|----------|")
        
        for i, (feature, data) in enumerate(sorted_features, 1):
            priority = "🔴 HIGH" if data["count"] >= 5 else "🟡 MEDIUM" if data["count"] >= 3 else "🟢 LOW"
            lines.append(f"| {i} | {feature} | {data['count']} | {priority} |")
        
        lines.append("")
        
        # Detailed breakdown
        lines.append("## 📝 Detailed Breakdown")
        lines.append("")
        
        for feature, data in sorted_features[:10]:
            lines.append(f"### {feature}")
            lines.append(f"- **Total Requests:** {data['count']}")
            lines.append(f"- **First Requested:** {data['first_seen'] or 'Unknown'}")
            lines.append(f"- **Example Contexts:**")
            for ctx in data["contexts"][:5]:
                lines.append(f"  - {ctx}")
            lines.append("")
    else:
        lines.append("*No data requests logged yet.*")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_wishlist_{timestamp}.md"
    filepath = os.path.join(TRAINING_REPORTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"📄 Data wishlist report saved: training_reports/{filename}")
    return filepath


if __name__ == "__main__":
    # Test report generation
    reporter = TrainingReporter(mode="historical")
    
    # Add some test predictions
    reporter.log_prediction(PredictionResult(
        player="LeBron James",
        team="LAL",
        prop="POINTS",
        line=25.5,
        prediction="OVER",
        confidence=8,
        actual_value=28,
        actual_outcome="OVER",
        is_correct=True,
        player_tier="STAR",
        context_tags=["weak_defense", "high_total"],
        reason="Star player vs weak defense with high game total"
    ))
    
    reporter.log_prediction(PredictionResult(
        player="Role Player",
        team="MIA",
        prop="POINTS",
        line=12.5,
        prediction="OVER",
        confidence=7,
        actual_value=8,
        actual_outcome="UNDER",
        is_correct=False,
        player_tier="ROLE",
        context_tags=["blowout_risk"],
        reason="Had edge but game was a blowout"
    ))
    
    reporter.log_case_study(CaseStudySummary(
        archetype="Role Player in Blowout",
        player="Role Player",
        prop="POINTS",
        mistake="Didn't account for reduced minutes in blowout",
        correction="When Spread > 12 AND USG% < 18%, cap confidence at 5",
        confidence_before=7,
        actual_outcome="UNDER"
    ))
    
    reporter.log_data_request(
        feature="Defender Matchup Data",
        context="Role Player POINTS prediction",
        impact="Would have shown elite defender assignment"
    )
    
    # Generate report
    reporter.generate_report({"start": "2025-11-20", "end": "2025-11-27"})
    reporter.print_summary()

