# 🏀 ML + LLM Hybrid Prediction System Roadmap

## Table of Contents
- [Phase 0: Model Assessment](#phase-0-model-assessment-current)
- [Phase 1: ML-Primary Architecture](#phase-1-ml-primary-architecture)
- [Phase 2: Tiered Prediction Modes](#phase-2-tiered-prediction-modes)
- [Phase 3: Specialized LLM Roles](#phase-3-specialized-llm-roles)
- [Phase 4: Implementation Timeline](#phase-4-implementation-timeline)
- [Phase 5: Decision Framework](#phase-5-decision-framework)
- [Phase 6: Cost & Performance Projections](#phase-6-cost--performance-projections)
- [Appendix: File Structure](#appendix-file-structure)

---

## Phase 0: Model Assessment (CURRENT)

> **Objective:** Validate XGBoost model performance before integration with Gemini.

### 0.1 Current System Inventory

| Component | Status | Location |
|-----------|--------|----------|
| XGBoost PTS Model | ✅ Trained | `ml_models/prop_models/pts_model.pkl` |
| XGBoost REB Model | ✅ Trained | `ml_models/prop_models/reb_model.pkl` |
| XGBoost AST Model | ✅ Trained | `ml_models/prop_models/ast_model.pkl` |
| XGBoost 3PM Model | ✅ Trained | `ml_models/prop_models/3pm_model.pkl` |
| Probability Calibrators | ✅ Trained | `ml_models/calibrators/*.pkl` |
| Meta-Ensemble | ⬜ Optional | `ml_models/meta_model.pkl` |
| Gemini Integration | ✅ Existing | `process/nba_ai_pipeline.py` |

### 0.2 Assessment Checklist

- [ ] **Step 1: Generate Holdout Test Set**
  - Use dates NOT in training data
  - Minimum 100+ props for statistical significance
  - Balanced across prop types (PTS, REB, AST, 3PM)

- [ ] **Step 2: Run ML Predictions on Holdout**
  ```bash
  cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
  source nba_env/bin/activate
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
  
  python -c "
  from ml_pipeline.inference import get_ml_predictor
  from ml_pipeline.evaluator import ModelEvaluator
  import pandas as pd
  
  # Load holdout data
  df = pd.read_csv('datasets/ml_training/YOUR_HOLDOUT_FILE.csv')
  
  # Get predictions
  predictor = get_ml_predictor()
  predictions = predictor.predict(df)
  
  # Evaluate
  ModelEvaluator.evaluate(predictions['outcome'], predictions['ml_prob_over'])
  "
  ```

- [ ] **Step 3: Record Baseline Metrics**

### 0.3 Target Metrics (Minimum Viable)

| Metric | Target | Meaning |
|--------|--------|---------|
| **Accuracy** | > 52% | Better than coin flip |
| **ROC-AUC** | > 0.54 | Discrimination ability |
| **Brier Score** | < 0.25 | Probability calibration |
| **Log Loss** | < 0.69 | Below random baseline |
| **ECE** | < 0.10 | Expected Calibration Error |

### 0.4 Metrics Recording Template

```markdown
## Model Assessment Results - [DATE]

### Overall Performance
| Metric | PTS | REB | AST | 3PM | Combined |
|--------|-----|-----|-----|-----|----------|
| Accuracy | | | | | |
| ROC-AUC | | | | | |
| Brier Score | | | | | |
| Log Loss | | | | | |
| ECE | | | | | |
| Sample Size | | | | | |

### Confidence Calibration
| Confidence Bucket | Predictions | Accuracy | Expected |
|-------------------|-------------|----------|----------|
| 1-3 (Low) | | | 50-55% |
| 4-6 (Medium) | | | 55-65% |
| 7-10 (High) | | | 65-80% |

### Decision Gate
- [ ] Accuracy > 52% for at least 2 prop types → Proceed to Phase 1
- [ ] ROC-AUC > 0.54 overall → Models have predictive signal
- [ ] Brier Score < 0.25 → Probabilities are reasonably calibrated
- [ ] If metrics fail → Return to training, increase data, tune hyperparameters
```

### 0.5 What "Good Enough" Looks Like

**Realistic Expectations for Sports Betting ML:**
- 52-55% accuracy = Profitable with proper bankroll management
- 55-58% accuracy = Strong edge, sustainable profits
- 58%+ accuracy = Elite performance (rare)

**Red Flags (Model Problems):**
- Accuracy < 50% → Model is harmful, do not deploy
- ROC-AUC < 0.50 → Model has no discrimination, random guessing
- Brier Score > 0.30 → Probabilities are poorly calibrated
- High accuracy but low ROC-AUC → Overfitting to majority class

### 0.6 Phase 0 Completion Criteria

| Criterion | Status |
|-----------|--------|
| Holdout test completed | ⬜ |
| At least 2 prop types with Accuracy > 52% | ⬜ |
| Overall ROC-AUC > 0.54 | ⬜ |
| Confidence calibration verified | ⬜ |
| Results documented | ⬜ |

**→ If ALL criteria met: Proceed to Phase 1**
**→ If criteria NOT met: Iterate on training data/features/hyperparameters**

---

## Phase 1: ML-Primary Architecture

> **Objective:** Make XGBoost the primary prediction engine, with Gemini for edge cases.

### 1.1 Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         FEATURE ENGINEERING              │
                    │       (nba_data_builder.py)              │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       XGBoost MODELS (Primary)           │
                    │  • Fast: <1ms per prop                   │
                    │  • Produces: P(OVER), Confidence, Edge   │
                    │  • 4 prop-specific models                │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       DECISION ROUTER                    │
                    │  Route based on ML confidence & context  │
                    └────┬────────────────────────────┬───────┘
                         │                            │
          ┌──────────────▼──────────────┐  ┌─────────▼────────────────┐
          │     HIGH CONFIDENCE          │  │    LOW/MEDIUM CONFIDENCE  │
          │   (ml_confidence >= 7)       │  │   OR COMPLEX CONTEXT      │
          │                              │  │   (ml_confidence < 7)     │
          │   ✅ Use ML prediction       │  │                           │
          │   directly                   │  │   📤 Send to Gemini       │
          │                              │  │   for contextual analysis │
          └──────────────┬──────────────┘  └─────────┬─────────────────┘
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       FINAL PREDICTIONS                  │
                    │  + Confidence + Reasoning                │
                    └─────────────────────────────────────────┘
```

### 1.2 Key Components to Build

| Component | File | Description |
|-----------|------|-------------|
| HybridPredictor | `ml_pipeline/hybrid_predictor.py` | Main orchestrator |
| DecisionRouter | `ml_pipeline/decision_router.py` | Routes props to ML or LLM |
| PredictionSource | `ml_pipeline/hybrid_predictor.py` | Tracks which system decided |

### 1.3 Routing Logic

```
IF ml_confidence >= 7 AND no_edge_case_flags:
    → Use ML directly (fast path)
    
ELIF ml_confidence >= 5:
    → Validate with Gemini (confirmation)
    
ELSE (ml_confidence < 5):
    → Gemini decides (LLM primary)
```

### 1.4 Edge Case Flags (Always Send to LLM)

| Flag | Condition | Reason |
|------|-----------|--------|
| `is_b2b` | True | Fatigue affects performance unpredictably |
| `blowout_risk` | \|Spread\| > 12 | Game script changes stat distribution |
| `volatile_player` | Reliability_Tag == 'VOLATILE' | High variance, ML less reliable |
| `unusual_line_move` | Line changed > 1 point | Market knows something |
| `injury_flagged` | Manual flag | News not in training data |

### 1.5 Phase 1 Deliverables

- [ ] `hybrid_predictor.py` created
- [ ] Routing thresholds configured
- [ ] Integration with existing pipeline
- [ ] A/B test: Hybrid vs ML-only vs LLM-only
- [ ] Documentation updated

---

## Phase 2: Tiered Prediction Modes

> **Objective:** Create operational modes for different use cases.

### 2.1 Mode Definitions

| Mode | Use Case | ML Calls | LLM Calls | Latency |
|------|----------|----------|-----------|---------|
| **Fast** | Live betting, screening | All | None | <1 sec |
| **Balanced** | Daily predictions | All | ~30% | ~20 sec |
| **Premium** | High-stakes analysis | All | All | ~60 sec |

### 2.2 Mode Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                    WHEN TO USE EACH MODE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🏃 FAST MODE                                                │
│  ├─ Live in-game betting (no time for LLM)                  │
│  ├─ Screening 100+ props quickly                            │
│  ├─ Real-time odds monitoring                               │
│  └─ When you trust ML confidence is high                    │
│                                                              │
│  ⚖️ BALANCED MODE (Recommended Default)                      │
│  ├─ Daily prediction slate                                  │
│  ├─ Standard betting workflow                               │
│  ├─ When you have 30-60 seconds per batch                   │
│  └─ Best accuracy/speed tradeoff                            │
│                                                              │
│  💎 PREMIUM MODE                                             │
│  ├─ Building high-stakes parlays                            │
│  ├─ Small slate (<20 props) analysis                        │
│  ├─ When you need full reasoning for each prop              │
│  └─ Complex multi-game scenarios                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation

```python
class HybridPredictor:
    def predict(self, df, mode='balanced'):
        if mode == 'fast':
            return self._predict_fast(df)
        elif mode == 'balanced':
            return self._predict_balanced(df)
        elif mode == 'premium':
            return self._predict_premium(df)
```

### 2.4 Phase 2 Deliverables

- [ ] Mode switching implemented
- [ ] CLI flags for mode selection
- [ ] Performance benchmarks for each mode
- [ ] Mode recommendation logic

---

## Phase 3: Specialized LLM Roles

> **Objective:** Use Gemini for high-value tasks where it excels.

### 3.1 Role Definitions

| Role | Trigger | Value Add |
|------|---------|-----------|
| **Edge Case Analyst** | Low ML confidence | Contextual judgment |
| **Parlay Validator** | Parlay construction | Correlation detection |
| **Reflection Engine** | Post-game misses | Learning from mistakes |
| **Meta-Learner** | Weekly schedule | System evolution |

### 3.2 Role: Edge Case Analyst

**When:** ML confidence < 7 OR edge case flag present

**What Gemini Does:**
- Reviews full context (B2B, matchup, injury news)
- Weighs factors ML can't see
- Provides reasoning for override

**Output:**
```json
{
  "prediction": "OVER",
  "confidence": 7,
  "reasoning": "Despite ML uncertainty, player has dominated this matchup historically",
  "override_ml": true
}
```

### 3.3 Role: Parlay Validator

**When:** Building multi-leg parlays

**What Gemini Does:**
- Checks for same-game correlations
- Identifies game script dependencies
- Flags hidden risks

**Output:**
```json
{
  "is_valid": true,
  "correlation_risk": "medium",
  "concerns": ["Two players from same game - if blowout, both may sit"],
  "recommended_adjustment": "Replace Leg 3 with player from different game"
}
```

### 3.4 Role: Reflection Engine

**When:** After games complete (daily)

**What Gemini Does:**
- Analyzes high-confidence misses
- Generates case studies
- Identifies ML blind spots

**Output:**
```json
{
  "archetype": "Star Player Rest Game",
  "ml_blind_spot": "ML didn't detect load management pattern",
  "correction_rule": "Flag players with 35+ MPG when team is locked in standings",
  "add_to_llm_flags": true
}
```

### 3.5 Role: Meta-Learner

**When:** Weekly (Sunday night)

**What Gemini Does:**
- Reviews all case studies from the week
- Identifies recurring patterns
- Updates system prompt
- Adjusts routing thresholds

**Output:**
```markdown
## Weekly Learning Summary

### Patterns Identified
1. B2B games: ML accuracy dropped to 48% → Increase LLM routing
2. High spreads (>14): LLM caught 3 blowout rests that ML missed

### Threshold Adjustments
- ML_HIGH_CONFIDENCE: 7 → 8 (be more conservative)
- Always LLM for players with USG% > 30%

### System Prompt Updates
Added: "Check for load management risk when spread > 10 and player > 32 MPG"
```

### 3.6 Phase 3 Deliverables

- [ ] Edge Case Analyst implemented
- [ ] Parlay Validator implemented
- [ ] Reflection Engine implemented
- [ ] Meta-Learner weekly job scheduled
- [ ] Case study auto-generation

---

## Phase 4: Implementation Timeline

### Week 1: Foundation (After Phase 0 Complete)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create `hybrid_predictor.py` skeleton | File created |
| 2 | Implement routing logic | `should_use_llm()` working |
| 3 | Integrate with existing `nba_ai_pipeline.py` | Hybrid mode functional |
| 4 | Test on historical data | Baseline metrics |
| 5 | Tune confidence thresholds | Optimal routing % |
| 6-7 | Deploy Balanced mode as default | Production ready |

### Week 2: Optimization

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Track ML vs LLM agreement rates | Stats dashboard |
| 3-4 | Identify which edge cases LLM helps | Edge case analysis |
| 5 | Tune ML_HIGH_CONFIDENCE threshold | Optimal threshold |
| 6-7 | Document findings | Updated roadmap |

### Week 3: Specialized Roles

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Implement parlay validation | `parlay_validator.py` |
| 3-4 | Implement ML miss reflection | `reflection_engine.py` |
| 5-7 | Build monitoring dashboard | Streamlit/CLI dashboard |

### Week 4: Continuous Learning

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Implement meta-learner | `meta_learner.py` |
| 3 | Schedule weekly learning job | Cron job configured |
| 4-5 | A/B test: Hybrid vs baselines | Test results |
| 6-7 | Final documentation | Complete system docs |

---

## Phase 5: Decision Framework

### 5.1 Prediction Routing Flowchart

```
                    ┌─────────────────┐
                    │   NEW PROP      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Run XGBoost ML  │
                    │ (instant)       │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ ML Confidence >= 7?           │
              └───────┬─────────────┬────────┘
                      │             │
                     YES           NO
                      │             │
                      ▼             ▼
         ┌────────────────┐  ┌────────────────┐
         │ Has edge case  │  │ Confidence 5-6? │
         │ flags?         │  └───────┬────────┘
         └───────┬───────┘          │
                 │                  │
            ┌────┴────┐        ┌────┴────┐
            NO       YES       YES      NO
            │         │         │        │
            ▼         ▼         ▼        ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐
     │ USE ML   │ │ LLM      │ │ LLM      │ │ LLM    │
     │ DIRECTLY │ │ VALIDATE │ │ VALIDATE │ │ DECIDE │
     └──────────┘ └──────────┘ └──────────┘ └────────┘
           │           │            │           │
           │           ▼            ▼           │
           │    ┌─────────────────────┐        │
           │    │ ML & LLM agree?     │        │
           │    └───────┬────────┬────┘        │
           │           YES      NO            │
           │            │        │             │
           │            ▼        ▼             │
           │     ┌─────────┐ ┌─────────┐      │
           │     │ USE ML  │ │ Higher  │      │
           │     │ (valid) │ │ conf    │      │
           │     │         │ │ wins    │      │
           │     └────┬────┘ └────┬────┘      │
           │          │           │           │
           └──────────┴───────────┴───────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ FINAL PREDICTION    │
                    │ + Source tracking   │
                    └─────────────────────┘
```

### 5.2 Threshold Configuration

```python
# ml_pipeline/config.py

HYBRID_CONFIG = {
    # Routing thresholds
    'ML_HIGH_CONFIDENCE': 7,      # Use ML directly above this
    'ML_LOW_CONFIDENCE': 5,       # Always use LLM below this
    
    # Edge case flags (always send to LLM)
    'EDGE_CASE_FLAGS': {
        'is_b2b': True,
        'spread_threshold': 12,    # |Spread| > this
        'volatile_player': True,   # Reliability_Tag == 'VOLATILE'
        'usg_threshold': 30,       # USG% > this (load management risk)
    },
    
    # Disagreement resolution
    'DISAGREEMENT_RULE': 'higher_confidence',  # or 'always_llm', 'always_ml'
    
    # Batch settings
    'LLM_BATCH_SIZE': 10,         # Props per Gemini call
    'LLM_RATE_LIMIT': 1.0,        # Seconds between calls
}
```

---

## Phase 6: Cost & Performance Projections

### 6.1 Scenario: 50 Props/Day

| Mode | LLM Calls | Cost/Day | Latency | Est. Accuracy |
|------|-----------|----------|---------|---------------|
| ML-Only | 0 | $0.00 | <1 sec | 54% |
| Hybrid (30%) | 5 batches | ~$0.01 | ~20 sec | 56% |
| Full LLM | 5 batches | ~$0.03 | ~50 sec | 55% |

*Assumes Gemini free tier (60 RPM) with batching*

### 6.2 ROI Calculation

```
Break-Even Analysis:
- If hybrid mode gives +2% accuracy
- On 50 props/day at $10 average bet
- That's 1 extra win per day
- 1 win × $10 × 0.91 (juice) = $9.10/day profit
- Cost: ~$0.01/day

ROI: 910x return on API costs
```

### 6.3 Performance Benchmarks (To Measure)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| ML-only accuracy | >52% | Holdout test |
| Hybrid accuracy | >54% | A/B test |
| LLM agreement rate | >70% | Track in production |
| LLM value-add rate | >5% | When LLM overrides, was it right? |

---

## Appendix: File Structure

### New Files to Create

```
ml_pipeline/
├── inference.py              # ✅ Existing
├── hybrid_predictor.py       # 🆕 Phase 1
├── decision_router.py        # 🆕 Phase 1
├── config.py                 # 🆕 Phase 1 (thresholds)
├── parlay_validator.py       # 🆕 Phase 3
├── reflection_engine.py      # 🆕 Phase 3
└── meta_learner.py           # 🆕 Phase 3

process/
├── nba_ai_pipeline.py        # ✏️ Update to use HybridPredictor
└── optimizer.py              # ✏️ Update to use parlay validation
```

### Updated Integration Points

```python
# In nba_ai_pipeline.py - replace direct Gemini calls with:

from ml_pipeline.hybrid_predictor import HybridPredictor

class ReflexivePredictionEngine:
    def __init__(self):
        self.hybrid = HybridPredictor(mode='balanced')
    
    def analyze_props(self, features_df):
        return self.hybrid.predict(features_df)
```

---

## Quick Reference Commands

```bash
# Phase 0: Model Assessment
cd "/Users/anindithram/Documents/NBA Player Prop Analyst"
source nba_env/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Evaluate models
python ml_pipeline/train_all_models.py --evaluate-only

# Generate holdout predictions
python -c "
from ml_pipeline.inference import get_ml_predictor
from ml_pipeline.evaluator import ModelEvaluator
import pandas as pd

df = pd.read_csv('datasets/ml_training/labeled_data_LATEST.csv')
predictor = get_ml_predictor()
predictions = predictor.predict(df)
print(ModelEvaluator.generate_report(predictions))
"
```

---

## Progress Tracker

| Phase | Status | Start Date | End Date | Notes |
|-------|--------|------------|----------|-------|
| Phase 0: Assessment | 🔄 In Progress | | | |
| Phase 1: ML-Primary | ⬜ Not Started | | | |
| Phase 2: Tiered Modes | ⬜ Not Started | | | |
| Phase 3: LLM Roles | ⬜ Not Started | | | |
| Phase 4: Timeline | ⬜ Not Started | | | |

---

*Last Updated: December 12, 2025*

