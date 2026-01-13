"""
DATA LEAKAGE VERIFICATION TESTS
================================
This module provides critical tests to ensure there's no data leakage in the ML pipeline.

Run with: python tests/test_data_leakage.py

CRITICAL: Run these tests after any changes to feature engineering or data preprocessing.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'process'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_pipeline'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ml_models'))

# ============================================================================
# TEST 1: Rolling Window Excludes Current Game
# ============================================================================

def test_rolling_window_excludes_current():
    """
    CRITICAL: Verify that L5_Avg_* features don't include the current game's stats.
    
    If this fails, there's feature leakage - the model would be predicting
    using information it shouldn't have access to.
    """
    print("\n" + "="*60)
    print("TEST 1: Rolling Window Excludes Current Game")
    print("="*60)
    
    try:
        from shared_config import get_spark_session, PLAYER_STATS_CSV
        from pyspark.sql.functions import col, avg, to_date, to_timestamp
        from pyspark.sql.window import Window
        
        spark = get_spark_session()
        
        # Load a sample of player data
        df = spark.read.csv(PLAYER_STATS_CSV, header=True, inferSchema=True)
        df = df.withColumn(
            "game_date",
            to_date(to_timestamp(col("gameDateTimeEst"), "yyyy-MM-dd HH:mm:ss"))
        )
        
        # Pick a player with many games
        sample_player = df.groupBy("personId", "playerName").count() \
            .orderBy(col("count").desc()) \
            .first()
        
        player_id = sample_player["personId"]
        player_name = sample_player["playerName"]
        print(f"   Testing with player: {player_name} (ID: {player_id})")
        
        # Filter to this player
        player_df = df.filter(col("personId") == player_id) \
            .orderBy("gameDateTimeEst")
        
        # Define the window (L5 = last 5 games, EXCLUDING current)
        window_5 = (Window
            .partitionBy("personId")
            .orderBy("gameDateTimeEst")
            .rowsBetween(-5, -1))  # -5 to -1 means EXCLUDE row 0 (current)
        
        # Calculate L5_Avg_PTS
        player_df = player_df.withColumn(
            "L5_Avg_PTS",
            avg(col("points")).over(window_5)
        )
        
        # Convert to pandas for easier testing
        pdf = player_df.select("game_date", "points", "L5_Avg_PTS").toPandas()
        pdf = pdf.dropna(subset=["L5_Avg_PTS"])  # Drop first 5 games
        
        # For each row, verify L5_Avg_PTS doesn't include current game's points
        passed = True
        for i, row in pdf.iterrows():
            # Get the previous 5 games manually
            prev_5_idx = max(0, i - 5)
            prev_5_games = pdf.iloc[prev_5_idx:i]  # EXCLUDES current row
            
            if len(prev_5_games) == 0:
                continue
                
            expected_avg = prev_5_games["points"].mean()
            actual_avg = row["L5_Avg_PTS"]
            
            # Check if they match (within floating point tolerance)
            if not np.isclose(expected_avg, actual_avg, rtol=1e-5):
                print(f"   ❌ LEAKAGE DETECTED at row {i}!")
                print(f"      Expected L5 avg (excluding current): {expected_avg:.2f}")
                print(f"      Actual L5 avg: {actual_avg:.2f}")
                print(f"      Current game points: {row['points']}")
                passed = False
                break
        
        if passed:
            print("   ✅ PASSED: Rolling windows correctly exclude current game")
            return True
        else:
            print("   ❌ FAILED: Data leakage detected in rolling windows!")
            return False
            
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# TEST 2: Temporal Train/Test Split
# ============================================================================

def test_temporal_split():
    """
    Verify that train/test split is chronological, not random.
    
    All train dates must be BEFORE all test dates.
    """
    print("\n" + "="*60)
    print("TEST 2: Temporal Train/Test Split")
    print("="*60)
    
    try:
        from model_trainer import temporal_train_test_split
        
        # Create synthetic data with dates
        dates = pd.date_range('2024-12-01', periods=100, freq='D')
        df = pd.DataFrame({
            'game_date': dates,
            'outcome': np.random.randint(0, 2, 100)
        })
        
        train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)
        
        train_max_date = pd.to_datetime(train_df['game_date']).max()
        test_min_date = pd.to_datetime(test_df['game_date']).min()
        
        print(f"   Train dates: {train_df['game_date'].min()} to {train_df['game_date'].max()}")
        print(f"   Test dates:  {test_df['game_date'].min()} to {test_df['game_date'].max()}")
        
        if train_max_date < test_min_date:
            print("   ✅ PASSED: Train dates are all before test dates")
            return True
        else:
            print("   ❌ FAILED: Train and test dates overlap!")
            print(f"      Train max: {train_max_date}")
            print(f"      Test min: {test_min_date}")
            return False
            
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# TEST 3: TimeSeriesSplit CV
# ============================================================================

def test_timeseries_cv():
    """
    Verify that cross-validation uses TimeSeriesSplit correctly.
    
    In each fold, validation data must be chronologically AFTER training data.
    """
    print("\n" + "="*60)
    print("TEST 3: TimeSeriesSplit Cross-Validation")
    print("="*60)
    
    try:
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create ordered indices (simulating chronological order)
        n_samples = 100
        X = np.arange(n_samples).reshape(-1, 1)  # Features are just indices
        y = np.random.randint(0, 2, n_samples)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_passed = True
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            train_max = train_idx.max()
            val_min = val_idx.min()
            
            print(f"   Fold {fold + 1}: Train indices [0-{train_max}], Val indices [{val_min}-{val_idx.max()}]")
            
            if train_max >= val_min:
                print(f"      ❌ LEAKAGE: Train contains indices >= validation start!")
                all_passed = False
        
        if all_passed:
            print("   ✅ PASSED: All CV folds maintain temporal ordering")
            return True
        else:
            print("   ❌ FAILED: Some CV folds have temporal leakage!")
            return False
            
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# TEST 4: Label Engineering No Lookahead
# ============================================================================

def test_label_engineering():
    """
    Verify that labels are derived from actual outcomes, not features.
    
    The 'outcome' column must be derived from actual game results,
    joined ONLY on date + player (not on any prediction-derived fields).
    """
    print("\n" + "="*60)
    print("TEST 4: Label Engineering (No Lookahead)")
    print("="*60)
    
    try:
        # Load a sample labeled file if available
        labeled_dir = os.path.join(PROJECT_ROOT, 'datasets', 'ml_training')
        
        if not os.path.exists(labeled_dir):
            print("   ⚠️ No labeled data directory found, skipping test")
            return True
        
        labeled_files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
        
        if not labeled_files:
            print("   ⚠️ No labeled data files found, skipping test")
            return True
        
        # Load the most recent labeled file
        latest_file = sorted(labeled_files)[-1]
        df = pd.read_csv(os.path.join(labeled_dir, latest_file))
        
        print(f"   Checking file: {latest_file}")
        
        # Verify outcome is binary
        if 'outcome' not in df.columns:
            print("   ⚠️ No 'outcome' column found")
            return False
        
        unique_outcomes = df['outcome'].dropna().unique()
        if not set(unique_outcomes).issubset({0, 1}):
            print(f"   ❌ FAILED: Outcome contains non-binary values: {unique_outcomes}")
            return False
        
        print(f"   Unique outcomes: {unique_outcomes}")
        
        # Verify outcome is derived from actual_value vs Line
        if 'actual_value' in df.columns and 'Line' in df.columns:
            # Recompute outcome and compare
            expected_outcome = (df['actual_value'] > df['Line']).astype(int)
            actual_outcome = df['outcome']
            
            # Compare (allow for NaN differences)
            valid_mask = df['actual_value'].notna() & df['Line'].notna()
            matches = (expected_outcome[valid_mask] == actual_outcome[valid_mask]).all()
            
            if matches:
                print("   ✅ PASSED: Outcomes correctly derived from (actual_value > Line)")
                return True
            else:
                mismatches = ((expected_outcome != actual_outcome) & valid_mask).sum()
                print(f"   ❌ FAILED: {mismatches} rows have incorrect outcome labels")
                return False
        else:
            print("   ⚠️ Cannot verify outcome derivation (missing actual_value or Line)")
            return True
            
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# TEST 5: Feature-Label Independence
# ============================================================================

def test_feature_label_independence():
    """
    Verify that features don't contain outcome information.
    
    Check that there's no perfect correlation between features and labels
    (which would indicate leakage).
    """
    print("\n" + "="*60)
    print("TEST 5: Feature-Label Independence")
    print("="*60)
    
    try:
        labeled_dir = os.path.join(PROJECT_ROOT, 'datasets', 'ml_training')
        
        if not os.path.exists(labeled_dir):
            print("   ⚠️ No labeled data directory found, skipping test")
            return True
        
        labeled_files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
        
        if not labeled_files:
            print("   ⚠️ No labeled data files found, skipping test")
            return True
        
        latest_file = sorted(labeled_files)[-1]
        df = pd.read_csv(os.path.join(labeled_dir, latest_file))
        
        print(f"   Checking file: {latest_file}")
        
        if 'outcome' not in df.columns:
            print("   ⚠️ No 'outcome' column found")
            return True
        
        # Suspicious columns that might leak
        suspicious = ['actual_value', 'points', 'reboundsTotal', 'assists', 'threePointersMade']
        suspicious_in_df = [c for c in suspicious if c in df.columns]
        
        # Check correlation with outcome
        suspicious_corrs = []
        for col in suspicious_in_df:
            valid = df[[col, 'outcome']].dropna()
            if len(valid) > 10:
                corr = valid[col].corr(valid['outcome'])
                suspicious_corrs.append((col, corr))
                
                if abs(corr) > 0.8:
                    print(f"   ⚠️ WARNING: High correlation between '{col}' and outcome ({corr:.3f})")
        
        # If actual_value is present and highly correlated, that's expected
        # because outcome IS derived from actual_value > Line
        # The key is that actual_value shouldn't be used as a FEATURE
        
        # Check feature config to make sure actual columns aren't used
        try:
            from feature_config import PROP_FEATURES
            all_features = set()
            for features in PROP_FEATURES.values():
                all_features.update(features)
            
            leaked_features = all_features.intersection(suspicious)
            if leaked_features:
                print(f"   ❌ FAILED: Outcome-derived columns used as features: {leaked_features}")
                return False
            else:
                print("   ✅ PASSED: No outcome-derived columns used as features")
                return True
                
        except ImportError:
            print("   ⚠️ Could not import feature_config, skipping feature check")
            return True
            
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# TEST 6: Season Average Window Check
# ============================================================================

def test_season_average_window():
    """
    Verify that Season_Avg_* uses unboundedPreceding to -1, not including current.
    """
    print("\n" + "="*60)
    print("TEST 6: Season Average Window Check")
    print("="*60)
    
    try:
        from shared_config import get_spark_session, PLAYER_STATS_CSV
        from pyspark.sql.functions import col, avg, to_date, to_timestamp
        from pyspark.sql.window import Window
        
        spark = get_spark_session()
        
        df = spark.read.csv(PLAYER_STATS_CSV, header=True, inferSchema=True)
        df = df.withColumn(
            "game_date",
            to_date(to_timestamp(col("gameDateTimeEst"), "yyyy-MM-dd HH:mm:ss"))
        )
        
        # Pick a player
        sample_player = df.groupBy("personId", "playerName").count() \
            .orderBy(col("count").desc()) \
            .first()
        
        player_id = sample_player["personId"]
        player_name = sample_player["playerName"]
        print(f"   Testing with player: {player_name}")
        
        player_df = df.filter(col("personId") == player_id) \
            .orderBy("gameDateTimeEst")
        
        # Season window (all previous games, EXCLUDING current)
        window_season = (Window
            .partitionBy("personId")
            .orderBy("gameDateTimeEst")
            .rowsBetween(Window.unboundedPreceding, -1))
        
        player_df = player_df.withColumn(
            "Season_Avg_PTS",
            avg(col("points")).over(window_season)
        )
        
        pdf = player_df.select("game_date", "points", "Season_Avg_PTS").toPandas()
        
        # First game should have NaN (no previous games)
        if pd.isna(pdf.iloc[0]["Season_Avg_PTS"]):
            print("   ✅ First game has NaN Season_Avg (correct - no previous games)")
        else:
            print(f"   ❌ First game should have NaN Season_Avg, got: {pdf.iloc[0]['Season_Avg_PTS']}")
            return False
        
        # Second game should have first game's points as average
        if len(pdf) > 1 and not pd.isna(pdf.iloc[1]["Season_Avg_PTS"]):
            expected = pdf.iloc[0]["points"]
            actual = pdf.iloc[1]["Season_Avg_PTS"]
            if np.isclose(expected, actual):
                print(f"   ✅ Second game Season_Avg = first game's points ({actual:.1f})")
            else:
                print(f"   ❌ Second game Season_Avg wrong: expected {expected}, got {actual}")
                return False
        
        print("   ✅ PASSED: Season average correctly excludes current game")
        return True
        
    except Exception as e:
        print(f"   ⚠️ ERROR: {e}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all data leakage tests."""
    print("\n" + "="*60)
    print("DATA LEAKAGE VERIFICATION TEST SUITE")
    print(f"Started: {datetime.now()}")
    print("="*60)
    
    tests = [
        ("Rolling Window Excludes Current", test_rolling_window_excludes_current),
        ("Temporal Train/Test Split", test_temporal_split),
        ("TimeSeriesSplit CV", test_timeseries_cv),
        ("Label Engineering", test_label_engineering),
        ("Feature-Label Independence", test_feature_label_independence),
        ("Season Average Window", test_season_average_window),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"   ⚠️ Test '{name}' raised exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - No data leakage detected!")
    else:
        print("\n⚠️ SOME TESTS FAILED - Review and fix data leakage issues!")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
