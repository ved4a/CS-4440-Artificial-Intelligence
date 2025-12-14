#!/usr/bin/env python3
"""
Phase 2 - EDA & Feature Engineering Agent

Usage:
    python phase2_feature_agent.py \
      --input ./artifacts/predict_online_gaming_clean.csv \
      --schema ./artifacts/predict_online_gaming_schema_validation.json \
      --output-dir ./artifacts \
      --target EngagementLevel

This script:
 - Loads cleaned data from Phase 1
 - Performs exploratory data analysis (EDA)
 - Engineers domain-specific features for gaming behavior prediction
 - Ranks features by predictive power (mutual information)
 - Outputs enhanced dataset + feature catalog + EDA report
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

# Scikit-learn for feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# -----------------------
# Constants
# -----------------------
NUMERIC_FEATURES = [
    "Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"
]

CATEGORICAL_FEATURES = ["Gender", "Location", "GameGenre", "GameDifficulty"]

TARGET_COLUMN = "EngagementLevel"

# -----------------------
# Step 1: Load Data & Validate
# -----------------------
# -----------------------
# Step 1: Load Data & Validate
# -----------------------
def load_clean_data(csv_path: str, schema_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the cleaned CSV from Phase 1 and its schema/validation JSON.
    Validates that the data contains expected columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned CSV not found: {csv_path}")
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema JSON not found: {schema_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load schema/validation info
    with open(schema_path, 'r') as f:
        schema_info = json.load(f)
    
    # Validate expected columns exist
    required_cols = ["PlayerID", TARGET_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns in cleaned data: {missing_cols}")
    
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Handle different schema structures from Phase 1
    validation_success = None
    if 'validation_result' in schema_info:
        validation_success = schema_info['validation_result'].get('success')
    elif 'validation' in schema_info:
        validation_success = schema_info['validation'].get('success')
    
    if validation_success is not None:
        print(f"✓ Phase 1 validation status: {validation_success}")
    else:
        print(f"✓ Schema loaded (validation status not found in schema)")
    
    return df, schema_info


# -----------------------
# Step 2: Descriptive Statistics
# -----------------------
def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive descriptive statistics for numeric and categorical features.
    Returns a dictionary with summary statistics.
    """
    stats_report = {
        "numeric_summary": {},
        "categorical_summary": {},
        "missing_data": {},
        "correlation_matrix": {}
    }
    
    # Numeric features
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe().to_dict()
        stats_report["numeric_summary"] = numeric_stats
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        stats_report["correlation_matrix"] = corr_matrix.to_dict()
    
    # Categorical features
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats_report["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_values": value_counts.head(10).to_dict(),
            "value_distribution": value_counts.to_dict()
        }
    
    # Missing data analysis
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            stats_report["missing_data"][col] = {
                "count": int(missing_count),
                "percentage": float(missing_count / len(df) * 100)
            }
    
    print(f"✓ Computed descriptive statistics for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    
    return stats_report


# -----------------------
# Step 3: Target Variable Analysis
# -----------------------
def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze the target variable distribution and class balance.
    Compute cross-tabulations with key categorical features.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    target_analysis = {
        "class_distribution": {},
        "class_balance": {},
        "cross_tabulations": {}
    }
    
    # Class distribution
    class_counts = df[target_col].value_counts()
    total = len(df)
    
    target_analysis["class_distribution"] = {
        "counts": class_counts.to_dict(),
        "percentages": (class_counts / total * 100).to_dict()
    }
    
    # Check imbalance
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = float(majority_class) / float(minority_class)
    
    target_analysis["class_balance"] = {
        "imbalance_ratio": float(imbalance_ratio),
        "is_balanced": bool(imbalance_ratio < 2.0),
        "recommendation": "Use stratified sampling" if imbalance_ratio >= 2.0 else "Classes are reasonably balanced"
    }
    
    # Cross-tabulations with categorical features
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    for col in categorical_cols:
        crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        target_analysis["cross_tabulations"][col] = crosstab.to_dict()
    
    print(f"✓ Analyzed target variable: {class_counts.to_dict()}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    return target_analysis


# -----------------------
# Step 4: Automated Feature Engineering
# -----------------------
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Create domain-specific engineered features for gaming behavior prediction.
    Returns enhanced dataframe and a list describing each engineered feature.
    """
    df_eng = df.copy()
    engineered_features = []
    
    # Helper function to register each feature
    def register_feature(name: str, formula: str, description: str):
        engineered_features.append({
            "name": name,
            "formula": formula,
            "description": description,
            "type": "numeric"
        })
    
    # 1. Session Intensity (sessions per week × avg session duration)
    if "SessionsPerWeek" in df.columns and "AvgSessionDurationMinutes" in df.columns:
        df_eng["SessionIntensity"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"]
        register_feature(
            "SessionIntensity",
            "SessionsPerWeek × AvgSessionDurationMinutes",
            "Total weekly playtime from session frequency and duration"
        )
    
    # 2. Achievements per hour played
    if "AchievementsUnlocked" in df.columns and "PlayTimeHours" in df.columns:
        df_eng["AchievementsPerHour"] = df["AchievementsUnlocked"] / (df["PlayTimeHours"] + 1)  # +1 to avoid division by zero
        register_feature(
            "AchievementsPerHour",
            "AchievementsUnlocked / (PlayTimeHours + 1)",
            "Achievement rate indicating player efficiency"
        )
    
    # 3. Purchase rate (purchases per session)
    if "InGamePurchases" in df.columns and "SessionsPerWeek" in df.columns:
        total_sessions = df["SessionsPerWeek"] * (df["PlayTimeHours"] / (df["AvgSessionDurationMinutes"] / 60 + 0.1))
        df_eng["PurchaseRate"] = df["InGamePurchases"] / (total_sessions + 1)
        register_feature(
            "PurchaseRate",
            "InGamePurchases / (estimated_total_sessions + 1)",
            "Spending behavior per gaming session"
        )
    
    # 4. Level progression rate (level per hour)
    if "PlayerLevel" in df.columns and "PlayTimeHours" in df.columns:
        df_eng["LevelProgressionRate"] = df["PlayerLevel"] / (df["PlayTimeHours"] + 1)
        register_feature(
            "LevelProgressionRate",
            "PlayerLevel / (PlayTimeHours + 1)",
            "How quickly player progresses through levels"
        )
    
    # 5. Engagement score (composite metric)
    if all(col in df.columns for col in ["SessionsPerWeek", "PlayTimeHours", "AchievementsUnlocked"]):
        df_eng["EngagementScore"] = (
            df["SessionsPerWeek"] * 0.3 +
            (df["PlayTimeHours"] / 10) * 0.4 +
            (df["AchievementsUnlocked"] / 50) * 0.3
        )
        register_feature(
            "EngagementScore",
            "Weighted combination of sessions, playtime, and achievements",
            "Composite engagement metric"
        )
    
    # 6. Age group binning
    if "Age" in df.columns:
        df_eng["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["Teen", "Young_Adult", "Adult", "Middle_Age", "Senior"]
        )
        engineered_features.append({
            "name": "AgeGroup",
            "formula": "pd.cut(Age, bins=[0,18,25,35,50,100])",
            "description": "Age segmented into demographic groups",
            "type": "categorical"
        })
    
    # 7. Playtime tier
    if "PlayTimeHours" in df.columns:
        df_eng["PlaytimeTier"] = pd.cut(
            df["PlayTimeHours"],
            bins=[0, 10, 50, 200, 1000],
            labels=["Casual", "Regular", "Hardcore", "Extreme"]
        )
        engineered_features.append({
            "name": "PlaytimeTier",
            "formula": "pd.cut(PlayTimeHours, bins=[0,10,50,200,1000])",
            "description": "Player type based on total playtime",
            "type": "categorical"
        })
    
    # 8. Purchase behavior flag
    if "InGamePurchases" in df.columns:
        df_eng["IsPayer"] = (df["InGamePurchases"] > 0).astype(int)
        register_feature(
            "IsPayer",
            "1 if InGamePurchases > 0 else 0",
            "Binary flag for whether player makes purchases"
        )
    
    # 9. Session consistency (low variance in session length suggests routine)
    if "AvgSessionDurationMinutes" in df.columns and "SessionsPerWeek" in df.columns:
        df_eng["SessionConsistency"] = df["AvgSessionDurationMinutes"] / (df["SessionsPerWeek"] + 1)
        register_feature(
            "SessionConsistency",
            "AvgSessionDurationMinutes / (SessionsPerWeek + 1)",
            "Ratio indicating session length consistency"
        )
    
    print(f"✓ Engineered {len(engineered_features)} new features")
    
    return df_eng, engineered_features


# -----------------------
# Step 5: Feature Quality Assessment
# -----------------------
def assess_feature_quality(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Compute mutual information scores to rank features by predictive power.
    Identify low-variance and highly correlated features.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Encode target if categorical
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    # Get all numeric features (original + engineered)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col and col != "PlayerID"]
    
    # Handle missing values for MI calculation
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create feature importance ranking
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Identify low-variance features (variance < threshold)
    low_variance_features = []
    variance_threshold = 0.01
    
    for col in numeric_cols:
        if X[col].var() < variance_threshold:
            low_variance_features.append(col)
    
    # Identify highly correlated feature pairs (correlation > 0.9)
    corr_matrix = X.corr().abs()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })
    
    quality_report = {
        "feature_importance": feature_importance.to_dict('records'),
        "low_variance_features": low_variance_features,
        "high_correlation_pairs": high_corr_pairs,
        "recommendations": {
            "top_features": feature_importance.head(10)['feature'].tolist(),
            "features_to_drop": low_variance_features + [pair['feature2'] for pair in high_corr_pairs]
        }
    }
    
    print(f"✓ Ranked {len(numeric_cols)} features by mutual information")
    print(f"  Top 5 features: {feature_importance.head(5)['feature'].tolist()}")
    
    return quality_report


# -----------------------
# Step 6: Generate Feature Catalog
# -----------------------
def generate_feature_catalog(
    original_stats: Dict[str, Any],
    engineered_features: List[Dict[str, Any]],
    quality_report: Dict[str, Any],
    target_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compile a comprehensive feature catalog documenting all features,
    their statistics, engineering logic, and recommendations.
    """
    catalog = {
        "generated_at": datetime.now().isoformat(),
        "original_features": {
            "numeric": list(original_stats["numeric_summary"].keys()),
            "categorical": list(original_stats["categorical_summary"].keys()),
            "statistics": original_stats
        },
        "engineered_features": engineered_features,
        "target_variable": target_analysis,
        "feature_quality": quality_report,
        "modeling_recommendations": {
            "recommended_features": quality_report["recommendations"]["top_features"],
            "features_to_drop": quality_report["recommendations"]["features_to_drop"],
            "preprocessing_notes": [
                "Apply standard scaling to numeric features before modeling",
                "One-hot encode categorical features (Gender, GameGenre, Location, GameDifficulty)",
                "Use stratified K-fold for cross-validation due to class distribution",
                "Consider SMOTE if class imbalance affects minority class performance"
            ]
        }
    }
    
    return catalog

# -----------------------
# Step 7: Save Artifacts
# -----------------------
def save_artifacts(
    output_dir: str,
    enhanced_df: pd.DataFrame,
    feature_catalog: Dict[str, Any],
    eda_report: Dict[str, Any]
) -> Dict[str, str]:
    """
    Save enhanced dataset, feature catalog, and EDA report.
    Returns paths to saved artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save enhanced dataset
    enhanced_csv_path = os.path.join(output_dir, "predict_online_gaming_enhanced.csv")
    enhanced_df.to_csv(enhanced_csv_path, index=False)
    
    # Helper function to convert numpy/pandas types to JSON-serializable types
    def make_json_serializable(obj):
        """Recursively convert numpy/pandas types to Python native types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Convert to JSON-serializable format
    feature_catalog_clean = make_json_serializable(feature_catalog)
    eda_report_clean = make_json_serializable(eda_report)
    
    # Save feature catalog
    catalog_path = os.path.join(output_dir, "feature_catalog.json")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(feature_catalog_clean, f, indent=2)
    
    # Save EDA report
    eda_path = os.path.join(output_dir, "eda_report.json")
    with open(eda_path, 'w', encoding='utf-8') as f:
        json.dump(eda_report_clean, f, indent=2)
    
    print(f"✓ Saved enhanced dataset: {enhanced_csv_path}")
    print(f"✓ Saved feature catalog: {catalog_path}")
    print(f"✓ Saved EDA report: {eda_path}")
    
    return {
        "enhanced_csv": enhanced_csv_path,
        "feature_catalog": catalog_path,
        "eda_report": eda_path
    }


# -----------------------
# Main Pipeline
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 2 - EDA & Feature Engineering Agent")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV from Phase 1")
    parser.add_argument("--schema", required=True, help="Path to schema/validation JSON from Phase 1")
    parser.add_argument("--output-dir", required=True, help="Directory to write enhanced dataset + reports")
    parser.add_argument("--target", default="EngagementLevel", help="Target column name")
    args = parser.parse_args()
    
    # Step 1: Load clean data
    df, schema_info = load_clean_data(args.input, args.schema)
    
    # Step 2: Compute descriptive statistics
    descriptive_stats = compute_descriptive_stats(df)
    
    # Step 3: Analyze target variable
    target_analysis = analyze_target_variable(df, args.target)
    
    # Step 4: Engineer features
    df_enhanced, engineered_features = engineer_features(df)
    
    # Step 5: Assess feature quality
    quality_report = assess_feature_quality(df_enhanced, args.target)
    
    # Step 6: Generate feature catalog
    feature_catalog = generate_feature_catalog(
        descriptive_stats,
        engineered_features,
        quality_report,
        target_analysis
    )
    
    # Compile EDA report
    eda_report = {
        "descriptive_statistics": descriptive_stats,
        "target_analysis": target_analysis,
        "feature_quality": quality_report
    }
    
    # Step 7: Save artifacts
    artifact_paths = save_artifacts(
        args.output_dir,
        df_enhanced,
        feature_catalog,
        eda_report
    )
    
    # Output orchestrator message
    orchestrator_message = {
        "phase": "feature_engineering",
        "status": "success",
        "artifacts": artifact_paths,
        "summary": {
            "input_rows": len(df),
            "output_rows": len(df_enhanced),
            "original_features": len(df.columns),
            "engineered_features": len(engineered_features),
            "total_features": len(df_enhanced.columns),
            "top_5_features": quality_report["recommendations"]["top_features"][:5],
            "target_balance": target_analysis["class_balance"]
        },
        "next_steps": [
            "Proceed to Phase 3: Model training with recommended features",
            "Use stratified sampling based on target distribution",
            "Apply feature scaling and one-hot encoding"
        ]
    }
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE - Orchestrator Message:")
    print("="*60)
    print(json.dumps(orchestrator_message, indent=2))


if __name__ == "__main__":
    main()