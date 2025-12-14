#!/usr/bin/env python3
"""
Phase 1 - Data Ingestion & Validation Agent (Great Expectations 1.9.2+)

Usage (example):
    python phase1_data_agent.py \
      --input predict_online_gaming.csv \
      --output-dir ./artifacts \
      --suite-name gaming_suite

This script:
 - loads the CSV specified by --input
 - canonicalizes and performs deterministic cleaning
 - programmatically creates a Great Expectations expectation suite (or reuses if exists)
 - runs GE validation and saves validation results
 - writes a cleaned CSV, expectation suite JSON, and a schema+validation JSON
 - prints a JSON "orchestrator_message" to stdout for orchestration

Requirements:
 - Python 3.9+
 - pandas
 - great_expectations>=1.9.2
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import pandas as pd

# Fail fast if Great Expectations is not present: this script **requires** GE.
try:
    import great_expectations as gx
    from great_expectations.data_context import EphemeralDataContext
    from great_expectations.core.expectation_suite import ExpectationSuite
except Exception as e:
    sys.stderr.write(
        "ERROR: Great Expectations is required for this script but is not importable.\n"
        "Install it with: pip install great_expectations>=1.9.2\n"
        f"Import error: {e}\n"
    )
    raise

# -----------------------
# Constants / defaults
# -----------------------
DEFAULT_ALLOWED_ENGAGEMENT = ["High", "Medium", "Low"]
DEFAULT_NUMERIC_CANDIDATES = [
    "Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"
]
# Simple deterministic rules (column -> (min, max or None))
RANGE_RULES = {
    "Age": (10, 99),
    "PlayTimeHours": (0, 1_000),
    "SessionsPerWeek": (0, 168),
    "AvgSessionDurationMinutes": (1, 24 * 60),
}

# -----------------------
# Helper functions
# -----------------------
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame. Fail if not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize and clean the DataFrame deterministically:
     - trim whitespace on object columns
     - normalize casing for Gender, GameGenre, Location
     - coerce numeric columns to numeric types (NaN on failure)
     - canonicalize EngagementLevel into allowed set (others -> NA)
     - remove rows violating simple deterministic rules
     - drop duplicate PlayerID rows (keep first)
    """
    df_clean = df.copy(deep=True)

    # Normalize string columns: strip whitespace, collapse internal whitespace
    str_cols = df_clean.select_dtypes(include="object").columns.tolist()
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Canonicalize common categorical columns
    if "Gender" in df_clean.columns:
        df_clean["Gender"] = df_clean["Gender"].str.title().replace({"M": "Male", "F": "Female"})

    if "GameGenre" in df_clean.columns:
        df_clean["GameGenre"] = df_clean["GameGenre"].str.upper()

    if "Location" in df_clean.columns:
        df_clean["Location"] = df_clean["Location"].str.upper()

    # Coerce numeric-like columns
    for col in DEFAULT_NUMERIC_CANDIDATES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Canonicalize EngagementLevel target
    if "EngagementLevel" in df_clean.columns:
        df_clean["EngagementLevel"] = df_clean["EngagementLevel"].astype(str).str.title().replace({
            "1": "High", "2": "Medium", "3": "Low"
        })
        df_clean.loc[~df_clean["EngagementLevel"].isin(DEFAULT_ALLOWED_ENGAGEMENT), "EngagementLevel"] = pd.NA

    # Apply simple deterministic rules and mark violating rows
    violations = pd.Series(False, index=df_clean.index)

    # check range rules
    for col, (min_v, max_v) in RANGE_RULES.items():
        if col in df_clean.columns:
            mask_ok = df_clean[col].between(min_v, max_v)
            violations = violations | (~mask_ok.fillna(False))

    # other simple checks
    if "PlayTimeHours" in df_clean.columns:
        violations = violations | (df_clean["PlayTimeHours"] < 0).fillna(False)
    if "PlayerID" in df_clean.columns:
        violations = violations | df_clean["PlayerID"].isna().fillna(False)

    if violations.any():
        df_clean = df_clean.loc[~violations].reset_index(drop=True)

    # Drop duplicates on PlayerID
    if "PlayerID" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["PlayerID"], keep="first").reset_index(drop=True)

    return df_clean


def create_ephemeral_context(output_dir: str) -> EphemeralDataContext:
    """
    Create an ephemeral GX context (no persistent configuration required).
    In GX 1.x, EphemeralDataContext is the recommended way for programmatic usage.
    """
    context = gx.get_context(mode="ephemeral")
    return context


def ensure_expectation_suite(context: EphemeralDataContext, suite_name: str, df_sample: pd.DataFrame) -> ExpectationSuite:
    """
    Ensure there is an expectation suite with the given name in the GX DataContext.
    If the suite exists, return it. Otherwise, create a programmatic suite with conservative expectations
    based on df_sample using the GX 1.x Fluent API.
    """
    # Check if suite exists
    try:
        suite = context.suites.get(name=suite_name)
        return suite
    except Exception:
        pass  # Suite doesn't exist, create it

    # Create a new suite using GX 1.x patterns (don't add to context yet)
    suite = gx.ExpectationSuite(name=suite_name)

    # Add a pandas datasource for validation
    datasource_name = "pandas_datasource"
    try:
        datasource = context.data_sources.add_pandas(name=datasource_name)
    except Exception:
        # Datasource might already exist
        datasource = context.data_sources.get(name=datasource_name)

    # Add a dataframe asset
    asset_name = "gaming_data"
    try:
        data_asset = datasource.add_dataframe_asset(name=asset_name)
    except Exception:
        data_asset = datasource.get_asset(name=asset_name)

    # Create a batch request with the sample dataframe
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "sample_batch"
    )
    
    # Get batch and create validator with the suite
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df_sample})
    validator = context.get_validator(batch=batch, expectation_suite=suite)

    # Build expectations on the sample DataFrame
    # Table-level expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=list(df_sample.columns)
    )

    # PlayerID expectations
    if "PlayerID" in df_sample.columns:
        validator.expect_column_values_to_not_be_null(column="PlayerID")
        validator.expect_column_values_to_be_unique(column="PlayerID")

    # EngagementLevel expectations
    if "EngagementLevel" in df_sample.columns:
        validator.expect_column_values_to_not_be_null(column="EngagementLevel")
        validator.expect_column_values_to_be_in_set(
            column="EngagementLevel",
            value_set=DEFAULT_ALLOWED_ENGAGEMENT
        )

    # Numeric range expectations (conservative)
    for col, (min_v, max_v) in RANGE_RULES.items():
        if col in df_sample.columns:
            validator.expect_column_values_to_be_between(
                column=col,
                min_value=min_v,
                max_value=max_v,
                mostly=0.95
            )

    # PlayTimeHours / SessionsPerWeek range
    if "PlayTimeHours" in df_sample.columns:
        validator.expect_column_values_to_be_between(
            column="PlayTimeHours",
            min_value=0,
            max_value=1_000,
            mostly=0.99
        )
    if "SessionsPerWeek" in df_sample.columns:
        validator.expect_column_values_to_be_between(
            column="SessionsPerWeek",
            min_value=0,
            max_value=168,
            mostly=0.99
        )

    # ensure GameGenre values are within observed set
    if "GameGenre" in df_sample.columns:
        unique_genres = df_sample["GameGenre"].dropna().unique().tolist()
        if unique_genres:
            validator.expect_column_values_to_be_in_set(
                column="GameGenre",
                value_set=unique_genres
            )

    # Save the suite to context
    context.suites.add(validator.expectation_suite)
    
    return validator.expectation_suite


def run_validation(context: EphemeralDataContext, df: pd.DataFrame, suite_name: str) -> Dict[str, Any]:
    """
    Run Great Expectations validation on `df` using suite `suite_name`.
    Returns the validation_result dictionary (serializable).
    Uses GX 1.x Fluent API patterns.
    """
    # Get or create the pandas datasource
    datasource_name = "pandas_datasource"
    try:
        datasource = context.data_sources.get(name=datasource_name)
    except Exception:
        datasource = context.data_sources.add_pandas(name=datasource_name)

    # Get or create the dataframe asset
    asset_name = "gaming_data"
    try:
        data_asset = datasource.get_asset(name=asset_name)
    except Exception:
        data_asset = datasource.add_dataframe_asset(name=asset_name)

    # Get or create batch definition
    try:
        batch_definition = data_asset.get_batch_definition("validation_batch")
    except Exception:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            "validation_batch"
        )

    # Create batch with the dataframe
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # Get validator
    validator = context.get_validator(
        batch=batch,
        expectation_suite_name=suite_name
    )

    # Run validation
    validation_result = validator.validate()
    
    # Convert to dictionary format
    return validation_result.to_json_dict()


def save_artifacts(output_dir: str, cleaned_df: pd.DataFrame, suite: ExpectationSuite, validation_result: Dict[str, Any]):
    """Save cleaned CSV, expectation suite JSON, and schema+validation JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    cleaned_csv_path = os.path.join(output_dir, "predict_online_gaming_clean.csv")
    suite_json_path = os.path.join(output_dir, "gaming_expectations.json")
    schema_json_path = os.path.join(output_dir, "predict_online_gaming_schema_validation.json")

    # save cleaned CSV
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    # suite JSON: write the suite dict if provided
    try:
        suite_dict = suite.to_json_dict()
        with open(suite_json_path, "w", encoding="utf-8") as f:
            json.dump(suite_dict, f, indent=2)
    except Exception:
        # best-effort: skip if suite can't be serialized
        suite_json_path = None

    # build schema summary
    schema_summary = {}
    for col in cleaned_df.columns:
        col_ser = cleaned_df[col]
        schema_summary[col] = {
            "dtype": str(col_ser.dtype),
            "n_missing": int(col_ser.isna().sum()),
            "n_unique": int(col_ser.nunique(dropna=True)),
            "sample_values": col_ser.dropna().unique().tolist()[:10]
        }

    out_obj = {
        "generated_at": datetime.utcnow().isoformat(),
        "cleaned_csv": cleaned_csv_path,
        "expectation_suite": suite_json_path,
        "schema": schema_summary,
        "validation_result": validation_result
    }

    with open(schema_json_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    return cleaned_csv_path, suite_json_path, schema_json_path


# -----------------------
# Main CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Phase1 Data Ingestion & Validation Agent")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write cleaned CSV + artifacts")
    parser.add_argument("--suite-name", default="gaming_suite", help="Great Expectations suite name")
    args = parser.parse_args()

    # 1) load raw CSV
    raw_df = load_csv(args.input)

    # 2) canonicalize / clean deterministically
    cleaned_df = canonicalize_dataframe(raw_df)

    # 3) initialize an ephemeral GX DataContext (GX 1.x pattern)
    context = create_ephemeral_context(args.output_dir)

    # 4) ensure the suite exists (or create from sample)
    suite = ensure_expectation_suite(
        context,
        args.suite_name,
        cleaned_df.sample(min(100, len(cleaned_df)), random_state=1)
    )

    # 5) run validation
    validation_result = run_validation(context, cleaned_df, args.suite_name)

    # 6) save artifacts
    cleaned_csv_path, suite_json_path, schema_json_path = save_artifacts(
        args.output_dir,
        cleaned_df,
        suite,
        validation_result
    )

    # 7) print orchestrator message JSON to stdout for orchestrator consumption
    orchestrator_message = {
        "artifact": {
            "clean_csv": cleaned_csv_path,
            "schema_validation": schema_json_path,
            "expectations": suite_json_path
        },
        "rows": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "validation": {
            "success": validation_result.get("success", None),
            "statistics": validation_result.get("statistics", {}),
            "results_summary": [
                {
                    "expectation_type": r.get("expectation_config", {}).get("expectation_type"),
                    "success": r.get("success"),
                    "result": r.get("result") if "result" in r else None
                }
                for r in validation_result.get("results", [])
            ]
        }
    }

    print(json.dumps(orchestrator_message, indent=2))


if __name__ == "__main__":
    main()