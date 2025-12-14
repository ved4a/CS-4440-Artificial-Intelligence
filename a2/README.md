# Agentic AI Project

Pipeline for the Predict Online Gaming Behavior project. The orchestrator runs ingestion → feature engineering → model training → guardrails → monitoring → explainability → recommendations, storing outputs under `artifacts/`.

## Prerequisites

-   Python 3.10+ on Windows (tested with PowerShell).
-   `predict_online_gaming.csv` present in the repository root.
-   Optional: a clean virtual environment to isolate dependencies.

## Setup (PowerShell)

```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the full pipeline

```
python smoke_orchestrator.py
```

This triggers the orchestrated end-to-end run and writes stage outputs to `artifacts/` (ingestion, feature catalogs, trained models, guardrail checks, monitoring logs, explainability assets, and recommendations).

## Re-run explainability only (optional)

```
python run_explainability.py
```

Uses the saved model artifacts (default `trainer_random_forest`) and regenerates global/local explanations plus guardrail summaries under `artifacts/explainability/`.

## Key outputs

-   `artifacts/ingestion` and `artifacts/predict_online_gaming_clean.csv`: cleaned data, schema, and validation reports.
-   `artifacts/modeling/models`: trained models, metadata, and probability exports.
-   `artifacts/guardrail` and `artifacts/monitoring`: evaluation checks, drift metrics, and monitoring history.
-   `artifacts/explainability` and `artifacts/recommendations`: SHAP summaries and recommended actions.
