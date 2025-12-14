#!/usr/bin/env python3
"""
Evaluator & Guardrail Agent (Phase 4)

This agent loads previously trained models, re-computes evaluation metrics,
compares every candidate against the designated baseline, runs SHAP-based
explanations, and performs adversarial synthetic tests to surface robustness
issues before deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_DEFAULT = "EngagementLevel"
DEFAULT_MIN_HIGH_RECALL = 0.60
DEFAULT_OUTPUT_DIR = Path("artifacts") / "guardrail"
DEFAULT_MAX_SHAP_SAMPLES = 25


@dataclass
class ModelArtifact:
    name: str
    agent_role: str
    metadata_path: Path
    model_path: Path
    probabilities_path: Path
    metadata: Dict[str, Any]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    for candidate in ("row_index", "index", "Unnamed: 0"):
        if candidate in df.columns:
            df.set_index(candidate, inplace=True)
            df.index.name = "row_index"
            break
    if df.index.name != "row_index":
        df.index.name = "row_index"
    return df


def load_feature_catalog(catalog_path: Path) -> Dict[str, Any]:
    with open(catalog_path, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    return catalog


def resolve_feature_names(
    catalog: Dict[str, Any],
    fallback_columns: Iterable[str],
    target: str,
) -> Tuple[List[str], List[str], List[str]]:
    feature_names = catalog.get("feature_names")
    numeric_features = catalog.get("numeric_features", [])
    categorical_features = catalog.get("categorical_features", [])

    if not feature_names:
        feature_names = [col for col in fallback_columns if col != target]

    if not numeric_features and not categorical_features:
        numeric_features = [col for col in feature_names if catalog.get("dtypes", {}).get(col) == "numeric"]
        categorical_features = [col for col in feature_names if col not in numeric_features]

    return list(feature_names), list(numeric_features), list(categorical_features)


def list_model_artifacts(models_dir: Path) -> List[ModelArtifact]:
    artifacts: List[ModelArtifact] = []
    for metadata_path in models_dir.glob("*_metadata.json"):
        base_name = metadata_path.stem.replace("_metadata", "")
        model_path = models_dir / f"{base_name}.joblib"
        probabilities_path = models_dir / f"{base_name}_probabilities.csv"

        if not model_path.exists() or not probabilities_path.exists():
            continue

        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        agent_role = metadata.get("agent_role", "unknown")
        artifacts.append(
            ModelArtifact(
                name=base_name,
                agent_role=agent_role,
                metadata_path=metadata_path,
                model_path=model_path,
                probabilities_path=probabilities_path,
                metadata=metadata,
            )
        )
    artifacts.sort(key=lambda art: art.name)
    return artifacts


def extract_class_order(prob_df: pd.DataFrame) -> List[str]:
    prob_cols = [col for col in prob_df.columns if col.startswith("prob_")]
    if not prob_cols:
        raise ValueError("Probability dataframe lacks 'prob_' columns.")
    return [col.replace("prob_", "") for col in prob_cols]


def compute_metrics(prob_df: pd.DataFrame, class_order: Sequence[str]) -> Dict[str, Any]:
    y_true = prob_df["true_label"].astype(str)
    y_pred = prob_df["predicted_label"].astype(str)

    metrics: Dict[str, Any] = {}
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_order,
        zero_division=0,
    )
    per_class: Dict[str, Any] = {}
    for idx, label in enumerate(class_order):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    metrics["per_class"] = per_class
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=class_order,
    ).tolist()
    return metrics


def threshold_check_high_recall(
    metrics: Dict[str, Any],
    focus_class: str,
    min_recall: float,
) -> Tuple[bool, Optional[str]]:
    class_stats = metrics.get("per_class", {}).get(focus_class)
    if class_stats is None:
        return False, f"Missing metrics for class '{focus_class}'."
    recall_value = class_stats.get("recall")
    if recall_value is None:
        return False, f"Recall unavailable for class '{focus_class}'."
    if recall_value < min_recall:
        return False, f"Recall {recall_value:.3f} below threshold {min_recall:.3f} for class '{focus_class}'."
    return True, None


def align_feature_frame(
    dataset: pd.DataFrame,
    feature_names: Sequence[str],
    row_indices: Sequence[Any],
) -> pd.DataFrame:
    missing_features = [col for col in feature_names if col not in dataset.columns]
    if missing_features:
        raise KeyError(f"Dataset missing expected features: {missing_features}")

    subset = dataset.loc[row_indices, feature_names]
    subset = subset.copy()
    return subset


def compute_shap_top_features(
    model: Any,
    feature_subset: pd.DataFrame,
    predicted_labels: Sequence[str],
    class_order: Sequence[str],
    num_features: int,
    max_samples: int,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    if len(feature_subset) == 0:
        return []

    sampled = feature_subset.copy()
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=random_state)

    try:
        background = feature_subset.sample(
            n=min(len(feature_subset), max(num_features * 3, 20)),
            random_state=random_state,
        )
    except ValueError:
        background = feature_subset.copy()

    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(sampled)
    except Exception as exc:  # pragma: no cover - SHAP fallback
        warnings.warn(f"SHAP computation failed: {exc}")
        return []

    explanations: List[Dict[str, Any]] = []
    sampled_indices = sampled.index.tolist()
    label_series = pd.Series(predicted_labels, index=feature_subset.index)
    label_subset = label_series.loc[sampled_indices]

    if isinstance(shap_values, list):
        shap_matrix = {
            class_order[idx]: explanation.values
            for idx, explanation in enumerate(shap_values)
            if idx < len(class_order)
        }
    else:
        shap_matrix = {class_order[0]: shap_values.values}

    for row_pos, row_index in enumerate(sampled_indices):
        predicted_label = str(label_subset.iloc[row_pos])
        shap_for_label = shap_matrix.get(predicted_label)
        if shap_for_label is None:
            # fall back to first available class
            shap_for_label = next(iter(shap_matrix.values()))
        row_shap = shap_for_label[row_pos]
        top_indices = np.argsort(np.abs(row_shap))[::-1][:num_features]

        top_features = []
        for feature_idx in top_indices:
            feature_name = sampled.columns[feature_idx]
            shap_value = float(row_shap[feature_idx])
            feature_value = sampled.iloc[row_pos, feature_idx]
            top_features.append(
                {
                    "feature": feature_name,
                    "value": _to_serializable(feature_value),
                    "shap": shap_value,
                }
            )

        explanations.append(
            {
                "row_index": _to_serializable(row_index),
                "predicted_label": predicted_label,
                "top_features": top_features,
            }
        )

    return explanations


def generate_adversarial_samples(
    dataset: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)

    baseline: Dict[str, Any] = {}
    for col in numeric_columns:
        baseline[col] = float(dataset[col].median())
    for col in categorical_columns:
        baseline[col] = _mode_value(dataset[col])

    quantiles: Dict[str, Dict[str, float]] = {}
    for col in numeric_columns:
        quantiles[col] = {
            "low": float(dataset[col].quantile(0.05)),
            "mid": float(dataset[col].median()),
            "high": float(dataset[col].quantile(0.95)),
        }

    scenarios = scenarios or {}
    def add_if_columns(name: str, updates: Dict[str, Any]) -> None:
        filtered_updates = {k: v for k, v in updates.items() if k in baseline}
        if filtered_updates:
            scenarios[name] = filtered_updates

    add_if_columns(
        "LowPlayHighSpend",
        {
            "PlayTimeHours": quantiles.get("PlayTimeHours", {}).get("low", 0.0),
            "SessionsPerWeek": quantiles.get("SessionsPerWeek", {}).get("low", 0.0),
            "InGamePurchases": quantiles.get("InGamePurchases", {}).get("high", 0.0),
        },
    )
    add_if_columns(
        "HighPlayLowAchievements",
        {
            "PlayTimeHours": quantiles.get("PlayTimeHours", {}).get("high", 0.0),
            "AchievementsUnlocked": quantiles.get("AchievementsUnlocked", {}).get("low", 0.0),
        },
    )
    add_if_columns(
        "HighLevelNoActivity",
        {
            "PlayerLevel": quantiles.get("PlayerLevel", {}).get("high", 0.0),
            "PlayTimeHours": quantiles.get("PlayTimeHours", {}).get("low", 0.0),
            "SessionsPerWeek": quantiles.get("SessionsPerWeek", {}).get("low", 0.0),
        },
    )
    add_if_columns(
        "LongSessionsNoPurchases",
        {
            "AvgSessionDurationMinutes": quantiles.get("AvgSessionDurationMinutes", {}).get("high", 0.0),
            "InGamePurchases": quantiles.get("InGamePurchases", {}).get("low", 0.0),
        },
    )

    rows: List[Dict[str, Any]] = []
    for scenario_name, updates in scenarios.items():
        row = baseline.copy()
        row.update(updates)
        row["_scenario"] = scenario_name
        rows.append(row)

    adversarial_df = pd.DataFrame(rows)
    adversarial_df.set_index("_scenario", inplace=True)
    return adversarial_df


def evaluate_model(
    artifact: ModelArtifact,
    dataset: pd.DataFrame,
    feature_names: Sequence[str],
    class_order: Sequence[str],
    args: argparse.Namespace,
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = joblib.load(artifact.model_path)
    prob_df = pd.read_csv(artifact.probabilities_path)
    metrics = compute_metrics(prob_df, class_order)

    threshold_passed, threshold_reason = threshold_check_high_recall(
        metrics=metrics,
        focus_class=args.focus_class,
        min_recall=args.min_high_recall,
    )

    deltas: Dict[str, Any] = {}
    if baseline_metrics is not None:
        deltas["balanced_accuracy_delta"] = float(
            metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        )
        baseline_recall = baseline_metrics.get("per_class", {}).get(args.focus_class, {}).get("recall")
        current_recall = metrics.get("per_class", {}).get(args.focus_class, {}).get("recall")
        if baseline_recall is not None and current_recall is not None:
            deltas["recall_delta"] = float(current_recall - baseline_recall)

    shap_explanations: List[Dict[str, Any]] = []
    shap_path: Optional[Path] = None
    if args.enable_shap:
        try:
            feature_subset = align_feature_frame(
                dataset=dataset,
                feature_names=feature_names,
                row_indices=prob_df["row_index"].tolist(),
            )
            shap_explanations = compute_shap_top_features(
                model=model,
                feature_subset=feature_subset,
                predicted_labels=prob_df["predicted_label"].tolist(),
                class_order=class_order,
                num_features=args.num_top_features,
                max_samples=args.max_shap_samples,
            )
            if shap_explanations:
                shap_path = args.output_dir / f"{artifact.name}_shap_top_features.json"
                with open(shap_path, "w", encoding="utf-8") as fh:
                    json.dump(shap_explanations, fh, indent=2)
        except Exception as exc:  # pragma: no cover - defensive logging
            warnings.warn(f"Failed to compute SHAP explanations for {artifact.name}: {exc}")

    adversarial_summary: List[Dict[str, Any]] = []
    adversarial_path: Optional[Path] = None
    if args.enable_adversarial:
        try:
            catalog = artifact.metadata.get("metadata", {})
            numeric_cols = catalog.get("numeric_features") or []
            categorical_cols = catalog.get("categorical_features") or []
            adversarial_df = generate_adversarial_samples(
                dataset=dataset,
                numeric_columns=numeric_cols or [col for col in feature_names if _is_numeric(dataset[col])],
                categorical_columns=categorical_cols or [col for col in feature_names if not _is_numeric(dataset[col])],
            )
            if not adversarial_df.empty:
                adv_predictions = model.predict(adversarial_df)
                adv_probabilities = model.predict_proba(adversarial_df)
                for idx, scenario in enumerate(adversarial_df.index):
                    proba_row = adv_probabilities[idx]
                    scenario_probs = {
                        class_label: float(proba_row[class_idx])
                        for class_idx, class_label in enumerate(class_order)
                    }
                    adversarial_summary.append(
                        {
                            "scenario": scenario,
                            "predicted_label": _to_serializable(adv_predictions[idx]),
                            "max_probability": float(np.max(proba_row)),
                            "probabilities": scenario_probs,
                        }
                    )
                adversarial_path = args.output_dir / f"{artifact.name}_adversarial_checks.json"
                with open(adversarial_path, "w", encoding="utf-8") as fh:
                    json.dump(adversarial_summary, fh, indent=2)
        except Exception as exc:  # pragma: no cover - defensive logging
            warnings.warn(f"Failed adversarial evaluation for {artifact.name}: {exc}")

    return {
        "name": artifact.name,
        "agent_role": artifact.agent_role,
        "metrics": metrics,
        "threshold_passed": threshold_passed,
        "threshold_reason": threshold_reason,
        "deltas_vs_baseline": deltas,
        "shap_top_features": str(shap_path.resolve()) if shap_path else None,
        "adversarial_results": str(adversarial_path.resolve()) if adversarial_path else None,
        "probabilities_path": str(artifact.probabilities_path.resolve()),
    }


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _mode_value(series: pd.Series) -> Any:
    if series.empty:
        return None
    try:
        return series.mode(dropna=True).iloc[0]
    except Exception:
        return series.iloc[0]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def orchestrate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = load_dataset(args.dataset)
    catalog = load_feature_catalog(args.feature_catalog)
    feature_names, numeric_cols, categorical_cols = resolve_feature_names(
        catalog=catalog,
        fallback_columns=dataset.columns,
        target=args.target,
    )
    if args.target in dataset.columns:
        dataset = dataset.drop(columns=[args.target])

    artifacts = list_model_artifacts(args.models_dir)
    if not artifacts:
        raise RuntimeError(f"No model artifacts found in {args.models_dir}.")

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_artifact = next(
        (art for art in artifacts if art.agent_role == "baseline" or art.name.startswith("baseline")), None
    )
    if baseline_artifact is None:
        raise RuntimeError("Baseline model artifact not found.")

    baseline_prob_df = pd.read_csv(baseline_artifact.probabilities_path)
    class_order = extract_class_order(baseline_prob_df)

    summary_models: Dict[str, Any] = {}

    baseline_result = evaluate_model(
        artifact=baseline_artifact,
        dataset=dataset,
        feature_names=feature_names,
        class_order=class_order,
        args=args,
        baseline_metrics=None,
    )
    summary_models[baseline_result["name"]] = baseline_result

    baseline_metrics = baseline_result["metrics"]

    for artifact in artifacts:
        if artifact.name == baseline_artifact.name:
            continue
        result = evaluate_model(
            artifact=artifact,
            dataset=dataset,
            feature_names=feature_names,
            class_order=class_order,
            args=args,
            baseline_metrics=baseline_metrics,
        )
        summary_models[result["name"]] = result

    summary = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "focus_class": args.focus_class,
        "min_high_recall": args.min_high_recall,
        "dataset": str(args.dataset.resolve()),
        "feature_catalog": str(args.feature_catalog.resolve()),
        "models_dir": str(args.models_dir.resolve()),
        "baseline_model": baseline_artifact.name,
        "models": summary_models,
    }

    summary_path = args.output_dir / "guardrail_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluator & Guardrail Agent (Phase 4)")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts") / "predict_online_gaming_enhanced.csv")
    parser.add_argument("--feature-catalog", type=Path, default=Path("artifacts") / "feature_catalog.json")
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts") / "modeling" / "models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--focus-class", type=str, default="High")
    parser.add_argument("--min-high-recall", type=float, default=DEFAULT_MIN_HIGH_RECALL)
    parser.add_argument("--num-top-features", type=int, default=3)
    parser.add_argument("--max-shap-samples", type=int, default=DEFAULT_MAX_SHAP_SAMPLES)
    parser.add_argument("--enable-shap", action="store_true")
    parser.add_argument("--enable-adversarial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)

    summary = orchestrate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()