import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from orchestrator_agent import AgentHandle, OrchestratorAgent
from explainability_agent import ExplainabilityAgent
from recommendation_agent import RecommendationAgent

try:
    from ingestion_agent import (
        canonicalize_dataframe,
        create_ephemeral_context,
        ensure_expectation_suite,
        load_csv,
        run_validation,
        save_artifacts as save_ingestion_artifacts,
    )
except Exception as exc:  # pragma: no cover - dependency guidance
    raise RuntimeError(
        "Failed to import ingestion_agent. Ensure Great Expectations is installed (pip install great_expectations)."
    ) from exc

try:
    from feature_agent import (
        analyze_target_variable,
        assess_feature_quality,
        compute_descriptive_stats,
        engineer_features,
        generate_feature_catalog,
        load_clean_data,
        save_artifacts as save_feature_artifacts,
    )
except Exception as exc:  # pragma: no cover - dependency guidance
    raise RuntimeError("Failed to import feature_agent; verify Phase 2 module is available.") from exc

from guardrail_agent import orchestrate as guardrail_orchestrate
from model_selection_agent import orchestrate_agents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)

RAW_DATA_PATH = Path("predict_online_gaming.csv")
ARTIFACT_ROOT = Path("artifacts")
SUITE_NAME = "gaming_suite"
BACKGROUND_SIZE = 100
GLOBAL_SAMPLE_SIZE = 256
LOCAL_SAMPLE_SIZE = 100
TARGET_COLUMN = "EngagementLevel"
MONITORING_DIR = ARTIFACT_ROOT / "monitoring"
MODEL_REGISTRY_DIR = ARTIFACT_ROOT / "model_registry"
GOVERNANCE_DIR = ARTIFACT_ROOT / "governance"
APPROVAL_ARCHIVE_DIR = GOVERNANCE_DIR / "archive"
DRIFT_THRESHOLD = 0.2
ACCURACY_TOLERANCE = 0.05
MONITORING_HISTORY_FILE = MONITORING_DIR / "metrics_history.jsonl"
BASELINE_FILE = MONITORING_DIR / "baseline_class_distribution.json"


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _serialize_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


def _compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    labels = sorted(set(y_true.dropna().unique()).union(set(y_pred.dropna().unique())))
    if not labels:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def _class_distribution(series: pd.Series) -> Dict[str, float]:
    return {str(label): float(freq) for label, freq in series.value_counts(normalize=True).to_dict().items()}


def _population_stability_index(
    baseline: Dict[str, float],
    current: Dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    psi = 0.0
    classes = set(baseline).union(current)
    for cls in classes:
        expected = max(baseline.get(cls, epsilon), epsilon)
        actual = max(current.get(cls, epsilon), epsilon)
        psi += (actual - expected) * math.log(actual / expected)
    return float(psi)


def _append_metrics_record(record: Dict[str, Any]) -> None:
    MONITORING_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MONITORING_HISTORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _guardrail_passed(summary: Dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    for key in ("passed", "success", "status", "overall_pass", "ok"):
        value = summary.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"pass", "passed", "success", "succeeded", "ok", "true"}
    return bool(summary)


def _resolve_model_artifact_paths(model_artifacts: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    probability_ref = model_artifacts.get("probabilities") or model_artifacts.get("probability")
    if probability_ref is None:
        raise KeyError("Model artifacts missing 'probabilities' entry.")
    return (
        Path(model_artifacts["model"]),
        Path(model_artifacts["metadata"]),
        Path(probability_ref),
    )


def _load_approval_decision(model_name: str) -> Optional[Tuple[Dict[str, Any], Path]]:
    if not GOVERNANCE_DIR.exists():
        return None
    candidates = [
        GOVERNANCE_DIR / f"{model_name}_approval.json",
        GOVERNANCE_DIR / "approval.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _read_json(candidate)
        if payload and payload.get("approved") and payload.get("model_name", model_name) == model_name:
            return payload, candidate
    return None


def _archive_approval(source: Path, model_name: str) -> None:
    if not source.exists():
        return
    _ensure_dirs(APPROVAL_ARCHIVE_DIR)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    target = APPROVAL_ARCHIVE_DIR / f"{model_name}_approval_{timestamp}.json"
    try:
        source.rename(target)
    except OSError:
        payload = _read_json(source) or {}
        _write_json(target, payload)
        if source.exists():
            source.unlink()


def _promote_model_if_approved(
    model_name: str,
    model_artifacts: Dict[str, Any],
    guardrail_summary: Dict[str, Any],
) -> Dict[str, Any]:
    if not _guardrail_passed(guardrail_summary):
        return {"promoted": False, "reason": "guardrail_not_passed"}
    model_path, metadata_path, probability_path = _resolve_model_artifact_paths(model_artifacts)
    decision = _load_approval_decision(model_name)
    if decision is None:
        template_path = _emit_approval_template(
            model_name=model_name,
            model_path=model_path,
            metadata_path=metadata_path,
            probability_path=probability_path,
            guardrail_summary=guardrail_summary,
        )
        return {
            "promoted": False,
            "reason": "approval_missing",
            "approval_template": str(template_path),
        }
    approval_payload, approval_path = decision
    record = {
        "model_name": model_name,
        "model_path": _serialize_path(model_path),
        "metadata_path": _serialize_path(metadata_path),
        "probabilities_path": _serialize_path(probability_path),
        "promoted_at": datetime.utcnow().isoformat(),
        "approved_by": approval_payload.get("approver"),
        "approval_timestamp": approval_payload.get("timestamp"),
        "guardrail_summary": guardrail_summary,
    }
    _ensure_dirs(MODEL_REGISTRY_DIR)
    pointer_path = MODEL_REGISTRY_DIR / "current_model.json"
    _write_json(pointer_path, record)
    _archive_approval(approval_path, model_name)
    return {"promoted": True, "record": record}


def _emit_approval_template(
    model_name: str,
    model_path: Path,
    metadata_path: Path,
    probability_path: Path,
    guardrail_summary: Dict[str, Any],
) -> Path:
    _ensure_dirs(GOVERNANCE_DIR)
    template_path = GOVERNANCE_DIR / f"{model_name}_approval_template.json"
    payload = {
        "model_name": model_name,
        "approved": False,
        "approver": "",
        "timestamp": "",
        "notes": "Review guardrail_summary and set approved=true before rerunning retraining/promotion.",
        "model_artifacts": {
            "model": _serialize_path(model_path),
            "metadata": _serialize_path(metadata_path),
            "probabilities": _serialize_path(probability_path),
        },
        "guardrail_summary": guardrail_summary,
    }
    _write_json(template_path, payload)
    return template_path


def ensure_data_available() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH.resolve()}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def run_ingestion_stage(_ctx):
    logger.info("Phase 1: Ingestion & GE validation")
    ensure_data_available()

    raw_df = load_csv(str(RAW_DATA_PATH))
    cleaned_df = canonicalize_dataframe(raw_df)
    if cleaned_df.empty:
        raise ValueError("Cleaned dataframe is empty after ingestion.")

    context = create_ephemeral_context(str(ARTIFACT_ROOT))
    sample_size = min(len(cleaned_df), 100)
    sample_df = cleaned_df.sample(n=sample_size, random_state=1) if sample_size > 0 else cleaned_df

    suite = ensure_expectation_suite(context, SUITE_NAME, sample_df)
    validation = run_validation(context, cleaned_df, SUITE_NAME)
    clean_csv, suite_path, schema_path = save_ingestion_artifacts(
        str(ARTIFACT_ROOT),
        cleaned_df,
        suite,
        validation,
    )

    ingestion_payload = {
        "clean_csv": str(clean_csv),
        "schema": str(schema_path),
        "expectations": str(suite_path) if suite_path else None,
        "rows": len(cleaned_df),
        "validation_success": validation.get("success"),
    }
    return {"ingestion": ingestion_payload}


def run_validation_stage(ctx):
    logger.info("Phase 1b: Schema validation checkpoint")
    ingestion = ctx["ingestion"]
    clean_csv = Path(ingestion["clean_csv"])
    schema_path = Path(ingestion["schema"])
    stats = {
        "clean_csv_exists": clean_csv.exists(),
        "schema_exists": schema_path.exists(),
        "rows": ingestion["rows"],
        "validation_success": ingestion["validation_success"],
    }
    return {"validation": stats}


def run_feature_stage(ctx):
    logger.info("Phase 2: Feature engineering")
    ingestion = ctx["ingestion"]
    df, _ = load_clean_data(ingestion["clean_csv"], ingestion["schema"])
    descriptive_stats = compute_descriptive_stats(df)
    target_analysis = analyze_target_variable(df, "EngagementLevel")
    df_enhanced, engineered_features = engineer_features(df)
    quality_report = assess_feature_quality(df_enhanced, "EngagementLevel")
    feature_catalog = generate_feature_catalog(
        descriptive_stats,
        engineered_features,
        quality_report,
        target_analysis,
    )
    eda_report = {
        "descriptive_statistics": descriptive_stats,
        "target_analysis": target_analysis,
        "feature_quality": quality_report,
    }
    artifacts = save_feature_artifacts(
        str(ARTIFACT_ROOT),
        df_enhanced,
        feature_catalog,
        eda_report,
    )
    summary = {
        "input_rows": len(df),
        "output_rows": len(df_enhanced),
        "engineered_feature_count": len(engineered_features),
        "top_features": quality_report["recommendations"]["top_features"][:5],
    }
    return {"features": {"artifacts": artifacts, "summary": summary}}


def run_model_stage(ctx):
    logger.info("Phase 3: Model selection & training")
    feature_artifacts = ctx["features"]["artifacts"]
    args = SimpleNamespace(
        input=str(feature_artifacts["enhanced_csv"]),
        feature_catalog=str(feature_artifacts["feature_catalog"]),
        output_dir=str(ARTIFACT_ROOT / "modeling"),
        target="EngagementLevel",
        agents=["baseline", "trainer"],
        test_size=0.2,
        random_state=42,
        optuna_trials=10,
    )
    message = orchestrate_agents(args)
    return {"modeling": message}


def run_guardrail_stage(ctx):
    logger.info("Phase 4: Guardrail evaluation")
    feature_artifacts = ctx["features"]["artifacts"]
    best_name = ctx["modeling"]["best_model"]["name"]
    model_info = ctx["modeling"]["artifacts"]["models"][best_name]["artifacts"]
    models_dir = Path(model_info["model"]).parent
    args = SimpleNamespace(
        dataset=Path(feature_artifacts["enhanced_csv"]),
        feature_catalog=Path(feature_artifacts["feature_catalog"]),
        models_dir=models_dir,
        output_dir=Path(ARTIFACT_ROOT) / "guardrail",
        target="EngagementLevel",
        focus_class="High",
        min_high_recall=0.6,
        num_top_features=3,
        max_shap_samples=10,
        enable_shap=False,
        enable_adversarial=False,
    )
    summary = guardrail_orchestrate(args)
    return {"evaluation": summary}


def _load_model_artifacts(modeling_ctx: Dict[str, Any]) -> Tuple[str, Path, Path, Path]:
    best_name = modeling_ctx["best_model"]["name"]
    model_info = modeling_ctx["artifacts"]["models"][best_name]["artifacts"]
    model_path, metadata_path, probability_path = _resolve_model_artifact_paths(model_info)
    return best_name, model_path, metadata_path, probability_path


def _load_local_explanations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Local explanation file not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if not df.empty and "entity_id" in df.columns:
        df = df.set_index("entity_id")
    return df


def run_monitoring_stage(ctx):
    logger.info("Phase 4b: Monitoring & drift detection")
    _ensure_dirs(MONITORING_DIR)
    feature_artifacts = ctx["features"]["artifacts"]
    modeling_ctx = ctx["modeling"]
    timestamp = datetime.utcnow().isoformat()

    best_name, _, _, prob_path = _load_model_artifacts(modeling_ctx)
    feature_df = pd.read_csv(feature_artifacts["enhanced_csv"])
    if "PlayerID" in feature_df.columns:
        feature_df = feature_df.set_index("PlayerID")

    prob_df = pd.read_csv(prob_path)
    if "PlayerID" in prob_df.columns:
        prob_df = prob_df.set_index("PlayerID")
    if "predicted_label" not in prob_df.columns:
        raise KeyError("Probability artifact missing 'predicted_label' column.")

    aligned = (
        feature_df.join(prob_df["predicted_label"], how="inner")
        .dropna(subset=[TARGET_COLUMN, "predicted_label"])
    )
    if aligned.empty:
        record = {
            "timestamp": timestamp,
            "model": best_name,
            "records": 0,
            "alert": True,
            "alert_reasons": ["no_records_available_for_monitoring"],
            "metrics_file": str(MONITORING_HISTORY_FILE),
            "baseline_file": str(BASELINE_FILE),
        }
        _append_metrics_record(record)
        return {"monitoring": record}

    metrics = _compute_classification_metrics(aligned[TARGET_COLUMN], aligned["predicted_label"])
    distribution = _class_distribution(aligned[TARGET_COLUMN])
    baseline = _read_json(BASELINE_FILE)
    baseline_created = False
    if baseline is None:
        baseline_created = True
        baseline = {
            "created_at": timestamp,
            "model_name": best_name,
            "accuracy": metrics["accuracy"],
            "class_distribution": distribution,
        }
        _write_json(BASELINE_FILE, baseline)

    psi = _population_stability_index(baseline["class_distribution"], distribution)
    alert_reasons: List[str] = []
    accuracy_drop = baseline["accuracy"] - metrics["accuracy"]
    if psi > DRIFT_THRESHOLD:
        alert_reasons.append(f"psi>{DRIFT_THRESHOLD}")
    if accuracy_drop > ACCURACY_TOLERANCE:
        alert_reasons.append(f"accuracy_drop>{ACCURACY_TOLERANCE}")

    record = {
        "timestamp": timestamp,
        "model": best_name,
        "records": int(len(aligned)),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "psi": psi,
        "accuracy_drop": accuracy_drop,
        "alert": bool(alert_reasons),
        "alert_reasons": alert_reasons,
        "class_distribution": distribution,
        "metrics_file": str(MONITORING_HISTORY_FILE),
        "baseline_file": str(BASELINE_FILE),
        "baseline_created": baseline_created,
        "baseline_exists": True,
    }
    _append_metrics_record(record)
    return {"monitoring": record}


def run_explainability_stage(ctx):
    logger.info("Phase 5: Explainability")
    feature_artifacts = ctx["features"]["artifacts"]
    modeling_ctx = ctx["modeling"]

    feature_df = pd.read_csv(feature_artifacts["enhanced_csv"])
    if "PlayerID" in feature_df.columns:
        feature_df = feature_df.set_index("PlayerID")

    best_name, model_path, metadata_path, prob_path = _load_model_artifacts(modeling_ctx)
    metadata = _read_json(metadata_path) or {}
    feature_names: List[str] = (
        metadata.get("feature_names")
        or metadata.get("features_used")
        or [col for col in feature_df.columns if col != TARGET_COLUMN]
    )
    class_labels: List[str] = metadata.get("class_labels") or []

    prob_df = pd.read_csv(prob_path)
    if "PlayerID" in prob_df.columns:
        prob_df = prob_df.set_index("PlayerID")
    elif "row_index" in prob_df.columns:
        prob_df = prob_df.set_index("row_index")

    if "predicted_label" not in prob_df.columns:
        raise KeyError("Probability artifact missing 'predicted_label' column.")

    predictions = prob_df["predicted_label"]
    if not class_labels:
        class_labels = sorted(predictions.dropna().unique().tolist())

    rename_map = {
        f"prob_{label}": label
        for label in class_labels
        if f"prob_{label}" in prob_df.columns
    }
    aligned_prob = prob_df.rename(columns=rename_map)

    common_index = feature_df.index.intersection(predictions.index)
    if common_index.empty:
        raise ValueError("Could not align feature and prediction indices for explainability.")

    feature_df = feature_df.loc[common_index, feature_names]
    predictions = predictions.loc[common_index]
    aligned_prob = aligned_prob.loc[common_index]

    agent = ExplainabilityAgent(
        artifact_dir=str(ARTIFACT_ROOT / "explainability" / best_name),
        llm_enabled=False,
    )
    background = (
        feature_df.sample(BACKGROUND_SIZE, random_state=42)
        if len(feature_df) > BACKGROUND_SIZE
        else feature_df.copy()
    )
    model = joblib.load(model_path)
    agent.register_model(
        model=model,
        feature_names=feature_names,
        class_labels=class_labels,
        background_frame=background,
        background_sample_size=len(background),
    )

    agent.compute_global_importance(
        feature_df,
        sample_size=min(GLOBAL_SAMPLE_SIZE, len(feature_df)),
    )

    subset_index = predictions.index
    if len(subset_index) > LOCAL_SAMPLE_SIZE:
        subset_index = predictions.sample(LOCAL_SAMPLE_SIZE, random_state=42).index

    local_batch_id = best_name
    local_df = agent.explain_batch(
        feature_frame=feature_df.loc[subset_index],
        predictions=predictions.loc[subset_index],
        probability_frame=aligned_prob.loc[subset_index],
        batch_id=local_batch_id,
    )
    guardrail_summary = agent.build_guardrail_summary(local_df)

    payload = {
        "model_id": best_name,
        "local_batch_id": local_batch_id,
        "records": int(len(local_df)),
        "global_importance_path": str(agent.artifact_dir / "global_importance.json"),
        "local_explanations_path": str(
            agent.artifact_dir / f"local_explanations_{local_batch_id}.jsonl"
        ),
        "guardrail_summary_path": str(agent.artifact_dir / "guardrail_summary.json"),
        "guardrail_summary": guardrail_summary,
    }
    return {"explainability": payload}


def run_recommendation_stage(ctx):
    logger.info("Phase 6: Recommendation / Action agent")
    explainability = ctx["explainability"]
    explanations_path = Path(explainability["local_explanations_path"])
    explanation_df = _load_local_explanations(explanations_path)
    batch_id = explainability.get("local_batch_id") or explainability.get("model_id")
    agent = RecommendationAgent(artifact_dir=str(ARTIFACT_ROOT / "recommendations"))
    rec_df = agent.generate_recommendations(explanation_df, batch_id=batch_id)
    stats = agent.summary(rec_df)
    rec_path = agent.artifact_dir / f"recommendations_{batch_id}.jsonl"
    payload = {
        "file": str(rec_path),
        "summary": stats,
        "total_recommendations": int(len(rec_df)),
        "batch_id": batch_id,
    }
    return {"recommendation": payload}


def run_retraining_stage(ctx):
    logger.info("Phase 7: Conditional retraining & promotion")
    monitoring_ctx = ctx.get("monitoring") or {}
    if not monitoring_ctx.get("alert"):
        logger.info("Retraining skipped (no monitoring alert).")
        return {"retraining": {"skipped": True, "reason": "no_alert_triggered"}}

    feature_artifacts = ctx["features"]["artifacts"]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    retrain_dir = ARTIFACT_ROOT / "retraining" / timestamp
    _ensure_dirs(retrain_dir)

    retrain_args = SimpleNamespace(
        input=str(feature_artifacts["enhanced_csv"]),
        feature_catalog=str(feature_artifacts["feature_catalog"]),
        output_dir=str(retrain_dir),
        target=TARGET_COLUMN,
        agents=["trainer"],
        test_size=0.2,
        random_state=42,
        optuna_trials=5,
    )
    new_model_ctx = orchestrate_agents(retrain_args)
    best_name = new_model_ctx["best_model"]["name"]
    model_info = new_model_ctx["artifacts"]["models"][best_name]["artifacts"]

    guardrail_args = SimpleNamespace(
        dataset=Path(feature_artifacts["enhanced_csv"]),
        feature_catalog=Path(feature_artifacts["feature_catalog"]),
        models_dir=Path(model_info["model"]).parent,
        output_dir=retrain_dir / "guardrail",
        target=TARGET_COLUMN,
        focus_class="High",
        min_high_recall=0.6,
        num_top_features=3,
        max_shap_samples=10,
        enable_shap=False,
        enable_adversarial=False,
    )
    guardrail_summary = guardrail_orchestrate(guardrail_args)
    promotion = _promote_model_if_approved(best_name, model_info, guardrail_summary)

    payload = {
        "best_model": new_model_ctx["best_model"],
        "artifacts": new_model_ctx["artifacts"],
        "guardrail_summary": guardrail_summary,
        "promotion": promotion,
        "output_dir": str(retrain_dir),
        "timestamp": timestamp,
    }
    return {"retraining": payload}


def run_full_pipeline() -> None:
    orch = build_orchestrator()
    context = orch.init_context()
    ingestion_result = run_ingestion_stage(context)
    validation_result = run_validation_stage(context)
    feature_result = run_feature_stage(context)
    modeling_result = run_model_stage(context)
    guardrail_result = run_guardrail_stage(context)
    monitoring_result = run_monitoring_stage(context)
    retraining_result = run_retraining_stage(context)

    logger.info("Pipeline run completed.")
    logger.info(f"Ingestion result: {ingestion_result}")
    logger.info(f"Validation result: {validation_result}")
    logger.info(f"Feature engineering result: {feature_result}")
    logger.info(f"Modeling result: {modeling_result}")
    logger.info(f"Guardrail result: {guardrail_result}")
    logger.info(f"Monitoring result: {monitoring_result}")
    logger.info(f"Retraining result: {retraining_result}")


def build_orchestrator() -> OrchestratorAgent:
    orch = OrchestratorAgent()
    orch.register_agent(AgentHandle("ingestion", "ingestion", run_ingestion_stage))
    orch.register_agent(
        AgentHandle(
            name="validation",
            stage="validation",
            runner=run_validation_stage,
            depends_on=["ingestion"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="feature_engineering",
            stage="feature_engineering",
            runner=run_feature_stage,
            depends_on=["ingestion"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="model_training",
            stage="model_training",
            runner=run_model_stage,
            depends_on=["features"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="evaluation",
            stage="evaluation",
            runner=run_guardrail_stage,
            depends_on=["modeling", "features"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="monitoring",
            stage="monitoring",
            runner=run_monitoring_stage,
            depends_on=["modeling", "features", "evaluation"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="explainability",
            stage="explainability",
            runner=run_explainability_stage,
            depends_on=["modeling", "features"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="recommendation",
            stage="recommendation",
            runner=run_recommendation_stage,
            depends_on=["explainability"],
        )
    )
    orch.register_agent(
        AgentHandle(
            name="retraining",
            stage="retraining",
            runner=run_retraining_stage,
            depends_on=["monitoring", "features"],
        )
    )
    return orch


def main() -> None:
    orch = build_orchestrator()
    run = orch.run(label="phase-1-7")
    print("Run status:", run.status)
    print("Context keys:", list(run.context.keys()))
    print("Events:", len(run.events))


if __name__ == "__main__":
    sys.exit(main())
