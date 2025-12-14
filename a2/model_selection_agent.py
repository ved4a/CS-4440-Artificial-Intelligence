#!/usr/bin/env python3
"""
Modeling Agents (ensemble)

This agent orchestrates three collaborating sub-agents to evaluate modeling
strategies for the online gaming engagement dataset:

- Baseline Agent: trains fast reference models (Logistic Regression, Decision Tree).
- Trainer Agent: fits stronger ensembles (Random Forest, XGBoost, LightGBM) and an
  optional neural classifier when libraries are available.
- HyperTune Agent: performs automated hyperparameter optimisation with Optuna or
  GridSearchCV when Optuna is not installed.

Each trained model emits calibrated probability vectors for the three
EngagementLevel classes, along with serialized artifacts and metadata that
captures feature usage, hyperparameters, and evaluation metrics.
"""

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    import optuna  # type: ignore

    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb  # type: ignore

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_OPTUNA_TRIALS = 40


@dataclass
class ModelResult:
    """Container for model evaluation artifacts."""

    name: str
    agent_role: str
    estimator: Any
    metrics: Dict[str, Any]
    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray
    hyperparameters: Dict[str, Any]
    calibrated: bool
    metadata: Dict[str, Any]


def load_dataset(csv_path: str, feature_catalog_path: Optional[str], target: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    catalog = {}
    recommended = None
    if feature_catalog_path and os.path.exists(feature_catalog_path):
        with open(feature_catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        recommended = catalog.get("modeling_recommendations", {}).get("recommended_features", [])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    return {"data": df, "catalog": catalog, "recommended_features": recommended}


def choose_features(df: pd.DataFrame, target: str, recommended: Optional[Sequence[str]]) -> Dict[str, Any]:
    exclude = {target, "PlayerID"}
    if recommended:
        feature_list = [col for col in recommended if col in df.columns]
        if not feature_list:
            warnings.warn("Recommended features missing in dataframe; falling back to full feature set")
    if not recommended or not feature_list:
        feature_list = [col for col in df.columns if col not in exclude]

    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for col in feature_list:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) <= 12:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)

    feature_frame = df[feature_list].copy()
    for col in numeric_features:
        if feature_frame[col].isna().any():
            feature_frame[col] = feature_frame[col].fillna(feature_frame[col].median())
    for col in categorical_features:
        feature_frame[col] = feature_frame[col].fillna("__missing__").astype(str)

    return {
        "X": feature_frame,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_list,
    }


def build_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
    transformers: List[Any] = []
    if numeric:
        transformers.append(("num", StandardScaler(), list(numeric)))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), list(categorical)))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = list(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba))
    except ValueError:
        metrics["log_loss"] = None
    try:
        class_count = y_proba.shape[1]
        brier_scores = [brier_score_loss((y_true == i).astype(int), y_proba[:, i]) for i in range(class_count)]
        metrics["brier_score"] = float(np.mean(brier_scores))
    except ValueError:
        metrics["brier_score"] = None
    try:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["roc_auc_ovr"] = None

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(label_encoder.classes_)))
    metrics["per_class"] = {
        label_encoder.classes_[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(label_encoder.classes_))
    }

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def train_model(
    name: str,
    agent_role: str,
    estimator: Any,
    calibrate: bool,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
) -> ModelResult:
    start = time.time()
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", estimator),
    ])
    model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3) if calibrate else pipeline
    model.fit(X_train, y_train)
    duration = time.time() - start

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except NotFittedError as exc:  # pragma: no cover
        raise RuntimeError(f"Model {name} failed to produce probabilities: {exc}")

    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_duration_seconds": float(duration),
        "evaluation_split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "features_used": list(feature_names),
        "calibrated": calibrate,
        "agent_role": agent_role,
    }

    hyperparameters = sanitize_params(estimator.get_params())

    return ModelResult(
        name=name,
        agent_role=agent_role,
        estimator=model,
        metrics=metrics,
        predictions=y_pred,
        probabilities=y_proba,
        true_labels=y_test,
        hyperparameters=hyperparameters,
        calibrated=calibrate,
        metadata=metadata,
    )


def run_baseline_agent(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
) -> List[ModelResult]:
    role = "baseline"
    results = []

    lr = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    results.append(
        train_model(
            name="baseline_logistic_regression",
            agent_role=role,
            estimator=lr,
            calibrate=False,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )
    )

    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=3,
        class_weight="balanced",
    )
    results.append(
        train_model(
            name="baseline_decision_tree",
            agent_role=role,
            estimator=dt,
            calibrate=True,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )
    )

    return results


def run_trainer_agent(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
) -> List[ModelResult]:
    role = "trainer"
    results: List[ModelResult] = []

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results.append(
        train_model(
            name="trainer_random_forest",
            agent_role=role,
            estimator=rf,
            calibrate=True,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )
    )

    if XGBOOST_AVAILABLE:
        xgb_estimator = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=350,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            use_label_encoder=False,
        )
        results.append(
            train_model(
                name="trainer_xgboost",
                agent_role=role,
                estimator=xgb_estimator,
                calibrate=True,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_names,
            )
        )

    if LIGHTGBM_AVAILABLE:
        lgb_estimator = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        results.append(
            train_model(
                name="trainer_lightgbm",
                agent_role=role,
                estimator=lgb_estimator,
                calibrate=True,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_names,
            )
        )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=64,
        max_iter=300,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=10,
    )
    results.append(
        train_model(
            name="trainer_mlp_classifier",
            agent_role=role,
            estimator=mlp,
            calibrate=True,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )
    )

    return results


def run_hypertune_agent(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
    optuna_trials: int,
) -> List[ModelResult]:
    role = "hypertune"
    results: List[ModelResult] = []

    if OPTUNA_AVAILABLE:
        def rf_objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 6, 24),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.6, 0.8, 1.0]),
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("classifier", model),
            ])
            scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="balanced_accuracy", n_jobs=-1)
            return float(scores.mean())

        rf_study = optuna.create_study(direction="maximize")
        rf_study.optimize(rf_objective, n_trials=optuna_trials, show_progress_bar=False)
        best_rf = RandomForestClassifier(**rf_study.best_params)
        results.append(
            train_model(
                name="hypertune_random_forest_optuna",
                agent_role=role,
                estimator=best_rf,
                calibrate=True,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_names,
            )
        )
        results[-1].metadata["optuna_best_value"] = float(rf_study.best_value)
        results[-1].metadata["optuna_trials"] = int(optuna_trials)

        if XGBOOST_AVAILABLE:
            def xgb_objective(trial: "optuna.Trial") -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "use_label_encoder": False,
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                }
                model = xgb.XGBClassifier(**params)
                pipeline = Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("classifier", model),
                ])
                scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="balanced_accuracy", n_jobs=-1)
                return float(scores.mean())

            xgb_study = optuna.create_study(direction="maximize")
            xgb_study.optimize(xgb_objective, n_trials=optuna_trials, show_progress_bar=False)
            best_xgb = xgb.XGBClassifier(**xgb_study.best_params)
            results.append(
                train_model(
                    name="hypertune_xgboost_optuna",
                    agent_role=role,
                    estimator=best_xgb,
                    calibrate=True,
                    preprocessor=preprocessor,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    label_encoder=label_encoder,
                    feature_names=feature_names,
                )
            )
            results[-1].metadata["optuna_best_value"] = float(xgb_study.best_value)
            results[-1].metadata["optuna_trials"] = int(optuna_trials)
    else:
        grid = {
            "classifier__n_estimators": [200, 400, 600],
            "classifier__max_depth": [8, 12, 16],
            "classifier__min_samples_split": [2, 6, 12],
        }
        rf_pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        search = GridSearchCV(
            rf_pipeline,
            param_grid=grid,
            cv=3,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_rf = RandomForestClassifier(
            n_estimators=search.best_params["classifier__n_estimators"],
            max_depth=search.best_params["classifier__max_depth"],
            min_samples_split=search.best_params["classifier__min_samples_split"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        results.append(
            train_model(
                name="hypertune_random_forest_grid",
                agent_role=role,
                estimator=tuned_rf,
                calibrate=True,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_names,
            )
        )
        results[-1].metadata["gridsearch_best_score"] = float(search.best_score_)
        results[-1].metadata["gridsearch_params"] = sanitize_params(search.best_params)

    return results


def probabilities_dataframe(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    label_encoder: LabelEncoder,
    sample_index: Sequence[Any],
    player_ids: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    columns = [f"prob_{cls}" for cls in label_encoder.classes_]
    proba_df = pd.DataFrame(probabilities, columns=columns)
    proba_df.insert(0, "predicted_label", label_encoder.inverse_transform(predictions))
    proba_df.insert(0, "true_label", label_encoder.inverse_transform(true_labels))
    proba_df.insert(0, "row_index", list(sample_index))
    if player_ids is not None:
        proba_df.insert(1, "PlayerID", list(player_ids))
    return proba_df


def save_artifacts(
    output_dir: str,
    results: List[ModelResult],
    label_encoder: LabelEncoder,
    feature_names: Sequence[str],
    test_index: Sequence[Any],
    player_ids: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "agent_name": "Modeling Agents (ensemble)",
        "models": {},
        "feature_names": list(feature_names),
        "classes": label_encoder.classes_.tolist(),
    }

    for result in results:
        model_path = models_dir / f"{result.name}.joblib"
        joblib.dump(result.estimator, model_path)

        proba_df = probabilities_dataframe(
            probabilities=result.probabilities,
            true_labels=result.true_labels,
            predictions=result.predictions,
            label_encoder=label_encoder,
            sample_index=test_index,
            player_ids=player_ids,
        )
        proba_path = models_dir / f"{result.name}_probabilities.csv"
        proba_df.to_csv(proba_path, index=False)

        metadata = {
            "model_name": result.name,
            "agent_role": result.agent_role,
            "metrics": result.metrics,
            "hyperparameters": result.hyperparameters,
            "calibrated": result.calibrated,
            "feature_names": list(feature_names),
            "class_labels": label_encoder.classes_.tolist(),
            "artifacts": {
                "model_path": str(model_path.resolve()),
                "probabilities_path": str(proba_path.resolve()),
            },
        }
        metadata.update(result.metadata)
        meta_path = models_dir / f"{result.name}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        summary["models"][result.name] = {
            "agent_role": result.agent_role,
            "balanced_accuracy": result.metrics.get("balanced_accuracy"),
            "macro_f1": result.metrics.get("macro_f1"),
            "calibrated": result.calibrated,
            "artifacts": {
                "model": str(model_path.resolve()),
                "metadata": str(meta_path.resolve()),
                "probabilities": str(proba_path.resolve()),
            },
        }

    comparison_rows = [
        {
            "model": result.name,
            "agent": result.agent_role,
            "balanced_accuracy": result.metrics.get("balanced_accuracy"),
            "macro_f1": result.metrics.get("macro_f1"),
        }
        for result in results
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = models_dir / "model_comparison.csv"
    comparison_df.sort_values(by="balanced_accuracy", ascending=False, inplace=True)
    comparison_df.to_csv(comparison_path, index=False)
    summary["comparison_csv"] = str(comparison_path.resolve())

    summary_path = Path(output_dir) / "modeling_agents_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": str(summary_path.resolve()),
        "comparison": str(comparison_path.resolve()),
        "models": summary["models"],
    }


def orchestrate_agents(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_info = load_dataset(args.input, args.feature_catalog, args.target)
    df: pd.DataFrame = dataset_info["data"]
    feature_info = choose_features(df, args.target, dataset_info["recommended_features"])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[args.target])
    if len(label_encoder.classes_) != 3:
        warnings.warn(
            "Target column does not have exactly three classes; probabilities will align with available classes.",
            RuntimeWarning,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_info["X"],
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    test_index = X_test.index.tolist()
    player_ids = None
    if "PlayerID" in df.columns:
        player_ids = df.loc[test_index, "PlayerID"].tolist()

    preprocessor = build_preprocessor(feature_info["numeric_features"], feature_info["categorical_features"])

    agents_requested = set(args.agents)
    results: List[ModelResult] = []

    if "baseline" in agents_requested:
        results.extend(
            run_baseline_agent(
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_info["feature_names"],
            )
        )

    if "trainer" in agents_requested:
        results.extend(
            run_trainer_agent(
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_info["feature_names"],
            )
        )

    if "tune" in agents_requested:
        results.extend(
            run_hypertune_agent(
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                feature_names=feature_info["feature_names"],
                optuna_trials=args.optuna_trials,
            )
        )

    if not results:
        raise RuntimeError("No agents executed; provide at least one agent in --agents")

    artifacts = save_artifacts(
        output_dir=args.output_dir,
        results=results,
        label_encoder=label_encoder,
        feature_names=feature_info["feature_names"],
        test_index=test_index,
        player_ids=player_ids,
    )

    best_result = max(results, key=lambda r: r.metrics.get("balanced_accuracy", 0.0))

    orchestrator_message = {
        "phase": "model_training",
        "status": "success",
        "best_model": {
            "name": best_result.name,
            "agent_role": best_result.agent_role,
            "metrics": best_result.metrics,
        },
        "agents_executed": sorted(list(agents_requested)),
        "metrics_ranked": {
            result.name: result.metrics for result in sorted(
                results,
                key=lambda r: r.metrics.get("balanced_accuracy", 0.0),
                reverse=True,
            )
        },
        "artifacts": artifacts,
        "split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "test_index": test_index,
        },
        "classes": label_encoder.classes_.tolist(),
    }

    return orchestrator_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modeling Agents (ensemble)")
    parser.add_argument("--input", required=True, help="Path to enhanced CSV dataset")
    parser.add_argument("--feature-catalog", help="Optional path to feature catalog JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to persist model artifacts")
    parser.add_argument("--target", default="EngagementLevel", help="Target column name")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["baseline", "trainer", "tune"],
        choices=["baseline", "trainer", "tune"],
        help="Subset of agents to execute",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Hold-out set size proportion")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for splitting")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS, help="Number of Optuna trials if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator_message = orchestrate_agents(args)

    print("\n" + "=" * 60)
    print("MODEL SELECTION SUMMARY")
    print("=" * 60)
    print(json.dumps(orchestrator_message, indent=2))


if __name__ == "__main__":
    main()
