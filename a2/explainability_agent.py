"""
Explainability agent for the Predict Online Gaming Behavior pipeline.

This module provides Phase 5 (Explainability Agent) capabilities:
    • Compute global and local SHAP explanations aligned with the modeling artifacts.
    • Persist explanation payloads for guardrails, reporting, and downstream agents.
    • Generate concise natural-language rationales using an optional free local LLM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
except ImportError:  # transformers is optional (only needed when LLM justifications are enabled)
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    hf_pipeline = None  # type: ignore[assignment]


@dataclass
class ExplanationResult:
    """Container for a single prediction explanation artifact."""
    entity_id: Any
    predicted_label: str
    probability: Optional[float]
    shap_values: Dict[str, float]
    top_features: List[Dict[str, Any]]
    justification: str
    feature_snapshot: Dict[str, Any]

    def to_serializable(self) -> Dict[str, Any]:
        return asdict(self)


class LLMJustifier:
    """Thin wrapper around a free Hugging Face causal LLM for explanation templating."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ) -> None:
        if hf_pipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for LLMJustifier. Install it or disable LLM explanations."
            )

        kwargs: Dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True  # requires bitsandbytes; falls back automatically if unavailable.

        logger.info("Loading free LLM model '%s' for explainability justifications.", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device:
            self.generator = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        else:
            self.generator = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def render(self, prompt: str) -> str:
        try:
            response = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
            )
            text = response[0]["generated_text"]
            # The pipeline returns prompt + completion; strip the prompt portion.
            completion = text[len(prompt) :].strip()
            return completion or text.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("LLM generation failed (%s); falling back to template.", exc)
            return ""


class ExplainabilityAgent:
    """Phase 5 explainability agent coordinating SHAP computation and narrative summaries."""

    def __init__(
        self,
        artifact_dir: str = "artifacts/explainability",
        llm_model: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.3",
        llm_enabled: bool = False,
        top_k: int = 3,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.model: Any = None
        self.feature_names: List[str] = []
        self.class_labels: List[str] = []
        self.background: Optional[pd.DataFrame] = None
        self.explainer: Optional[Any] = None
        self.top_k = top_k

        self.llm = None
        if llm_enabled and llm_model:
            try:
                self.llm = LLMJustifier(model_name=llm_model)
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning(
                    "LLMJustifier could not be initialized (%s). Falling back to templated messages.", exc
                )

    # ------------------------------------------------------------------
    # Registration & explainer bootstrap
    # ------------------------------------------------------------------
    def register_model(
        self,
        model: Any,
        feature_names: Sequence[str],
        class_labels: Sequence[str],
        background_frame: pd.DataFrame,
        background_sample_size: int = 200,
    ) -> None:
        """Attach a fitted classifier to the agent and bootstrap a SHAP explainer."""
        if background_frame.empty:
            raise ValueError("background_frame must contain data for explainer baseline.")

        self.model = model
        self.feature_names = list(feature_names)
        self.class_labels = list(class_labels)

        self.background = background_frame[self.feature_names].copy()
        if background_sample_size and len(self.background) > background_sample_size:
            self.background = self.background.sample(background_sample_size, random_state=42)

        self.explainer = self._build_explainer()
        logger.info(
            "ExplainabilityAgent registered model %s with %d features and %d classes.",
            type(model).__name__,
            len(self.feature_names),
            len(self.class_labels),
        )

    def _build_explainer(self) -> Any:
        if self.model is None or self.background is None:
            raise RuntimeError("Model and background data must be registered before building the explainer.")

        # Prefer tree-based explainers when supported.
        if hasattr(self.model, "estimators_") or hasattr(self.model, "tree_"):
            logger.info("Using TreeExplainer for model %s.", type(self.model).__name__)
            try:
                return shap.TreeExplainer(self.model)
            except Exception as exc:
                logger.info(
                    "TreeExplainer initialization failed (%s); falling back to KernelExplainer.",
                    exc,
                )

        logger.info("Using KernelExplainer for model %s.", type(self.model).__name__)
        predict_fn = self._make_predict_function()
        background = self.background[self.feature_names]
        return shap.KernelExplainer(predict_fn, background)

    # ------------------------------------------------------------------
    # Global explainability
    # ------------------------------------------------------------------
    def compute_global_importance(
        self,
        reference_frame: pd.DataFrame,
        sample_size: int = 2048,
        artifact_name: str = "global_importance.json",
    ) -> Dict[str, Any]:
        """Generate global SHAP summaries and persist them for guardrails/reporting."""
        self._ensure_ready()

        reference = reference_frame[self.feature_names]
        if len(reference) > sample_size:
            reference = reference.sample(sample_size, random_state=42)

        shap_values = self._compute_shap(reference)
        per_class_arrays = self._normalize_shap(shap_values)

        summaries: Dict[str, Any] = {"classes": {}, "feature_names": self.feature_names}
        for class_idx, class_label in enumerate(self.class_labels):
            abs_means = np.mean(np.abs(per_class_arrays[class_idx]), axis=0)
            feature_importance = [
                {"feature": feat, "importance": float(val)}
                for feat, val in sorted(
                    zip(self.feature_names, abs_means),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            summaries["classes"][class_label] = feature_importance

        artifact_path = self.artifact_dir / artifact_name
        artifact_path.write_text(json.dumps(summaries, indent=2))
        logger.info("Persisted global SHAP summary to %s.", artifact_path)
        return summaries

    # ------------------------------------------------------------------
    # Local explainability
    # ------------------------------------------------------------------
    def explain_batch(
        self,
        feature_frame: pd.DataFrame,
        predictions: pd.Series,
        probability_frame: Optional[pd.DataFrame] = None,
        batch_id: str = "batch",
    ) -> pd.DataFrame:
        """Attach local SHAP explanations and optional LLM justifications to predictions."""
        self._ensure_ready()

        features = feature_frame[self.feature_names]
        shap_values = self._compute_shap(features)
        per_class_arrays = self._normalize_shap(shap_values)

        explanations: List[ExplanationResult] = []
        for row_position, (row_idx, feature_row) in enumerate(features.iterrows()):
            predicted_label = predictions.loc[row_idx]
            class_index = self._label_to_index(predicted_label)
            shap_vector = per_class_arrays[class_index][row_position]
            shap_map = {
                feat: float(value) for feat, value in zip(self.feature_names, shap_vector)
            }
            top_features = self._extract_top_features(feature_row, shap_vector)
            probability = None
            if probability_frame is not None:
                probability = self._lookup_probability(
                    probability_frame=probability_frame,
                    entity_id=row_idx,
                    predicted_label=predicted_label,
                )

            justification = self._render_justification(
                predicted_label,
                probability,
                top_features,
            )

            result = ExplanationResult(
                entity_id=row_idx,
                predicted_label=predicted_label,
                probability=probability,
                shap_values=shap_map,
                top_features=top_features,
                justification=justification,
                feature_snapshot=self._safe_feature_snapshot(feature_row),
            )
            explanations.append(result)

        df = pd.DataFrame([exp.to_serializable() for exp in explanations]).set_index("entity_id")
        artifact_path = self.artifact_dir / f"local_explanations_{batch_id}.jsonl"
        with artifact_path.open("w", encoding="utf-8") as handle:
            for exp in explanations:
                handle.write(json.dumps(exp.to_serializable()) + "\n")
        logger.info("Persisted %d local explanations to %s.", len(explanations), artifact_path)
        return df

    # ------------------------------------------------------------------
    # Guardrail integration helpers
    # ------------------------------------------------------------------
    def build_guardrail_summary(self, local_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate basic statistics for guardrail checks."""
        stats = {
            "total_records": int(len(local_df)),
            "class_distribution": local_df.groupby("predicted_label").size().to_dict(),
            "average_confidence": float(local_df["probability"].dropna().mean())
            if "probability" in local_df and not local_df["probability"].dropna().empty
            else None,
        }
        artifact_path = self.artifact_dir / "guardrail_summary.json"
        artifact_path.write_text(json.dumps(stats, indent=2))
        logger.info("Persisted guardrail explainability summary to %s.", artifact_path)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self.model is None or self.explainer is None or not self.feature_names:
            raise RuntimeError("ExplainabilityAgent is not registered with a model and explainer.")

    def _compute_shap(self, features: pd.DataFrame) -> Any:
        if isinstance(self.explainer, shap.TreeExplainer):
            return self.explainer.shap_values(features)
        return self.explainer.shap_values(features.to_numpy())  # KernelExplainer

    def _normalize_shap(self, shap_values: Any) -> List[np.ndarray]:
        if isinstance(shap_values, list):
            return [np.array(values) for values in shap_values]
        values = np.asarray(shap_values)
        if values.ndim == 3:  # (n_samples, n_classes, n_features)
            return [values[:, idx, :] for idx in range(values.shape[1])]
        if values.ndim == 2:  # (n_samples, n_features) -> single class
            return [values]
        raise ValueError("Unsupported SHAP output shape.")

    def _make_predict_function(self):
        predict_proba = getattr(self.model, "predict_proba", None)
        if predict_proba is None:
            raise AttributeError("Model must expose predict_proba for KernelExplainer usage.")

        def _predict(data: Any) -> np.ndarray:
            if isinstance(data, pd.DataFrame):
                frame = data[self.feature_names]
            else:
                frame = pd.DataFrame(data, columns=self.feature_names)
            return predict_proba(frame)

        return _predict

    def _label_to_index(self, label: Any) -> int:
        try:
            return self.class_labels.index(label)
        except ValueError as exc:
            raise KeyError(f"Label '{label}' not found in registered class_labels.") from exc

    def _extract_top_features(self, feature_row: pd.Series, shap_vector: np.ndarray) -> List[Dict[str, Any]]:
        rankings = sorted(
            zip(self.feature_names, shap_vector, feature_row[self.feature_names]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_entries: List[Dict[str, Any]] = []
        for feature, attribution, raw_value in rankings[: self.top_k]:
            top_entries.append(
                {
                    "feature": feature,
                    "attribution": float(attribution),
                    "feature_value": self._safe_value(raw_value),
                }
            )
        return top_entries

    def _lookup_probability(
        self,
        probability_frame: pd.DataFrame,
        entity_id: Any,
        predicted_label: str,
    ) -> Optional[float]:
        try:
            row = probability_frame.loc[entity_id]
        except KeyError:
            logger.debug("Probability row missing for entity_id %s", entity_id)
            return None

        if isinstance(row, pd.DataFrame):  # duplicate indices
            row = row.iloc[0]

        for candidate in (predicted_label, f"prob_{predicted_label}"):
            if candidate in row.index:
                value = row[candidate]
                return float(value) if pd.notna(value) else None

        return None

    def _render_justification(
        self,
        predicted_label: str,
        probability: Optional[float],
        top_features: List[Dict[str, Any]],
    ) -> str:
        bullet_points = "\n".join(
            f"- {t['feature']} ({'+' if t['attribution'] >= 0 else ''}{t['attribution']:.3f}) | value={t['feature_value']}"
            for t in top_features
        )
        confidence_text = f"{probability:.3f}" if probability is not None else "n/a"
        base_prompt = (
            "You are an analytics assistant for an online gaming retention project. "
            "Craft a concise, evidence-based justification for a prediction. "
            "Keep it under 60 words, cite only the supplied features, and avoid hallucinations.\n\n"
            f"Predicted engagement class: {predicted_label}\n"
            f"Confidence: {confidence_text}\n"
            "Top contributing features:\n"
            f"{bullet_points}\n"
            "Explanation:"
        )

        if self.llm:
            completion = self.llm.render(base_prompt).strip()
            if completion:
                return completion

        top_tokens = ", ".join(
            f"{item['feature']} ({item['attribution']:.3f})" for item in top_features
        )
        confidence_phrase = f" with probability {probability:.2f}" if probability is not None else ""
        return (
            f"Predicted {predicted_label}{confidence_phrase} driven by {top_tokens}. "
            "Feature attributions derived from SHAP."
        )

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value

    def _safe_feature_snapshot(self, feature_row: pd.Series) -> Dict[str, Any]:
        return {feat: self._safe_value(feature_row[feat]) for feat in self.feature_names}