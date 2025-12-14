"""
Recommendation / Action agent for the Predict Online Gaming Behavior pipeline.

Phase 6 responsibilities:
    • Convert engagement predictions plus explainability context into actionable interventions.
    • Attach evidentiary feature drivers and confidence scores to each recommendation.
    • Persist recommendation payloads for orchestrator, guardrail review, and closed-loop tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRecord:
    """Container capturing the action plan and evidence for a single player."""
    entity_id: Any
    predicted_label: str
    confidence: Optional[float]
    primary_action: str
    secondary_actions: List[str]
    evidence: List[str]
    expected_impact: str
    justification: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> Dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "primary_action": self.primary_action,
            "secondary_actions": self.secondary_actions,
            "evidence": self.evidence,
            "expected_impact": self.expected_impact,
            "justification": self.justification,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class RecommendationAgent:
    """Phase 6 Action / Intervention agent translating predictions into retention tactics."""

    def __init__(
        self,
        artifact_dir: str = "artifacts/recommendations",
        action_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        min_confidence: float = 0.0,
        history_window: int = 30,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.history_window = history_window

        self.action_catalog = action_catalog or {
            "High": {
                "primary": "Provide loyalty rewards bundle to sustain engagement.",
                "secondary": [
                    "Invite to exclusive tournaments.",
                    "Surface leaderboard progress notifications.",
                ],
                "impact": "Maintains momentum for highly engaged players; prevents churn due to stagnation.",
            },
            "Medium": {
                "primary": "Trigger achievement-nudge campaign highlighting near-miss goals.",
                "secondary": [
                    "Email personalized session streak challenges.",
                    "Offer limited-time cooperative quests to deepen involvement.",
                ],
                "impact": "Targets latent interest to elevate engagement toward High classification.",
            },
            "Low": {
                "primary": "Enroll player in win-back quest with starter incentives.",
                "secondary": [
                    "Provide discount on starter bundles.",
                    "Deploy in-app coach tips to reduce perceived difficulty.",
                ],
                "impact": "Mitigates risk of churn by lowering friction and adding immediate value.",
            },
        }

    def generate_recommendations(
        self,
        explanation_frame: pd.DataFrame,
        batch_id: str,
        probability_column: str = "probability",
    ) -> pd.DataFrame:
        """Create recommendation records from ExplainabilityAgent outputs."""
        if explanation_frame.empty:
            raise ValueError("explanation_frame is empty; cannot generate recommendations.")

        records: List[RecommendationRecord] = []
        for entity_id, row in explanation_frame.iterrows():
            confidence = None
            if probability_column in row and pd.notna(row[probability_column]):
                confidence = float(row[probability_column])
                if confidence < self.min_confidence:
                    logger.debug(
                        "Skipping entity %s due to confidence %.3f below threshold %.3f.",
                        entity_id,
                        confidence,
                        self.min_confidence,
                    )
                    continue

            label = str(row["predicted_label"])
            catalog_entry = self.action_catalog.get(label)
            if catalog_entry is None:
                logger.debug("No action catalog entry for label '%s'; skipping entity %s.", label, entity_id)
                continue

            evidence = self._format_evidence(row.get("top_features", []))
            justification = row.get("justification", "")
            metadata = {
                "shap_values": row.get("shap_values"),
                "feature_snapshot": row.get("feature_snapshot"),
                "history_window_days": self.history_window,
            }

            record = RecommendationRecord(
                entity_id=entity_id,
                predicted_label=label,
                confidence=confidence,
                primary_action=catalog_entry["primary"],
                secondary_actions=list(catalog_entry.get("secondary", [])),
                evidence=evidence,
                expected_impact=catalog_entry["impact"],
                justification=justification,
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
            records.append(record)

        df = pd.DataFrame([rec.to_serializable() for rec in records]).set_index("entity_id")
        output_path = self.artifact_dir / f"recommendations_{batch_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec.to_serializable()) + "\n")
        logger.info("Persisted %d recommendations to %s.", len(records), output_path)
        return df

    def summary(self, recommendations: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate recommendations for monitoring dashboards and guardrails."""
        if recommendations.empty:
            return {"total_recommendations": 0}

        label_counts = recommendations.groupby("predicted_label").size().to_dict()
        avg_confidence = (
            float(recommendations["confidence"].dropna().mean())
            if "confidence" in recommendations and not recommendations["confidence"].dropna().empty
            else None
        )
        stats = {
            "total_recommendations": int(len(recommendations)),
            "label_distribution": label_counts,
            "average_confidence": avg_confidence,
        }
        summary_path = self.artifact_dir / "recommendation_summary.json"
        summary_path.write_text(json.dumps(stats, indent=2))
        logger.info("Persisted recommendation summary to %s.", summary_path)
        return stats

    @staticmethod
    def _format_evidence(top_features: Any) -> List[str]:
        evidence: List[str] = []
        if isinstance(top_features, list):
            for feature_info in top_features:
                feature = feature_info.get("feature")
                attribution = feature_info.get("attribution")
                value = feature_info.get("feature_value")
                if feature is None or attribution is None:
                    continue
                sign = "+" if attribution >= 0 else ""
                evidence.append(f"{feature} {sign}{attribution:.3f} (value={value})")
        return evidence