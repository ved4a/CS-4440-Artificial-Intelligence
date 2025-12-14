"""
Orchestrator / Coordinator agent (Phase 7) for the Predict Online Gaming Behavior pipeline.

Responsibilities:
    • Route execution across phases, manage shared context, and enforce run ordering.
    • Maintain a lightweight model registry and structured logs.
    • Trigger retraining workflows when monitoring signals indicate drift.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentHandle:
    """Configuration for a registered pipeline agent."""
    name: str
    stage: str
    runner: Callable[[Dict[str, Any]], Dict[str, Any]]
    max_retries: int = 1
    depends_on: List[str] = field(default_factory=list)


@dataclass
class PipelineRun:
    """Captures state for a single orchestrated execution."""
    run_id: str
    label: str
    version: int
    created_at: str
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"


class OrchestratorAgent:
    """Phase 7 orchestrator coordinating multi-agent execution and lifecycle management."""

    def __init__(
        self,
        registry_path: str = "artifacts/orchestrator/model_registry.json",
        log_dir: str = "artifacts/orchestrator/logs",
        stage_order: Optional[List[str]] = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stage_order = stage_order or [
            "ingestion",
            "validation",
            "feature_engineering",
            "model_training",
            "evaluation",
            "monitoring",
            "explainability",
            "recommendation",
            "retraining",
        ]
        self._agents: Dict[str, AgentHandle] = {}
        self._registry: List[Dict[str, Any]] = self._load_registry()
        self._version = len(self._registry)

    # ------------------------------------------------------------------
    # Agent registration & orchestration
    # ------------------------------------------------------------------
    def register_agent(self, handle: AgentHandle) -> None:
        if handle.name in self._agents:
            raise ValueError(f"Agent '{handle.name}' already registered.")
        self._agents[handle.name] = handle
        logger.debug("Registered agent '%s' for stage '%s'.", handle.name, handle.stage)

    def run(
        self,
        label: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineRun:
        run_id = str(uuid.uuid4())
        state = PipelineRun(
            run_id=run_id,
            label=label,
            version=self._version + 1,
            created_at=datetime.utcnow().isoformat(),
            context=initial_context.copy() if initial_context else {},
            events=[],
        )
        logger.info("Starting orchestrated run %s (%s).", run_id, label)

        try:
            for stage in self.stage_order:
                stage_agents = [
                    handle for handle in self._agents.values() if handle.stage == stage
                ]
                if not stage_agents:
                    continue

                for handle in stage_agents:
                    self._await_dependencies(state, handle.depends_on)
                    self._execute_agent(state, handle)

            state.status = "succeeded"
            logger.info("Run %s completed successfully.", run_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            state.status = "failed"
            logger.exception("Run %s failed: %s", run_id, exc)
            state.events.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "run_failed",
                    "details": {"error": str(exc)},
                }
            )
            raise
        finally:
            self._persist_run(state)

        return state

    # ------------------------------------------------------------------
    # Registry & retraining management
    # ------------------------------------------------------------------
    def trigger_retraining(
        self,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "retraining_triggered",
            "details": {"reason": reason, "payload": payload or {}},
        }
        logger.warning("Retraining triggered: %s", reason)
        self._append_registry_event(event)

    def check_for_drift(self, context: Dict[str, Any]) -> bool:
        monitoring = context.get("monitoring") or {}
        drift_flag = monitoring.get("drift_detected", False)
        if drift_flag:
            drift_details = monitoring.get("drift_report", {})
            self.trigger_retraining("data_or_model_drift", drift_details)
        return bool(drift_flag)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_agent(self, state: PipelineRun, handle: AgentHandle) -> None:
        attempt = 0
        while attempt < handle.max_retries:
            attempt += 1
            try:
                logger.info(
                    "Running agent '%s' (stage=%s, attempt=%d).",
                    handle.name,
                    handle.stage,
                    attempt,
                )
                result = handle.runner(state.context)
                state.context.update(result or {})
                state.events.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": "agent_completed",
                        "details": {
                            "agent": handle.name,
                            "stage": handle.stage,
                            "attempt": attempt,
                            "result_keys": list((result or {}).keys()),
                        },
                    }
                )
                if result and "monitoring" in result:
                    self.check_for_drift(state.context)
                return
            except Exception as exc:
                logger.warning(
                    "Agent '%s' failed on attempt %d: %s",
                    handle.name,
                    attempt,
                    exc,
                )
                state.events.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": "agent_failed",
                        "details": {
                            "agent": handle.name,
                            "stage": handle.stage,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    }
                )
                if attempt >= handle.max_retries:
                    raise

    def _await_dependencies(self, state: PipelineRun, dependencies: List[str]) -> None:
        missing = [dep for dep in dependencies if dep not in state.context]
        if missing:
            raise RuntimeError(
                f"Dependencies {missing} were not satisfied before execution."
            )

    def _persist_run(self, state: PipelineRun) -> None:
        log_path = self.log_dir / f"run_{state.run_id}.json"
        log_path.write_text(json.dumps(state.__dict__, indent=2))
        registry_entry = {
            "run_id": state.run_id,
            "label": state.label,
            "version": state.version,
            "created_at": state.created_at,
            "status": state.status,
            "artifacts": state.context.get("artifacts"),
            "metrics": state.context.get("metrics"),
            "notes": state.context.get("notes"),
        }
        self._registry.append(registry_entry)
        self._version = len(self._registry)
        self.registry_path.write_text(json.dumps(self._registry, indent=2))
        logger.debug("Persisted run %s to registry.", state.run_id)

    def _append_registry_event(self, event: Dict[str, Any]) -> None:
        events_path = self.registry_path.parent / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _load_registry(self) -> List[Dict[str, Any]]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return []