"""
evolution/experimentation_engine.py — Layer 7, Module 5
═══════════════════════════════════════════════════════════════════════════════
A/B testing framework for agent capabilities.

Runs controlled experiments: splits traffic between a *control*
(current) and *treatment* (proposed change), measures KPI deltas,
and recommends adoption or rollback.

Safety: experiments never exceed a configured traffic allocation
and automatically stop on degradation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ── Data types ────────────────────────────────────────────────────────────────


class ExperimentStatus(Enum):
    DRAFT = auto()
    RUNNING = auto()
    PAUSED = auto()
    CONCLUDED = auto()
    ADOPTED = auto()
    ROLLED_BACK = auto()


class MetricDirection(Enum):
    HIGHER_IS_BETTER = auto()
    LOWER_IS_BETTER = auto()


@dataclass
class ExperimentMetric:
    """Definition of a KPI to be tracked."""

    name: str
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    min_delta_pct: float = 5.0  # minimum improvement % to declare winner


@dataclass
class ExperimentSample:
    """One observation in either arm of the experiment."""

    arm: str  # "control" or "treatment"
    metric_values: Dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Experiment:
    """An A/B experiment definition."""

    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    name: str = ""
    target_capability: str = ""
    hypothesis: str = ""
    metrics: List[ExperimentMetric] = field(default_factory=list)
    traffic_pct: float = 10.0  # % of traffic routed to treatment
    max_samples: int = 100
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    concluded_at: Optional[float] = None
    control_samples: List[ExperimentSample] = field(default_factory=list)
    treatment_samples: List[ExperimentSample] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None


# ── Engine ────────────────────────────────────────────────────────────────────


class ExperimentationEngine:
    """
    Manages experiment lifecycle: create → start → record → conclude → adopt/rollback.
    """

    MAX_CONCURRENT = 3
    MAX_TRAFFIC_PCT = 30.0  # safety cap: never more than 30% in treatment
    DEGRADATION_FLOOR = 0.15  # stop if treatment is 15% worse on any metric

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._total_concluded = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        target_capability: str,
        hypothesis: str = "",
        metrics: List[ExperimentMetric] = None,
        traffic_pct: float = 10.0,
        max_samples: int = 100,
    ) -> Experiment:
        traffic_pct = min(traffic_pct, self.MAX_TRAFFIC_PCT)
        exp = Experiment(
            name=name,
            target_capability=target_capability,
            hypothesis=hypothesis,
            metrics=metrics
            or [
                ExperimentMetric("success_rate", MetricDirection.HIGHER_IS_BETTER),
                ExperimentMetric(
                    "latency_ms", MetricDirection.LOWER_IS_BETTER, min_delta_pct=10.0
                ),
            ],
            traffic_pct=traffic_pct,
            max_samples=max_samples,
        )
        self._experiments[exp.experiment_id] = exp
        return exp

    def start(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.DRAFT:
            return False
        running = sum(
            1
            for e in self._experiments.values()
            if e.status == ExperimentStatus.RUNNING
        )
        if running >= self.MAX_CONCURRENT:
            return False
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = time.time()
        return True

    def pause(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False
        exp.status = ExperimentStatus.PAUSED
        return True

    def resume(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.PAUSED:
            return False
        exp.status = ExperimentStatus.RUNNING
        return True

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_sample(
        self,
        experiment_id: str,
        arm: str,
        metric_values: Dict[str, float],
    ) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False
        if arm not in ("control", "treatment"):
            return False
        sample = ExperimentSample(arm=arm, metric_values=metric_values)
        if arm == "control":
            exp.control_samples.append(sample)
        else:
            exp.treatment_samples.append(sample)

        # Auto-stop on degradation
        if self._check_degradation(exp):
            self._conclude(exp, forced=True)

        # Auto-conclude when enough samples
        total = len(exp.control_samples) + len(exp.treatment_samples)
        if total >= exp.max_samples and exp.status == ExperimentStatus.RUNNING:
            self._conclude(exp)

        return True

    def route_arm(self, experiment_id: str, random_value: float = 0.5) -> str:
        """Decide which arm a given request should go to.

        *random_value* should be a uniform [0, 1) random value.
        Returns 'treatment' if the request falls in the treatment bucket,
        otherwise 'control'.
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return "control"
        if random_value * 100 < exp.traffic_pct:
            return "treatment"
        return "control"

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _aggregate(
        self, samples: List[ExperimentSample], metric_name: str
    ) -> Optional[float]:
        vals = [
            s.metric_values.get(metric_name)
            for s in samples
            if metric_name in s.metric_values
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _check_degradation(self, exp: Experiment) -> bool:
        """Return True if treatment is significantly worse."""
        if len(exp.treatment_samples) < 5:
            return False
        for m in exp.metrics:
            ctrl = self._aggregate(exp.control_samples, m.name)
            treat = self._aggregate(exp.treatment_samples, m.name)
            if ctrl is None or treat is None:
                continue
            if m.direction == MetricDirection.HIGHER_IS_BETTER:
                if ctrl > 0 and (ctrl - treat) / ctrl > self.DEGRADATION_FLOOR:
                    return True
            else:
                if ctrl > 0 and (treat - ctrl) / ctrl > self.DEGRADATION_FLOOR:
                    return True
        return False

    def _conclude(self, exp: Experiment, forced: bool = False) -> None:
        exp.status = ExperimentStatus.CONCLUDED
        exp.concluded_at = time.time()
        self._total_concluded += 1

        result: Dict[str, Any] = {"forced_stop": forced, "metrics": {}}
        winner = "control"
        treatment_wins = 0

        for m in exp.metrics:
            ctrl = self._aggregate(exp.control_samples, m.name)
            treat = self._aggregate(exp.treatment_samples, m.name)
            metric_result: Dict[str, Any] = {
                "control_avg": ctrl,
                "treatment_avg": treat,
                "winner": None,
            }
            if ctrl is not None and treat is not None and ctrl != 0:
                if m.direction == MetricDirection.HIGHER_IS_BETTER:
                    delta = (treat - ctrl) / abs(ctrl) * 100
                else:
                    delta = (ctrl - treat) / abs(ctrl) * 100
                metric_result["delta_pct"] = round(delta, 2)
                if delta >= m.min_delta_pct:
                    metric_result["winner"] = "treatment"
                    treatment_wins += 1
                elif delta <= -m.min_delta_pct:
                    metric_result["winner"] = "control"
                else:
                    metric_result["winner"] = "tie"
            result["metrics"][m.name] = metric_result

        if not forced and treatment_wins == len(exp.metrics):
            winner = "treatment"
        result["overall_winner"] = winner
        exp.result = result

    def conclude(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False
        self._conclude(exp)
        return True

    # ── Adoption ──────────────────────────────────────────────────────────────

    def adopt(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.CONCLUDED:
            return False
        exp.status = ExperimentStatus.ADOPTED
        return True

    def rollback(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status not in (
            ExperimentStatus.CONCLUDED,
            ExperimentStatus.ADOPTED,
        ):
            return False
        exp.status = ExperimentStatus.ROLLED_BACK
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def running_experiments(self) -> List[Experiment]:
        return [
            e
            for e in self._experiments.values()
            if e.status == ExperimentStatus.RUNNING
        ]

    def snapshot(self) -> Dict[str, Any]:
        statuses: Dict[str, int] = {}
        for e in self._experiments.values():
            key = e.status.name.lower()
            statuses[key] = statuses.get(key, 0) + 1
        return {
            "total_experiments": len(self._experiments),
            "statuses": statuses,
            "total_concluded": self._total_concluded,
        }
