"""Shared data shapes for the evolution pipeline (PLAN.md §3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    step: int
    features: Dict[str, Any]
    action: Dict[str, Any]
    milestones_reached: List[str]


@dataclass
class Trajectory:
    task_id: str
    seed: int
    library_version: str
    success: bool
    steps: List[StepRecord]
    keyframe_paths: List[str] = field(default_factory=list)

    def first_missing_milestone(self, ordered_milestones: List[str]) -> Optional[str]:
        reached = set()
        for s in self.steps:
            reached.update(s.milestones_reached)
        for m in ordered_milestones:
            if m not in reached:
                return m
        return None

    def divergence_step(self, missing_milestone: str, prior_milestone: Optional[str]) -> int:
        """Heuristic estimate of "when things should have started going
        differently": the step right after the last-reached milestone
        before `missing_milestone`, or 0 if there wasn't one.

        NOTE: this is a simple stand-in for what a diagnose-stage LLM would
        actually estimate (Zetta's paper calls the real quantity the
        "Earliest Observable Divergence", tEOD, and distinguishes it from
        the true causal point). A real critic often can't fire *before*
        this step -- it may need a few steps to confirm a stall is really a
        stall. Don't be surprised if evaluate_shadow_replay's "at or before
        divergence_step" check is stricter than a real threshold-based
        critic can satisfy; loosen the comparison (e.g. divergence_step +
        a small confirmation window) once you're tuning real bundles on
        the workstation and can see this concretely."""
        if prior_milestone is None:
            return 0
        for s in reversed(self.steps):
            if prior_milestone in s.milestones_reached:
                return s.step + 1
        return 0


@dataclass
class FailureCluster:
    task_id: str
    first_missing_milestone: str
    seeds: List[int]
    representative_seed: int


@dataclass
class Diagnosis:
    cluster_id: str
    mechanism: str
    first_missing_milestone: str
    divergence_step: int
    evidence: str


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    cluster_id: str
    bundle_id: str
    approved: bool
    gates: List[GateResult]
    reason: str
