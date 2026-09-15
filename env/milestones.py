"""Per-task milestone checks (PLAN.md §3, stage 2 clustering key).

A milestone is a boolean function of the feature dict computed once per
step by env.libero_env.LiberoEnv.features(). This module only defines the
*shape*; the actual thresholds have to be tuned per task once you can
watch real telemetry on the workstation (PLAN.md §8 Weeks 1-2), and belong
in configs/manifest.yaml / a small per-task wiring module -- not hardcoded
here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

MilestoneCheck = Callable[[Dict[str, Any]], bool]


def lift_above(feature: str, threshold: float) -> MilestoneCheck:
    """True once `feature` (typically an object's height above the table)
    exceeds `threshold`."""
    return lambda features: features.get(feature, 0.0) > threshold


def near(feature_a: str, feature_b: str, tolerance: float) -> MilestoneCheck:
    """True once two scalar features (e.g. an object's z vs. a target
    surface's z) are within `tolerance` of each other."""
    return lambda features: abs(features.get(feature_a, 1e9) - features.get(feature_b, -1e9)) < tolerance


def above_threshold(feature: str, op: str, threshold: Any) -> MilestoneCheck:
    """Generic single-feature check, e.g. above_threshold('gripper_open', '==', False)."""
    from runtime.critic import eval_predicate

    return lambda features: eval_predicate((feature, op, threshold), features)


class MilestoneTracker:
    """Evaluates an ordered set of milestones once per step and records the
    first step each one becomes true -- this is exactly what stage 2
    (evolution.cluster) groups failures by."""

    def __init__(self, milestones: Dict[str, MilestoneCheck]):
        self.milestones = milestones
        self.reached_at: Dict[str, int] = {}

    def reset(self) -> None:
        self.reached_at.clear()

    def step(self, step: int, features: Dict[str, Any]) -> None:
        for name, check in self.milestones.items():
            if name not in self.reached_at and check(features):
                self.reached_at[name] = step

    def first_missing(self, ordered_names: List[str]) -> Optional[str]:
        for name in ordered_names:
            if name not in self.reached_at:
                return name
        return None
