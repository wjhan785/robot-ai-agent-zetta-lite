"""Stage 5: shadow replay -- offline check that a candidate critic detects
the failures it claims to, and never fires on successful trajectories.
Read-only: this validates detection only, never recovery -- a replayed
critic doesn't get to act, so it can't establish that the *recovery* would
have worked, only that the *critic* would have noticed (PLAN.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from evolution.models import GateResult, Trajectory
from runtime.critic import Critic, Rule


@dataclass
class ShadowReplayReport:
    detect_rate: float
    false_trigger_rate: float
    lead_times: List[int] = field(default_factory=list)
    passed: bool = False


def evaluate_shadow_replay(
    critic_rule: Rule,
    target_failures: List[Trajectory],
    success_controls: List[Trajectory],
    divergence_steps: Dict[int, int],   # seed -> divergence_step, for target_failures
    min_detect_rate: float = 0.8,
    max_false_trigger_rate: float = 0.0,
) -> ShadowReplayReport:
    detected = 0
    lead_times: List[int] = []
    for traj in target_failures:
        critic = Critic(id="shadow", rule=critic_rule)
        first_trigger = None
        for s in traj.steps:
            if critic.step(s.features):
                first_trigger = s.step
                break
        divergence = divergence_steps.get(traj.seed)
        if first_trigger is not None and divergence is not None and first_trigger <= divergence:
            detected += 1
            lead_times.append(divergence - first_trigger)

    false_triggers = 0
    for traj in success_controls:
        critic = Critic(id="shadow", rule=critic_rule)
        if any(critic.step(s.features) for s in traj.steps):
            false_triggers += 1

    detect_rate = detected / len(target_failures) if target_failures else 0.0
    false_trigger_rate = false_triggers / len(success_controls) if success_controls else 0.0
    passed = detect_rate >= min_detect_rate and false_trigger_rate <= max_false_trigger_rate
    return ShadowReplayReport(detect_rate, false_trigger_rate, lead_times, passed)


def to_gate_result(report: ShadowReplayReport) -> GateResult:
    return GateResult(
        name="shadow_replay",
        passed=report.passed,
        detail={
            "detect_rate": report.detect_rate,
            "false_trigger_rate": report.false_trigger_rate,
            "lead_times": report.lead_times,
        },
    )
