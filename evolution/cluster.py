"""Stage 2: cluster failed trajectories by first missing milestone
(PLAN.md §3). Pass rule: pick the largest cluster."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from evolution.models import FailureCluster, Trajectory


def cluster_failures(trajectories: List[Trajectory], ordered_milestones: List[str]) -> List[FailureCluster]:
    groups: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.success:
            continue
        missing = t.first_missing_milestone(ordered_milestones)
        if missing is None:
            continue  # reached every milestone but still failed -- not milestone-diagnosable, skip
        groups[missing].append(t)

    clusters: List[FailureCluster] = []
    for milestone, group in groups.items():
        prior = prior_milestone(milestone, ordered_milestones)
        # representative = median by divergence step, so the LLM sees a "typical" failure, not an outlier
        by_divergence = sorted(group, key=lambda t: t.divergence_step(milestone, prior))
        representative = by_divergence[len(by_divergence) // 2]
        clusters.append(
            FailureCluster(
                task_id=group[0].task_id,
                first_missing_milestone=milestone,
                seeds=sorted(t.seed for t in group),
                representative_seed=representative.seed,
            )
        )

    clusters.sort(key=lambda c: len(c.seeds), reverse=True)
    return clusters


def prior_milestone(milestone: str, ordered: List[str]) -> Optional[str]:
    """The milestone immediately before `milestone` in `ordered`, or None if
    it's first -- shared with evolution.campaign._run_gates, which needs
    the same notion to compute a failure's divergence_step for gating."""
    idx = ordered.index(milestone)
    return ordered[idx - 1] if idx > 0 else None
