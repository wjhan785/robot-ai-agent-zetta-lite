"""Stage 1 (and the live episodes same_seed/regression/held_out need):
run episodes and collect Trajectory records (PLAN.md §3).

Two backends:
- `fake_rollout`: deterministic synthetic trajectories, no sim/policy at
  all. Used by `--backend local` purely to exercise the rest of the
  pipeline's plumbing -- see PLAN.md §5/§6 and the module docstring caveat
  below about what it can and can't demonstrate.
- `real_rollout`: the actual LIBERO env + Pi0.5 policy, via runtime.loop.
  Only runs on the TC1 workstation (env/policy/infra all need the GPU box).
"""
from __future__ import annotations

import random
from typing import Any, Callable, List

from evolution.models import StepRecord, Trajectory
from runtime.loop import Bundle, RuntimeLoop
from runtime.role1 import Role1


def fake_rollout(
    task_id: str,
    seeds: List[int],
    ordered_milestones: List[str],
    library_version: str,
    base_success_rate: float = 0.5,
    rng_seed: int = 0,
    steps_per_gap: int = 3,
    stall_extra_steps: int = 6,
) -> List[Trajectory]:
    """Deterministic synthetic data -- no sim/policy involved at all. Each
    episode progresses through `ordered_milestones` at a fixed cadence
    (`steps_per_gap` steps between each, tracked as feature
    `steps_since_progress` which resets to 0 the instant a milestone is
    reached). Failures simply stop progressing partway through and run on
    for `stall_extra_steps` more steps without resetting the counter -- a
    toy stand-in for "the arm got stuck". This is what makes a
    stall-detecting critic like `["steps_since_progress", ">=", 5]`
    meaningfully (and honestly) detectable in tests without any real
    simulator.

    CAVEAT (see also evolution.models.Trajectory.divergence_step): the
    heuristic divergence_step used elsewhere in the pipeline is "right
    after the last reached milestone", but a stall-detecting critic can
    only fire a few steps *into* the stall, once it's distinguishable from
    normal progress. So a canned bundle replayed via evolution.shadow_replay
    against this data may legitimately fail the "detect at or before
    divergence_step" check even though it correctly detects the stall --
    that's an intentional, documented rough edge, not a bug to silently
    paper over. The CLI's `--backend local` dry run is a plumbing smoke
    test for exactly this reason; see tests/test_campaign.py for a
    hand-crafted example that *is* engineered to pass every gate.
    """
    trajectories = []
    for seed in seeds:
        seed_rng = random.Random(f"{rng_seed}:{seed}")
        success = seed_rng.random() < base_success_rate
        n_reach = len(ordered_milestones) if success else seed_rng.randint(1, len(ordered_milestones))

        steps: List[StepRecord] = []
        step_idx = 0
        for k in range(n_reach):
            for g in range(steps_per_gap):
                milestones_here = [ordered_milestones[k]] if g == steps_per_gap - 1 else []
                steps.append(StepRecord(
                    step=step_idx, features={"steps_since_progress": g}, action={},
                    milestones_reached=milestones_here,
                ))
                step_idx += 1
        if not success:
            for g in range(stall_extra_steps):
                steps.append(StepRecord(
                    step=step_idx, features={"steps_since_progress": steps_per_gap + g}, action={},
                    milestones_reached=[],
                ))
                step_idx += 1

        trajectories.append(Trajectory(
            task_id=task_id, seed=seed, library_version=library_version, success=success, steps=steps,
        ))
    return trajectories


def real_rollout(
    make_env: Callable[[int], Any],
    make_policy: Callable[[], Any],
    bundles: List[Bundle],
    seeds: List[int],
    task_id: str,
    library_version: str,
    role1_factory: Callable[[], Role1],
) -> List[Trajectory]:
    """Runs sequentially -- see infra/env_worker.py + infra/policy_worker.py
    for the parallel Ray version used once G1-G4 (PLAN.md §4) are wired
    up. `make_env(seed)` must return an env that logs each step's telemetry
    to `.step_log` (a List[StepRecord]) as it runs -- see env/libero_env.py.
    """
    trajectories: List[Trajectory] = []
    for seed in seeds:
        env = make_env(seed)
        policy = make_policy()
        loop = RuntimeLoop(policy=policy, env=env, bundles=bundles, role1=role1_factory())
        result = loop.run_episode()
        trajectories.append(Trajectory(
            task_id=task_id, seed=seed, library_version=library_version,
            success=result.success, steps=list(getattr(env, "step_log", [])),
        ))
    return trajectories
