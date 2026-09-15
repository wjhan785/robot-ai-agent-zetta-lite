from evolution.models import StepRecord, Trajectory
from evolution.shadow_replay import evaluate_shadow_replay

CRITIC_RULE = ("x", ">=", 5)


def _traj(seed: int, success: bool, xs: list) -> Trajectory:
    steps = [StepRecord(step=i, features={"x": x}, action={}, milestones_reached=[]) for i, x in enumerate(xs)]
    return Trajectory(task_id="t", seed=seed, library_version="v0", success=success, steps=steps)


def test_detects_failures_and_ignores_successes():
    failures = [_traj(1, False, [0, 1, 2, 3, 4, 5, 6]), _traj(2, False, [0, 2, 4, 5, 6])]
    successes = [_traj(3, True, [0, 1, 2, 3, 4, 3, 2])]
    # both failures first hit x>=5 at or before the divergence_step we declare here
    divergence_steps = {1: 5, 2: 4}

    report = evaluate_shadow_replay(CRITIC_RULE, failures, successes, divergence_steps)

    assert report.detect_rate == 1.0
    assert report.false_trigger_rate == 0.0
    assert report.passed


def test_false_trigger_on_a_success_fails_the_gate():
    failures = [_traj(1, False, [0, 1, 2, 3, 4, 5, 6])]
    successes = [_traj(2, True, [0, 1, 5, 1, 2])]  # transiently crosses the threshold too
    divergence_steps = {1: 5}

    report = evaluate_shadow_replay(CRITIC_RULE, failures, successes, divergence_steps)

    assert report.false_trigger_rate == 1.0
    assert not report.passed


def test_trigger_after_divergence_is_not_counted_as_detected():
    failures = [_traj(1, False, [0, 1, 2, 3, 4, 5, 6])]  # first hits 5 at step 5
    successes = []
    divergence_steps = {1: 2}  # declared divergence is earlier than the actual trigger

    report = evaluate_shadow_replay(CRITIC_RULE, failures, successes, divergence_steps)

    assert report.detect_rate == 0.0
    assert not report.passed


def test_empty_inputs_do_not_crash():
    report = evaluate_shadow_replay(CRITIC_RULE, [], [], {})
    assert report.detect_rate == 0.0
    assert report.false_trigger_rate == 0.0
