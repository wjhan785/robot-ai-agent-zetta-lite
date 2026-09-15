from runtime.recovery import RecoveryActor, RecoveryStep


class _FakeCtx:
    """Minimal ExecutionContext: tracks a scalar 'height' that `lift`
    increases and a call log, enough to exercise budgets/re-entry."""

    def __init__(self):
        self.height = 0.0
        self.calls = []

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0):
        self.height += dz
        self.calls.append(("move_delta", dx, dy, dz))

    def rotate_wrist(self, delta_rad):
        self.calls.append(("rotate_wrist", delta_rad))

    def set_gripper(self, open):
        self.calls.append(("set_gripper", open))

    def object_pose(self, name):
        return (0.0, 0.0, 0.0)

    def eef_pose(self):
        return (0.0, 0.0, self.height)

    def query_policy(self, n_chunks=1):
        self.calls.append(("query_policy", n_chunks))


def test_recovery_runs_all_steps_and_reenters():
    ctx = _FakeCtx()
    actor = RecoveryActor(step_budget=10)
    # threshold only satisfied after BOTH lifts (0.05 + 0.05 == 0.10), so
    # this actually exercises "ran every step then re-entered", unlike a
    # threshold any single step could already satisfy
    steps = [RecoveryStep("lift", {"dz": 0.05}), RecoveryStep("lift", {"dz": 0.05})]
    outcome = actor.run(ctx, steps, reentry_rule=("height", ">=", 0.10), feature_fn=lambda: {"height": ctx.height})
    assert outcome.completed
    assert outcome.steps_run == 2
    assert outcome.reentered


def test_recovery_stops_early_once_reentry_condition_met():
    ctx = _FakeCtx()
    actor = RecoveryActor(step_budget=10)
    steps = [
        RecoveryStep("lift", {"dz": 0.10}),
        RecoveryStep("lift", {"dz": 0.10}),
        RecoveryStep("lift", {"dz": 0.10}),
    ]
    outcome = actor.run(ctx, steps, reentry_rule=("height", ">=", 0.10), feature_fn=lambda: {"height": ctx.height})
    assert outcome.steps_run == 1  # stopped right after the first lift crossed the threshold
    assert outcome.reentered


def test_recovery_respects_step_budget():
    ctx = _FakeCtx()
    actor = RecoveryActor(step_budget=2)
    steps = [RecoveryStep("lift", {"dz": 0.01}) for _ in range(5)]
    outcome = actor.run(ctx, steps, reentry_rule=("height", ">=", 999), feature_fn=lambda: {"height": ctx.height})
    assert not outcome.completed
    assert outcome.steps_run == 2
    assert not outcome.reentered


def test_recovery_without_reentry_rule_always_reenters():
    ctx = _FakeCtx()
    actor = RecoveryActor(step_budget=10)
    steps = [RecoveryStep("lift", {"dz": 0.01})]
    outcome = actor.run(ctx, steps, reentry_rule=None, feature_fn=lambda: {"height": ctx.height})
    assert outcome.reentered
