from policy.mock_policy import MockPolicy
from runtime.critic import Critic
from runtime.loop import Bundle, RuntimeLoop
from runtime.recovery import RecoveryStep
from runtime.role1 import Role1


class _FakeEnv:
    """Succeeds at `max_steps` only if `recovered` has been set by then --
    `recovered` only gets set by a recovery's `lift` tool call, so success
    is a direct proxy for "did the runtime loop actually recover"."""

    def __init__(self, max_steps: int = 5):
        self.step_idx = 0
        self.max_steps = max_steps
        self.recovered = False

    def reset(self):
        self.step_idx = 0
        self.recovered = False
        return {}

    def step(self, action):
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        return {}, done, done and self.recovered

    def features(self):
        return {"step": self.step_idx, "recovered": self.recovered}

    # ExecutionContext protocol
    def move_delta(self, dx=0.0, dy=0.0, dz=0.0):
        self.recovered = True

    def rotate_wrist(self, delta_rad):
        pass

    def set_gripper(self, open):
        pass

    def object_pose(self, name):
        return (0.0, 0.0, 0.0)

    def eef_pose(self):
        return (0.0, 0.0, 0.0)

    def query_policy(self, n_chunks=1):
        pass


def test_episode_fails_without_a_bundle():
    env = _FakeEnv(max_steps=5)
    loop = RuntimeLoop(policy=MockPolicy(seed=0), env=env, bundles=[], role1=Role1(allowed_tools=set()))
    result = loop.run_episode()
    assert not result.success


def test_bundle_recovers_and_episode_succeeds():
    env = _FakeEnv(max_steps=5)
    bundle = Bundle(
        id="stall-fix",
        critic=Critic(id="stall-fix", rule=("step", "==", 3)),
        recovery_steps=[RecoveryStep("lift", {"dz": 0.05})],
        reentry_rule=("recovered", "==", True),
    )
    role1 = Role1(allowed_tools={"lift"})
    loop = RuntimeLoop(policy=MockPolicy(seed=0), env=env, bundles=[bundle], role1=role1)
    result = loop.run_episode()

    assert result.success
    kinds = [e.kind for e in result.events]
    assert "critic_fired" in kinds
    assert "recovery_start" in kinds
    assert "recovery_end" in kinds
    assert "role1_reject" not in kinds


def test_role1_rejects_bundle_with_disallowed_tool():
    env = _FakeEnv(max_steps=5)
    bundle = Bundle(
        id="stall-fix",
        critic=Critic(id="stall-fix", rule=("step", "==", 3)),
        recovery_steps=[RecoveryStep("lift", {"dz": 0.05})],
        reentry_rule=("recovered", "==", True),
    )
    role1 = Role1(allowed_tools=set())  # lift not allowed
    loop = RuntimeLoop(policy=MockPolicy(seed=0), env=env, bundles=[bundle], role1=role1)
    result = loop.run_episode()

    assert not result.success
    kinds = [e.kind for e in result.events]
    assert "role1_reject" in kinds
    assert "recovery_start" not in kinds
