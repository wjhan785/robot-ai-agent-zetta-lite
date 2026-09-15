from runtime.role1 import Role1


def make_role1(**kwargs):
    defaults = dict(allowed_tools={"lift", "retreat"}, max_recoveries_per_episode=2, cooldown_steps=30)
    defaults.update(kwargs)
    return Role1(**defaults)


def test_approves_known_tools():
    role1 = make_role1()
    decision = role1.decide(proposed_tools=["lift"], step=0)
    assert decision.approved


def test_rejects_unknown_tools():
    role1 = make_role1()
    decision = role1.decide(proposed_tools=["teleport"], step=0)
    assert not decision.approved
    assert "allowlist" in decision.reason


def test_rejects_when_recovery_already_active():
    role1 = make_role1()
    role1.on_recovery_start(step=0)
    decision = role1.decide(proposed_tools=["lift"], step=1)
    assert not decision.approved
    assert "already running" in decision.reason


def test_enforces_cooldown():
    role1 = make_role1(cooldown_steps=30)
    role1.on_recovery_start(step=0)
    role1.on_recovery_end()
    decision = role1.decide(proposed_tools=["lift"], step=10)
    assert not decision.approved
    assert "cooldown" in decision.reason

    decision_later = role1.decide(proposed_tools=["lift"], step=31)
    assert decision_later.approved


def test_enforces_recovery_budget():
    role1 = make_role1(max_recoveries_per_episode=1, cooldown_steps=0)
    role1.on_recovery_start(step=0)
    role1.on_recovery_end()
    decision = role1.decide(proposed_tools=["lift"], step=1)
    assert not decision.approved
    assert "budget exhausted" in decision.reason


def test_reset_clears_budget_and_cooldown():
    role1 = make_role1(max_recoveries_per_episode=1, cooldown_steps=5)
    role1.on_recovery_start(step=0)
    role1.on_recovery_end()
    role1.reset()
    decision = role1.decide(proposed_tools=["lift"], step=1)
    assert decision.approved
