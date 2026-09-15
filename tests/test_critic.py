import pytest

from runtime.critic import Critic, eval_predicate, eval_rule


def test_eval_predicate_operators():
    features = {"x": 5}
    assert eval_predicate(("x", "==", 5), features)
    assert not eval_predicate(("x", "!=", 5), features)
    assert eval_predicate(("x", "<", 6), features)
    assert eval_predicate(("x", "<=", 5), features)
    assert eval_predicate(("x", ">", 4), features)
    assert eval_predicate(("x", ">=", 5), features)


def test_eval_predicate_unknown_feature_raises():
    with pytest.raises(KeyError):
        eval_predicate(("missing", "==", 1), {"x": 1})


def test_eval_predicate_unknown_op_raises():
    with pytest.raises(ValueError):
        eval_predicate(("x", "~=", 1), {"x": 1})


def test_eval_rule_all():
    rule = {"all": [("x", ">", 0), ("y", "<", 10)]}
    assert eval_rule(rule, {"x": 1, "y": 5})
    assert not eval_rule(rule, {"x": 1, "y": 20})


def test_eval_rule_any():
    rule = {"any": [("x", ">", 100), ("y", "<", 10)]}
    assert eval_rule(rule, {"x": 1, "y": 5})
    assert not eval_rule(rule, {"x": 1, "y": 20})


def test_eval_rule_nested():
    rule = {"all": [("x", ">", 0), {"any": [("y", "==", 1), ("y", "==", 2)]}]}
    assert eval_rule(rule, {"x": 1, "y": 2})
    assert not eval_rule(rule, {"x": 1, "y": 3})


def test_eval_rule_malformed_raises():
    with pytest.raises(ValueError):
        eval_rule({"nope": []}, {"x": 1})


def test_critic_edge_triggers_once_per_window():
    critic = Critic(id="c1", rule=("x", ">=", 5))
    fires = [critic.step({"x": v}) for v in [0, 3, 5, 6, 7, 3, 5]]
    # fires once when it first crosses the threshold, not on every step
    # it stays true, then can fire again after dropping and re-crossing
    assert fires == [False, False, True, False, False, False, True]


def test_critic_reset_rearms():
    critic = Critic(id="c1", rule=("x", ">=", 5))
    assert critic.step({"x": 5}) is True
    assert critic.step({"x": 5}) is False
    critic.reset()
    assert critic.step({"x": 5}) is True
