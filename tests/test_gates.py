import pytest

from evolution.gates import (
    PairedOutcome,
    evaluate_held_out,
    evaluate_regression,
    evaluate_same_seed,
    one_sided_exact_mcnemar,
)


def test_same_seed_passes_when_rescue_rate_meets_threshold():
    paired = [
        PairedOutcome(seed=1, parent_success=False, candidate_success=True),
        PairedOutcome(seed=2, parent_success=False, candidate_success=True),
        PairedOutcome(seed=3, parent_success=False, candidate_success=False),
    ]
    gate = evaluate_same_seed(paired, min_rescue_rate=0.5)
    assert gate.passed
    assert gate.detail["rescued"] == 2
    assert gate.detail["regressed"] == 0


def test_same_seed_fails_on_any_regression():
    paired = [
        PairedOutcome(seed=1, parent_success=True, candidate_success=False),
        PairedOutcome(seed=2, parent_success=False, candidate_success=True),
    ]
    gate = evaluate_same_seed(paired, min_rescue_rate=0.0)
    assert not gate.passed
    assert gate.detail["regressed"] == 1


def test_regression_gate_passes_with_no_drop():
    paired = [PairedOutcome(seed=i, parent_success=True, candidate_success=True) for i in range(5)]
    gate = evaluate_regression(paired, allow_success_drop=0)
    assert gate.passed
    assert gate.detail["drop"] == 0


def test_regression_gate_fails_on_a_drop():
    paired = [
        PairedOutcome(seed=1, parent_success=True, candidate_success=True),
        PairedOutcome(seed=2, parent_success=True, candidate_success=False),
    ]
    gate = evaluate_regression(paired, allow_success_drop=0)
    assert not gate.passed
    assert gate.detail["drop"] == 1


def test_held_out_gate_requires_min_gain():
    paired = [
        PairedOutcome(seed=1, parent_success=False, candidate_success=True),
        PairedOutcome(seed=2, parent_success=True, candidate_success=True),
    ]
    gate = evaluate_held_out(paired, min_gain=1)
    assert gate.passed
    assert gate.detail["gain"] == 1


def test_held_out_gate_fails_with_no_gain():
    paired = [PairedOutcome(seed=i, parent_success=True, candidate_success=True) for i in range(10)]
    gate = evaluate_held_out(paired, min_gain=1)
    assert not gate.passed
    assert gate.detail["gain"] == 0


@pytest.mark.parametrize(
    "b,c,expected",
    [
        (0, 0, 1.0),   # no discordant pairs at all -> p=1
        (1, 0, 0.5),   # one candidate-only win, no parent-only wins: P(X>=1 | n=1) = 0.5
        (2, 0, 0.25),  # P(X>=2 | n=2) = comb(2,2)/4 = 0.25
    ],
)
def test_mcnemar_known_values(b, c, expected):
    paired = (
        [PairedOutcome(seed=i, parent_success=False, candidate_success=True) for i in range(b)]
        + [PairedOutcome(seed=100 + i, parent_success=True, candidate_success=False) for i in range(c)]
    )
    assert one_sided_exact_mcnemar(paired) == pytest.approx(expected)
