"""Stages 6-8: live gates (PLAN.md §3). Same-seed and regression compare
raw counts; held-out adds a one-sided exact McNemar test, reported for
context -- with n=10 held-out episodes it is informational, not the
pass/fail criterion (see PLAN.md §3 stage 8, manifest.yaml gates.held_out).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import List

from evolution.models import GateResult


@dataclass
class PairedOutcome:
    seed: int
    parent_success: bool
    candidate_success: bool


def one_sided_exact_mcnemar(paired: List[PairedOutcome]) -> float:
    """P(candidate beats parent by this many discordant wins or more | no
    true difference), i.e. P(X >= b) under X ~ Binomial(b+c, 0.5), where
    b = candidate-only wins, c = parent-only wins."""
    b = sum(1 for p in paired if p.candidate_success and not p.parent_success)
    c = sum(1 for p in paired if p.parent_success and not p.candidate_success)
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def evaluate_same_seed(paired: List[PairedOutcome], min_rescue_rate: float = 0.5) -> GateResult:
    rescued = sum(1 for p in paired if p.candidate_success and not p.parent_success)
    regressed = sum(1 for p in paired if p.parent_success and not p.candidate_success)
    rate = rescued / len(paired) if paired else 0.0
    passed = rate >= min_rescue_rate and regressed == 0
    return GateResult(
        "same_seed", passed,
        {"rescued": rescued, "regressed": regressed, "rescue_rate": rate, "n": len(paired)},
    )


def evaluate_regression(paired: List[PairedOutcome], allow_success_drop: int = 0) -> GateResult:
    parent_successes = sum(p.parent_success for p in paired)
    candidate_successes = sum(p.candidate_success for p in paired)
    drop = parent_successes - candidate_successes
    passed = drop <= allow_success_drop
    return GateResult(
        "regression", passed,
        {"parent_successes": parent_successes, "candidate_successes": candidate_successes, "drop": drop},
    )


def evaluate_held_out(paired: List[PairedOutcome], min_gain: int = 1, alpha: float = 0.025) -> GateResult:
    parent_successes = sum(p.parent_success for p in paired)
    candidate_successes = sum(p.candidate_success for p in paired)
    gain = candidate_successes - parent_successes
    p_value = one_sided_exact_mcnemar(paired)
    passed = gain >= min_gain
    return GateResult(
        "held_out", passed,
        {
            "parent_successes": parent_successes,
            "candidate_successes": candidate_successes,
            "gain": gain,
            "p_value": p_value,
            "alpha": alpha,
            "note": "n is small at this scale; p_value is informational, not the pass rule",
        },
    )
