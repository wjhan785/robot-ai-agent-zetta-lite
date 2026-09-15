"""Recovery Actor: runs an approved bundle's steps against a bounded budget
(PLAN.md §3). Only ever invoked after Role1.decide() returns approved=True
(see runtime.loop) -- nothing here decides *whether* to recover, only *how*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from runtime.critic import Rule, eval_rule
from runtime.tools import ExecutionContext, get_tool


@dataclass
class RecoveryStep:
    tool: str
    args: Dict[str, Any]


@dataclass
class RecoveryOutcome:
    completed: bool
    steps_run: int
    reentered: bool
    reason: str


class RecoveryActor:
    def __init__(self, step_budget: int = 80):
        self.step_budget = step_budget

    def run(
        self,
        ctx: ExecutionContext,
        steps: List[RecoveryStep],
        reentry_rule: Optional[Rule],
        feature_fn: Callable[[], Dict[str, Any]],
    ) -> RecoveryOutcome:
        """`feature_fn()` must reflect ctx's *current* state (called again
        after every tool call, since each one mutates the environment)."""
        steps_run = 0
        for step in steps:
            if steps_run >= self.step_budget:
                return RecoveryOutcome(False, steps_run, False, "step budget exhausted")
            spec = get_tool(step.tool)
            clamped = spec.clamp(step.args)
            spec.handler(ctx, **clamped)
            steps_run += 1
            if reentry_rule is not None and eval_rule(reentry_rule, feature_fn()):
                return RecoveryOutcome(True, steps_run, True, "re-entry condition met early")

        if reentry_rule is None:
            return RecoveryOutcome(True, steps_run, True, "no re-entry rule; ran all steps")
        reentered = eval_rule(reentry_rule, feature_fn())
        reason = "ran all steps" if reentered else "budget spent, re-entry condition still not met"
        return RecoveryOutcome(True, steps_run, reentered, reason)
