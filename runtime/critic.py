"""Runtime critic: a cheap, declarative rule evaluated every control step
(PLAN.md §3). No LLM in this file -- the LLM only ever *writes* a rule
(evolution.propose); this module just evaluates one.

The same Rule shape is used live (runtime.loop) and offline
(evolution.shadow_replay replays the exact same rule over recorded
trajectories), so a rule's behaviour never differs between validation and
production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

Predicate = Tuple[str, str, Any]                     # (feature, op, value)
Rule = Union[Predicate, Dict[str, List["Rule"]]]      # predicate, {"all": [...]}, or {"any": [...]}

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def eval_predicate(pred: Predicate, features: Dict[str, Any]) -> bool:
    feature, op, value = pred
    if feature not in features:
        raise KeyError(f"critic referenced unknown feature {feature!r}; have: {sorted(features)}")
    if op not in _OPS:
        raise ValueError(f"unsupported operator {op!r}; allowed: {sorted(_OPS)}")
    return _OPS[op](features[feature], value)


def _is_predicate(rule: Any) -> bool:
    return (
        isinstance(rule, (list, tuple))
        and len(rule) == 3
        and isinstance(rule[1], str)
        and rule[1] in _OPS
    )


def eval_rule(rule: Rule, features: Dict[str, Any]) -> bool:
    if _is_predicate(rule):
        return eval_predicate(rule, features)  # type: ignore[arg-type]
    if not isinstance(rule, dict) or len(rule) != 1:
        raise ValueError(f"malformed rule: {rule!r}")
    if "all" in rule:
        return all(eval_rule(r, features) for r in rule["all"])
    if "any" in rule:
        return any(eval_rule(r, features) for r in rule["any"])
    raise ValueError(f"malformed rule: {rule!r} (expected 'all' or 'any')")


@dataclass
class Critic:
    """One evolved critic: a rule plus edge-triggering so it fires exactly
    once per contiguous window where the rule holds, not once per step."""

    id: str
    rule: Rule
    _armed: bool = True

    def step(self, features: Dict[str, Any]) -> bool:
        fired = eval_rule(self.rule, features)
        should_trigger = fired and self._armed
        self._armed = not fired
        return should_trigger

    def reset(self) -> None:
        self._armed = True
