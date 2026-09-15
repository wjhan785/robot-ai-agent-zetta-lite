"""Role1: rule-based gatekeeper between a Critic's proposal and the Recovery
Actor (PLAN.md §3, `runtime.role1_mode: rule_based` in manifest.yaml).

Kept deliberately dumb: fixed tool allowlist, a recovery budget per
episode, and a cooldown -- no LLM in this loop. An LLM-driven Role1 is a
documented stretch experiment (PLAN.md Week 10), not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Role1Decision:
    approved: bool
    reason: str


@dataclass
class Role1:
    allowed_tools: Set[str]
    max_recoveries_per_episode: int = 2
    cooldown_steps: int = 30

    _recoveries_used: int = field(default=0, repr=False)
    _last_recovery_step: int = field(default=-10_000, repr=False)
    _recovery_active: bool = field(default=False, repr=False)

    def decide(self, *, proposed_tools: List[str], step: int) -> Role1Decision:
        if self._recovery_active:
            return Role1Decision(False, "a recovery is already running")
        if self._recoveries_used >= self.max_recoveries_per_episode:
            return Role1Decision(False, "recovery budget exhausted for this episode")
        if step - self._last_recovery_step < self.cooldown_steps:
            return Role1Decision(False, "still within cooldown of the last recovery")
        unknown = [t for t in proposed_tools if t not in self.allowed_tools]
        if unknown:
            return Role1Decision(False, f"tool(s) not on allowlist: {unknown}")
        return Role1Decision(True, "approved")

    def on_recovery_start(self, step: int) -> None:
        self._recovery_active = True
        self._recoveries_used += 1
        self._last_recovery_step = step

    def on_recovery_end(self) -> None:
        self._recovery_active = False

    def reset(self) -> None:
        self._recoveries_used = 0
        self._last_recovery_step = -10_000
        self._recovery_active = False
