"""Loop 1 (PLAN.md §3): the per-step runtime loop.

VLA acts -> Critic checks -> Role1 approves/rejects -> Recovery Actor runs
-> re-entry back to the VLA. This module is backend-agnostic: it only
needs an env shaped like runtime.tools.ExecutionContext plus
.reset()/.step(action)/.features(), and a policy with
.reset()/.act(obs)/.flush_chunk().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from runtime.critic import Critic, Rule
from runtime.recovery import RecoveryActor, RecoveryStep
from runtime.role1 import Role1


@dataclass
class Bundle:
    """One evolved critic+recovery pair, as produced by evolution.propose
    and validated against schemas/bundle.schema.json before it ever
    reaches here."""

    id: str
    critic: Critic
    recovery_steps: List[RecoveryStep]
    reentry_rule: Optional[Rule]


@dataclass
class EpisodeEvent:
    step: int
    kind: str    # "critic_fired" | "role1_reject" | "recovery_start" | "recovery_end"
    detail: str


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    events: List[EpisodeEvent] = field(default_factory=list)


class RuntimeLoop:
    def __init__(
        self,
        policy: Any,
        env: Any,
        bundles: List[Bundle],
        role1: Role1,
        recovery_actor: Optional[RecoveryActor] = None,
        feature_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        max_steps: int = 300,
    ):
        self.policy = policy
        self.env = env
        self.bundles = bundles
        self.role1 = role1
        self.recovery_actor = recovery_actor or RecoveryActor()
        self.feature_fn = feature_fn or (lambda: env.features())
        self.max_steps = max_steps

    def run_episode(self) -> EpisodeResult:
        obs = self.env.reset()
        self.policy.reset()
        self.role1.reset()
        for b in self.bundles:
            b.critic.reset()

        events: List[EpisodeEvent] = []
        for step in range(self.max_steps):
            action = self.policy.act(obs)
            obs, done, success = self.env.step(action)

            features = self.feature_fn()
            for bundle in self.bundles:
                if not bundle.critic.step(features):
                    continue
                events.append(EpisodeEvent(step, "critic_fired", bundle.id))

                decision = self.role1.decide(
                    proposed_tools=[s.tool for s in bundle.recovery_steps], step=step
                )
                if not decision.approved:
                    events.append(EpisodeEvent(step, "role1_reject", decision.reason))
                    continue

                self.role1.on_recovery_start(step)
                events.append(EpisodeEvent(step, "recovery_start", bundle.id))
                outcome = self.recovery_actor.run(
                    self.env, bundle.recovery_steps, bundle.reentry_rule, self.feature_fn
                )
                self.role1.on_recovery_end()
                events.append(EpisodeEvent(step, "recovery_end", outcome.reason))
                if outcome.reentered:
                    self.policy.flush_chunk()

            if done:
                return EpisodeResult(success, step + 1, events)

        return EpisodeResult(False, self.max_steps, events)
