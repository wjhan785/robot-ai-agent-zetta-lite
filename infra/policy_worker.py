"""Ray actor holding the frozen Pi0.5 policy on the GPU (PLAN.md §4, G1-G3).
Only importable on the TC1 workstation (needs `ray` and openpi).
"""
from __future__ import annotations

from typing import Any, Dict

import ray


@ray.remote(num_gpus=1)
class PolicyWorker:
    def __init__(self, action_chunk: int = 10, exec_steps: int = 5):
        from policy.pi05_policy import Pi05Policy  # deferred: openpi only on TC1

        self._policy = Pi05Policy(action_chunk=action_chunk, exec_steps=exec_steps)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """NOTE (PLAN.md §4 G2): true dynamic batching needs observations
        stacked into one forward pass *inside* Pi05Policy, not just
        request-coalesced here -- infra/batcher.py shows the
        request-coalescing shape; wiring a real batched forward pass into
        policy/pi05_policy.py is a TODO for once you're timing this on
        TC1 (see scripts/bench_throughput.py)."""
        return self._policy.act(obs)

    def reset(self) -> None:
        self._policy.reset()

    def flush_chunk(self) -> None:
        self._policy.flush_chunk()
