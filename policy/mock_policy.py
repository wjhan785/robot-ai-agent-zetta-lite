"""A scripted/noisy stand-in for Pi0.5 -- no GPU, no openpi, no LIBERO
needed. Used for local dry runs, tests, and the runtime.loop smoke tests.
See policy/pi05_policy.py for the real thing, which only runs on TC1.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional


class MockPolicy:
    """Emits small random Cartesian deltas; occasionally "drops" the
    object by opening the gripper, so the runtime loop and evolution
    pipeline have something to detect/fix in tests without a real sim."""

    def __init__(self, seed: Optional[int] = None, drop_prob: float = 0.0):
        self._rng = random.Random(seed)
        self.drop_prob = drop_prob

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        drop = self._rng.random() < self.drop_prob
        return {
            "dx": self._rng.uniform(-0.02, 0.02),
            "dy": self._rng.uniform(-0.02, 0.02),
            "dz": self._rng.uniform(-0.02, 0.02),
            "gripper_open": drop,
        }

    def flush_chunk(self) -> None:
        pass
