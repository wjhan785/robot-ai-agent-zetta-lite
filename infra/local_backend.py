"""The `--backend local` counterpart to infra/env_worker.py +
infra/policy_worker.py (PLAN.md §4/§6): no Ray, sequential. Just re-exports
evolution.rollout.real_rollout so evolution/campaign.py has one obvious
import path regardless of which backend was selected on the CLI."""
from __future__ import annotations

from evolution.rollout import real_rollout as sequential_rollout  # noqa: F401
