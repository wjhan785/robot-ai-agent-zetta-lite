"""Throughput benchmark for the FYP results table (PLAN.md §4).

Only runs on TC1 (needs the real env/policy). Times a handful of episodes
under increasing levels of optimisation and appends rows to a CSV:

    (a) sequential          -- infra.local_backend.sequential_rollout, no Ray
    (b) +G1 (Ray split)     -- infra.env_worker + infra.policy_worker, 1 env at a time
    (c) +G2/G3 (batching)   -- N parallel env workers feeding one batched PolicyWorker
    (d) +G4 (chunking)      -- exec_steps>1 (already the default; compare exec_steps=1 vs >1)
    (e) +G7 (stretch)       -- fewer denoising steps / quantisation, if implemented

Usage:
    python scripts/bench_throughput.py --config configs/manifest.yaml --episodes 4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from evolution.campaign import load_config


def _time_it(fn, *args, **kwargs) -> float:
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def run_sequential(config: Dict[str, Any], task: Dict[str, Any], n_episodes: int) -> float:
    from env.libero_env import LiberoEnv
    from evolution.rollout import real_rollout
    from policy.pi05_policy import Pi05Policy
    from runtime.role1 import Role1
    from runtime.tools import TOOL_CATALOG

    policy_cfg = config["policy"]
    seeds = list(range(config["seeds"]["dev"]["start"], config["seeds"]["dev"]["start"] + n_episodes))

    def make_env(seed: int) -> LiberoEnv:
        return LiberoEnv(suite=task["suite"], task_name=task["name"], init_state=seed)

    def make_policy() -> Pi05Policy:
        return Pi05Policy(action_chunk=policy_cfg["action_chunk"], exec_steps=policy_cfg["exec_steps"])

    def role1_factory() -> Role1:
        return Role1(allowed_tools=set(TOOL_CATALOG))

    return _time_it(
        real_rollout, make_env, make_policy, [], seeds, task["id"], "bench", role1_factory
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--out", default="runs/throughput.csv")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    task = config["tasks"][0]

    rows: List[Dict[str, Any]] = []

    elapsed = run_sequential(config, task, args.episodes)
    rows.append({
        "config": "sequential",
        "episodes": args.episodes,
        "seconds": round(elapsed, 2),
        "episodes_per_min": round(args.episodes / elapsed * 60, 2) if elapsed > 0 else float("inf"),
    })

    # TODO (PLAN.md §4 G1-G7, Week 4): add the Ray-batched configurations
    # here once infra/env_worker.py + infra/policy_worker.py are wired
    # into a comparable entry point -- they're written but not yet driven
    # by this script. Follow the same _time_it(...) pattern.

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "episodes", "seconds", "episodes_per_min"])
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"appended to {out_path}")


if __name__ == "__main__":
    main()
