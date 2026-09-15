"""Ray actor wrapping one LIBERO env (PLAN.md §4, G1). One process per env,
pinned to a CPU core. Only importable where `ray` and `env.libero_env`
are installed -- i.e. only on the TC1 workstation, after slurm/setup_env.sh.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import ray


@ray.remote(num_cpus=1)
class EnvWorker:
    def __init__(self, suite: str, task_name: str):
        from env.libero_env import LiberoEnv  # deferred: only importable on TC1

        self._make_env = lambda seed: LiberoEnv(suite=suite, task_name=task_name, init_state=seed)
        self._env = None

    def reset(self, seed: int) -> Dict[str, Any]:
        self._env = self._make_env(seed)
        return self._env.reset()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
        return self._env.step(action)

    def features(self) -> Dict[str, Any]:
        return self._env.features()

    def step_log(self):
        return self._env.step_log

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Lets a (remote, single-process) RecoveryActor act on this env
        without shipping the whole env object back and forth -- see
        runtime/recovery.py for the ExecutionContext protocol this
        satisfies once G1 is wired into runtime/loop.py's execution
        context for the parallel path."""
        getattr(self._env, tool_name)(**args)
