"""Real LIBERO environment wrapper (PLAN.md §2/§3). Only importable on the
TC1 workstation once slurm/setup_env.sh has installed `libero`.

VERIFY on first use: exact benchmark/env constructor kwargs and the
observation dict's key names can vary slightly across LIBERO versions --
the shapes below match the commonly-documented `libero.libero` API. Fix up
import paths/kwargs here once you can see real error messages on TC1, not
by guessing further offline. `object_pose()` in particular is a stub --
it needs this task's actual MuJoCo body names, which are only knowable
once you can inspect a running task.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.milestones import MilestoneTracker
from evolution.models import StepRecord


class LiberoEnv:
    def __init__(
        self,
        suite: str,
        task_name: str,
        init_state: int,
        milestone_checks: Optional[Dict[str, Any]] = None,
        camera_size: int = 224,
    ):
        from libero.libero import benchmark, get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore

        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[suite]()
        task_id = self._task_index_by_name(task_name)
        task = self._task_suite.get_task(task_id)
        self._init_states = self._task_suite.get_task_init_states(task_id)

        bddl_file = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=camera_size, camera_widths=camera_size
        )

        self._init_state_idx = init_state
        self._milestones = MilestoneTracker(milestone_checks or {})
        self.step_log: List[StepRecord] = []
        self._step_idx = 0
        self._last_raw_obs: Dict[str, Any] = {}

    def _task_index_by_name(self, task_name: str) -> int:
        for i in range(self._task_suite.n_tasks):
            if self._task_suite.get_task(i).name == task_name:
                return i
        raise KeyError(f"task {task_name!r} not found in suite (check configs/manifest.yaml)")

    def reset(self) -> Dict[str, Any]:
        self._env.reset()
        self._env.set_init_state(self._init_states[self._init_state_idx])
        self._milestones.reset()
        self.step_log = []
        self._step_idx = 0
        self._last_raw_obs = {}
        return self._obs_dict()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
        libero_action = self._to_libero_action(action)
        raw_obs, reward, done, info = self._env.step(libero_action)
        self._last_raw_obs = raw_obs
        features = self.features()
        self._milestones.step(self._step_idx, features)

        success = bool(done and info.get("success", reward > 0))
        self.step_log.append(
            StepRecord(
                step=self._step_idx,
                features=features,
                action=action,
                milestones_reached=[m for m, s in self._milestones.reached_at.items() if s == self._step_idx],
            )
        )
        self._step_idx += 1
        return self._obs_dict(), bool(done), success

    def _to_libero_action(self, action: Dict[str, Any]) -> List[float]:
        # LIBERO's action space: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
        return [
            action.get("dx", 0.0), action.get("dy", 0.0), action.get("dz", 0.0),
            action.get("d_roll", 0.0), action.get("d_pitch", 0.0), action.get("d_yaw", 0.0),
            1.0 if action.get("gripper_open", False) else -1.0,
        ]

    def _obs_dict(self) -> Dict[str, Any]:
        return {"agentview_image": self._last_raw_obs.get("agentview_image")}

    def features(self) -> Dict[str, Any]:
        """Cheap, named numeric features for the critic/milestones. TODO:
        confirm exact obs keys (robot0_eef_pos etc.) against your
        installed LIBERO version and extend with whatever object poses
        this task's milestones actually need (configs/manifest.yaml)."""
        raw = self._last_raw_obs
        eef = raw.get("robot0_eef_pos", [0.0, 0.0, 0.0])
        gripper_qpos = raw.get("robot0_gripper_qpos", [0.0])
        return {
            "eef_x": eef[0], "eef_y": eef[1], "eef_z": eef[2],
            "gripper_open": bool(len(gripper_qpos) and gripper_qpos[0] > 0.02),
            "step": self._step_idx,
        }

    # -- runtime.tools.ExecutionContext protocol -----------------------
    def move_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.step({"dx": dx, "dy": dy, "dz": dz})

    def rotate_wrist(self, delta_rad: float) -> None:
        self.step({"d_yaw": delta_rad})

    def set_gripper(self, open: bool) -> None:
        self.step({"gripper_open": open})

    def object_pose(self, name: str) -> Tuple[float, float, float]:
        """TODO: map `name` (e.g. "target") to this task's actual MuJoCo
        body name and read `self._env.sim.data.body_xpos[...]` -- body
        names are task/BDDL-specific and only knowable once you can
        inspect a real task on TC1. Left unimplemented on purpose rather
        than guessed, since a wrong silent guess here is worse than a
        loud NotImplementedError."""
        raise NotImplementedError(
            f"wire up object_pose({name!r}) once you can inspect real MuJoCo body names on TC1"
        )

    def eef_pose(self) -> Tuple[float, float, float]:
        f = self.features()
        return (f["eef_x"], f["eef_y"], f["eef_z"])

    def query_policy(self, n_chunks: int = 1) -> None:
        """No-op here: runtime.loop.RuntimeLoop already calls
        policy.flush_chunk() after a recovery re-enters, which is what
        actually forces a fresh policy query on the next step. This
        exists only to satisfy the ExecutionContext protocol for the
        `vla_chunk` tool."""
        return None
