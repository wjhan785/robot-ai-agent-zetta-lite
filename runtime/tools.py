"""Bounded recovery tools (PLAN.md §3, Runtime loop).

Every tool is a small, clamped primitive. The Propose stage (LLM) can only
reference tools by name+args from TOOL_CATALOG (schemas/bundle.schema.json
enforces the name; ToolSpec.clamp() enforces the numeric bounds at
execution time) -- that's the whole of this project's "light safety".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol, Tuple


class ExecutionContext(Protocol):
    """Whatever the recovery actor needs to actually move the robot / query
    privileged sim state. Implemented by env.libero_env.LiberoEnv on the
    real cluster, and by small fakes in tests."""

    def move_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None: ...
    def rotate_wrist(self, delta_rad: float) -> None: ...
    def set_gripper(self, open: bool) -> None: ...
    def object_pose(self, name: str) -> Tuple[float, float, float]: ...
    def eef_pose(self) -> Tuple[float, float, float]: ...
    def query_policy(self, n_chunks: int = 1) -> None: ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    bounds: Dict[str, Tuple[float, float]]   # param -> (min, max), inclusive
    handler: Callable[..., None]

    def clamp(self, args: Dict[str, Any]) -> Dict[str, Any]:
        clamped = dict(args)
        for key, (lo, hi) in self.bounds.items():
            if key in clamped:
                clamped[key] = max(lo, min(hi, clamped[key]))
        return clamped


def _move_delta(ctx: ExecutionContext, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
    ctx.move_delta(dx=dx, dy=dy, dz=dz)


def _move_to_object(
    ctx: ExecutionContext, obj: str, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0
) -> None:
    ox, oy, oz = ctx.object_pose(obj)
    ex, ey, ez = ctx.eef_pose()
    ctx.move_delta(dx=(ox + offset_x) - ex, dy=(oy + offset_y) - ey, dz=(oz + offset_z) - ez)


def _set_gripper(ctx: ExecutionContext, open: bool) -> None:
    ctx.set_gripper(open=open)


def _rotate_wrist(ctx: ExecutionContext, delta_rad: float) -> None:
    ctx.rotate_wrist(delta_rad=delta_rad)


def _lift(ctx: ExecutionContext, dz: float = 0.08) -> None:
    ctx.move_delta(dz=dz)


def _retreat(ctx: ExecutionContext, dz: float = 0.1, dx: float = -0.05) -> None:
    ctx.move_delta(dx=dx, dz=dz)


def _vla_chunk(ctx: ExecutionContext, n: int = 1) -> None:
    ctx.query_policy(n_chunks=n)


TOOL_CATALOG: Dict[str, ToolSpec] = {
    "move_delta": ToolSpec(
        "move_delta",
        "Cartesian translation of the end effector, clamped to +-10cm per axis.",
        {"dx": (-0.10, 0.10), "dy": (-0.10, 0.10), "dz": (-0.10, 0.10)},
        _move_delta,
    ),
    "move_to_object": ToolSpec(
        "move_to_object",
        "Move to a named object's pose (+ offset) using privileged sim state.",
        {"offset_x": (-0.10, 0.10), "offset_y": (-0.10, 0.10), "offset_z": (-0.10, 0.10)},
        _move_to_object,
    ),
    "set_gripper": ToolSpec(
        "set_gripper", "Open or close the gripper.", {}, _set_gripper,
    ),
    "rotate_wrist": ToolSpec(
        "rotate_wrist",
        "Rotate the wrist joint, clamped to +-45 degrees (radians).",
        {"delta_rad": (-0.785, 0.785)},
        _rotate_wrist,
    ),
    "lift": ToolSpec(
        "lift", "Move straight up, clamped to 0-15cm.", {"dz": (0.0, 0.15)}, _lift,
    ),
    "retreat": ToolSpec(
        "retreat",
        "Back away and lift slightly, clamped to a small range.",
        {"dz": (0.0, 0.15), "dx": (-0.15, 0.0)},
        _retreat,
    ),
    "vla_chunk": ToolSpec(
        "vla_chunk",
        "Re-query the frozen Pi0.5 policy for n action chunks (1-4).",
        {"n": (1, 4)},
        _vla_chunk,
    ),
}


def get_tool(name: str) -> ToolSpec:
    try:
        return TOOL_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool {name!r}; allowed: {sorted(TOOL_CATALOG)}") from exc
