import pytest

from runtime.tools import TOOL_CATALOG, get_tool


def test_get_tool_unknown_raises():
    with pytest.raises(KeyError):
        get_tool("teleport")


def test_clamp_restricts_out_of_bounds_values():
    spec = TOOL_CATALOG["move_delta"]
    clamped = spec.clamp({"dx": 5.0, "dy": -5.0, "dz": 0.02})
    assert clamped["dx"] == pytest.approx(0.10)
    assert clamped["dy"] == pytest.approx(-0.10)
    assert clamped["dz"] == pytest.approx(0.02)  # untouched, already in bounds


def test_clamp_ignores_unbounded_args():
    spec = TOOL_CATALOG["set_gripper"]
    clamped = spec.clamp({"open": True})
    assert clamped == {"open": True}


def test_all_catalog_tools_have_a_handler():
    for name, spec in TOOL_CATALOG.items():
        assert spec.name == name
        assert callable(spec.handler)


class _RecordingCtx:
    def __init__(self):
        self.calls = []

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0):
        self.calls.append(("move_delta", dx, dy, dz))

    def rotate_wrist(self, delta_rad):
        self.calls.append(("rotate_wrist", delta_rad))

    def set_gripper(self, open):
        self.calls.append(("set_gripper", open))

    def object_pose(self, name):
        return (1.0, 2.0, 3.0)

    def eef_pose(self):
        return (0.0, 0.0, 0.0)

    def query_policy(self, n_chunks=1):
        self.calls.append(("query_policy", n_chunks))


def test_move_to_object_computes_delta_from_poses():
    ctx = _RecordingCtx()
    spec = TOOL_CATALOG["move_to_object"]
    spec.handler(ctx, obj="target", offset_z=0.02)
    assert ctx.calls == [("move_delta", 1.0, 2.0, 3.02)]


def test_lift_and_retreat_use_defaults():
    ctx = _RecordingCtx()
    TOOL_CATALOG["lift"].handler(ctx)
    TOOL_CATALOG["retreat"].handler(ctx)
    assert ctx.calls == [
        ("move_delta", 0.0, 0.0, 0.08),
        ("move_delta", -0.05, 0.0, 0.1),
    ]
