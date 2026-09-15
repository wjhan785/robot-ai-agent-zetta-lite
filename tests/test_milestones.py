from env.milestones import MilestoneTracker, above_threshold, lift_above, near


def test_lift_above():
    check = lift_above("obj_z", 0.05)
    assert not check({"obj_z": 0.02})
    assert check({"obj_z": 0.10})


def test_near():
    check = near("obj_z", "target_z", tolerance=0.01)
    assert check({"obj_z": 0.50, "target_z": 0.505})
    assert not check({"obj_z": 0.50, "target_z": 0.60})


def test_above_threshold_wraps_eval_predicate():
    check = above_threshold("gripper_open", "==", False)
    assert check({"gripper_open": False})
    assert not check({"gripper_open": True})


def test_milestone_tracker_records_first_step_only():
    tracker = MilestoneTracker({"grasped": lift_above("obj_z", 0.05)})
    tracker.step(0, {"obj_z": 0.0})
    tracker.step(1, {"obj_z": 0.10})
    tracker.step(2, {"obj_z": 0.20})  # still above threshold, but already recorded
    assert tracker.reached_at == {"grasped": 1}


def test_milestone_tracker_first_missing():
    tracker = MilestoneTracker({"a": lambda f: True, "b": lambda f: False})
    tracker.step(0, {})
    assert tracker.first_missing(["a", "b", "c"]) == "b"


def test_milestone_tracker_reset_clears_progress():
    tracker = MilestoneTracker({"a": lambda f: True})
    tracker.step(0, {})
    assert "a" in tracker.reached_at
    tracker.reset()
    assert tracker.reached_at == {}
