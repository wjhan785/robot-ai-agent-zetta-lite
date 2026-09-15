from evolution.cluster import cluster_failures
from evolution.models import StepRecord, Trajectory

MILESTONES = ["a", "b", "c"]


def _traj(seed: int, success: bool, reached: list) -> Trajectory:
    steps = [
        StepRecord(step=i * 10, features={}, action={}, milestones_reached=[m])
        for i, m in enumerate(reached)
    ]
    return Trajectory(task_id="t", seed=seed, library_version="v0", success=success, steps=steps)


def test_clusters_group_by_first_missing_milestone():
    trajectories = [
        _traj(1, False, ["a"]),          # missing "b"
        _traj(2, False, ["a"]),          # missing "b"
        _traj(3, False, ["a", "b"]),     # missing "c"
        _traj(4, True, ["a", "b", "c"]),  # success, excluded
    ]
    clusters = cluster_failures(trajectories, MILESTONES)
    by_milestone = {c.first_missing_milestone: c for c in clusters}
    assert set(by_milestone["b"].seeds) == {1, 2}
    assert by_milestone["c"].seeds == [3]


def test_largest_cluster_first():
    trajectories = [
        _traj(1, False, ["a"]),
        _traj(2, False, ["a"]),
        _traj(3, False, ["a"]),
        _traj(4, False, ["a", "b"]),
    ]
    clusters = cluster_failures(trajectories, MILESTONES)
    assert clusters[0].first_missing_milestone == "b"  # 3 seeds > 1 seed
    assert len(clusters[0].seeds) == 3


def test_failures_that_reached_everything_are_skipped():
    trajectories = [_traj(1, False, ["a", "b", "c"])]  # failed despite reaching all milestones
    clusters = cluster_failures(trajectories, MILESTONES)
    assert clusters == []


def test_representative_seed_is_a_cluster_member():
    trajectories = [_traj(1, False, ["a"]), _traj(2, False, ["a"]), _traj(3, False, ["a"])]
    clusters = cluster_failures(trajectories, MILESTONES)
    assert clusters[0].representative_seed in clusters[0].seeds
