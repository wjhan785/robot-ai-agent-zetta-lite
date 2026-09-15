"""End-to-end test of the composite `evolve` stage against a hand-crafted
critic that IS engineered to pass every gate -- unlike the CLI's
`--backend local` dry run (which uses a fixed canned response regardless
of the data and typically gets rejected, see evolution/campaign.py's
_LOCAL_* constants and evolution/rollout.py's fake_rollout docstring),
this proves the pipeline CAN approve a genuinely good bundle end to end.
"""
import json
import shutil
from pathlib import Path

from evolution.campaign import CampaignState, campaign_dir, save_trajectories, stage_cluster, stage_evolve
from evolution.rollout import fake_rollout
from llm.client import FakeLLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS_PER_GAP = 3  # must match fake_rollout's default


def _config():
    return {
        "campaign": {"name": "test-campaign"},
        "tasks": [{
            "id": "task_a", "suite": "libero_10", "name": "X",
            "instruction": "x", "milestones": ["a", "b", "c"],
        }],
        "seeds": {"total_per_task": 30, "held_out": {"start": 0, "end": 9}, "dev": {"start": 10, "end": 29}},
        "llm": {"provider": "deepseek", "model": "deepseek-chat", "supports_images": False, "max_propose_rounds": 3},
        "gates": {
            "shadow_replay": {"min_detect_rate": 0.8, "max_false_trigger_rate": 0.0},
            "same_seed": {"min_rescue_rate": 0.5},
            "regression": {"allow_success_drop": 0},
            "held_out": {"min_gain": 1, "mcnemar_alpha": 0.025},
        },
    }


def test_full_local_campaign_round_approves_a_good_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPO_ROOT / "prompts", tmp_path / "prompts")

    config = _config()
    task = config["tasks"][0]
    base = campaign_dir(config)

    trajectories = fake_rollout(
        task_id=task["id"], seeds=list(range(10, 30)), ordered_milestones=task["milestones"],
        library_version="v0", base_success_rate=0.5, rng_seed=1,
    )
    save_trajectories(base, task["id"], "v0", trajectories)
    assert any(t.success for t in trajectories)
    assert any(not t.success for t in trajectories)

    clusters = stage_cluster(config)
    assert clusters, "fake_rollout(rng_seed=1) should produce at least one failure cluster"

    diagnose_response = json.dumps({
        "mechanism": "arm stalled mid-approach",
        "first_missing_milestone": clusters[0].first_missing_milestone,
        "divergence_step": 0,
        "evidence": "steps_since_progress kept climbing without ever resetting",
    })
    # >= STEPS_PER_GAP is the exact boundary fake_rollout's synthetic data
    # supports: a success's steps_since_progress never exceeds
    # STEPS_PER_GAP-1 (it resets at every milestone), while a failure's
    # stall block starts at exactly STEPS_PER_GAP -- right at
    # divergence_step -- so this fires at or before divergence on every
    # failure and never on a success. See evolution/rollout.py's
    # fake_rollout docstring for why a laxer threshold (e.g. 5) would NOT
    # satisfy the "at or before divergence" check.
    propose_response = json.dumps({
        "id": "stall-fix",
        "critic": ["steps_since_progress", ">=", STEPS_PER_GAP],
        "recovery": [{"tool": "lift", "args": {"dz": 0.05}}],
        "reentry": ["steps_since_progress", "<", STEPS_PER_GAP],
    })
    llm = FakeLLMClient([diagnose_response, propose_response])

    decision = stage_evolve(config, backend="local", llm=llm)

    assert decision is not None
    assert decision.approved, decision.reason
    assert (tmp_path / "skills" / "v1" / "stall-fix.json").exists()
    assert (tmp_path / "skills" / "v1" / "decision.json").exists()

    state = CampaignState.load(base / "state.json")
    assert state.library_version == "v1"
    assert clusters[0].first_missing_milestone in state.completed_clusters


def test_evolve_is_a_no_op_once_every_cluster_is_completed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPO_ROOT / "prompts", tmp_path / "prompts")

    config = _config()
    task = config["tasks"][0]
    base = campaign_dir(config)
    save_trajectories(base, task["id"], "v0", [])  # no trajectories at all -> no clusters
    stage_cluster(config)

    decision = stage_evolve(config, backend="local", llm=FakeLLMClient(["{}"]))
    assert decision is None
