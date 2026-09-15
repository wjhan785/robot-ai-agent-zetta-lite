import json
from pathlib import Path

import pytest

from evolution.diagnose import diagnose, summarize_trajectory
from evolution.models import FailureCluster, StepRecord, Trajectory
from evolution.propose import ProposalError, propose
from llm.client import FakeLLMClient

DIAGNOSE_PROMPT = Path("prompts/diagnose.md").read_text(encoding="utf-8")
PROPOSE_PROMPT = Path("prompts/propose.md").read_text(encoding="utf-8")


def _traj(seed: int, success: bool) -> Trajectory:
    steps = [
        StepRecord(step=0, features={"x": 0}, action={}, milestones_reached=["a"]),
        StepRecord(step=10, features={"x": 1}, action={}, milestones_reached=(["b"] if success else [])),
    ]
    return Trajectory(task_id="task_a", seed=seed, library_version="v0", success=success, steps=steps)


def test_summarize_trajectory_lists_milestones_in_order():
    text = summarize_trajectory(_traj(1, True), "label")
    assert "milestone `a` reached" in text
    assert "milestone `b` reached" in text


def test_diagnose_parses_llm_json_into_a_diagnosis():
    cluster = FailureCluster(task_id="task_a", first_missing_milestone="b", seeds=[1, 2], representative_seed=1)
    response = json.dumps({
        "mechanism": "the gripper opened too early",
        "first_missing_milestone": "b",
        "divergence_step": 5,
        "evidence": "gripper_open flips true before lift completes",
    })
    llm = FakeLLMClient([response])
    result = diagnose(llm, cluster, _traj(1, False), _traj(2, True), DIAGNOSE_PROMPT)

    assert result.mechanism == "the gripper opened too early"
    assert result.divergence_step == 5
    assert result.cluster_id == "task_a:b"
    # the prompt actually got the cluster's task_id and milestone substituted in
    assert "task_a" in llm.calls[0]["user"]
    assert "`b`" in llm.calls[0]["user"]


VALID_PROPOSE_RESPONSE = json.dumps({
    "id": "bundle-1",
    "critic": ["x", ">=", 5],
    "recovery": [{"tool": "lift", "args": {"dz": 0.05}}],
    "reentry": ["x", "<", 5],
})


def _diagnosis():
    from evolution.models import Diagnosis

    return Diagnosis(cluster_id="task_a:b", mechanism="m", first_missing_milestone="b", divergence_step=5, evidence="e")


def test_propose_returns_a_validated_bundle():
    llm = FakeLLMClient([VALID_PROPOSE_RESPONSE])
    bundle = propose(llm, _diagnosis(), PROPOSE_PROMPT, round_id=1)
    assert bundle["id"] == "bundle-1"


def test_propose_raises_on_malformed_json():
    llm = FakeLLMClient(["not json at all"])
    with pytest.raises(ProposalError):
        propose(llm, _diagnosis(), PROPOSE_PROMPT, round_id=1)


def test_propose_raises_on_unknown_tool():
    bad = json.dumps({
        "id": "bundle-1",
        "critic": ["x", ">=", 5],
        "recovery": [{"tool": "teleport", "args": {}}],
        "reentry": ["x", "<", 5],
    })
    llm = FakeLLMClient([bad])
    with pytest.raises(ProposalError):
        propose(llm, _diagnosis(), PROPOSE_PROMPT, round_id=1)


def test_propose_feeds_previous_feedback_into_the_prompt():
    llm = FakeLLMClient([VALID_PROPOSE_RESPONSE])
    propose(llm, _diagnosis(), PROPOSE_PROMPT, round_id=2, previous_feedback="round 1 used an unknown tool")
    assert "round 1 used an unknown tool" in llm.calls[0]["user"]
