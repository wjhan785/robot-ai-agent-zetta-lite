import json

import pytest

from evolution.models import GateResult
from evolution.promote import decide, promote


def test_decide_approves_when_all_gates_pass():
    gates = [GateResult("shadow_replay", True), GateResult("same_seed", True)]
    decision = decide("cluster-1", "bundle-1", gates)
    assert decision.approved
    assert decision.reason == "all gates passed"


def test_decide_rejects_and_names_failed_gates():
    gates = [GateResult("shadow_replay", True), GateResult("same_seed", False)]
    decision = decide("cluster-1", "bundle-1", gates)
    assert not decision.approved
    assert "same_seed" in decision.reason


def test_promote_refuses_a_rejected_decision(tmp_path):
    decision = decide("cluster-1", "bundle-1", [GateResult("shadow_replay", False)])
    with pytest.raises(ValueError):
        promote({"id": "bundle-1"}, decision, tmp_path, "v0", "v1")


def test_promote_writes_bundle_and_decision_and_carries_forward_prior_version(tmp_path):
    (tmp_path / "v0").mkdir()
    (tmp_path / "v0" / "old-bundle.json").write_text(json.dumps({"id": "old-bundle"}), encoding="utf-8")

    decision = decide("cluster-1", "new-bundle", [GateResult("shadow_replay", True)])
    dst = promote({"id": "new-bundle", "critic": []}, decision, tmp_path, "v0", "v1")

    assert dst == tmp_path / "v1"
    assert (dst / "old-bundle.json").exists()   # carried forward from v0
    assert (dst / "new-bundle.json").exists()   # the newly approved bundle
    assert (dst / "decision.json").exists()

    written = json.loads((dst / "new-bundle.json").read_text(encoding="utf-8"))
    assert written["id"] == "new-bundle"
