import pytest

from evolution.propose import ProposalError, validate_bundle

VALID_BUNDLE = {
    "id": "bundle-1",
    "critic": ["steps_since_progress", ">=", 5],
    "recovery": [{"tool": "lift", "args": {"dz": 0.05}}],
    "reentry": ["steps_since_progress", "<", 5],
}


def test_valid_bundle_passes():
    validate_bundle(VALID_BUNDLE)  # should not raise


def test_valid_bundle_with_nested_rules():
    bundle = dict(VALID_BUNDLE)
    bundle["critic"] = {"all": [["a", ">", 0], {"any": [["b", "==", 1], ["b", "==", 2]]}]}
    validate_bundle(bundle)  # should not raise


def test_missing_required_field_rejected():
    bundle = {k: v for k, v in VALID_BUNDLE.items() if k != "reentry"}
    with pytest.raises(ProposalError):
        validate_bundle(bundle)


def test_unknown_top_level_field_rejected():
    bundle = dict(VALID_BUNDLE, extra_field="not allowed")
    with pytest.raises(ProposalError):
        validate_bundle(bundle)


def test_unknown_tool_name_rejected_by_schema():
    bundle = dict(VALID_BUNDLE, recovery=[{"tool": "teleport", "args": {}}])
    with pytest.raises(ProposalError):
        validate_bundle(bundle)


def test_bad_operator_rejected():
    bundle = dict(VALID_BUNDLE, critic=["x", "~=", 1])
    with pytest.raises(ProposalError):
        validate_bundle(bundle)


def test_recovery_step_missing_args_rejected():
    bundle = dict(VALID_BUNDLE, recovery=[{"tool": "lift"}])
    with pytest.raises(ProposalError):
        validate_bundle(bundle)
