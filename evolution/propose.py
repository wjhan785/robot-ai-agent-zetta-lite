"""Stage 4: Propose (LLM). Produces a bundle JSON, validated against
schemas/bundle.schema.json and the tool allowlist before it's ever
accepted (PLAN.md §3)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

from evolution.models import Diagnosis
from llm.client import LLMClient
from runtime.tools import TOOL_CATALOG

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "bundle.schema.json"


class ProposalError(Exception):
    """Raised when the LLM's response isn't usable -- bad JSON, fails the
    bundle schema, or references a tool that isn't on the allowlist. The
    caller (evolution.campaign) feeds the message back as `previous_feedback`
    on the next round."""


def _load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def describe_tool_catalog() -> str:
    return "\n".join(f"- {name}: {spec.description} bounds={spec.bounds}" for name, spec in TOOL_CATALOG.items())


def propose(
    llm: LLMClient,
    diagnosis: Diagnosis,
    prompt_template: str,
    round_id: int,
    previous_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    user_prompt = prompt_template.format(
        mechanism=diagnosis.mechanism,
        first_missing_milestone=diagnosis.first_missing_milestone,
        divergence_step=diagnosis.divergence_step,
        evidence=diagnosis.evidence,
        tool_catalog=describe_tool_catalog(),
        round_id=round_id,
        previous_feedback=previous_feedback or "(first attempt for this cluster)",
    )
    raw = llm.complete(
        system=(
            "You are a repair-proposal agent. Output ONLY a single JSON object "
            "matching the given bundle schema -- one frozen critic+recovery "
            "bundle, nothing else."
        ),
        user=user_prompt,
        json_mode=True,
    )
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"response was not valid JSON: {exc}") from exc

    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: Dict[str, Any]) -> None:
    schema = _load_schema()
    try:
        jsonschema.validate(bundle, schema)
    except jsonschema.ValidationError as exc:
        raise ProposalError(f"bundle failed schema validation: {exc.message}") from exc

    for step in bundle["recovery"]:
        if step["tool"] not in TOOL_CATALOG:
            raise ProposalError(f"unknown tool {step['tool']!r}")
