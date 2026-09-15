"""Stage 3: Diagnose (LLM). Explanation only -- it never writes or executes
code (PLAN.md §3). Input is a text summary of telemetry/milestones, not
images, unless manifest.yaml's llm.supports_images is confirmed true and
prompts/diagnose.md is updated to include them (PLAN.md Context).
"""
from __future__ import annotations

import json

from evolution.models import Diagnosis, FailureCluster, Trajectory
from llm.client import LLMClient


def summarize_trajectory(traj: Trajectory, label: str) -> str:
    lines = [f"## {label} (seed={traj.seed}, success={traj.success})"]
    seen: set = set()
    for s in traj.steps:
        for m in s.milestones_reached:
            if m not in seen:
                seen.add(m)
                lines.append(f"- step {s.step}: milestone `{m}` reached")
    if traj.steps:
        last = traj.steps[-1]
        lines.append(f"- final step {last.step}: features={json.dumps(last.features)}")
    else:
        lines.append("- no steps recorded")
    return "\n".join(lines)


def diagnose(
    llm: LLMClient,
    cluster: FailureCluster,
    representative: Trajectory,
    reference_success: Trajectory,
    prompt_template: str,
) -> Diagnosis:
    user_prompt = prompt_template.format(
        task_id=cluster.task_id,
        first_missing_milestone=cluster.first_missing_milestone,
        n_failures=len(cluster.seeds),
        failure_summary=summarize_trajectory(representative, "Representative failure"),
        success_summary=summarize_trajectory(reference_success, "Successful reference"),
    )
    raw = llm.complete(
        system=(
            "You are a diagnostic agent for a robot manipulation policy. "
            "Explain the ONE most likely observable causal failure mechanism. "
            "Do not propose code, tools, or fixes -- that is a separate stage."
        ),
        user=user_prompt,
        json_mode=True,
    )
    data = json.loads(raw)
    return Diagnosis(
        cluster_id=f"{cluster.task_id}:{cluster.first_missing_milestone}",
        mechanism=data["mechanism"],
        first_missing_milestone=data.get("first_missing_milestone", cluster.first_missing_milestone),
        divergence_step=int(data["divergence_step"]),
        evidence=data.get("evidence", ""),
    )
