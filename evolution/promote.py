"""Stage 9: approve/reject a candidate bundle based on all gate results
(PLAN.md §3). Approved bundles are copied into skills/v{n+1}/; rejected
ones are reported so stage 4 can retry with feedback."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from evolution.models import Decision, GateResult


def decide(cluster_id: str, bundle_id: str, gates: List[GateResult]) -> Decision:
    approved = all(g.passed for g in gates)
    failed = [g.name for g in gates if not g.passed]
    reason = "all gates passed" if approved else f"failed gate(s): {', '.join(failed)}"
    return Decision(cluster_id=cluster_id, bundle_id=bundle_id, approved=approved, gates=gates, reason=reason)


def promote(
    bundle: Dict[str, Any],
    decision: Decision,
    skills_dir: Path,
    from_version: str,
    to_version: str,
) -> Path:
    if not decision.approved:
        raise ValueError(f"refusing to promote a rejected bundle: {decision.reason}")

    src = skills_dir / from_version
    dst = skills_dir / to_version
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for item in src.glob("*.json"):
            if item.name != "decision.json":
                shutil.copy2(item, dst / item.name)

    (dst / f"{bundle['id']}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (dst / "decision.json").write_text(
        json.dumps(
            {
                "cluster_id": decision.cluster_id,
                "bundle_id": decision.bundle_id,
                "reason": decision.reason,
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in decision.gates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dst
