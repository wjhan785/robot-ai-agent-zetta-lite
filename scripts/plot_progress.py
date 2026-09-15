"""Plot held-out success rate per skill library version (PLAN.md §8 Weeks
8-9: "plot success rate per library version, v0 -> v1 -> v2"). CPU-only
(reads JSON already produced by evolution.campaign); needs matplotlib.

Usage:
    python scripts/plot_progress.py --config configs/manifest.yaml --out runs/progress.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from evolution.campaign import campaign_dir, load_config


def collect_decisions(base: Path) -> List[Dict]:
    """Reads every skills/v*/decision.json in submission order (by
    version number) -- each one records the gate detail dict, which
    includes held-out gain/success counts once real held-out rollouts are
    wired in (PLAN.md §7 note in evolution/campaign.py's _run_gates)."""
    skills_dir = Path("skills")
    rows = []
    if not skills_dir.exists():
        return rows
    versions = sorted(
        (d for d in skills_dir.iterdir() if d.is_dir() and d.name.startswith("v")),
        key=lambda d: int(d.name.lstrip("v")),
    )
    for version_dir in versions:
        decision_path = version_dir / "decision.json"
        if decision_path.exists():
            data = json.loads(decision_path.read_text(encoding="utf-8"))
            rows.append({"version": version_dir.name, **data})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="runs/progress.png")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    base = campaign_dir(config)
    rows = collect_decisions(base)
    if not rows:
        print("no skills/v*/decision.json found yet -- run a campaign first (PLAN.md §7)")
        return

    import matplotlib.pyplot as plt

    versions = [r["version"] for r in rows]
    held_out_gains = []
    for r in rows:
        held_out = next((g for g in r["gates"] if g["name"] == "held_out"), None)
        held_out_gains.append(held_out["detail"].get("gain", 0) if held_out else 0)

    fig, ax = plt.subplots()
    ax.plot(versions, held_out_gains, marker="o")
    ax.set_xlabel("skill library version")
    ax.set_ylabel("held-out gain (episodes) vs. previous version")
    ax.set_title("Zetta-Lite evolution progress")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
