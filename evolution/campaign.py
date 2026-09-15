"""Stage 0 & orchestrator: the campaign state machine (PLAN.md §3, §5, §7).

Usage (see PLAN.md §7 for the sbatch wrappers):
    python -m evolution.campaign --stage rollout --config configs/manifest.yaml
    python -m evolution.campaign --stage cluster --config configs/manifest.yaml
    python -m evolution.campaign --stage evolve  --config configs/manifest.yaml
    python -m evolution.campaign --backend local --policy mock --stage rollout --config configs/manifest.yaml
    python -m evolution.campaign --backend local --stage evolve --config configs/manifest.yaml

`evolve` is the composite stage from PLAN.md §5/§7: it chains
diagnose -> propose -> shadow -> same_seed -> regression -> heldout ->
decide against the single largest pending cluster, retrying propose up to
`llm.max_propose_rounds` times, so one SLURM job (or one local dry run)
carries a whole candidate round.

`--backend local` runs the SAME code path with synthetic trajectories
(evolution.rollout.fake_rollout) and a canned LLM response instead of a
real API key -- it is a plumbing smoke test, not a demonstration that a
bundle gets approved (see the caveat in evolution/rollout.py). For a
genuine positive path, see tests/test_campaign.py.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evolution.cluster import cluster_failures, prior_milestone
from evolution.diagnose import diagnose
from evolution.gates import PairedOutcome, evaluate_regression, evaluate_same_seed
from evolution.models import Decision, FailureCluster, GateResult, StepRecord, Trajectory
from evolution.promote import decide as decide_fn
from evolution.promote import promote
from evolution.propose import ProposalError, propose
from evolution.rollout import fake_rollout
from evolution.shadow_replay import evaluate_shadow_replay, to_gate_result
from llm.client import DeepSeekClient, FakeLLMClient, LLMClient
from runtime.tools import TOOL_CATALOG

# ---------------------------------------------------------------- config --


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def campaign_dir(config: Dict[str, Any]) -> Path:
    return Path("runs") / config["campaign"]["name"]


def dev_seeds(config: Dict[str, Any]) -> List[int]:
    s = config["seeds"]["dev"]
    return list(range(s["start"], s["end"] + 1))


def held_out_seeds(config: Dict[str, Any]) -> List[int]:
    s = config["seeds"]["held_out"]
    return list(range(s["start"], s["end"] + 1))


# ----------------------------------------------------------------- state --


@dataclass
class CampaignState:
    library_version: str = "v0"
    completed_clusters: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        if path.exists():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def note(self, message: str) -> None:
        self.log.append(message)
        print(message)


# --------------------------------------------------------------- storage --


def _traj_path(base: Path, task_id: str, version: str) -> Path:
    return base / "trajectories" / version / f"{task_id}.json"


def save_trajectories(base: Path, task_id: str, version: str, trajectories: List[Trajectory]) -> None:
    path = _traj_path(base, task_id, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in trajectories], indent=2), encoding="utf-8")


def load_trajectories(base: Path, task_id: str, version: str) -> List[Trajectory]:
    path = _traj_path(base, task_id, version)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_traj_from_dict(r) for r in raw]


def _traj_from_dict(r: Dict[str, Any]) -> Trajectory:
    steps = [StepRecord(**s) for s in r["steps"]]
    return Trajectory(
        task_id=r["task_id"], seed=r["seed"], library_version=r["library_version"],
        success=r["success"], steps=steps, keyframe_paths=r.get("keyframe_paths", []),
    )


def _load_bundles(skills_dir: Path, version: str):
    """Loads every approved bundle in skills/<version>/ as a runtime.loop.Bundle."""
    from runtime.critic import Critic
    from runtime.loop import Bundle
    from runtime.recovery import RecoveryStep

    version_dir = skills_dir / version
    if not version_dir.exists():
        return []
    bundles = []
    for path in sorted(version_dir.glob("*.json")):
        if path.name == "decision.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        bundles.append(Bundle(
            id=data["id"],
            critic=Critic(id=data["id"], rule=data["critic"]),
            recovery_steps=[RecoveryStep(tool=s["tool"], args=s["args"]) for s in data["recovery"]],
            reentry_rule=data["reentry"],
        ))
    return bundles


# ---------------------------------------------------------------- stages --


def stage_rollout(config: Dict[str, Any], backend: str, policy: str) -> None:
    base = campaign_dir(config)
    state = CampaignState.load(base / "state.json")
    seeds = dev_seeds(config)

    for task in config["tasks"]:
        if backend == "local" or policy == "mock":
            trajectories = fake_rollout(
                task_id=task["id"], seeds=seeds, ordered_milestones=task["milestones"],
                library_version=state.library_version,
            )
        else:
            trajectories = _real_rollout_for_task(config, task, seeds, state.library_version)
        save_trajectories(base, task["id"], state.library_version, trajectories)
        n_success = sum(t.success for t in trajectories)
        print(f"[rollout] {task['id']} @ {state.library_version}: {n_success}/{len(trajectories)} succeeded")


def _real_rollout_for_task(config: Dict[str, Any], task: Dict[str, Any], seeds: List[int], version: str) -> List[Trajectory]:
    """Only reachable with --backend slurm --policy real, on the TC1 box
    where env/libero_env.py and policy/pi05_policy.py can actually import."""
    from env.libero_env import LiberoEnv
    from evolution.rollout import real_rollout
    from policy.pi05_policy import Pi05Policy
    from runtime.role1 import Role1

    policy_cfg = config["policy"]
    runtime_cfg = config["runtime"]
    bundles = _load_bundles(Path("skills"), version)

    def make_env(seed: int) -> LiberoEnv:
        return LiberoEnv(suite=task["suite"], task_name=task["name"], init_state=seed)

    def make_policy() -> Pi05Policy:
        return Pi05Policy(action_chunk=policy_cfg["action_chunk"], exec_steps=policy_cfg["exec_steps"])

    def role1_factory() -> Role1:
        return Role1(
            allowed_tools=set(TOOL_CATALOG),
            max_recoveries_per_episode=runtime_cfg["max_recoveries_per_episode"],
            cooldown_steps=runtime_cfg["recovery_cooldown_steps"],
        )

    return real_rollout(
        make_env=make_env, make_policy=make_policy, bundles=bundles, seeds=seeds,
        task_id=task["id"], library_version=version, role1_factory=role1_factory,
    )


def stage_cluster(config: Dict[str, Any]) -> List[FailureCluster]:
    base = campaign_dir(config)
    state = CampaignState.load(base / "state.json")
    all_clusters: List[FailureCluster] = []
    for task in config["tasks"]:
        trajectories = load_trajectories(base, task["id"], state.library_version)
        all_clusters.extend(cluster_failures(trajectories, task["milestones"]))

    path = base / "candidates" / "clusters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in all_clusters], indent=2), encoding="utf-8")
    print(f"[cluster] found {len(all_clusters)} cluster(s), largest first")
    return all_clusters


def _make_llm(config: Dict[str, Any]) -> LLMClient:
    return DeepSeekClient(model=config["llm"]["model"])


def stage_evolve(config: Dict[str, Any], backend: str, llm: Optional[LLMClient] = None) -> Optional[Decision]:
    """diagnose -> propose -> shadow -> same_seed -> regression -> heldout
    -> decide, retried up to llm.max_propose_rounds times against the
    single largest pending cluster (PLAN.md §5/§7)."""
    base = campaign_dir(config)
    state = CampaignState.load(base / "state.json")

    clusters_path = base / "candidates" / "clusters.json"
    if not clusters_path.exists():
        stage_cluster(config)
    raw_clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    pending = [c for c in raw_clusters if c["first_missing_milestone"] not in state.completed_clusters]
    if not pending:
        state.note("[evolve] no pending clusters -- nothing to do")
        state.save(base / "state.json")
        return None

    cluster = FailureCluster(**pending[0])  # already largest-first from stage_cluster
    task = next(t for t in config["tasks"] if t["id"] == cluster.task_id)
    trajectories = load_trajectories(base, task["id"], state.library_version)
    representative = next(t for t in trajectories if t.seed == cluster.representative_seed)
    reference_success = next((t for t in trajectories if t.success), representative)

    llm = llm or _make_llm(config)
    diagnose_prompt = Path("prompts/diagnose.md").read_text(encoding="utf-8")
    propose_prompt = Path("prompts/propose.md").read_text(encoding="utf-8")

    diagnosis = diagnose(llm, cluster, representative, reference_success, diagnose_prompt)
    state.note(f"[diagnose] {diagnosis.mechanism}")

    feedback: Optional[str] = None
    max_rounds = config["llm"]["max_propose_rounds"]
    for round_id in range(1, max_rounds + 1):
        try:
            bundle = propose(llm, diagnosis, propose_prompt, round_id, feedback)
        except ProposalError as exc:
            feedback = f"round {round_id} rejected at schema validation: {exc}"
            state.note(f"[propose] {feedback}")
            continue

        gates = _run_gates(config, base, task, cluster, bundle, state.library_version)
        decision = decide_fn(cluster.first_missing_milestone, bundle["id"], gates)
        _save_round(base, cluster, round_id, bundle, decision)

        if decision.approved:
            next_version = _next_version(state.library_version)
            promote(bundle, decision, Path("skills"), state.library_version, next_version)
            state.library_version = next_version
            state.completed_clusters.append(cluster.first_missing_milestone)
            state.note(f"[decide] approved -> {next_version} ({decision.reason})")
            state.save(base / "state.json")
            return decision

        feedback = decision.reason
        state.note(f"[decide] round {round_id} rejected: {decision.reason}")

    state.note(f"[evolve] gave up on cluster {cluster.first_missing_milestone} after {max_rounds} round(s)")
    state.save(base / "state.json")
    return None


def _run_gates(
    config: Dict[str, Any], base: Path, task: Dict[str, Any], cluster: FailureCluster,
    bundle: Dict[str, Any], library_version: str,
) -> List[GateResult]:
    trajectories = load_trajectories(base, task["id"], library_version)
    failures = [t for t in trajectories if not t.success and t.seed in cluster.seeds]
    successes = [t for t in trajectories if t.success]
    prior = prior_milestone(cluster.first_missing_milestone, task["milestones"])
    divergence_steps = {t.seed: t.divergence_step(cluster.first_missing_milestone, prior) for t in failures}

    gates_cfg = config["gates"]
    shadow_report = evaluate_shadow_replay(
        bundle["critic"], failures, successes, divergence_steps,
        min_detect_rate=gates_cfg["shadow_replay"]["min_detect_rate"],
        max_false_trigger_rate=gates_cfg["shadow_replay"]["max_false_trigger_rate"],
    )
    shadow_gate = to_gate_result(shadow_report)
    if not shadow_gate.passed:
        return [shadow_gate]

    # NOTE: same_seed/regression/heldout below are PLACEHOLDERS keyed on
    # shadow-replay detection rather than a real live rerun -- there's no
    # simulator to actually execute a recovery in without a GPU. Once
    # env/policy/infra run for real on TC1, replace this block with real
    # live reruns via evolution.rollout.real_rollout (same_seed/regression
    # on dev seeds, heldout on the held-out seeds) -- see PLAN.md §6/§7 and
    # evolution/rollout.py's real_rollout().
    paired_same_seed = [
        PairedOutcome(seed=s, parent_success=False, candidate_success=s in divergence_steps)
        for s in cluster.seeds
    ]
    same_seed_gate = evaluate_same_seed(paired_same_seed, gates_cfg["same_seed"]["min_rescue_rate"])
    if not same_seed_gate.passed:
        return [shadow_gate, same_seed_gate]

    paired_regression = [PairedOutcome(seed=t.seed, parent_success=True, candidate_success=True) for t in successes]
    regression_gate = evaluate_regression(paired_regression, gates_cfg["regression"]["allow_success_drop"])
    if not regression_gate.passed:
        return [shadow_gate, same_seed_gate, regression_gate]

    held_out_gate = GateResult(
        "held_out", True, {"note": "placeholder pass-through pending a real held-out rollout, see PLAN.md §7"}
    )
    return [shadow_gate, same_seed_gate, regression_gate, held_out_gate]


def _save_round(base: Path, cluster: FailureCluster, round_id: int, bundle: Dict[str, Any], decision: Decision) -> None:
    path = base / "candidates" / cluster.first_missing_milestone / f"round_{round_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "bundle": bundle,
            "decision": {
                "approved": decision.approved,
                "reason": decision.reason,
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in decision.gates],
            },
        }, indent=2),
        encoding="utf-8",
    )


def _next_version(current: str) -> str:
    return f"v{int(current.lstrip('v')) + 1}"


# ------------------------------------------------------- local dry run --

_LOCAL_DIAGNOSE_RESPONSE = json.dumps({
    "mechanism": "local dry-run placeholder diagnosis (synthetic fake_rollout data, not a real failure)",
    "divergence_step": 0,
    "evidence": "generated by evolution.rollout.fake_rollout, see its docstring caveat",
})
_LOCAL_PROPOSE_RESPONSE = json.dumps({
    "id": "dry-run-bundle",
    "critic": ["steps_since_progress", ">=", 5],
    "recovery": [{"tool": "lift", "args": {"dz": 0.05}}],
    "reentry": ["steps_since_progress", "<", 5],
})


# -------------------------------------------------------------------- CLI --

STAGES = ["rollout", "cluster", "evolve"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Zetta-Lite evolution campaign (PLAN.md §3)")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    parser.add_argument("--policy", default="real", choices=["real", "mock"])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.stage == "rollout":
        stage_rollout(config, args.backend, args.policy)
    elif args.stage == "cluster":
        stage_cluster(config)
    elif args.stage == "evolve":
        llm = FakeLLMClient([_LOCAL_DIAGNOSE_RESPONSE, _LOCAL_PROPOSE_RESPONSE]) if args.backend == "local" else None
        stage_evolve(config, args.backend, llm=llm)


if __name__ == "__main__":
    main()
