# Zetta-Lite

A scaled-down, from-scratch reimplementation of the ideas in
[Zetta-Embodiment](https://github.com/air-embodied-brain/Zetta-Embodiment)
([paper, arXiv:2608.16590](https://arxiv.org/abs/2608.16590)): a frozen VLA
(Pi0.5) wrapped by small, LLM-written runtime critics and recovery
programs, which only join a growing skill library after passing a series
of offline and live gates. FYP learning project — see **[PLAN.md](PLAN.md)**
for the full design, scope decisions, and rationale. This file is just a
quickstart.

## Local dev (this PC)

```bash
conda activate learning
pip install -r requirements.txt
pytest -q
```

Every test is CPU-only and runs in well under a second — they cover
`runtime/` (critic, role1, recovery, loop), `evolution/` (cluster, shadow
replay, gates, propose/diagnose against a `FakeLLMClient`, schema
validation, promote, and a full end-to-end `evolve` round against
synthetic data), `env/milestones.py`, and `infra/batcher.py`. Nothing in
`tests/` imports LIBERO, openpi, or Ray.

To exercise the whole pipeline's plumbing without a real sim or API key:

```bash
python -m evolution.campaign --backend local --policy mock --stage rollout --config configs/manifest.yaml
python -m evolution.campaign --backend local --stage cluster --config configs/manifest.yaml
python -m evolution.campaign --backend local --stage evolve  --config configs/manifest.yaml
```

This uses synthetic trajectories (`evolution/rollout.py`'s `fake_rollout`)
and a canned LLM response — it's a smoke test that nothing crashes and
`state.json`/`runs/` get written correctly, **not** a demonstration that a
bundle gets approved (the canned response is static regardless of the
data, so `evolve` typically ends in a clean, expected rejection after 5
rounds — see the docstrings in `evolution/rollout.py` and
`evolution/campaign.py` for exactly why, and `tests/test_campaign.py` for
a hand-crafted example that _is_ engineered to pass every gate).

## On Remote Workstation

```bash
sbatch slurm/setup_env.sh                      # once
sbatch --job-name=rollout slurm/gpu_job.sh rollout
sbatch --job-name=evolve  slurm/gpu_job.sh evolve
```

See [PLAN.md §7](PLAN.md#7-remote-execution-via-slurm-tc1) for the QoS
limits, cluster etiquette, and job templates.

## Layout

| Path                         | What                                                                           | Testable here?                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `runtime/`                   | Loop 1: critic, role1, recovery, tools, loop                                   | ✅                                                                                |
| `evolution/`                 | Loops 2-3: cluster, diagnose, propose, shadow_replay, gates, promote, campaign | ✅ (real env/policy paths need TC1)                                               |
| `env/`, `policy/`            | LIBERO + Pi0.5 wrappers                                                        | `milestones.py`/`mock_policy.py` only — `libero_env.py`/`pi05_policy.py` need TC1 |
| `infra/`                     | Ray env/policy workers + dynamic batcher (PLAN.md §4)                          | `batcher.py` only                                                                 |
| `llm/`                       | DeepSeek client + `FakeLLMClient` for tests                                    | ✅                                                                                |
| `slurm/`                     | `sbatch` templates                                                             | —                                                                                 |
| `configs/manifest.yaml`      | The one frozen config for a campaign                                           | —                                                                                 |
| `schemas/bundle.schema.json` | What a proposed bundle must look like                                          | —                                                                                 |
| `skills/v0/`                 | The empty starting skill library                                               | —                                                                                 |

Known TODOs (flagged inline in the relevant file, not guessed at):
`env/libero_env.py`'s `object_pose()`, exact LIBERO task-name strings in
`configs/manifest.yaml`, and openpi server wiring in `policy/pi05_policy.py`
— all need a real LIBERO/openpi install on TC1 to verify, not further
guessing offline (PLAN.md §8 Weeks 1-2).
