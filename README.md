# exp-delta: RTC Tail-Delay Experiment Repo

This repository is an experiment-focused fork for evaluating **real-time chunking (RTC)** under **long-tail inference delay** and **tail-control** settings.

Base project context:
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)

## What Is Added In This Repo

- `src/eval_flow.py` now supports:
  - per-chunk sampled delay profiles (`fixed` / `mixture`)
  - optional tail cap (`tail_cap`)
  - extra metrics:
    - `d_p50`, `d_p95`, `d_p99`
    - `p_violate`
    - `max_delta_action`, `max_ddelta_action`
- Experiment scripts:
  - `exp/run_stepA_baseline.py`
  - `exp/run_stepD_tail_v1.py`
  - `exp/summarize_tail_v1.py`
- Planning/report docs:
  - `exp/exp_draft.txt`
  - `exp/exp_plan_executable.md`
  - `exp_runs/tail_v1/report.md`

## Environment

This repo has been run with:
- Python: `./.conda/envs/rtc-kinetix/bin/python`
- GPU: NVIDIA RTX 4070
- CUDA toolkit path: `/usr/local/cuda-12.9`

Recommended runtime env vars:

```bash
export CUDA_ROOT=/usr/local/cuda-12.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
```

Quick device check:

```bash
nvidia-smi
./.conda/envs/rtc-kinetix/bin/python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

## Quick Start (Current Experimental Workflow)

## Step A: Baseline evaluation sanity check

Runs single-config baseline (`naive`, `d=0`, `execute_horizon=1`) and writes:
- `exp_runs/stepA_baseline/results.csv`
- `exp_runs/stepA_baseline/preview.mp4`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/run_stepA_baseline.py
```

## Step D: Tail experiment matrix

Runs:
- profiles: `long_tail`, `tail_controlled`
- methods: `naive`, `realtime`
- seeds: `0..4`

Outputs:
- per-run csv files under `exp_runs/tail_v1/<profile>/<method>/seed_<n>/results.csv`
- merged table: `exp_runs/tail_v1/results_all.csv`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/run_stepD_tail_v1.py
```

## Step E: Aggregation and report tables

Outputs:
- `exp_runs/tail_v1/summary.csv`
- `exp_runs/tail_v1/summary.md`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/summarize_tail_v1.py
```

## Current Main Result Snapshot

See `exp_runs/tail_v1/report.md` for full interpretation. In short:
- Tail control reduces delay-tail risk (`d_p99: 4 -> 2`, `p_violate: ~0.04 -> 0`).
- `realtime` is more stable than `naive` on return and smoothness proxies.
- `solved` is still `0` in this checkpoint regime.

## Directory Guide

- `src/`: training/eval code (`eval_flow.py` contains delay profile support)
- `exp/`: reproducible experiment scripts and planning docs
- `exp_runs/`: generated experiment outputs
- `logs-expert/`, `logs-bc/`: checkpoints and training artifacts

## Known Caveats

- In some environments, forcing `JAX_PLATFORMS=gpu` can fail; prefer automatic backend selection.
- `num_evals=2048` may OOM on RTX 4070 for some configs; use `256` or `128` for iteration.
- Current model checkpoint quality limits solve rate interpretation (many runs remain `solved=0`).

## Upstream/Origin Layout

This repo is intended to be used with:
- `upstream`: original source repository
- `origin`: your experiment repository (`phi-media-lab/exp-delta`)

Typical sync flow:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```
