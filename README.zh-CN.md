# exp-delta：RTC 尾延迟实验仓库

[English](README.md) | [中文](README.zh-CN.md)

本仓库是一个面向实验的方法学分支，主要用于评估 **Real-Time Chunking (RTC)** 在**长尾推理延迟**与**尾延迟控制（tail control）**场景下的表现。

基础论文背景：
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)

## 本仓库的新增内容

- `src/eval_flow.py` 已支持：
  - 按 chunk 采样延迟 profile（`fixed` / `mixture`）
  - 可选延迟硬截断（`tail_cap`）
  - 新增统计指标：
    - `d_p50`, `d_p95`, `d_p99`
    - `p_violate`
    - `max_delta_action`, `max_ddelta_action`
- 实验脚本：
  - `exp/run_stepA_baseline.py`
  - `exp/run_stepD_tail_v1.py`
  - `exp/summarize_tail_v1.py`
- 规划与报告：
  - `exp/exp_draft.txt`
  - `exp/exp_plan_executable.md`
  - `exp_runs/tail_v1/report.md`

## 运行环境

当前已验证环境：
- Python：`./.conda/envs/rtc-kinetix/bin/python`
- GPU：NVIDIA RTX 4070
- CUDA 路径：`/usr/local/cuda-12.9`

推荐环境变量：

```bash
export CUDA_ROOT=/usr/local/cuda-12.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
```

快速确认设备：

```bash
nvidia-smi
./.conda/envs/rtc-kinetix/bin/python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

## 快速开始（当前实验流程）

## Step A：基线评估可用性检查

运行单配置 baseline（`naive`, `d=0`, `execute_horizon=1`），输出：
- `exp_runs/stepA_baseline/results.csv`
- `exp_runs/stepA_baseline/preview.mp4`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/run_stepA_baseline.py
```

## Step D：尾延迟实验矩阵

运行维度：
- profile：`long_tail`, `tail_controlled`
- method：`naive`, `realtime`
- seed：`0..4`

输出：
- 每次运行结果：`exp_runs/tail_v1/<profile>/<method>/seed_<n>/results.csv`
- 合并结果：`exp_runs/tail_v1/results_all.csv`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/run_stepD_tail_v1.py
```

## Step E：结果汇总与报告表

输出：
- `exp_runs/tail_v1/summary.csv`
- `exp_runs/tail_v1/summary.md`

```bash
./.conda/envs/rtc-kinetix/bin/python exp/summarize_tail_v1.py
```

## 当前结果快照

完整解释见：`exp_runs/tail_v1/report.md`。当前主要结论：
- tail control 能显著降低尾部延迟风险（`d_p99: 4 -> 2`，`p_violate: ~0.04 -> 0`）。
- `realtime` 在 return 与平滑性 proxy 上优于 `naive`。
- 当前 checkpoint 范围下 `solved` 仍为 `0`。

## 目录说明

- `src/`：训练与评估代码（`eval_flow.py` 含延迟 profile 支持）
- `exp/`：可复现实验脚本与实验规划文档
- `exp_runs/`：实验输出与汇总结果
- `logs-expert/`, `logs-bc/`：训练和 checkpoint 产物

## 已知注意事项

- 部分环境中强制 `JAX_PLATFORMS=gpu` 可能初始化失败，建议使用自动后端选择。
- 在 RTX 4070 上，部分配置 `num_evals=2048` 可能 OOM；实验迭代建议 `256` 或 `128`。
- 当前策略强度限制了 `solved` 的解释力度（可能长期为 0）。

## 远程仓库协作建议（origin / upstream）

建议保持：
- `upstream`：原始上游仓库
- `origin`：你自己的实验仓库（如 `phi-media-lab/exp-delta`）

同步上游常用命令：

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```
