# RTC Tail-Delay 可执行实验计划（v1）

## 1. 实验目标

- 目标：验证在 **长尾推理延迟** 下，`realtime` 方法是否相对 `naive` 更稳；并验证“tail control”是否能进一步改善结果。
- 主问题：
  1. `long_tail` 分布下，`realtime` vs `naive` 的 solve/return/jerk 差异。
  2. `tail_controlled` 相比 `long_tail`，是否降低 `d_p99` 和违规率，并提升或保持 solve。

## 2. 成功标准（必须满足）

- `primary`: `returned_episode_solved`（越高越好）。
- `secondary`: `returned_episode_returns`、`max_ddelta_action`（jerk proxy，越低越好）。
- 通过判据（针对 `realtime`）：
  - 在 `tail_controlled` 下，相比 `long_tail`：
    - `d_p99` 下降；
    - `p_violate`（`d > H - s`）下降；
    - `returned_episode_solved` 不下降（允许波动 ±1 个标准差）。

## 3. 固定实验协议

- 关卡：先只跑 `worlds/l/grasp_easy.json`（单关卡快速闭环）。
- seeds：`5`（`0,1,2,3,4`）。
- 每次评估 rollout：`num_evals=256`（当前 4070 上 2048 易 OOM）。
- 对比方法：首轮只跑 `naive`、`realtime`。
- 模型：固定稳定小模型（避免 XLA crash）。
  - `action_chunk_size=4`
  - `channel_dim=64`
  - `channel_hidden_dim=128`
  - `token_hidden_dim=32`
  - `num_layers=2`

## 4. 运行环境约束

- 使用解释器：`./.conda/envs/rtc-kinetix/bin/python`
- 环境变量：
  - `CUDA_ROOT=/usr/local/cuda-12.9`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `WANDB_MODE=disabled`
- 不要强制设置 `JAX_PLATFORMS=gpu`（当前环境存在初始化异常风险）。

## 5. 实施步骤

## Step A: Baseline 可用性检查（必须先过）

- 输入 checkpoint：`logs-bc/dummy-0qgv560m/7/policies/worlds_l_grasp_easy.pkl`（或你指定的新 checkpoint）。
- 先在固定延迟 `d=0` 评估 `naive` 一次，确认非崩溃且能出 `results.csv`。
- 验收：
  - 产物存在：`eval_output_* / results.csv`、`preview.mp4`
  - 指标字段完整：`returned_episode_lengths/returns/solved`

## Step B: 实现 delay profile（代码改造）

- 文件：`src/eval_flow.py`
- 改造目标（最小可用）：
  1. 新增 `DelayProfileConfig`：
     - `mode: "fixed" | "mixture"`
     - `fixed_delay: int`
     - `p_spike: float`
     - `d_spike: int`
     - `base_delays: tuple[int, ...]`
     - `base_probs: tuple[float, ...]`
     - `tail_cap: int | None`
  2. `EvalConfig` 新增：
     - `delay_profile: DelayProfileConfig | None = None`
  3. 在 `execute_chunk` 内每次采样 `d_t`（`jax.random`），替代常量 `config.inference_delay`。
  4. `tail control` 首版只做硬截断：`d_t = min(d_t, tail_cap)`（若 `tail_cap` 非空）。
  5. 记录延迟统计中间量：每个 chunk 的 `d_t`。

## Step C: 新增统计输出（代码改造）

- 文件：`src/eval_flow.py`
- 在 `return_info` 增加字段：
  - `d_p50`, `d_p95`, `d_p99`
  - `p_violate`（定义：`d_t > action_chunk_size - execute_horizon`）
  - `max_delta_action`（`max ||a_t - a_{t-1}||`）
  - `max_ddelta_action`（`max ||a_t - 2a_{t-1} + a_{t-2}||`）

## Step D: 运行实验矩阵（首轮）

- profile 定义：
  - `long_tail`:
    - `base_delays=(0,1)`, `base_probs=(0.9,0.1)`
    - `p_spike=0.05`, `d_spike=4`
    - `tail_cap=None`
  - `tail_controlled`:
    - 同上，但 `tail_cap=2`

- 方法：
  - `naive`
  - `realtime`（`prefix_attention_schedule="exp"`）

- seeds：`0..4`

- 结果目录建议：
  - `exp_runs/tail_v1/<profile>/<method>/seed_<n>/results.csv`

## Step E: 汇总分析

- 汇总脚本（新建建议）：`exp/summarize_tail_v1.py`
- 生成：
  - `summary.csv`（每组 mean/std）
  - `summary.md`（结论表）

## 6. 命令模板

## 6.1 单次评估模板

```bash
export CUDA_ROOT=/usr/local/cuda-12.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

./.conda/envs/rtc-kinetix/bin/python src/eval_flow.py \
  --run-path logs-bc/<RUN_NAME> \
  --level-paths worlds/l/grasp_easy.json \
  --config.num-evals 256 \
  --config.num-flow-steps 5 \
  --config.model.action-chunk-size 4 \
  --config.model.channel-dim 64 \
  --config.model.channel-hidden-dim 128 \
  --config.model.token-hidden-dim 32 \
  --config.model.num-layers 2 \
  --output-dir exp_runs/tmp_eval
```

## 6.2 批量实验模板（建议写 shell 脚本）

- 新建：`exp/run_tail_v1.sh`
- 循环维度：
  - `profile in {long_tail, tail_controlled}`
  - `method in {naive, realtime}`
  - `seed in {0..4}`

## 7. 结果表结构（必须统一）

- 主表字段：
  - `profile,method,seed,level`
  - `returned_episode_solved`
  - `returned_episode_returns`
  - `returned_episode_lengths`
  - `d_p50,d_p95,d_p99`
  - `p_violate`
  - `max_delta_action,max_ddelta_action`

## 8. 风险与回退

- 风险 1：GPU OOM
  - 回退：`num_evals 256 -> 128`，保持其余不变。
- 风险 2：XLA 编译 crash
  - 回退：维持小模型配置，不切回默认大模型。
- 风险 3：solve 全 0
  - 回退：先只比较 `return + jerk + violation`，并补一轮更强 checkpoint。

## 9. 交付物清单

- 代码：
  - `src/eval_flow.py`（delay profile + 统计）
  - `exp/run_tail_v1.sh`
  - `exp/summarize_tail_v1.py`
- 结果：
  - `exp_runs/tail_v1/.../results.csv`
  - `exp_runs/tail_v1/summary.csv`
  - `exp_runs/tail_v1/summary.md`

## 10. 执行顺序（一天内最小闭环）

1. Step A baseline 可用性检查（30 分钟）
2. Step B+C 代码改造（2-3 小时）
3. Step D 首轮矩阵跑完（2-4 小时，取决于显卡负载）
4. Step E 汇总与结论（30-60 分钟）
