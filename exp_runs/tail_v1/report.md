# RTC Tail v1 实验报告

## 1. 实验背景与目标

本实验面向 `real-time-chunking-kinetix` 的评估链路，验证以下两个问题：

1. 在**长尾推理延迟**下，`realtime` 相对 `naive` 是否更稳、更平滑。
2. 加入**tail control**（对延迟做上界控制）后，是否能显著降低尾部延迟风险指标。

本轮是方法学验证（Step D/E），重点是“延迟分布与控制指标是否按预期变化”，不是最终任务 solved SOTA。

---

## 2. 实验配置

### 2.1 代码与配置范围

- 评估实现：`src/eval_flow.py`
- 新增能力：
  - `DelayProfileConfig`（`fixed` / `mixture`）
  - per-chunk 延迟采样 `d_t`
  - `tail_cap`（硬截断）
  - 新指标：`d_p50/p95/p99`、`p_violate`、`max_delta_action`、`max_ddelta_action`

### 2.2 运行协议

- 关卡：`worlds/l/grasp_easy.json`
- checkpoint：`logs-bc/dummy-0qgv560m/7/policies/worlds_l_grasp_easy.pkl`
- seeds：`0,1,2,3,4`（共 5 个）
- `num_evals=256`
- `execute_horizon=1`
- 模型（稳定小模型）：
  - `action_chunk_size=4`
  - `channel_dim=64`
  - `channel_hidden_dim=128`
  - `token_hidden_dim=32`
  - `num_layers=2`

### 2.3 实验矩阵

- profiles:
  - `long_tail`: `p_spike=0.05`, `d_spike=4`, `base_delays=(0,1)`, `base_probs=(0.9,0.1)`, `tail_cap=None`
  - `tail_controlled`: 同上，但 `tail_cap=2`
- methods:
  - `naive`
  - `realtime`（`prefix_attention_schedule="exp"`）

总运行数：`2 profiles x 2 methods x 5 seeds = 20`

---

## 3. 输出产物

- 原始逐 run 结果：`exp_runs/tail_v1/<profile>/<method>/seed_<n>/results.csv`
- 汇总原表：`exp_runs/tail_v1/results_all.csv`
- 统计汇总：`exp_runs/tail_v1/summary.csv`
- 文本摘要：`exp_runs/tail_v1/summary.md`

---

## 4. 结果总览（5 seeds，mean ± std）

| profile | method | solved | return | length | d_p99 | p_violate | max_ddelta_action |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_tail | naive | 0.000000 ± 0.000000 | -0.024219 ± 0.009726 | 245.300000 ± 4.343926 | 4.000000 ± 0.000000 | 0.039844 ± 0.010482 | 4.492304 ± 0.022017 |
| long_tail | realtime | 0.000000 ± 0.000000 | -0.006250 ± 0.005241 | 253.191406 ± 1.074410 | 4.000000 ± 0.000000 | 0.039844 ± 0.010482 | 3.749981 ± 0.027765 |
| tail_controlled | naive | 0.000000 ± 0.000000 | -0.025000 ± 0.011254 | 245.363281 ± 4.181742 | 2.000000 ± 0.000000 | 0.000000 ± 0.000000 | 4.494214 ± 0.022169 |
| tail_controlled | realtime | 0.000000 ± 0.000000 | -0.005469 ± 0.004454 | 253.810156 ± 1.065575 | 2.000000 ± 0.000000 | 0.000000 ± 0.000000 | 3.743051 ± 0.019483 |

---

## 5. 结果解释

## 5.1 Tail control 是否有效：是

- `d_p99`：从 `4.0` 降到 `2.0`（naive/realtime 两组都成立）。
- `p_violate`：从 `0.039844` 降到 `0.0`（naive/realtime 两组都成立）。

解释：  
`tail_cap=2` 直接限制了延迟尾部，等价于把“高延迟 spike”裁剪掉，故尾部风险指标显著改善，且结果稳定（std=0）。

## 5.2 realtime 相对 naive 是否更稳：是

在 `long_tail` 下：
- return：`-0.024219 -> -0.006250`（提升）
- length：`245.30 -> 253.19`（更长）
- max_ddelta_action：`4.492304 -> 3.749981`（更平滑）

在 `tail_controlled` 下：
- return：`-0.025000 -> -0.005469`（提升）
- length：`245.36 -> 253.81`（更长）
- max_ddelta_action：`4.494214 -> 3.743051`（更平滑）

解释：  
在当前策略能力范围内，`realtime` 持续表现出更好的轨迹稳定性与回报表现，且这种优势在有无 tail control 的两种 profile 下都存在。

## 5.3 solved 仍为 0 的含义

- 四组 `returned_episode_solved` 全部为 `0`。
- 这说明当前 checkpoint 在该关卡上仍未达到解关阈值，但并不否定 tail 方法学结论。

本轮实验可支持的结论是：
- **延迟分布控制机制有效**（从指标层面）。
- **realtime 相对 naive 的稳定性优势成立**（在非解关 regime 下）。

本轮实验不能支持的结论是：
- “tail control 提升 solved”或“方法达到可解任务水平”。

---

## 6. 有效性与局限

1. 单关卡（`grasp_easy`）外推性有限。  
2. `num_evals=256` 是显存与稳定性折中，不是最高统计精度配置。  
3. 使用稳定小模型而非默认大模型，绝对性能不代表完整训练配置上限。  
4. solved 全 0 会限制对“任务完成率增益”的结论强度。

---

## 7. 结论

本次 Tail v1 实验达成了预期的工程与方法学目标：

- Step D 实验矩阵完整执行（20 runs）。
- Step C 指标新增生效并可复用。
- tail control 对尾部风险的抑制非常明确（`d_p99` 与 `p_violate` 同步改善）。
- realtime 在回报与平滑性上稳定优于 naive。

建议进入下一阶段：

1. 扩展到多关卡（至少 3-5 个 level）。
2. 提升策略强度（使用更强 checkpoint），把 solved 拉到非零后再复验 tail 结论。
3. 在 Step F 中增加 `hard_masking` / `bid`，并单独做 `adaptive execute_horizon` 消融。
