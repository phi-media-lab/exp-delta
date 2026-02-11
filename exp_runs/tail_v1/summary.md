# Tail v1 Summary

- source: `exp_runs/tail_v1/results_all.csv`
- runs: `20`

## Group Means ± Std

| profile | method | solved | return | length | d_p99 | p_violate | max_ddelta_action |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_tail | naive | 0.000000 ± 0.000000 | -0.024219 ± 0.009726 | 245.300000 ± 4.343926 | 4.000000 ± 0.000000 | 0.039844 ± 0.010482 | 4.492304 ± 0.022017 |
| long_tail | realtime | 0.000000 ± 0.000000 | -0.006250 ± 0.005241 | 253.191406 ± 1.074410 | 4.000000 ± 0.000000 | 0.039844 ± 0.010482 | 3.749981 ± 0.027765 |
| tail_controlled | naive | 0.000000 ± 0.000000 | -0.025000 ± 0.011254 | 245.363281 ± 4.181742 | 2.000000 ± 0.000000 | 0.000000 ± 0.000000 | 4.494214 ± 0.022169 |
| tail_controlled | realtime | 0.000000 ± 0.000000 | -0.005469 ± 0.004454 | 253.810156 ± 1.065575 | 2.000000 ± 0.000000 | 0.000000 ± 0.000000 | 3.743051 ± 0.019483 |

## Key Deltas (mean)

- Tail control effect on `d_p99` (naive): 4.000 -> 2.000
- Tail control effect on `p_violate` (naive): 0.039844 -> 0.000000
- Tail control effect on `d_p99` (realtime): 4.000 -> 2.000
- Tail control effect on `p_violate` (realtime): 0.039844 -> 0.000000
- Realtime vs naive under long_tail (`return`): -0.024219 -> -0.006250
- Realtime vs naive under tail_controlled (`return`): -0.025000 -> -0.005469