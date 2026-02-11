import csv
import math
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path("exp_runs/tail_v1/results_all.csv")
OUTPUT_CSV = Path("exp_runs/tail_v1/summary.csv")
OUTPUT_MD = Path("exp_runs/tail_v1/summary.md")

METRICS = [
    "returned_episode_solved",
    "returned_episode_returns",
    "returned_episode_lengths",
    "d_p50",
    "d_p95",
    "d_p99",
    "p_violate",
    "max_delta_action",
    "max_ddelta_action",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def main():
    rows = list(csv.DictReader(INPUT_PATH.open()))
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["profile"], row["method"])
        for metric in METRICS:
            grouped[key][metric].append(float(row[metric]))

    summary_rows: list[dict[str, str]] = []
    for (profile, method), metrics in sorted(grouped.items()):
        out: dict[str, str] = {"profile": profile, "method": method, "n": str(len(metrics[METRICS[0]]))}
        for metric in METRICS:
            vals = metrics[metric]
            out[f"{metric}_mean"] = f"{mean(vals):.6f}"
            out[f"{metric}_std"] = f"{std(vals):.6f}"
        summary_rows.append(out)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    by_key = {(r["profile"], r["method"]): r for r in summary_rows}

    def val(profile: str, method: str, metric: str) -> float:
        return float(by_key[(profile, method)][f"{metric}_mean"])

    lines = [
        "# Tail v1 Summary",
        "",
        f"- source: `{INPUT_PATH}`",
        f"- runs: `{len(rows)}`",
        "",
        "## Group Means ± Std",
        "",
        "| profile | method | solved | return | length | d_p99 | p_violate | max_ddelta_action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            "| "
            f"{r['profile']} | {r['method']} | "
            f"{r['returned_episode_solved_mean']} ± {r['returned_episode_solved_std']} | "
            f"{r['returned_episode_returns_mean']} ± {r['returned_episode_returns_std']} | "
            f"{r['returned_episode_lengths_mean']} ± {r['returned_episode_lengths_std']} | "
            f"{r['d_p99_mean']} ± {r['d_p99_std']} | "
            f"{r['p_violate_mean']} ± {r['p_violate_std']} | "
            f"{r['max_ddelta_action_mean']} ± {r['max_ddelta_action_std']} |"
        )

    lines += [
        "",
        "## Key Deltas (mean)",
        "",
        f"- Tail control effect on `d_p99` (naive): "
        f"{val('long_tail', 'naive', 'd_p99'):.3f} -> {val('tail_controlled', 'naive', 'd_p99'):.3f}",
        f"- Tail control effect on `p_violate` (naive): "
        f"{val('long_tail', 'naive', 'p_violate'):.6f} -> {val('tail_controlled', 'naive', 'p_violate'):.6f}",
        f"- Tail control effect on `d_p99` (realtime): "
        f"{val('long_tail', 'realtime', 'd_p99'):.3f} -> {val('tail_controlled', 'realtime', 'd_p99'):.3f}",
        f"- Tail control effect on `p_violate` (realtime): "
        f"{val('long_tail', 'realtime', 'p_violate'):.6f} -> {val('tail_controlled', 'realtime', 'p_violate'):.6f}",
        f"- Realtime vs naive under long_tail (`return`): "
        f"{val('long_tail', 'naive', 'returned_episode_returns'):.6f} -> "
        f"{val('long_tail', 'realtime', 'returned_episode_returns'):.6f}",
        f"- Realtime vs naive under tail_controlled (`return`): "
        f"{val('tail_controlled', 'naive', 'returned_episode_returns'):.6f} -> "
        f"{val('tail_controlled', 'realtime', 'returned_episode_returns'):.6f}",
    ]

    OUTPUT_MD.write_text("\n".join(lines))
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
