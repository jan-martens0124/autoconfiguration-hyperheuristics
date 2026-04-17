#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_raw_results(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "instance": int(row["instance"]),
                    "objective": float(row["objective"]),
                }
            )
    return rows


def pick_algorithms(rows):
    algos = sorted({row["algorithm"] for row in rows})
    if len(algos) != 2:
        raise ValueError(f"Expected exactly 2 algorithms, found {len(algos)}: {algos}")
    return algos


def build_instance_plots(raw_rows, output_dir: Path):
    algo_a, algo_b = pick_algorithms(raw_rows)
    instances = sorted({row["instance"] for row in raw_rows})
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    written = []

    for instance in instances:
        inst_rows = [r for r in raw_rows if r["instance"] == instance]
        a_rows = sorted([r for r in inst_rows if r["algorithm"] == algo_a], key=lambda r: r["seed"])
        b_rows = sorted([r for r in inst_rows if r["algorithm"] == algo_b], key=lambda r: r["seed"])

        a_seeds = [r["seed"] for r in a_rows]
        a_vals = [r["objective"] for r in a_rows]
        b_seeds = [r["seed"] for r in b_rows]
        b_vals = [r["objective"] for r in b_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        ax1.plot(a_seeds, a_vals, marker="o", linewidth=2, label=algo_a, color="#4C78A8")
        ax1.plot(b_seeds, b_vals, marker="s", linewidth=2, label=algo_b, color="#F58518")
        ax1.set_title(f"Instance {instance}: objective by seed")
        ax1.set_xlabel("Seed")
        ax1.set_ylabel("Objective (lower is better)")
        ax1.legend()

        box = ax2.boxplot([a_vals, b_vals], tick_labels=[algo_a, algo_b], patch_artist=True)
        box["boxes"][0].set_facecolor("#4C78A8")
        box["boxes"][1].set_facecolor("#F58518")
        ax2.set_title(f"Instance {instance}: distribution across seeds")
        ax2.set_ylabel("Objective (lower is better)")

        fig.suptitle(f"Algorithm Comparison for Instance {instance}", fontsize=14, fontweight="bold")
        out_file = output_dir / f"instance_{instance}_comparison.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        written.append(out_file)

    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one comparison plot per instance from a raw_results.csv file."
    )
    parser.add_argument("raw_results", type=Path, help="Path to raw_results.csv")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("comparison_plots_by_instance"),
        help="Output directory for per-instance images (default: comparison_plots_by_instance)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_raw_results(args.raw_results)
    if not rows:
        raise ValueError("Input CSV is empty.")
    written = build_instance_plots(rows, args.output_dir)
    print(f"Created {len(written)} plots in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
