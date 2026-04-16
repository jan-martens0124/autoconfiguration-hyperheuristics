#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

SEEDS = [1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239]
INSTANCES = list(range(9))

COMMANDS = {
    "RNRunner": [
        "java",
        "-jar",
        "./RNRunner.jar",
        "2",
        "2024",
        "{seed}",
        "{instance}",
        "-d",
        "0.9297",
        "0.4394",
        "0.9845",
        "-i",
        "0.3162",
        "0.8701",
        "0.444",
        "-t",
        "10000",
    ],
    "CCFRunner-multi": [
        "java",
        "-jar",
        "./CCFRunner-multi.jar",
        "2",
        "2024",
        "{seed}",
        "{instance}",
        "-d",
        "0.9297",
        "0.4394",
        "0.9845",
        "-i",
        "0.3162",
        "0.8701",
        "0.444",
        "-p0.3393",
        "-t",
        "10000",
    ],
}


def run_once(algorithm: str, seed: int, instance: int) -> float:
    cmd = [part.format(seed=seed, instance=instance) for part in COMMANDS[algorithm]]
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"No output for {algorithm}, seed={seed}, instance={instance}")
    return float(output.splitlines()[-1].strip())


def summarize(rows):
    by_algo_instance = {}
    by_algo = {}

    for row in rows:
        ai_key = (row["algorithm"], row["instance"])
        by_algo_instance.setdefault(ai_key, []).append(row["objective"])
        by_algo.setdefault(row["algorithm"], []).append(row["objective"])

    per_instance = []
    for (algorithm, instance), values in sorted(by_algo_instance.items(), key=lambda x: (x[0][0], x[0][1])):
        per_instance.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    overall = []
    for algorithm, values in sorted(by_algo.items()):
        overall.append(
            {
                "algorithm": algorithm,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    return per_instance, overall


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(raw_rows, per_instance_rows, output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    instances = sorted({row["instance"] for row in per_instance_rows})

    rn_means = [next(r["mean"] for r in per_instance_rows if r["algorithm"] == "RNRunner" and r["instance"] == i) for i in instances]
    rn_stds = [next(r["std"] for r in per_instance_rows if r["algorithm"] == "RNRunner" and r["instance"] == i) for i in instances]
    ccf_means = [next(r["mean"] for r in per_instance_rows if r["algorithm"] == "CCFRunner-multi" and r["instance"] == i) for i in instances]
    ccf_stds = [next(r["std"] for r in per_instance_rows if r["algorithm"] == "CCFRunner-multi" and r["instance"] == i) for i in instances]

    ax1.errorbar(instances, rn_means, yerr=rn_stds, marker="o", capsize=3, label="RNRunner")
    ax1.errorbar(instances, ccf_means, yerr=ccf_stds, marker="s", capsize=3, label="CCFRunner-multi")
    ax1.set_title("Mean objective per instance (8 seeds)")
    ax1.set_xlabel("Instance")
    ax1.set_ylabel("Objective (lower is better)")
    ax1.legend()

    rn_all = [r["objective"] for r in raw_rows if r["algorithm"] == "RNRunner"]
    ccf_all = [r["objective"] for r in raw_rows if r["algorithm"] == "CCFRunner-multi"]
    box = ax2.boxplot([rn_all, ccf_all], tick_labels=["RNRunner", "CCFRunner-multi"], patch_artist=True)
    box["boxes"][0].set_facecolor("#4C78A8")
    box["boxes"][1].set_facecolor("#F58518")
    ax2.set_title("Overall distribution across all runs")
    ax2.set_ylabel("Objective (lower is better)")

    fig.suptitle("Algorithm Comparison: RNRunner vs CCFRunner-multi", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)


def main():
    out_dir = BASE_DIR / "comparison_results"
    out_dir.mkdir(exist_ok=True)

    raw_rows = []
    total = len(COMMANDS) * len(SEEDS) * len(INSTANCES)
    done = 0

    for algorithm in ["RNRunner", "CCFRunner-multi"]:
        for seed in SEEDS:
            for instance in INSTANCES:
                value = run_once(algorithm, seed, instance)
                raw_rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "instance": instance,
                        "objective": value,
                    }
                )
                done += 1
                print(f"[{done:3}/{total}] {algorithm} seed={seed} instance={instance} -> {value:.6f}", flush=True)

    per_instance_rows, overall_rows = summarize(raw_rows)

    write_csv(
        out_dir / "raw_results.csv",
        raw_rows,
        ["algorithm", "seed", "instance", "objective"],
    )
    write_csv(
        out_dir / "summary_by_instance.csv",
        per_instance_rows,
        ["algorithm", "instance", "mean", "std", "min", "max"],
    )
    write_csv(
        out_dir / "summary_overall.csv",
        overall_rows,
        ["algorithm", "mean", "std", "min", "max"],
    )

    make_plot(raw_rows, per_instance_rows, out_dir / "comparison_plot.png")

    print("\nCompleted.")
    print(f"Results folder: {out_dir}")


if __name__ == "__main__":
    main()
