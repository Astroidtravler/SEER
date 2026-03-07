"""Compare PPO baseline and MCTS variants under aligned budgets.

This script launches commands provided by user and aggregates official dev metrics.
"""

import argparse
import json
import os
import subprocess
from statistics import mean, pstdev
import math
import random


def normal_approx_ci(values, alpha=0.05):
    if not values:
        return (float("nan"), float("nan"))
    mu = mean(values)
    if len(values) < 2:
        return (mu, mu)
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    se = pstdev(values) / math.sqrt(len(values))
    return (mu - z * se, mu + z * se)


def paired_t_like_stat(a, b):
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b)]
    mu = mean(diffs)
    sd = pstdev(diffs)
    if sd == 0:
        return float("inf") if mu > 0 else float("-inf")
    return mu / (sd / math.sqrt(len(diffs)))


def paired_bootstrap_ci(a, b, alpha=0.05, iters=2000, seed=42):
    if len(a) != len(b) or len(a) == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    samples = []
    for _ in range(iters):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo = int((alpha / 2.0) * (iters - 1))
    hi = int((1.0 - alpha / 2.0) * (iters - 1))
    return (samples[lo], samples[hi])


def paired_permutation_pvalue(a, b, iters=5000, seed=42):
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(sum(diffs) / len(diffs))
    cnt = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / len(diffs)) >= obs:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def run_cmd(cmd: str):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def latest_output_dir(out_root: str) -> str:
    dirs = [os.path.join(out_root, d) for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
    return sorted(dirs, key=os.path.getmtime)[-1]


def load_dev_acc(run_dir: str, backend: str) -> float:
    if backend == "mcts":
        path = os.path.join(run_dir, "epoch_tree", "dev", "mcts_eval", "scores-dev.metrics.json")
    else:
        path = os.path.join(run_dir, "epoch_tree", "dev", "best_metrics.json")
    if not os.path.exists(path):
        return float("nan")
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("proof-overall", {}).get("acc", float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_cmd", type=str, required=True)
    parser.add_argument("--mcts_cmd", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--out_root", type=str, default="../../output_dir/etree_task1/test")
    parser.add_argument("--save_json", type=str, default="baseline_compare_summary.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--perm_iters", type=int, default=5000)
    parser.add_argument("--stat_seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    result = {"ppo": [], "mcts": []}

    for seed in seeds:
        ppo = f"{args.ppo_cmd} --seed {seed}"
        run_cmd(ppo)
        ppo_dir = latest_output_dir(args.out_root)
        result["ppo"].append(load_dev_acc(ppo_dir, backend="ppo"))

        mcts = f"{args.mcts_cmd} --seed {seed}"
        run_cmd(mcts)
        mcts_dir = latest_output_dir(args.out_root)
        result["mcts"].append(load_dev_acc(mcts_dir, backend="mcts"))

    ppo_ci = normal_approx_ci(result["ppo"], alpha=args.alpha)
    mcts_ci = normal_approx_ci(result["mcts"], alpha=args.alpha)
    summary = {
        "seeds": seeds,
        "alpha": args.alpha,
        "ppo": {"mean": mean(result["ppo"]), "std": pstdev(result["ppo"]) if len(seeds) > 1 else 0.0, "ci_low": ppo_ci[0], "ci_high": ppo_ci[1], "vals": result["ppo"]},
        "mcts": {"mean": mean(result["mcts"]), "std": pstdev(result["mcts"]) if len(seeds) > 1 else 0.0, "ci_low": mcts_ci[0], "ci_high": mcts_ci[1], "vals": result["mcts"]},
        "delta_mean": mean(result["mcts"]) - mean(result["ppo"]),
        "paired_t_stat": paired_t_like_stat(result["mcts"], result["ppo"]),
        "delta_bootstrap_ci": paired_bootstrap_ci(
            result["mcts"], result["ppo"], alpha=args.alpha, iters=args.bootstrap_iters, seed=args.stat_seed
        ),
        "paired_permutation_pvalue": paired_permutation_pvalue(
            result["mcts"], result["ppo"], iters=args.perm_iters, seed=args.stat_seed
        ),
    }

    with open(args.save_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
