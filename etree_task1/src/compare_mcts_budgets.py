"""Run budget-matched ablations for MCTS variants with multi-seed aggregation.

Example:
python compare_mcts_budgets.py \
  --base_cmd "python train.py --solver_backend mcts --do_train False --do_dev True --do_test False" \
  --simulations 50 --seeds 42,43,44
"""

import argparse
import csv
import json
import os
import subprocess
from statistics import mean, pstdev
import math



def normal_approx_ci(values, alpha=0.05):
    if not values:
        return (float("nan"), float("nan"))
    mu = mean(values)
    if len(values) < 2:
        return (mu, mu)
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    se = pstdev(values) / math.sqrt(len(values))
    return (mu - z * se, mu + z * se)

def run_cmd(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def load_metric(output_dir: str) -> float:
    score_path = os.path.join(output_dir, "epoch_tree", "dev", "mcts_eval", "scores-dev.metrics.json")
    if not os.path.exists(score_path):
        return float("nan")
    with open(score_path, "r") as f:
        metric = json.load(f)
    return metric.get("proof-overall", {}).get("acc", float("nan"))


def load_theory_stats(output_dir: str):
    pred_path = os.path.join(output_dir, "epoch_tree", "dev", "mcts_prediction_dev.jsonl")
    if not os.path.exists(pred_path):
        return {
            "bellman_residual_mean": float("nan"),
            "ucb_calibration_mean": float("nan"),
            "weighted_parent_var_proxy_mean": float("nan"),
            "ece_proxy_mean": float("nan"),
            "temperature_scale_mean": float("nan"),
        }
    vals = {
        "bellman_residual_mean": [],
        "ucb_calibration_mean": [],
        "weighted_parent_var_proxy_mean": [],
        "ece_proxy_mean": [],
        "temperature_scale_mean": [],
    }
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            s = item.get("search_stats", {})
            for k in vals:
                v = s.get(k, None)
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
    out = {}
    for k, arr in vals.items():
        out[k] = mean(arr) if arr else float("nan")
    return out


def latest_output_dir(out_root: str) -> str:
    task_dirs = [os.path.join(out_root, d) for d in os.listdir(out_root)]
    task_dirs = [d for d in task_dirs if os.path.isdir(d)]
    return sorted(task_dirs, key=os.path.getmtime)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cmd", type=str, required=True)
    parser.add_argument("--result_csv", type=str, default="mcts_budget_compare.csv")
    parser.add_argument("--result_json", type=str, default="mcts_budget_compare.json")
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--out_root", type=str, default="../../output_dir/etree_task1/test")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    variants = [
        ("mcts_tree", "--mcts_strict_dag_backprop False --mcts_weighted_return False --mcts_entropy_ucb False"),
        ("mcts_dag", "--mcts_strict_dag_backprop True --mcts_weighted_return False --mcts_entropy_ucb False"),
        ("mcts_dag_weighted", "--mcts_strict_dag_backprop True --mcts_weighted_return True --mcts_entropy_ucb False"),
        ("mcts_full", "--mcts_strict_dag_backprop True --mcts_weighted_return True --mcts_entropy_ucb True"),
    ]

    per_run_rows = []
    summary = []

    for name, extra in variants:
        vals = []
        residual_vals, calib_vals, varproxy_vals = [], [], []
        ece_vals, temp_vals = [], []
        for seed in seeds:
            cmd = f"{args.base_cmd} --seed {seed} --mcts_simulations {args.simulations} {extra}"
            run_cmd(cmd)
            latest = latest_output_dir(args.out_root)
            acc = load_metric(latest)
            th = load_theory_stats(latest)
            vals.append(acc)
            residual_vals.append(th["bellman_residual_mean"])
            calib_vals.append(th["ucb_calibration_mean"])
            varproxy_vals.append(th["weighted_parent_var_proxy_mean"])
            ece_vals.append(th["ece_proxy_mean"])
            temp_vals.append(th["temperature_scale_mean"])
            per_run_rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "simulations": args.simulations,
                    "proof_overall_acc": acc,
                    "bellman_residual_mean": th["bellman_residual_mean"],
                    "ucb_calibration_mean": th["ucb_calibration_mean"],
                    "weighted_parent_var_proxy_mean": th["weighted_parent_var_proxy_mean"],
                    "ece_proxy_mean": th["ece_proxy_mean"],
                    "temperature_scale_mean": th["temperature_scale_mean"],
                    "output_dir": latest,
                }
            )

        ci_low, ci_high = normal_approx_ci(vals, alpha=args.alpha)
        summary.append(
            {
                "variant": name,
                "simulations": args.simulations,
                "num_seeds": len(vals),
                "alpha": args.alpha,
                "acc_mean": mean(vals) if vals else float("nan"),
                "acc_std": pstdev(vals) if len(vals) > 1 else 0.0,
                "acc_ci_low": ci_low,
                "acc_ci_high": ci_high,
                "acc_values": vals,
                "bellman_residual_mean": mean(residual_vals) if residual_vals else float("nan"),
                "ucb_calibration_mean": mean(calib_vals) if calib_vals else float("nan"),
                "weighted_parent_var_proxy_mean": mean(varproxy_vals) if varproxy_vals else float("nan"),
                "ece_proxy_mean": mean(ece_vals) if ece_vals else float("nan"),
                "temperature_scale_mean": mean(temp_vals) if temp_vals else float("nan"),
            }
        )

    with open(args.result_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "simulations",
                "proof_overall_acc",
                "bellman_residual_mean",
                "ucb_calibration_mean",
                "weighted_parent_var_proxy_mean",
                "ece_proxy_mean",
                "temperature_scale_mean",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(per_run_rows)

    with open(args.result_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {args.result_csv}")
    print(f"Saved: {args.result_json}")


if __name__ == "__main__":
    main()
