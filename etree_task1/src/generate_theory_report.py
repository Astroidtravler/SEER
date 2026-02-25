"""Generate paper-oriented theory report from MCTS prediction outputs.

Reads mcts_prediction_{split}.jsonl and summarizes theorem-aligned indicators:
- Bellman residual statistics (contraction proxy)
- eta-contraction gap and conservative-gap
- calibration scaling statistics
- graph state reuse ratio
"""

import argparse
import json
import math
from statistics import mean, pstdev


def _safe_stats(vals):
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "count": len(vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--save_json", type=str, default="theory_report.json")
    args = parser.parse_args()

    metrics = {
        "bellman_residual_mean": [],
        "bellman_residual_max": [],
        "theory_contraction_ratio_mean": [],
        "ucb_calibration_mean": [],
        "theory_eta_contraction_gap_mean": [],
        "theory_conservative_gap_mean": [],
        "theory_unique_state_ratio": [],
        "weighted_parent_var_proxy_mean": [],
        "ece_proxy_mean": [],
        "temperature_scale_mean": [],
        "dag_equivalence_violation_count": [],
        "dag_equivalence_check_count": [],
    }

    with open(args.prediction_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            s = item.get("search_stats", {})
            for k in metrics:
                v = s.get(k, None)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    metrics[k].append(float(v))

    report = {k: _safe_stats(v) for k, v in metrics.items()}

    # Simple theorem-facing checklist
    report["checks"] = {
        "residual_bounded_signal": report["bellman_residual_mean"]["mean"],
        "contraction_ratio_signal": report["theory_contraction_ratio_mean"]["mean"],
        "eta_gap_signal": report["theory_eta_contraction_gap_mean"]["mean"],
        "calibration_signal": report["ucb_calibration_mean"]["mean"],
        "ece_signal": report["ece_proxy_mean"]["mean"],
        "temperature_signal": report["temperature_scale_mean"]["mean"],
        "dag_reuse_signal": report["theory_unique_state_ratio"]["mean"],
        "dag_equivalence_violation": report["dag_equivalence_violation_count"]["mean"],
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved theory report to {args.save_json}")


if __name__ == "__main__":
    main()
