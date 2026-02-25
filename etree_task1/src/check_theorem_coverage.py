"""Check whether code paths corresponding to Theorem 1/2/3 are implemented."""

import argparse
import json
import re


def has(path, patterns):
    txt = open(path, 'r', encoding='utf-8').read()
    return all(re.search(p, txt) for p in patterns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_json', type=str, default='theorem_coverage_check.json')
    args = p.parse_args()

    checks = {
        'theorem1_weighted_return_impl': has(
            'mcts_solver.py',
            [r'weight_mode', r'inv_var', r'weighted_parent_var_proxy_mean', r'objective_mode == "conservative"'],
        ),
        'theorem2_parametric_operator_impl': has(
            'mcts_solver.py',
            [r'def _theory_operator', r'mcts_theory_eta', r'objective_mode == "parametric"'],
        ),
        'theorem3_calibrated_entropy_ucb_impl': has(
            'mcts_solver.py',
            [r'def _compute_ece_proxy', r'def _calibrated_entropy_scale', r'calibration_mode == "ece_temp"'],
        ),
        'theorem3_cli_flags_impl': has(
            'arguments.py',
            [r'mcts_calibration_mode', r'mcts_ece_bins', r'mcts_calibration_min_t', r'mcts_calibration_max_t'],
        ),
    }

    out = {
        'checks': checks,
        'all_pass': all(checks.values()),
    }
    with open(args.save_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved theorem coverage check to {args.save_json}")


if __name__ == '__main__':
    main()
