"""Check theorem-to-implementation correspondence for strict proof claims."""

import argparse
import json
import re


def has(path, patterns):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    return all(re.search(p, txt) for p in patterns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_json', type=str, default='theorem_coverage_check.json')
    args = p.parse_args()

    checks = {
        'theorem1_parametric_contraction_impl': has(
            'mcts_solver.py',
            [
                r'def _validate_theorem_assumptions',
                r'gamma in \[0,1\)',
                r'def _theory_operator',
                r'theory_parametric_convex_violation_count',
                r'objective_mode == "parametric"',
            ],
        ),
        'theorem2_conservative_order_impl': has(
            'mcts_solver.py',
            [
                r'objective_mode == "conservative"',
                r'conservative_parent = min',
                r'theory_conservative_order_violation_count',
            ],
        ),
        'theorem3_dag_equivalence_impl': has(
            'mcts_node.py',
            [r'def canonical_state', r'Two states are equivalent iff'],
        ) and has(
            'mcts_solver.py',
            [r'def _check_dag_equivalence', r'dag_equivalence_violation_count', r'child_hash in self.node_table'],
        ),
        'theorem_calibration_flags_impl': has(
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
