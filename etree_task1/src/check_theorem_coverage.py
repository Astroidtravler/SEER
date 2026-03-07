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
        'theorem4_calibration_flags_impl': has(
            'arguments.py',
            [r'mcts_calibration_mode', r'mcts_ece_bins', r'mcts_calibration_min_t', r'mcts_calibration_max_t'],
        ),
        'theorem5_budget_lagrangian_impl': has(
            'arguments.py',
            [r'mcts_budget_mode', r'mcts_budget_value', r'mcts_theory5_lambda'],
        ) and has(
            'mcts_solver.py',
            [r'def _theory5_cost', r'theory5_lagrangian', r'theory5_feasible_indicator'],
        ),
        'theorem6_structure_lockin_impl': has(
            'arguments.py',
            [r'mcts_theory6_rho', r'mcts_theory6_pmin'],
        ) and has(
            'mcts_solver.py',
            [
                r'def _action_coverage_key',
                r'def _record_action_coverage',
                r'theory6_coverage_rate_qhat',
                r'theory6_lockin_upper_bound_prod',
                r'theory6_lockin_upper_bound_minq',
            ],
        ),
        'innovation_merge_measurement_impl': has(
            'arguments.py',
            [
                r'mcts_merge_paradigm',
                r'mcts_measurement_model',
                r'mcts_self_consistency_samples',
                r'mcts_gls_mode',
            ],
        ) and has(
            'mcts_solver.py',
            [
                r'def _accept_merge_by_hypothesis_test',
                r'def _aggregate_measurements',
                r'merge_hypothesis_accept_count',
                r'self_consistency_mean_std',
            ],
        ),
        'innovation_q_controller_closed_loop_impl': has(
            'arguments.py',
            [
                r'mcts_q_controller',
                r'mcts_q_target',
                r'mcts_q_min_candidates',
                r'mcts_q_max_candidates',
                r'mcts_q_explore_prob',
            ],
        ) and has(
            'mcts_solver.py',
            [
                r'def _q_controller_expand_policy',
                r'generate_actions\(',
                r'q_controller_expand_candidates_mean',
                r'q_controller_diverse_expand_count',
            ],
        ),
        'innovation_gls_search_layer_impl': has(
            'mcts_solver.py',
            [
                r'gls_parent_applied_count',
                r'gls_backup_applied_count',
                r'def _aggregate_parent_value',
                r'def _backup_aggregate',
            ],
        ),
        'innovation_2plus2_strong_alignment_impl': has(
            'mcts_node.py',
            [r'"depth"', r'"trace"', r'"latest"'],
        ) and has(
            'mcts_solver.py',
            [r'def _action_coverage_key', r'logical_parent_ids'],
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
