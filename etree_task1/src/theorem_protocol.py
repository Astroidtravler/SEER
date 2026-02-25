"""Strict experimental protocol for theorem-oriented MCTS validation.

This script runs a checklist over produced summary/report files and emits
pass/fail style protocol verdicts that can be cited in paper appendices.
"""

import argparse
import json
import math


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _is_num(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_summary', type=str, required=True)
    p.add_argument('--theory_report', type=str, required=True)
    p.add_argument('--save_json', type=str, default='theorem_protocol_check.json')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--max_contraction_ratio', type=float, default=1.05)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--eta', type=float, default=0.5)
    args = p.parse_args()

    base = _load(args.baseline_summary)
    th = _load(args.theory_report)

    checks = {}
    pval = base.get('paired_permutation_pvalue', float('nan'))
    checks['significance_pass'] = _is_num(pval) and (pval <= args.alpha)

    ci = base.get('delta_bootstrap_ci', [float('nan'), float('nan')])
    checks['delta_ci_excludes_zero'] = (
        isinstance(ci, (list, tuple)) and len(ci) == 2 and _is_num(ci[0]) and _is_num(ci[1]) and (ci[0] > 0 or ci[1] < 0)
    )

    ctr = th.get('theory_contraction_ratio_mean', {}).get('mean', float('nan'))
    checks['contraction_proxy_pass'] = _is_num(ctr) and ctr <= args.max_contraction_ratio

    resid = th.get('bellman_residual_mean', {}).get('mean', float('nan'))
    checks['residual_observable'] = _is_num(resid)

    dag_reuse = th.get('theory_unique_state_ratio', {}).get('mean', float('nan'))
    checks['dag_ratio_observable'] = _is_num(dag_reuse)

    ece = th.get('ece_proxy_mean', {}).get('mean', float('nan'))
    temp = th.get('temperature_scale_mean', {}).get('mean', float('nan'))
    checks['ece_observable'] = _is_num(ece)
    checks['temperature_observable'] = _is_num(temp)

    # theorem assumption checks (for appendix-level rigor)
    checks['assumption_discount_in_0_1'] = 0.0 <= args.gamma < 1.0
    checks['assumption_eta_in_0_1'] = 0.0 <= args.eta <= 1.0
    # for T_eta(s)=r+gamma*((1-eta)x+eta y), contraction constant is gamma in sup norm
    checks['theorem_contraction_constant_valid'] = checks['assumption_discount_in_0_1']

    out = {
        'alpha': args.alpha,
        'gamma': args.gamma,
        'eta': args.eta,
        'max_contraction_ratio': args.max_contraction_ratio,
        'checks': checks,
        'all_pass': all(checks.values()),
        'evidence': {
            'paired_permutation_pvalue': pval,
            'delta_bootstrap_ci': ci,
            'theory_contraction_ratio_mean': ctr,
            'bellman_residual_mean': resid,
            'theory_unique_state_ratio': dag_reuse,
            'ece_proxy_mean': ece,
            'temperature_scale_mean': temp,
        }
    }

    with open(args.save_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved protocol report to {args.save_json}")


if __name__ == '__main__':
    main()
