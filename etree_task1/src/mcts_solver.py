import logging
import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple

from llm_engine import LLMEngine
from mcts_node import Action, MCTSNode

logger = logging.getLogger(__name__)


class MCTSSolver:
    def __init__(self, args):
        self.args = args
        self.max_simulations = getattr(args, "mcts_simulations", 50)
        self.gamma = getattr(args, "gamma", 0.95)
        self.c_puct = getattr(args, "c_puct", 1.5)
        self.max_depth = getattr(args, "mcts_max_depth", 8)
        self.expand_fallback_reward = getattr(args, "mcts_fallback_reward", -0.2)
        self.trace_topk = getattr(args, "mcts_trace_topk", 3)

        self.enable_weighted_return = getattr(args, "mcts_weighted_return", True)
        self.weight_mode = getattr(args, "mcts_weight_mode", "prior")
        self.enable_entropy_ucb = getattr(args, "mcts_entropy_ucb", True)
        self.entropy_coef = getattr(args, "mcts_entropy_coef", 1.0)
        self.enable_dag_backprop = getattr(args, "mcts_strict_dag_backprop", True)
        self.dag_update_mode = getattr(args, "mcts_dag_update_mode", "value_only")
        self.objective_mode = getattr(args, "mcts_objective_mode", "graph_td")
        self.entropy_source = getattr(args, "mcts_entropy_source", "hybrid")
        self.backup_operator = getattr(args, "mcts_backup_operator", "mean")
        self.backup_tau = max(1e-6, float(getattr(args, "mcts_backup_tau", 1.0)))
        self.theory_eta = min(1.0, max(0.0, float(getattr(args, "mcts_theory_eta", 0.5))))
        self.track_bellman_residual = bool(getattr(args, "mcts_track_bellman_residual", True))
        self.calibration_beta = max(0.0, float(getattr(args, "mcts_calibration_beta", 0.5)))
        self.calibration_mode = getattr(args, "mcts_calibration_mode", "heuristic")
        self.ece_bins = max(2, int(getattr(args, "mcts_ece_bins", 10)))
        self.calib_min_t = max(1e-3, float(getattr(args, "mcts_calibration_min_t", 0.7)))
        self.calib_max_t = max(self.calib_min_t, float(getattr(args, "mcts_calibration_max_t", 3.0)))
        self.budget_mode = getattr(args, "mcts_budget_mode", "none")
        self.budget_value = float(getattr(args, "mcts_budget_value", 0.0))
        self.track_structure_quality = bool(getattr(args, "mcts_track_structure_quality", True))
        self.track_proof_theory = bool(getattr(args, "mcts_track_proof_theory", True))
        self.theory5_lambda = max(0.0, float(getattr(args, "mcts_theory5_lambda", 0.0)))
        self.theory6_rho = min(1.0, max(0.0, float(getattr(args, "mcts_theory6_rho", 1.0))))
        self.theory6_pmin = min(1.0, max(1e-9, float(getattr(args, "mcts_theory6_pmin", 0.05))))
        self.merge_paradigm = getattr(args, "mcts_merge_paradigm", "hash")
        self.merge_alpha = min(0.5, max(1e-6, float(getattr(args, "mcts_merge_alpha", 0.05))))
        self.merge_min_effect = max(0.0, float(getattr(args, "mcts_merge_min_effect", 0.05)))
        self.measurement_model = getattr(args, "mcts_measurement_model", "single")
        self.self_consistency_samples = max(1, int(getattr(args, "mcts_self_consistency_samples", 3)))
        self.self_consistency_trim_ratio = min(0.4, max(0.0, float(getattr(args, "mcts_self_consistency_trim_ratio", 0.2))))
        self.gls_mode = getattr(args, "mcts_gls_mode", "none")

        self._validate_theorem_assumptions()

        self.rollout_policy = getattr(args, "mcts_rollout_policy", "max_ucb")
        self.reward_backend = getattr(args, "mcts_reward_backend", "llm_judge_discrete")

        self.llm = LLMEngine(args)

        self.node_table: Dict[str, MCTSNode] = {}
        self.stats = {}
        self._coverage_seen_keys = set()
        self._coverage_total_actions = 0
        self._coverage_neff = 0
        self._coverage_min_prefix_qhat = 1.0
        self._lockin_product_bound = 1.0

    def _validate_theorem_assumptions(self) -> None:
        """Fail fast when theorem assumptions are violated.

        The strict theorem statements used in the project require:
        1) discount factor gamma in [0, 1)
        2) eta in [0, 1]
        """
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"Theorem assumption A1 violated: expected gamma in [0,1), got {self.gamma}")
        if not (0.0 <= self.theory_eta <= 1.0):
            raise ValueError(f"Theorem assumption A2 violated: expected eta in [0,1], got {self.theory_eta}")
        if not (0.0 <= self.theory6_rho <= 1.0):
            raise ValueError(f"Theorem assumption A6.2 violated: expected rho in [0,1], got {self.theory6_rho}")
        if not (0.0 < self.theory6_pmin <= 1.0):
            raise ValueError(f"Theorem assumption A6.3 violated: expected pmin in (0,1], got {self.theory6_pmin}")
        if self.merge_paradigm not in {"hash", "hypothesis_test"}:
            raise ValueError(f"Invalid merge paradigm: {self.merge_paradigm}")
        if self.measurement_model not in {"single", "self_consistency"}:
            raise ValueError(f"Invalid measurement model: {self.measurement_model}")
        if self.gls_mode not in {"none", "low_rank", "cluster"}:
            raise ValueError(f"Invalid gls mode: {self.gls_mode}")

    def _reset_search_stats(self):
        self.stats = {
            "reject_too_few_premises": 0,
            "reject_missing_premises": 0,
            "reject_empty_conclusion": 0,
            "reject_duplicate_conclusion": 0,
            "reject_bad_candidate": 0,
            "merge_hits": 0,
            "fallback_empty_expansion": 0,
            "selection_trace": [],
            "graph_backprop_updates": 0,
            "rollout_policy": self.rollout_policy,
            "reward_backend": self.reward_backend,
            "dag_update_mode": self.dag_update_mode,
            "objective_mode": self.objective_mode,
            "entropy_source": self.entropy_source,
            "backup_operator": self.backup_operator,
            "theory_eta": self.theory_eta,
            "budget_mode": self.budget_mode,
            "budget_value": self.budget_value,
            "elapsed_time_s": 0.0,
            "llm_calls": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "weighted_parent_terms": 0,
            "weighted_max_weight_mean": 0.0,
            "weighted_parent_count_mean": 0.0,
            "weighted_entropy_mean": 0.0,
            "weighted_parent_var_proxy_mean": 0.0,
            "bellman_residual_mean": 0.0,
            "bellman_residual_max": 0.0,
            "bellman_residual_count": 0,
            "ucb_calibration_mean": 0.0,
            "ucb_calibration_count": 0,
            "calibration_mode": self.calibration_mode,
            "ece_proxy_mean": 0.0,
            "ece_proxy_count": 0,
            "temperature_scale_mean": 0.0,
            "temperature_scale_count": 0,
            "theory_eta_contraction_gap_mean": 0.0,
            "theory_eta_contraction_gap_count": 0,
            "theory_conservative_gap_mean": 0.0,
            "theory_conservative_gap_count": 0,
            "theory_unique_state_ratio": 0.0,
            "theory_contraction_ratio_mean": 0.0,
            "theory_contraction_ratio_count": 0,
            "theory_contraction_bound": self.gamma,
            "dag_equivalence_violation_count": 0,
            "dag_equivalence_check_count": 0,
            "theory_parametric_convex_violation_count": 0,
            "theory_conservative_order_violation_count": 0,
            "theory5_lambda": self.theory5_lambda,
            "theory5_budget_B": self.budget_value,
            "theory5_cost": 0.0,
            "theory5_reward_proxy": 0.0,
            "theory5_lagrangian": 0.0,
            "theory5_feasible_indicator": 0.0,
            "theory6_rho": self.theory6_rho,
            "theory6_pmin": self.theory6_pmin,
            "theory6_total_action_visits_T": 0,
            "theory6_effective_expansions_neff": 0,
            "theory6_coverage_rate_qhat": 0.0,
            "theory6_min_prefix_qhat": 0.0,
            "theory6_lockin_upper_bound_prod": 1.0,
            "theory6_lockin_upper_bound_minq": 1.0,
            "merge_paradigm": self.merge_paradigm,
            "measurement_model": self.measurement_model,
            "gls_mode": self.gls_mode,
            "merge_hypothesis_accept_count": 0,
            "merge_hypothesis_reject_count": 0,
            "self_consistency_mean_std": 0.0,
            "self_consistency_count": 0,
        }

    def search(self, initial_data_item):
        self._reset_search_stats()
        self.node_table = {}
        self.llm.reset_counters()
        self._coverage_seen_keys = set()
        self._coverage_total_actions = 0
        self._coverage_neff = 0
        self._coverage_min_prefix_qhat = 1.0
        self._lockin_product_bound = 1.0
        start_time = time.time()

        root = MCTSNode(data_item=initial_data_item)
        root_hash = root.state_hash()
        self.node_table[root_hash] = root

        for sim_idx in range(self.max_simulations):
            path: List[MCTSNode] = []
            node = self._select(root, path)

            used_fallback = False
            if not node.is_terminal() and len(path) < self.max_depth:
                expanded = self._expand(node)
                if expanded:
                    node = self._pick_rollout_child(parent=node, expanded=expanded)
                    self._record_action_coverage(node)
                    path.append(node)
                else:
                    self.stats["fallback_empty_expansion"] += 1
                    used_fallback = True

            reward = self.expand_fallback_reward if used_fallback else self._evaluate_reward(node)
            node.r_t = reward
            self._backpropagate(path, reward)

            if sim_idx < 3:
                self._log_path_trace(path)

            if self._budget_reached(start_time):
                self.stats["budget_stop_simulation"] = sim_idx + 1
                break

        self._finalize_runtime_stats(start_time)
        return self._extract_best_tree(root)

    def _evaluate_reward(self, node: MCTSNode) -> float:
        if self.reward_backend == "heuristic_soft":
            return self._heuristic_soft_reward(node)
        if self.measurement_model == "single":
            if self.reward_backend == "llm_judge_soft":
                return self.llm.evaluate_state_raw(node)
            return self.llm.evaluate_state(node)

        samples = []
        for _ in range(self.self_consistency_samples):
            if self.reward_backend == "llm_judge_soft":
                samples.append(float(self.llm.evaluate_state_raw(node)))
            else:
                samples.append(float(self.llm.evaluate_state(node)))
        return self._aggregate_measurements(samples)

    def _aggregate_measurements(self, samples: List[float]) -> float:
        if not samples:
            return 0.0
        mu = sum(samples) / len(samples)
        var = sum((x - mu) ** 2 for x in samples) / max(1, len(samples) - 1)
        std = math.sqrt(max(0.0, var))
        self.stats["self_consistency_mean_std"] += std
        self.stats["self_consistency_count"] += 1

        if self.gls_mode == "low_rank":
            weights = [1.0 / max(1e-6, abs(x - mu) + 1e-3) for x in samples]
            sw = sum(weights)
            return sum(w * x for w, x in zip(weights, samples)) / max(1e-6, sw)
        if self.gls_mode == "cluster":
            pos = [x for x in samples if x >= mu]
            neg = [x for x in samples if x < mu]
            chosen = pos if len(pos) >= len(neg) else neg
            if not chosen:
                chosen = samples
            return sum(chosen) / len(chosen)

        if len(samples) <= 2:
            return mu
        k = int(len(samples) * self.self_consistency_trim_ratio)
        vals = sorted(samples)
        trimmed = vals[k: len(vals) - k] if len(vals) - 2 * k > 0 else vals
        return sum(trimmed) / len(trimmed)

    def _heuristic_soft_reward(self, node: MCTSNode) -> float:
        if not node.latest_conclusion_id:
            return 0.0
        con = node.id2sent.get(node.latest_conclusion_id, "")
        if not con or not node.H:
            return -0.2
        a = set(con.lower().split())
        b = set(node.H.lower().split())
        inter = len(a & b)
        union = max(1, len(a | b))
        jacc = inter / union
        return 2.0 * jacc - 1.0  # map to [-1,1]

    def _pick_rollout_child(self, parent: MCTSNode, expanded: List[MCTSNode]) -> MCTSNode:
        if self.rollout_policy == "sample":
            return random.choice(expanded)
        if self.rollout_policy == "max_prior":
            return max(expanded, key=lambda n: n.prior_p)
        # default max_ucb
        return max(expanded, key=lambda n: self._ucb_score(parent, n))

    def _log_path_trace(self, path):
        trace = []
        for n in path:
            trace.append(
                {
                    "id": n.data_id,
                    "action": n.action_taken,
                    "q": round(n.q_value, 4),
                    "visits": n.visits,
                    "prior": round(n.prior_p, 4),
                }
            )
        self.stats["selection_trace"].append(trace)

    def _select(self, node, path):
        cur = node
        path.append(cur)
        while cur.children and not cur.is_terminal() and len(path) < self.max_depth:
            scored = []
            for action, child in cur.children.items():
                scored.append((action, child, self._ucb_score(cur, child)))
            scored.sort(key=lambda x: x[2], reverse=True)
            if logger.isEnabledFor(logging.DEBUG):
                topk = scored[: self.trace_topk]
                logger.debug(
                    "Select top actions: %s",
                    [
                        {
                            "action": a,
                            "ucb": round(s, 4),
                            "q": round(c.q_value, 4),
                            "visit": c.visits,
                            "prior": round(c.prior_p, 4),
                        }
                        for a, c, s in topk
                    ],
                )
            _, cur, _ = scored[0]
            self._record_action_coverage(cur)
            path.append(cur)
        return cur

    def _expand(self, node):
        context = node.get_context_for_llm()
        candidates = self.llm.generate_actions(context)
        if not candidates:
            return []

        new_children = []
        for cand in candidates:
            action_str = cand.get("action_str", "")
            pre_ids = cand.get("pre_ids", [])
            conclusion = cand.get("conclusion", "")
            if not action_str or not pre_ids or not conclusion:
                self.stats["reject_bad_candidate"] += 1
                continue

            child = MCTSNode(parent=node, action=action_str)
            is_ok, reason, _ = child.validate_action(pre_ids, conclusion)
            if not is_ok:
                key = f"reject_{reason}"
                self.stats[key] = self.stats.get(key, 0) + 1
                continue

            if not child.apply_action(pre_ids, conclusion):
                self.stats["reject_bad_candidate"] += 1
                continue

            p = float(cand.get("probability", 0.0))
            child.prior_p = max(1e-6, p)
            child_hash = child.state_hash()

            if child_hash in self.node_table:
                merged = self.node_table[child_hash]
                self._check_dag_equivalence(child, merged)
                if self._accept_merge_by_hypothesis_test(candidate=child, merged=merged):
                    merged.add_parent(node, edge_prior=child.prior_p)
                    node.children[action_str] = merged
                    self.stats["merge_hits"] += 1
                    self.stats["merge_hypothesis_accept_count"] += 1
                    continue
                self.stats["merge_hypothesis_reject_count"] += 1
                # fallback to hash merge to preserve DAG consistency under canonical hash identity.
                merged.add_parent(node, edge_prior=child.prior_p)
                node.children[action_str] = merged
                self.stats["merge_hits"] += 1
                continue

            child.add_parent(node, edge_prior=child.prior_p)
            self.node_table[child_hash] = child
            node.children[action_str] = child
            new_children.append(child)

        return new_children

    def _accept_merge_by_hypothesis_test(self, candidate: MCTSNode, merged: MCTSNode) -> bool:
        if self.merge_paradigm == "hash":
            return True
        # Merge as hypothesis testing: H0 = same latent value distribution.
        n1 = max(1, int(candidate.visits))
        n2 = max(1, int(merged.visits))
        if n1 < 2 or n2 < 2:
            return True
        m1, m2 = float(candidate.V_t), float(merged.V_t)
        pooled_se = math.sqrt((1.0 / n1) + (1.0 / n2))
        if pooled_se <= 1e-9:
            return abs(m1 - m2) <= self.merge_min_effect
        z = abs(m1 - m2) / pooled_se
        # two-sided z critical approximation
        z_critical = 1.96 if abs(self.merge_alpha - 0.05) < 1e-9 else 2.576
        return (z <= z_critical) or (abs(m1 - m2) <= self.merge_min_effect)

    def _check_dag_equivalence(self, candidate: MCTSNode, merged: MCTSNode) -> None:
        """Runtime invariant check for Graph-MDP state equivalence under transposition merge.

        If two nodes share hash, their canonical states should match exactly.
        """
        self.stats["dag_equivalence_check_count"] += 1
        if candidate.canonical_state() != merged.canonical_state():
            self.stats["dag_equivalence_violation_count"] += 1


    def _action_coverage_key(self, node: MCTSNode) -> Tuple[str, ...]:
        if node.logical_parent_ids:
            return tuple(sorted(str(x) for x in node.logical_parent_ids))
        action_text = str(node.action_taken or "")
        return (action_text.strip().lower(),)

    def _record_action_coverage(self, node: MCTSNode) -> None:
        # Theorem 6 instrumentation: prefix coverage q_hat and lock-in probability upper bound.
        qhat_prev = 1.0
        if self._coverage_total_actions > 0:
            qhat_prev = self._coverage_neff / float(self._coverage_total_actions)
        factor = 1.0 - self.theory6_rho * qhat_prev * self.theory6_pmin
        factor = min(1.0, max(0.0, factor))
        self._lockin_product_bound *= factor

        self._coverage_total_actions += 1
        k = self._action_coverage_key(node)
        if k not in self._coverage_seen_keys:
            self._coverage_seen_keys.add(k)
            self._coverage_neff += 1

        qhat = self._coverage_neff / float(self._coverage_total_actions)
        self._coverage_min_prefix_qhat = min(self._coverage_min_prefix_qhat, qhat)

    def _aggregate_parent_value(self, node: MCTSNode) -> float:
        if not node.parents:
            return 0.0

        parent_vals = []
        weights = []
        var_proxy_terms = []
        for parent in node.parents:
            parent_vals.append(parent.V_t)
            ph = parent.node_hash or parent.state_hash()
            prior = node.incoming_prior.get(ph, 1e-6)
            # variance proxy used for theory-oriented analysis: uncertainty decreases with visits
            var_proxy = 1.0 / max(1.0, float(parent.visits))
            var_proxy_terms.append(var_proxy)
            if self.weight_mode == "visit":
                w = max(1e-6, float(parent.visits))
            elif self.weight_mode == "hybrid":
                w = max(1e-6, float(parent.visits) * prior)
            elif self.weight_mode == "inv_var":
                # inverse-variance proxy: reliable parents should get larger weights.
                # We use visit-scaled uncertainty as a practical estimator.
                w = max(1e-6, 1.0 / max(var_proxy, 1e-6))
            else:
                w = max(1e-6, prior)
            weights.append(w)

        if not self.enable_weighted_return:
            return sum(parent_vals) / max(1, len(parent_vals))

        s = sum(weights)
        norm_weights = [w / max(s, 1e-6) for w in weights]
        self.stats["weighted_parent_terms"] += 1
        self.stats["weighted_max_weight_mean"] += max(norm_weights)
        self.stats["weighted_parent_count_mean"] += len(parent_vals)
        self.stats["weighted_entropy_mean"] += self._normalized_entropy(norm_weights)
        if var_proxy_terms:
            self.stats["weighted_parent_var_proxy_mean"] += sum(var_proxy_terms) / len(var_proxy_terms)
        return sum(v * w for v, w in zip(parent_vals, norm_weights))

    def _backup_aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        if self.backup_operator == "max":
            return max(values)
        if self.backup_operator == "softmax":
            scaled = [v / self.backup_tau for v in values]
            m = max(scaled)
            exps = [math.exp(v - m) for v in scaled]
            z = sum(exps)
            if z <= 0:
                return sum(values) / len(values)
            weights = [e / z for e in exps]
            return sum(w * v for w, v in zip(weights, values))
        return sum(values) / len(values)

    def _theory_operator(self, node: MCTSNode, running: float, parent_agg: float, conservative_parent: float) -> float:
        # T_eta(s) = r + gamma * ((1-eta) * running + eta * parent_agg)
        # conservative term is reserved for robustness control via objective_mode.
        del conservative_parent
        mixed = (1.0 - self.theory_eta) * running + self.theory_eta * parent_agg
        lo = min(running, parent_agg) - 1e-9
        hi = max(running, parent_agg) + 1e-9
        if not (lo <= mixed <= hi):
            self.stats["theory_parametric_convex_violation_count"] += 1
        if self.track_proof_theory:
            self.stats["theory_eta_contraction_gap_mean"] += abs(running - parent_agg)
            self.stats["theory_eta_contraction_gap_count"] += 1
        return node.r_t + self.gamma * mixed

    def _compute_backup_target(self, node: MCTSNode, running: float) -> float:
        parent_agg = self._aggregate_parent_value(node)
        conservative_parent = min([p.V_t for p in node.parents], default=parent_agg)
        if conservative_parent - parent_agg > 1e-9:
            self.stats["theory_conservative_order_violation_count"] += 1
        if self.track_proof_theory:
            self.stats["theory_conservative_gap_mean"] += abs(parent_agg - conservative_parent)
            self.stats["theory_conservative_gap_count"] += 1
        if self.objective_mode == "seer":
            return node.r_t + self.gamma * parent_agg
        if self.objective_mode == "conservative":
            return node.r_t + self.gamma * conservative_parent
        if self.objective_mode == "parametric":
            return self._theory_operator(node, running, parent_agg, conservative_parent)
        # graph_td
        return node.r_t + self.gamma * self._backup_aggregate([running, parent_agg])

    def _backpropagate(self, path: List[MCTSNode], rollout_reward: float) -> None:
        running = rollout_reward
        for node in reversed(path):
            target = self._compute_backup_target(node, running)
            if self.track_bellman_residual:
                residual = abs(target - node.V_t)
                prev = node.last_bellman_residual
                if prev is not None and prev > 1e-12:
                    ratio = residual / prev
                    self.stats["theory_contraction_ratio_mean"] += ratio
                    self.stats["theory_contraction_ratio_count"] += 1
                node.last_bellman_residual = residual
                self.stats["bellman_residual_mean"] += residual
                self.stats["bellman_residual_max"] = max(self.stats["bellman_residual_max"], residual)
                self.stats["bellman_residual_count"] += 1
            node.accumulate(target, inc_visit=True)
            running = target

        if not self.enable_dag_backprop or not path:
            return

        q = deque(path)
        seen = set()
        while q:
            node = q.popleft()
            h = node.node_hash or node.state_hash()
            if h in seen:
                continue
            seen.add(h)

            graph_target = self._compute_backup_target(node, node.V_t)
            if self.track_bellman_residual:
                residual = abs(graph_target - node.V_t)
                prev = node.last_bellman_residual
                if prev is not None and prev > 1e-12:
                    ratio = residual / prev
                    self.stats["theory_contraction_ratio_mean"] += ratio
                    self.stats["theory_contraction_ratio_count"] += 1
                node.last_bellman_residual = residual
                self.stats["bellman_residual_mean"] += residual
                self.stats["bellman_residual_max"] = max(self.stats["bellman_residual_max"], residual)
                self.stats["bellman_residual_count"] += 1
            if self.dag_update_mode == "visit_and_value":
                node.accumulate(graph_target, inc_visit=True)
            else:
                node.set_bootstrap_value(graph_target)
            self.stats["graph_backprop_updates"] += 1

            for parent in node.parents:
                q.append(parent)

    def _normalized_entropy(self, probs: List[float]) -> float:
        if len(probs) <= 1:
            return 0.0
        s = sum(max(1e-12, p) for p in probs)
        norm = [max(1e-12, p) / s for p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for p in norm)
        max_ent = math.log(len(norm) + 1e-12)
        return entropy / max(max_ent, 1e-6)

    def _parent_entropy(self, parent: MCTSNode) -> float:
        prior_probs = [max(1e-12, ch.prior_p) for ch in parent.children.values()]
        post_probs = [1.0 / max(1.0, ch.visits) for ch in parent.children.values()]
        if self.entropy_source == "prior":
            return self._normalized_entropy(prior_probs)
        if self.entropy_source == "posterior":
            return self._normalized_entropy(post_probs)
        return 0.5 * (self._normalized_entropy(prior_probs) + self._normalized_entropy(post_probs))

    def _compute_ece_proxy(self, parent: MCTSNode) -> float:
        children = list(parent.children.values())
        if len(children) <= 1:
            return 0.0
        priors = [max(1e-12, ch.prior_p) for ch in children]
        s = sum(priors)
        priors = [p / max(1e-12, s) for p in priors]
        post = [max(1e-12, float(ch.visits)) for ch in children]
        ps = sum(post)
        post = [p / max(1e-12, ps) for p in post]

        # reliability-bin style ECE proxy using priors as confidence and posterior as empirical frequency.
        bins = self.ece_bins
        bucket = [[] for _ in range(bins)]
        for conf, freq in zip(priors, post):
            idx = min(bins - 1, int(conf * bins))
            bucket[idx].append((conf, freq))

        ece = 0.0
        for b in bucket:
            if not b:
                continue
            avg_conf = sum(x for x, _ in b) / len(b)
            avg_freq = sum(y for _, y in b) / len(b)
            mass = len(b) / float(len(children))
            ece += abs(avg_conf - avg_freq) * mass
        return max(0.0, min(1.0, ece))

    def _calibrated_entropy_scale(self, parent: MCTSNode, entropy: float) -> float:
        if self.calibration_mode == "ece_temp":
            ece = self._compute_ece_proxy(parent)
            # higher ECE -> larger temperature and more exploration.
            temp = min(self.calib_max_t, max(self.calib_min_t, 1.0 + self.calibration_beta * ece))
            scale = 1.0 + self.entropy_coef * entropy * temp
            self.stats["ece_proxy_mean"] += ece
            self.stats["ece_proxy_count"] += 1
            self.stats["temperature_scale_mean"] += temp
            self.stats["temperature_scale_count"] += 1
            self.stats["ucb_calibration_mean"] += scale
            self.stats["ucb_calibration_count"] += 1
            return scale

        calib = 1.0 + self.calibration_beta * abs(entropy - 0.5)
        scale = 1.0 + self.entropy_coef * entropy * calib
        self.stats["ucb_calibration_mean"] += scale
        self.stats["ucb_calibration_count"] += 1
        return scale

    def _ucb_score(self, parent, child):
        q_value = child.q_value
        entropy_scale = 1.0
        if self.enable_entropy_ucb:
            ent = self._parent_entropy(parent)
            entropy_scale = self._calibrated_entropy_scale(parent, ent)
        exploration = (
            self.c_puct
            * entropy_scale
            * child.prior_p
            * math.sqrt(max(1, parent.visits))
            / (1 + child.visits)
        )
        return q_value + exploration


    def _theory5_cost(self) -> float:
        if self.budget_mode == "wall_clock":
            return float(self.stats.get("elapsed_time_s", 0.0))
        if self.budget_mode == "llm_calls":
            return float(self.stats.get("llm_calls", 0.0))
        # default surrogate cost when budget mode is disabled
        return float(self.stats.get("llm_calls", 0.0))

    def _extract_best_tree(self, root):
        proof = []
        cur = root
        depth = 0
        while cur.children and depth < self.max_depth:
            best_action, nxt = max(cur.children.items(), key=lambda item: item[1].visits)
            if Action.end in best_action.lower():
                break
            if nxt.proof_str:
                proof.append(nxt.proof_str[-1])
            cur = nxt
            depth += 1

        triples = dict(root.id2sent)
        for step in proof:
            parts = step.split("->")
            if len(parts) != 2:
                continue
            rhs = parts[1].strip()
            if ":" in rhs:
                int_id, con = rhs.split(":", 1)
                triples[int_id.strip()] = con.strip()

        out_stats = {
            "simulations": self.max_simulations,
            "expanded_nodes": len(self.node_table),
            "weighted_return": self.enable_weighted_return,
            "weight_mode": self.weight_mode,
            "entropy_ucb": self.enable_entropy_ucb,
            "strict_dag_backprop": self.enable_dag_backprop,
        }

        if self.stats["weighted_parent_terms"] > 0:
            n = float(self.stats["weighted_parent_terms"])
            self.stats["weighted_max_weight_mean"] = self.stats["weighted_max_weight_mean"] / n
            self.stats["weighted_parent_count_mean"] = self.stats["weighted_parent_count_mean"] / n
            self.stats["weighted_entropy_mean"] = self.stats["weighted_entropy_mean"] / n
            self.stats["weighted_parent_var_proxy_mean"] = self.stats["weighted_parent_var_proxy_mean"] / n

        if self.stats["bellman_residual_count"] > 0:
            self.stats["bellman_residual_mean"] = (
                self.stats["bellman_residual_mean"] / float(self.stats["bellman_residual_count"])
            )
        if self.stats["ucb_calibration_count"] > 0:
            self.stats["ucb_calibration_mean"] = (
                self.stats["ucb_calibration_mean"] / float(self.stats["ucb_calibration_count"])
            )
        if self.stats["ece_proxy_count"] > 0:
            self.stats["ece_proxy_mean"] = (
                self.stats["ece_proxy_mean"] / float(self.stats["ece_proxy_count"])
            )
        if self.stats["temperature_scale_count"] > 0:
            self.stats["temperature_scale_mean"] = (
                self.stats["temperature_scale_mean"] / float(self.stats["temperature_scale_count"])
            )
        if self.stats["theory_eta_contraction_gap_count"] > 0:
            self.stats["theory_eta_contraction_gap_mean"] = (
                self.stats["theory_eta_contraction_gap_mean"]
                / float(self.stats["theory_eta_contraction_gap_count"])
            )
        if self.stats["theory_conservative_gap_count"] > 0:
            self.stats["theory_conservative_gap_mean"] = (
                self.stats["theory_conservative_gap_mean"]
                / float(self.stats["theory_conservative_gap_count"])
            )
        self.stats["theory_unique_state_ratio"] = float(len(self.node_table)) / max(
            1.0, float(self.max_simulations)
        )
        self.stats["theory6_total_action_visits_T"] = int(self._coverage_total_actions)
        self.stats["theory6_effective_expansions_neff"] = int(self._coverage_neff)
        if self._coverage_total_actions > 0:
            self.stats["theory6_coverage_rate_qhat"] = self._coverage_neff / float(self._coverage_total_actions)
            self.stats["theory6_min_prefix_qhat"] = self._coverage_min_prefix_qhat
        self.stats["theory6_lockin_upper_bound_prod"] = self._lockin_product_bound
        minq = self.stats.get("theory6_min_prefix_qhat", 0.0)
        self.stats["theory6_lockin_upper_bound_minq"] = (
            max(0.0, 1.0 - self.theory6_rho * self.theory6_pmin * minq)
            ** max(0, self._coverage_total_actions)
        )
        if self.stats["theory_contraction_ratio_count"] > 0:
            self.stats["theory_contraction_ratio_mean"] = (
                self.stats["theory_contraction_ratio_mean"]
                / float(self.stats["theory_contraction_ratio_count"])
            )
        if self.stats["self_consistency_count"] > 0:
            self.stats["self_consistency_mean_std"] = (
                self.stats["self_consistency_mean_std"] / float(self.stats["self_consistency_count"])
            )

        if self.track_structure_quality:
            out_stats["structure_quality"] = self._structure_quality_metrics(root, proof)

        reward_proxy = 1.0 if root.H and any(root.H == v for v in triples.values()) else 0.0
        cost_val = self._theory5_cost()
        budget_B = self.budget_value if self.budget_mode != "none" else 0.0
        self.stats["theory5_reward_proxy"] = reward_proxy
        self.stats["theory5_cost"] = cost_val
        self.stats["theory5_budget_B"] = budget_B
        self.stats["theory5_lambda"] = self.theory5_lambda
        self.stats["theory5_lagrangian"] = reward_proxy - self.theory5_lambda * (cost_val - budget_B)
        self.stats["theory5_feasible_indicator"] = 1.0 if (self.budget_mode == "none" or cost_val <= budget_B) else 0.0

        out_stats.update(self.stats)

        return {
            "id": root.data_id,
            "proof": "; ".join(proof),
            "meta": {"triples": triples},
            "search_stats": out_stats,
        }

    def _finalize_runtime_stats(self, start_time: float) -> None:
        elapsed = time.time() - start_time
        self.stats["elapsed_time_s"] = elapsed
        counters = self.llm.get_counters()
        self.stats["llm_calls"] = counters.get("llm_calls", 0)
        self.stats["llm_prompt_tokens"] = counters.get("llm_prompt_tokens", 0)
        self.stats["llm_completion_tokens"] = counters.get("llm_completion_tokens", 0)

    def _budget_reached(self, start_time: float) -> bool:
        if self.budget_mode == "none" or self.budget_value <= 0:
            return False
        if self.budget_mode == "wall_clock":
            return (time.time() - start_time) >= self.budget_value
        if self.budget_mode == "llm_calls":
            return self.llm.get_counters().get("llm_calls", 0) >= int(self.budget_value)
        return False

    def _structure_quality_metrics(self, root: MCTSNode, proof_steps: List[str]) -> Dict[str, float]:
        num_nodes = max(1, len(self.node_table))
        num_edges = 0
        depth = {root: 0}
        q = deque([root])
        visited = set([root])
        max_depth = 0
        while q:
            cur = q.popleft()
            d = depth[cur]
            max_depth = max(max_depth, d)
            for _, ch in cur.children.items():
                num_edges += 1
                if ch not in visited:
                    visited.add(ch)
                    depth[ch] = d + 1
                    q.append(ch)

        merge_hits = float(self.stats.get("merge_hits", 0))
        rejects = (
            self.stats.get("reject_too_few_premises", 0)
            + self.stats.get("reject_missing_premises", 0)
            + self.stats.get("reject_empty_conclusion", 0)
            + self.stats.get("reject_duplicate_conclusion", 0)
            + self.stats.get("reject_bad_candidate", 0)
        )
        attempts = rejects + num_edges

        return {
            "num_nodes": float(num_nodes),
            "num_edges": float(num_edges),
            "avg_branching_factor": float(num_edges) / max(1.0, float(num_nodes)),
            "max_graph_depth": float(max_depth),
            "merge_reuse_ratio": merge_hits / max(1.0, merge_hits + num_nodes),
            "redundancy_reject_ratio": float(rejects) / max(1.0, float(attempts)),
            "proof_steps": float(len(proof_steps)),
        }
