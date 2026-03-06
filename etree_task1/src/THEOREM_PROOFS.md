# MCTS Graph-MDP 理论命题（严格版）

> 本文件给出与当前实现一致的形式化命题、假设与证明草图，供论文正文/附录直接引用。

## 记号与实现对应

- 参数化备份算子（`objective_mode=parametric`）在代码中实现为：
  \[
  T_\eta(s) = r(s) + \gamma\big((1-\eta)x + \eta y\big),
  \]
  其中 \(x\) 对应 `running`，\(y\) 对应 `parent_agg`，\(\eta\in[0,1]\)。
- 参见：`mcts_solver.py::_validate_theorem_assumptions`、`_theory_operator` 与 `_compute_backup_target`。

## 假设

- **A1 (折扣因子有界)**: \(0\le \gamma < 1\)。
- **A2 (混合系数有界)**: \(0\le \eta\le 1\)。
- **A3 (值函数有界)**: 所有状态值落在有界集合内（由奖励有界和折扣机制保证）。
- **A4 (等价状态合并一致性)**: DAG transposition 合并仅发生在同一 canonical state hash 下，且后续更新共享同一状态统计。

## 命题 1：参数化算子在 \(\ell_\infty\) 范数下是 \(\gamma\)-压缩

令 \(V_1, V_2\) 为两个值函数，定义
\[
(T_\eta V)(s)=r(s)+\gamma\left((1-\eta)\,x(V,s)+\eta\,y(V,s)\right),
\]
其中 \(x,y\) 为对 \(V\) 的 1-Lipschitz 映射（在本实现中对应 rollout 与父聚合的线性/soft 聚合近似）。则：
\[
\|T_\eta V_1 - T_\eta V_2\|_\infty
\le \gamma\|V_1-V_2\|_\infty.
\]

### 证明
\[
\begin{aligned}
|T_\eta V_1 - T_\eta V_2|
&= \gamma\left|(1-\eta)(x_1-x_2)+\eta(y_1-y_2)\right|\\
&\le \gamma\left((1-\eta)|x_1-x_2|+\eta|y_1-y_2|\right)\\
&\le \gamma\left((1-\eta)+\eta\right)\|V_1-V_2\|_\infty\\
&=\gamma\|V_1-V_2\|_\infty.
\end{aligned}
\]
取上确界即得结论。

## 命题 2：Conservative 模式相对均值聚合的上偏风险更小

设父节点估计存在重尾正偏噪声，`conservative` 使用 \(\min\) 聚合。则其上偏误差满足：
\[
\mathbb{E}[\max(0,\hat V_{cons}-V^*)] \le \mathbb{E}[\max(0,\hat V_{mean}-V^*)].
\]

### 证明思路
在同一父值样本上，\(\min\le \text{mean}\)，故对任意阈值 \(V^*\) 有
\(\max(0,\min-V^*)\le \max(0,\text{mean}-V^*)\)。两侧取期望即得。

## 命题 3：DAG 等价状态合并不改变最优值（在 A4 下）

### 等价关系定义
定义状态等价关系 \(\sim\)：
\[
s_1 \sim s_2 \iff \text{canonical\_state}(s_1)=\text{canonical\_state}(s_2),
\]
其中 `canonical_state` 由 `(hypothesis, facts, used)` 的规范化有序表示组成。

### 命题
若两条轨迹映射到同一 canonical state hash，且更新在该 hash 下共享同一状态统计，则 Tree 与 DAG 的 Bellman 固定点一致，故最优值与最优策略不变。

### 证明
设原树 MDP 状态空间为 \(\mathcal S\)，按 \(\sim\) 构造商空间 \(\bar{\mathcal S}=\mathcal S/\sim\)。

1. **奖励保持**：若 \(s_1\sim s_2\)，canonical 表示一致，故可行动作集合与一步语义状态一致，奖励定义相同。
2. **转移保持**：对任意动作 \(a\)，\(s_1,s_2\) 在 canonical 规则下转移到同一等价类 \([s']\)。
3. **Bellman 方程同构**：原空间 Bellman 算子在等价类上映射为商空间 Bellman 算子，且值函数满足
   \(V(s)=\bar V([s])\)。

因此树搜索到图搜索（transposition merge）仅是状态空间商化，不改变 Bellman 固定点；故最优值与最优策略保持不变。

### 代码一致性检查
- `mcts_node.py::canonical_state()` 给出等价关系定义。
- `mcts_solver.py::_check_dag_equivalence()` 在每次合并时检查 invariant。
- 指标 `dag_equivalence_violation_count` 应为 0。
- 参数化凸组合不变量：`theory_parametric_convex_violation_count` 应为 0。
- conservative 顺序不变量：`theory_conservative_order_violation_count` 应为 0。

## 与实验协议的对应

- 命题 1 对应：`theory_contraction_ratio_mean`、`bellman_residual_mean/max`。
- 命题 2 对应：`theory_conservative_gap_mean` + baseline delta 显著性。
- 命题 3 对应：`theory_unique_state_ratio` 与性能/预算联合对比。
- 严格检查入口：`theorem_protocol.py`（`all_pass`）。

## 定理 5：预算受限目标的拉格朗日表示与 Pareto 等价（实现对应）

- 预算变量：`budget_mode` + `budget_value`；成本代理 `C(π)` 在实现中由 `_theory5_cost()` 给出（wall_clock 或 llm_calls）。
- 拉格朗日系数：`mcts_theory5_lambda`。
- 运行期统计：`theory5_reward_proxy`、`theory5_cost`、`theory5_budget_B`、`theory5_lagrangian`、`theory5_feasible_indicator`。
- 含义：固定 `B` 扫 `λ`，或固定 `λ` 扫 `B`，可得到经验 Pareto 前沿近似。

## 定理 6：结构质量中间变量与 lock-in 上界（实现对应）

- 覆盖 key：`_action_coverage_key()` 以推理动作的 premise-id 集合作为 \
  \(\kappa(a)\) 的可观测实现。
- 有效展开数与覆盖率：`theory6_effective_expansions_neff`、`theory6_total_action_visits_T`、`theory6_coverage_rate_qhat`。
- 前缀下界与锁死上界：`theory6_min_prefix_qhat`、`theory6_lockin_upper_bound_prod`、`theory6_lockin_upper_bound_minq`。
- 常数参数：`mcts_theory6_rho`（A6.2 系数）和 `mcts_theory6_pmin`（命题3给出的条件采样下界）。

## 创新范式补充：Merge as Hypothesis Testing + Self-Consistency Measurement

- Merge 机制：`mcts_merge_paradigm=hypothesis_test` 时，使用 `_accept_merge_by_hypothesis_test()` 对候选节点与已存在节点做“同分布/同值”近似检验（z-score + effect-size 门控），并记录
  `merge_hypothesis_accept_count` / `merge_hypothesis_reject_count`。
- Measurement 机制：`mcts_measurement_model=self_consistency` 时，对奖励进行多次测量后由 `_aggregate_measurements()` 聚合；可选
  `mcts_gls_mode in {none, low_rank, cluster}` 对应鲁棒均值/低秩近似加权/簇一致性聚合。
- 与 q-controller 的关系：`q_hat` 与 lock-in 上界仍通过 `theory6_*` 指标追踪；上述两机制可视作提高 key 覆盖与测量稳定性的工程实现。
