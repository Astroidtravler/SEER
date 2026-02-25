# MCTS 研究原型执行进展

## Progress
- [x] Step 1: 最小闭环（MCTS backend + 预测文件输出）
- [x] Step 2: 稳健性补丁（动作过滤统计、空扩展 fallback、可解释 trace）
- [x] Step 3: 对齐 SEER 官方评测格式（`$proof$ = ...` + 直接调用官方 scorer）
- [x] Step 4: 论文向增强（Weighted return / Entropy UCB / Strict DAG backprop）
- [x] Step 5: 固定预算消融脚本（`compare_mcts_budgets.py`）
- [x] Step 6: 统计一致性与公平比较增强（DAG value-only 更新、rollout policy、baseline 对比脚本）

## 新增关键实现
1. **DAG 反传统计一致性**
   - 新增 `--mcts_dag_update_mode {value_only,visit_and_value}`。
   - 默认 `value_only`：图传播仅更新 bootstrap value，避免单次 rollout 的重复 visits 记账。

2. **Rollout 决策可复现化**
   - 新增 `--mcts_rollout_policy {max_ucb,max_prior,sample}`。
   - 默认 `max_ucb`，不再默认随机扩展。

3. **奖励后端可插拔**
   - 新增 `--mcts_reward_backend {llm_judge_discrete,llm_judge_soft,heuristic_soft}`。

4. **公平比较工具**
   - `compare_mcts_budgets.py`：支持多 seed 聚合（均值/方差）。
   - `compare_baselines.py`：支持 PPO 与 MCTS 的同配置多 seed 对比。

## 快速运行示例
```bash
python train.py --solver_backend mcts --do_train False --do_dev True --do_test False \
  --mcts_rollout_policy max_ucb --mcts_dag_update_mode value_only

python compare_mcts_budgets.py \
  --base_cmd "python train.py --solver_backend mcts --do_train False --do_dev True --do_test False" \
  --simulations 50 --seeds 42,43,44

python compare_baselines.py \
  --ppo_cmd "python train.py --solver_backend ppo --do_dev True --do_test False" \
  --mcts_cmd "python train.py --solver_backend mcts --do_train False --do_dev True --do_test False"
```

- [x] Step 7: Novelty 路线图实装（objective/entropy-source/backup-operator/CI统计）

## 路线图实装细节（新增）
1. **统一目标函数开关**
   - 新增 `--mcts_objective_mode {seer,graph_td,conservative}`，将原始SEER式、Graph-TD式与保守式回报统一在同一实现中。
2. **校准熵来源开关**
   - 新增 `--mcts_entropy_source {prior,posterior,hybrid}`，支持先验熵、后验熵与混合熵用于Entropy-UCB。
3. **图备份算子开关**
   - 新增 `--mcts_backup_operator {mean,max,softmax}` 与 `--mcts_backup_tau`，用于控制Graph-TD中的聚合算子。
4. **统计显著性基础设施**
   - `compare_mcts_budgets.py`、`compare_baselines.py` 新增 `--alpha` 与正态近似置信区间输出；baseline对比新增 paired t-like 统计量。


- [x] Step 8: 补齐未完成三点（预算协议/结构质量指标/加权结构回报可审计证据）

## Step 8 实装
1. **预算受限最优性协议（工程实现）**
   - 新增 `--mcts_budget_mode {none,llm_calls,wall_clock}` 与 `--mcts_budget_value`。
   - 搜索循环支持预算触发提前停止，并在 `search_stats` 输出 `budget_stop_simulation`、`elapsed_time_s`、`llm_calls`、token 统计。
2. **结构质量指标扩展**
   - 在 `search_stats.structure_quality` 新增图结构指标：`num_nodes`、`num_edges`、`avg_branching_factor`、`max_graph_depth`、`merge_reuse_ratio`、`redundancy_reject_ratio`、`proof_steps`。
3. **加权结构回报可审计证据**
   - 在加权父聚合时输出统计证据：`weighted_parent_terms`、`weighted_max_weight_mean`、`weighted_parent_count_mean`，用于理论分析与附录审计。

## 论文证明版（新增）
1. **参数化 Bellman 算子 `T_eta`（工程实现）**
   - 新增 `--mcts_objective_mode parametric` 与 `--mcts_theory_eta`，在 `mcts_solver.py` 中实现
     `T_eta(s)=r(s)+gamma*((1-eta) * running + eta * parent_agg)`。
2. **近似压缩映射的经验证据**
   - 新增 Bellman 残差统计：`bellman_residual_mean`、`bellman_residual_max`、`bellman_residual_count`。
3. **校准熵 UCB 的理论代理证据**
   - 新增 `--mcts_calibration_beta` 与 `ucb_calibration_mean`，记录熵校准放缩均值。
4. **论文表格直出统计**
   - `compare_mcts_budgets.py` 新增理论列导出：`bellman_residual_mean`、`ucb_calibration_mean`、`weighted_parent_var_proxy_mean`。

## 论文证明版补充（本次）
1. **定理假设对应的运行指标**
   - 新增 `theory_eta_contraction_gap_mean`：用于观测 `|running - parent_agg|` 的收缩代理。
   - 新增 `theory_conservative_gap_mean`：用于观测 `|parent_agg - conservative_parent|` 的稳健偏差代理。
   - 新增 `theory_unique_state_ratio`：用于观测 DAG 状态复用强度（unique states / simulations）。
2. **证明报告脚本**
   - 新增 `generate_theory_report.py`，从 `mcts_prediction_<split>.jsonl` 聚合上述理论指标并输出 JSON 报告。
3. **严格实验协议脚本**
   - 新增 `theorem_protocol.py`：联合 `compare_baselines.py` 的显著性结果与 `generate_theory_report.py` 的理论指标，
     输出 `all_pass` 协议检查（置换检验 p 值、bootstrap CI、收缩代理阈值等）。
4. **ECE/温度校准理论版 UCB（新增）**
   - 新增 `--mcts_calibration_mode {heuristic,ece_temp}`、`--mcts_ece_bins`、`--mcts_calibration_min_t`、`--mcts_calibration_max_t`。
   - 在 `ece_temp` 模式下，以 ECE 代理驱动温度缩放并注入 UCB 探索项；输出 `ece_proxy_mean` 与 `temperature_scale_mean` 供论文分析。
5. **形式化命题与证明文本（新增）**
   - 新增 `THEOREM_PROOFS.md`，包含参数化算子压缩性、保守聚合风险界、DAG 合并等价性命题与证明草图，并给出与代码指标的一一对应。
6. **Graph-MDP DAG 等价不变性检查（新增）**
   - 在 transposition merge 时新增 runtime invariant 检查：`dag_equivalence_violation_count`。
   - 理论报告与协议脚本加入 `dag_equivalence_violation_count`，要求其为 0。

示例：
```bash
python generate_theory_report.py \
  --prediction_file ../../output_dir/etree_task1/test/.../epoch_tree/dev/mcts_prediction_dev.jsonl \
  --save_json theory_report_dev.json

python theorem_protocol.py \
  --baseline_summary baseline_compare_summary.json \
  --theory_report theory_report_dev.json \
  --save_json theorem_protocol_check.json
```

## FAQ：改进“可证明加权结构回报”是否要改原代码？
结论：**要改，但改动范围可控，不需要推翻 PPO 主线**。

建议最小改动点：
1. `mcts_solver.py`
   - 增加“证明友好”的统计量输出（如权重分布分位数、父节点估计方差代理）。
   - 增加 `weight_mode` 的一致化归一逻辑，保证实验可比。
2. `arguments.py`
   - 增加理论实验开关（如方差代理类型、保守温度等），避免硬编码。
3. `compare_mcts_budgets.py` / `compare_baselines.py`
   - 导出与命题对应的统计列（方差、CI、paired 统计），用于论文表格直接引用。

不建议改动：
- 旧 PPO 训练主链（`RL_solver.py` 等）可保持不变，避免影响基线复现。
