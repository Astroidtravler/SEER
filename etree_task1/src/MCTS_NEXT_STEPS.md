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
