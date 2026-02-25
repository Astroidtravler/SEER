# Theorem Proofs (Items 1-3 Only, Strict Version)

> 说明：本文件只覆盖三项理论命题（加权结构回报、统一 Bellman 算子族、校准熵 UCB）。
> **不覆盖第 4 点（DAG 等价最优值不变）**，以满足“证明文档不覆盖第四点”的要求。

---

## 0. 统一记号

- 状态 \(s\) 的即时奖励为 \(r(s)\)，折扣系数 \(\gamma\in[0,1)\)。
- 父状态集合为 \(\mathcal P(s)\)，父值估计为 \(V_j\)（\(j\in\mathcal P(s)\)）。
- 加权父聚合：
  \[
  \bar V_w(s) = \sum_{j\in\mathcal P(s)} w_j V_j,\quad w_j\ge 0,\ \sum_j w_j=1.
  \]
- 参数化 Bellman 算子（代码 `parametric` 模式）：
  \[
  (T_\eta V)(s)=r(s)+\gamma\big((1-\eta)X_V(s)+\eta Y_V(s)\big),\quad \eta\in[0,1],
  \]
  其中 \(X_V\) 对应 rollout/`running`，\(Y_V\) 对应父聚合 `parent_agg`。
- 熵正则 UCB（校准版）写成：
  \[
  U(s,a)=Q(s,a)+c\,P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}\cdot \Phi(\mathcal H(s),\operatorname{CalErr}(s)),
  \]
  其中 \(\operatorname{CalErr}\) 是校准误差（代码中以 ECE proxy 近似）。

---

## 1) 加权结构回报：方差最优与保守上偏界

### 假设 A1（噪声模型）
父值估计满足
\[
V_j = \mu_j + \varepsilon_j,
\]
其中 \(\mathbb E[\varepsilon_j]=0\)，\(\operatorname{Var}(\varepsilon_j)=\sigma_j^2\)，且 \(\varepsilon_j\) 近似独立。

### 命题 1A（逆方差加权最小方差）
在约束 \(\sum_j w_j=1\) 下，线性无偏估计器
\[
\hat V = \sum_j w_j V_j
\]
的方差最小解为
\[
w_j^*=\frac{\sigma_j^{-2}}{\sum_k \sigma_k^{-2}},
\]
对应最小方差
\[
\operatorname{Var}(\hat V^*)=\frac{1}{\sum_k \sigma_k^{-2}}.
\]

#### 证明
目标函数为
\[
\min_w\ \sum_j w_j^2\sigma_j^2\quad s.t.\ \sum_j w_j=1.
\]
拉格朗日函数
\[
\mathcal L(w,\lambda)=\sum_j w_j^2\sigma_j^2+\lambda\left(\sum_j w_j-1\right).
\]
一阶条件：
\[
\frac{\partial \mathcal L}{\partial w_j}=2w_j\sigma_j^2+\lambda=0
\Rightarrow
w_j=-\frac{\lambda}{2\sigma_j^2}.
\]
代入约束得
\[
-\frac{\lambda}{2}\sum_j \sigma_j^{-2}=1
\Rightarrow
-\frac{\lambda}{2}=\frac{1}{\sum_j\sigma_j^{-2}}.
\]
故
\[
w_j^*=\frac{\sigma_j^{-2}}{\sum_k\sigma_k^{-2}}.
\]
方差表达式直接代入即得。

### 命题 1B（conservative 聚合上偏风险更紧）
设 \(V_{cons}=\min_j V_j\)，\(V_{avg}=\frac1m\sum_jV_j\)。对任意阈值 \(v\) 有
\[
\max(0,V_{cons}-v)\le \max(0,V_{avg}-v).
\]
因此
\[
\mathbb E\big[\max(0,V_{cons}-v)\big]\le
\mathbb E\big[\max(0,V_{avg}-v)\big].
\]

#### 证明
由 \(\min_jV_j\le \frac1m\sum_jV_j\) 和函数 \(f(x)=\max(0,x-v)\) 单调不减可直接推出不等式，取期望即得。

### 与代码对应
- `mcts_weight_mode=inv_var` 对应逆不确定性权重近似。  
- `objective_mode=conservative` 对应保守聚合路径。  
- 相关统计：`weighted_parent_var_proxy_mean`、`theory_conservative_gap_mean`。

---

## 2) 统一 Bellman 算子族：压缩映射性质

### 假设 A2（Lipschitz）
对任意 \(V_1,V_2\)，存在
\[
\|X_{V_1}-X_{V_2}\|_\infty\le \|V_1-V_2\|_\infty,
\quad
\|Y_{V_1}-Y_{V_2}\|_\infty\le \|V_1-V_2\|_\infty.
\]

### 命题 2（\(T_\eta\) 为 \(\gamma\)-压缩）
在 A2 与 \(\eta\in[0,1]\) 下，
\[
\|T_\eta V_1 - T_\eta V_2\|_\infty
\le \gamma\|V_1-V_2\|_\infty.
\]

#### 证明
\[
\begin{aligned}
|T_\eta V_1-T_\eta V_2|
&=\gamma\left|(1-\eta)(X_1-X_2)+\eta(Y_1-Y_2)\right|\\
&\le \gamma\left((1-\eta)|X_1-X_2|+\eta|Y_1-Y_2|\right)\\
&\le \gamma\left((1-\eta)+\eta\right)\|V_1-V_2\|_\infty\\
&=\gamma\|V_1-V_2\|_\infty.
\end{aligned}
\]
取上确界即得。

### 推论（偏差-方差权衡）
\(\eta\) 增大时更依赖父聚合（通常方差降低、偏差可能上升）；\(\eta\) 减小时更依赖 rollout（偏差可能降低、方差可能上升）。

### 与代码对应
- `objective_mode=parametric` + `mcts_theory_eta`。
- 指标：`theory_eta_contraction_gap_mean`、`bellman_residual_mean/max`、`theory_contraction_ratio_mean`。

---

## 3) 校准误差驱动的熵正则 UCB

### 假设 A3（校准误差可估）
存在状态级校准误差 \(\Delta_{cal}(s)\in[0,1]\)，代码中用 ECE proxy 近似：
\[
\widehat\Delta_{cal}(s)\approx \operatorname{ECE}(s).
\]

### 定义（校准温度）
\[
\tau(s)=\operatorname{clip}\left(1+\beta\widehat\Delta_{cal}(s),\ \tau_{min},\tau_{max}\right).
\]

### 命题 3（误校准增大时探索单调增强）
令
\[
\Phi(\mathcal H,\widehat\Delta_{cal}) = 1 + c_H\,\mathcal H\,\tau(s),
\]
其中 \(c_H>0\)，则当 \(\widehat\Delta_{cal}\) 增大且未触及 clip 上界时：
\[
\frac{\partial \Phi}{\partial \widehat\Delta_{cal}} = c_H\,\mathcal H\,\beta \ge 0.
\]
故误校准越大，探索项越大，降低高置信错先验被过早锁定的风险。

#### 证明
由 \(\tau(s)=1+\beta\widehat\Delta_{cal}(s)\)（未触顶）直接代入并求导。

### 与代码对应
- `mcts_calibration_mode=ece_temp`、`mcts_ece_bins`、`mcts_calibration_min_t/max_t`。
- 指标：`ece_proxy_mean`、`temperature_scale_mean`、`ucb_calibration_mean`。

---

## 投稿建议

- 正文给命题与关键证明；附录放完整假设与推导细节。  
- 实验表格对应输出：
  - 命题 1：`weighted_parent_var_proxy_mean`、`theory_conservative_gap_mean`；
  - 命题 2：`theory_contraction_ratio_mean`、`bellman_residual_mean/max`；
  - 命题 3：`ece_proxy_mean`、`temperature_scale_mean`、`ucb_calibration_mean`。
