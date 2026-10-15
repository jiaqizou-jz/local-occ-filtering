# Experiment 1b: adversarial Case E

日期 2026-10-15。

## 1. 计划里字面的 false negative 构造不出来

计划要求"选 $K_t$ 使 $\ell_{\rm occ}$ 与 $\ell_{\rm dyn}$ 反向，$\ell_{\rm rev}\approx 0$ 而 $\ell_{\rm dyn}\neq 0$"。
用解析 ratio（`toy/linear_gaussian.py` 新增 `true_reverse_log_ratio`，rollout 的 state marginal 是逐步矩已知的高斯混合，三项都精确）在 target behavior $(K_t,\sigma_{a,t})$ 上网格搜索，Case C 的 dynamics shift 下 $E|\ell_{\rm rev}|/E|\ell_{\rm dyn}|$ 的最小值是 0.98，取在 source 自己的 behavior 上；再给 Domain 加常数漂移 $d$（可分离的 dynamics shift）和动作偏置 $b$，在 $(d_t,K_t,b_t,\sigma_{a,t})$ 上搜索，最小值仍 ≥ 1.03。

原因：$\ell_{\rm dyn}(s,a,s')=(\mu_t-\mu_s)\,s'/\sigma^2-(\mu_t^2-\mu_s^2)/2\sigma^2$，当 $\mu_t-\mu_s$ 依赖 $(s,a)$ 时交叉项不能写成 $f(s,a)+g(C)$，而 $\ell_{\rm occ}-\ell_{\rm marg}$ 只能抵消可分离的部分。平稳线性高斯下还能证明 $P_t(s,a\mid s')=P_s(s,a\mid s')$ 蕴含联合分布相等，即 reverse conditional 在这个家族里是完备统计量。**幅度层面，LoDADA 的失效是单向的：只有误报（Case B），没有漏报。**

搜索脚本：`python -m toy.design_case_e`，输出：

```
# magnitude: can any target behavior drive E|l_rev| below E|l_dyn|?  (C dynamics, grid over K and sigma_a)
K -0.80 sigma_a 0.30 ratio  0.98        abs_dyn   1.408  abs_rev   1.382  rho   0.964  fn   0.000  fp   0.000  
K -1.00 sigma_a 0.30 ratio  0.99        abs_dyn   1.411  abs_rev   1.395  rho   0.943  fn   0.000  fp   0.001  
K -0.60 sigma_a 0.30 ratio  1.00        abs_dyn   1.409  abs_rev   1.409  rho   0.965  fn   0.000  fp   0.000  
K -1.20 sigma_a 0.30 ratio  1.02        abs_dyn   1.419  abs_rev   1.442  rho   0.907  fn   0.003  fp   0.008  
K -0.80 sigma_a 0.45 ratio  1.03        abs_dyn   1.436  abs_rev   1.479  rho   0.926  fn   0.006  fp   0.002  

# ranking: target behavior noise sweep (C dynamics, K = source)
sigma_a 0.30                            abs_dyn   1.408  abs_rev   1.382  rho   0.964  fn   0.000  fp   0.000  
sigma_a 0.20                            abs_dyn   1.331  abs_rev   1.622  rho   0.832  fn   0.000  fp   0.059  
sigma_a 0.15                            abs_dyn   1.268  abs_rev   2.234  rho   0.632  fn   0.026  fp   0.147  
sigma_a 0.10                            abs_dyn   1.199  abs_rev   4.357  rho   0.351  fn   0.167  fp   0.257  
sigma_a 0.05                            abs_dyn   1.130  abs_rev  17.083  rho   0.048  fn   0.400  fp   0.351  

# chosen case E
E                                       abs_dyn   1.199  abs_rev   4.357  rho   0.351  fn   0.167  fp   0.257
```

## 2. 排序层面的漏报存在：Case E

对 filtering 真正要紧的是逐样本排序。target behavior 噪声比 source 小时（$\sigma_{a,t}=0.1$ vs 0.3，即 target 是 expert 数据、source 是 medium-replay，ODRL 的典型配置），**理想的** reverse score 也会漏掉 17% 最该过滤的样本、误报 26% 不该过滤的样本，Spearman 掉到 0.35（上表 sigma_a sweep，是 dose-response）。

Case E 定义：Case C 的 dynamics（A 0.9→0.6，B 0.5→0.8），target behavior $K=-0.8$（同 source），$\sigma_a=0.1$。

## 3. 估计器在 Case E 上的表现（5 seeds）

| score | abs | Spearman | FN | FP |
|---|---|---|---|---|
| 理想 reverse（解析） | 4.395 ± 0.025 | 0.350 ± 0.005 | 0.165 ± 0.005 | 0.257 ± 0.002 |
| LoDADA reverse | 4.008 ± 0.156 | 0.338 ± 0.017 | 0.168 ± 0.011 | 0.254 ± 0.009 |
| ours local corrected | 0.800 ± 0.035 | 0.703 ± 0.055 | 0.202 ± 0.038 | 0.055 ± 0.016 |
| DARA global（对 $s'$ 级真值） | 1.224 ± 0.062 | 0.883 ± 0.016 | 0.040 ± 0.011 | 0.014 ± 0.006 |

- reverse estimator 几乎完美复现理想 reverse 的失效：漏报和误报都是 score 本身的性质。
- corrected 修好了误报（0.254 → 0.055）和排序（0.34 → 0.70），**没有修好漏报**（0.202 vs 0.168）。
- corrected 的幅度 0.80 明显低于 C 的 1.23 和真 cluster-level 均值 1.2，估计器在 E 上丢失了量级。

## 4. 漏报在哪里：支撑诊断（seed 0）

把 held-out source 样本按解析的 $\log p_t(s,a)/p_s(s,a)$ 分三份：

```
case E seed 0, held-out source transitions split by analytic occupancy ratio log p_t(s,a)/p_s(s,a)
tercile                           n mean|truth|  rho_rev rho_local rho_dara |err|_local |err|_dara
off-support (low p_t/p_s)     16667       1.644    0.228     0.641    0.745       1.880      1.579
middle                        16666       1.343    0.778     0.770    0.984       0.845      0.540
on-support (high p_t/p_s)     16667       0.609    0.668     0.885    0.975       0.261      0.403

fraction of the top-25% most shifted transitions in each tercile: [0.388, 0.449, 0.163]
rev    false negatives: 1965 of 12500 shifted; share of those that are off-support tercile: 0.000; mean |local| on missed: 1.553, mean |truth| on missed: 1.901
local  false negatives: 2750 of 12500 shifted; share of those that are off-support tercile: 0.851; mean |local| on missed: 0.383, mean |truth| on missed: 3.149

mean action magnitude |a|: all 0.286 | top-25% shifted 0.452 | target data 0.121 | target 99th pct 0.458
```

- ours 的漏报 85% 在 target 支撑之外的 tercile；这些样本真实 shift 均值 3.15，ours 只给 0.38。cluster 级残差 $g(s,a,C)$ 只在两域都有数据的地方可识别，支撑外退化到 0，而 Case E 里 dynamics shift 随 $|a|$ 增大（$\mu_t-\mu_s=-0.3s+0.3a$），最该过滤的样本恰好 $|a|$ 大、在 target 的窄动作带之外。
- reverse 的漏报 0% 在支撑外，机制不同：是 occupancy 项在支撑内的逐样本抵消。ours 在 reverse 漏掉的样本上均值 1.55，能抓住它们。两者失效区域互补。
- DARA 在支撑外的 Spearman 0.745 高于 ours 的 0.641：$s'$ 级 log ratio 在这个 toy 里是 $(s,a,s')$ 的二次函数，MLP 外推得比 categorical cluster 残差好。这是 1 维 toy 的特性，高维 $s'$ 下未必成立。

## 5. 对 go/no-go 的意义

1b 的字面判据（$\ell_{\rm rev}\approx0$ 且 `abs_local` 仍非零）不成立，但不是因为 corrected 失败，而是这种 false negative 在该家族里不存在。改用逐样本判据后：

- reverse 的漏报真实存在且是固有的 → **成立**
- corrected 修误报、修排序 → **成立**
- corrected 修漏报 → **不成立**，漏报转移到 target 支撑外，是任何 occupancy-corrected 估计器的可识别性上限，不是 cluster 方法特有的缺陷，但 DARA 在这个 toy 里外推得更好

需要决定：filtering 阶段如何处理支撑外样本（corrected score 对它们没有信息，默认 0 意味着"保留"）。
