# Hopper 管线冒烟结果（surrogate 数据，不是 ODRL 基准数据）

日期 2026-10-15。这里的数据由 `locomotion/collect_surrogate.py` 在 gymnasium hopper 里生成：source 为带噪周期控制器（动作噪声 0.6，60k），target 为同一控制器族、噪声 0.2（12k），episode 多在 20–50 步内摔倒。它只用来验证管线和真值定义，**不能**替代 D4RL / ODRL 数据下结论。完整数字见 `summary.txt`，逐 seed 见 `hopper_*.csv`，最后一个 seed 的逐样本 score 见 `hopper_*_scores.npz`。

设置：steps 5000，K = 10（target s' 上 kmeans），3 seeds，held-out 为 source 的 20% episode。score 均为 log p_t/p_s 估计。

## 三个对后续有约束力的发现

1. **gravity 不适合做 local 真值。** 一步内重力变化只把 z 速度平移 Δg·dt（飞行相），D_true 的 10/50/90/99 分位几乎相等（×0.5 时 0.09/0.13/0.13/0.13）。没有逐样本差异可排序，Spearman 无意义（三个 score 都是 ±0.3 的噪声）。Experiment 3 改用 friction：接触相 D_true 非零（p90 ≈ 0.7），飞行相为 0（约一半样本），是真正 state-dependent 的 shift。
2. **cluster 分辨率低。** K = 10 时 dynamics shift 只让 1.6%（gravity ×0.5）到 6%（friction ×2.0）的 held-out 样本换 cluster。cluster 级的真 ratio log P_t(C|s,a)/P_s(C|s,a) 对绝大多数样本 ≈ 0，LoDADA 和 ours 都受此限制，DARA 的 s' 级 score 不受。真实数据上要扫 K 并考虑 target s' 分布更集中时 flip 率是否更高。
3. **occupancy 混淆。** friction 实验里纯 occupancy score 对 D_true 的 Spearman 就有 0.40–0.48，和 reverse / ours / DARA 同量级，说明"是否接触"与 occupancy 强相关，所有 score 都能借 occupancy 猜中大部分排序。只看接触相内部：ours 和 reverse 的 Spearman ≈ 0 或为负，DARA +0.29 / +0.67。

## 对计划判据的对应

- Experiment 2 类比（同物理、不同 behavior）：abs_local 0.61 ± 0.04，abs_dara 0.91，abs_rev 9.86 ± 0.60。corrected 比 reverse 更接近 0 → 判据在 surrogate 上成立，但 reverse 的 9.9 主要是 surrogate 两域 occupancy 近乎不相交导致 logit 饱和，真实数据上差距会小得多。
- Experiment 3 类比（friction）：接触 vs 飞行的检测 AUC，×0.5：rev 0.80 / local 0.83 / dara 0.83；×2.0：rev 0.85 / local 0.90 / dara 0.77。ours 在"是否有 shift"的检测上最好或持平，在"shift 多大"的排序上不如 DARA。
