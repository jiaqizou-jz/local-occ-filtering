# 验证计划：LoDADA 的 reverse-conditional score 会不会把 occupancy shift 当成 dynamics shift？

核心问题（first question）：

> LoDADA 比较的是 $P_s(s,a\mid s')$ vs $P_t(s,a\mid s')$。当 $P_s(s'\mid s,a)=P_t(s'\mid s,a)$ 但 $P_s(s,a)\neq P_t(s,a)$ 时，这个 score 是否仍然非零（false positive）？反过来，occupancy 能否恰好抵消真实 dynamics shift（false negative）？

分解：$\ell_{\rm rev}=\ell_{\rm dyn}+\ell_{\rm occ}-\ell_{\rm marginal}$。我们的 corrected score 是
$\hat\ell_{\rm dyn}(s,a,C)=\operatorname{logit}D_{SAC}(s,a,C)-\operatorname{logit}D_{SA}(s,a)\approx\log\frac{P_t(C\mid s,a)}{P_s(C\mid s,a)}$，$C=g(s')$ 为 target-fit 的共享聚类。

## 仓库结构

```
adaptation/                    # discrepancy estimation，与 RL 完全分离
  networks.py                  # 2x256 MLP + balanced source/target 训练（logit ≈ log p_t/p_s）
  clustering.py                # C = g(s')：target-fit kmeans / 1d quantile bins，两域共用
  occupancy_classifier.py      # D_SA
  sas_classifier.py            # D_SAS（DARC/DARA global baseline）
  joint_cluster_classifier.py  # D_SAC = f_occ(s,a) + g(s,a,embed C)，g 即 local dynamics residual
  reverse_classifier.py        # LoDADA baseline：每 cluster 一个 D_k(s,a)，cluster 内 balanced 训练
  score.py                     # 三种 log ratio
  filtering.py                 # exp(-alpha d) / quantile mask，仅 downstream RL 用
toy/
  linear_gaussian.py           # s' = A s + B a + eps, a = K s + eta；解析 KL / 逐样本 log ratio / cluster log ratio
  generate_dataset.py          # 四格 case A-D
  evaluate_ratio.py            # Experiment 1
  test_ratio_identity.py       # Experiment 0
algo/offline_offline/local_occ_iql.py   # IQL + 一次性 source 过滤（Experiment 4）
config/mujoco/local_occ_iql/   # 继承 iql 配置 + num_clusters / classifier_steps / keep_fraction
```

## 实验阶梯与 go/no-go

| # | 实验 | 命令 | 通过判据 |
|---|---|---|---|
| 0 | estimator 恒等式：joint − occ = conditional | `python -m pytest toy/test_ratio_identity.py` | Spearman > 0.9（DARA vs 真 $s'$ ratio；ours vs 真 cluster ratio；用 Spearman 是因为真 cluster ratio 有 −27 的截断尾部，logit 跟不上） |
| 1 | toy 四格矩阵 | `python -m toy.evaluate_ratio` | **Case B**：`abs_rev` 明显 > 0 且 `abs_local` ≈ `abs_rev(A)`；**Case D**：`rho_local` > `rho_rev`，多 seed 稳定 |
| 1b | adversarial false negative | 待写：选 $K_t$ 使 $\ell_{\rm occ}$ 与 $\ell_{\rm dyn}$ 反向 | $\ell_{\rm rev}\approx0$ 而 `abs_local` 仍非零 |
| 2 | ODRL 无 dynamics shift + behavior shift | `--env hopper-gravity --shift_level 1.0 --srctype medium-replay --tartype expert`（同 physics） | corrected score 分布比 reverse 更接近 0 |
| 3 | ODRL dynamics + behavior shift | gravity 0.5 / 2.0 + 不同 dataset quality | Spearman(ours, $D_{\rm true}$) > Spearman(LoDADA, $D_{\rm true}$)，$D_{\rm true}$ 由 probe (s,a) 在两个 simulator 里 Monte Carlo |
| 4 | downstream IQL | `python train.py --policy local_occ_iql --mode 3 --env hopper-gravity ...` | $J_t(\pi_{\rm ours})>J_t(\pi_{\rm LoDADA})$，同 budget / 同 seed |

只有 0–3 为正才进入 4。

## 指标约定

- 所有 score 均为 $\log p_t/p_s$ 估计，正 = 更像 target；discrepancy $d=-\ell$。
- Case A/B 真值恒为 0，Spearman 无定义（nan），只看 `abs_*`。
- Experiment 1 评估用 held-out source rollouts（seed+100），不用训练数据。

## 训练细节（已固定）

- source/target 每 minibatch 各半（`networks.train_domain_classifier`），无 prior correction。
- LoDADA baseline 在每个 cluster 内部 balanced 训练，使 logit ≈ $\log P_t(s,a\mid C)/P_s(s,a\mid C)$。
- joint classifier 中 $f_{\rm occ}$ 对 joint 分支 detach，残差 $g$ 的识别只由 occ loss 决定。
- cluster 由 target $s'$ 拟合，source/target 用同一 centroid / edges 分配。
- 第一阶段只看连续 score，不做 hard filtering。

## 结果记录

### Experiment 0（2026-10-15，seed 0，Case D，steps 3000，K = 10）

`python -m pytest toy/test_ratio_identity.py` 通过。对应的 Spearman：DARA vs 真 $s'$ ratio 0.986，ours vs 真 cluster ratio 0.958（与下表 seed 0 Case D 的 `rho_dara` / `rho_local` 相同量）。

### Experiment 1（2026-10-15，5 seeds，steps 3000，K = 10，held-out source 50k）

完整表格与逐 seed 数据：[results/exp1_toy_matrix_seeds5.md](../results/exp1_toy_matrix_seeds5.md)、[results/exp1_toy_matrix_seeds5.csv](../results/exp1_toy_matrix_seeds5.csv)。

| case | true_kl | abs_rev | abs_dara | abs_local | rho_rev | rho_local | rho_dara |
|---|---|---|---|---|---|---|---|
| A | 0.000 ± 0.000 | 0.189 ± 0.008 | 0.052 ± 0.015 | 0.048 ± 0.018 | nan | nan | nan |
| B | 0.000 ± 0.000 | 0.605 ± 0.036 | 0.123 ± 0.033 | 0.062 ± 0.007 | nan | nan | nan |
| C | 1.413 ± 0.018 | 1.556 ± 0.085 | 1.455 ± 0.056 | 1.231 ± 0.067 | 0.938 ± 0.008 | 0.966 ± 0.003 | 0.985 ± 0.004 |
| D | 1.413 ± 0.018 | 1.765 ± 0.060 | 1.391 ± 0.049 | 1.147 ± 0.056 | 0.894 ± 0.005 | 0.965 ± 0.005 | 0.983 ± 0.006 |

- 第一关（Case B）：`abs_rev` 0.605 ± 0.036 vs `abs_local` 0.062 ± 0.007，每个 seed 约 10 倍；`abs_local` 与 Case A 的 0.048 ± 0.018 同量级 → **通过，5/5 seed**
- 第二关（Case D）：`rho_local` 0.965 ± 0.005 > `rho_rev` 0.894 ± 0.005，逐 seed 差值 +0.06 到 +0.08 → **通过，5/5 seed**
- 首轮单 seed 结果（seed 0）被完整复现，已并入上表
- 未做：1b adversarial，K / N_t / steps 扫描，多维状态

## 待做

- [x] Experiment 0（pytest 通过）
- [x] Experiment 1 多 seed（5 seeds，两关均通过）
- [ ] Experiment 1b adversarial case E
- [ ] ODRL 侧：`adaptation/` 接 ODRL dataset，写 Layer-1 probe-state Monte Carlo 真值
- [ ] `local_occ_iql.py` 目前是 quantile hard filter，downstream 阶段再换 `exp_weights`
- [ ] 环境：ODRL 要 python 3.8 / mujoco-py / d4rl，toy 阶段 python 3.13 + torch 2.x 即可（本次 torch 2.14 cpu）
- [ ] 可选：Experiment 1 扫 K ∈ {5, 10, 20}、N_t ∈ {1k, 5k, 20k}，看 corrected 的优势是否随 target 样本量 / cluster 粗细变化
