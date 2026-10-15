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
| 0 | estimator 恒等式：joint − occ = conditional | `python -m pytest toy/test_ratio_identity.py` | Pearson > 0.9（DARA vs 真 $s'$ ratio；ours vs 真 cluster ratio） |
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

## 首轮结果（2026-10-15，seed 0，steps 3000，K = 10，held-out source 50k）

```
case      true_kl    abs_rev   abs_dara  abs_local    rho_rev  rho_local   rho_dara
A           0.000      0.203      0.064      0.019        nan        nan        nan
B           0.000      0.580      0.153      0.050        nan        nan        nan
C           1.410      1.559      1.415      1.201      0.946      0.967      0.980
D           1.410      1.838      1.353      1.122      0.898      0.958      0.986
```

- 第一关（Case B）：reverse 0.58 vs corrected 0.05，corrected 与 Case A 的 0.02 同量级 → 成立
- 第二关（Case D）：Spearman 0.958 (ours) > 0.898 (LoDADA) → 成立，但只有 1 个 seed
- 单 seed，未做 1b adversarial，未扫 K / N_t；不能据此下结论

## 待做

- [ ] Experiment 1 多 seed（`run(seeds = 5)`）
- [ ] Experiment 1b adversarial case E
- [ ] ODRL 侧：`adaptation/` 接 ODRL dataset，写 Layer-1 probe-state Monte Carlo 真值
- [ ] `local_occ_iql.py` 目前是 quantile hard filter，downstream 阶段再换 `exp_weights`
- [ ] 环境：ODRL 要 python 3.8 / mujoco-py / d4rl，toy 阶段本机 python 3.13 + torch 2.8 即可
