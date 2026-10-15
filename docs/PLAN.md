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
  linear_gaussian.py           # s' = A s + B a + d + eps, a = K s + b + eta；解析 KL / 逐样本 log ratio / cluster log ratio / occupancy / 理想 reverse ratio
  generate_dataset.py          # 四格 case A-D + adversarial case E
  design_case_e.py             # Experiment 1b：在解析 ratio 上搜索 target behavior
  evaluate_ratio.py            # Experiment 1 + 1b，含逐样本 FN / FP 率
  test_ratio_identity.py       # Experiment 0
locomotion/
  data.py                      # D4RL / ODRL hdf5 → (s, a, s')，按 episode 划 held-out，source 统计量标准化
  truth.py                     # gymnasium hopper（= gym/ODRL 模型的局部坐标版）+ 程序化 gravity / friction shift；D_true(s,a) = 两模拟器 s' 的归一化 L2 距离
  evaluate_ratio.py            # Experiment 2 / 3：kmeans(target s') + 四个估计器，报 abs / Spearman / FN / FP
  collect_surrogate.py         # 仅冒烟用的模拟器数据
algo/offline_offline/local_occ_iql.py   # IQL + 一次性 source 过滤（Experiment 4）
config/mujoco/local_occ_iql/   # 继承 iql 配置 + num_clusters / classifier_steps / keep_fraction
```

## 实验阶梯与 go/no-go

| # | 实验 | 命令 | 通过判据 |
|---|---|---|---|
| 0 | estimator 恒等式：joint − occ = conditional | `python -m pytest toy/test_ratio_identity.py` | Spearman > 0.9（DARA vs 真 $s'$ ratio；ours vs 真 cluster ratio；用 Spearman 是因为真 cluster ratio 有 −27 的截断尾部，logit 跟不上） |
| 1 | toy 四格矩阵 | `python -m toy.evaluate_ratio` | **Case B**：`abs_rev` 明显 > 0 且 `abs_local` ≈ `abs_rev(A)`；**Case D**：`rho_local` > `rho_rev`，多 seed 稳定 |
| 1b | adversarial false negative（Case E） | `python -m toy.design_case_e` 设计，`python -m toy.evaluate_ratio --cases E --seeds 5` 评估 | 原判据 $\ell_{\rm rev}\approx0$ 而 `abs_local` 仍非零 **在该家族不可构造**（见结果记录）；改为逐样本：`fn_rev_true` 明显 > 0，且 `fn_local` < `fn_rev`、`fp_local` < `fp_rev` |
| 2 | ODRL 无 dynamics shift + behavior shift | `python -m locomotion.evaluate_ratio --source hopper_medium_replay-v2.hdf5 --target hopper_expert-v2.hdf5`（D4RL 两个数据集，同 physics） | corrected score 分布比 reverse 更接近 0 |
| 3 | ODRL dynamics + behavior shift | **friction** 0.5 / 2.0（gravity 的一步 $D_{\rm true}$ 几乎是常数，见结果记录）+ 不同 dataset quality；`--shift friction --level 0.5` | Spearman(ours, $D_{\rm true}$) > Spearman(LoDADA, $D_{\rm true}$)，$D_{\rm true}$ 由 probe (s,a) 在两个 simulator 里单步得到（确定性模拟器，无需 Monte Carlo）；附接触 vs 飞行的检测 AUC |
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

### Experiment 1 + 1b（2026-10-15，5 seeds，steps 3000，K = 10，held-out source 50k）

完整表格：[results/exp1_toy_matrix_seeds5.md](../results/exp1_toy_matrix_seeds5.md)；Case E 构造、支撑诊断与解读：[results/exp1b_case_e.md](../results/exp1b_case_e.md)。

| case | true_kl | abs_rev | abs_dara | abs_local | abs_rev_true |
|---|---|---|---|---|---|
| A | 0.000 ± 0.000 | 0.189 ± 0.008 | 0.052 ± 0.015 | 0.048 ± 0.018 | 0.000 ± 0.000 |
| B | 0.000 ± 0.000 | 0.605 ± 0.036 | 0.123 ± 0.033 | 0.062 ± 0.007 | 0.468 ± 0.005 |
| C | 1.413 ± 0.018 | 1.556 ± 0.085 | 1.455 ± 0.056 | 1.231 ± 0.067 | 1.382 ± 0.009 |
| D | 1.413 ± 0.018 | 1.765 ± 0.060 | 1.391 ± 0.049 | 1.147 ± 0.056 | 1.673 ± 0.015 |
| E | 1.413 ± 0.018 | 4.008 ± 0.156 | 1.224 ± 0.062 | 0.800 ± 0.035 | 4.395 ± 0.025 |

| case | rho_rev | rho_local | rho_dara | rho_rev_true |
|---|---|---|---|---|
| A | nan | nan | nan | nan |
| B | nan | nan | nan | nan |
| C | 0.938 ± 0.008 | 0.966 ± 0.003 | 0.985 ± 0.004 | 0.964 ± 0.001 |
| D | 0.894 ± 0.005 | 0.965 ± 0.005 | 0.983 ± 0.006 | 0.919 ± 0.001 |
| E | 0.338 ± 0.017 | 0.703 ± 0.055 | 0.883 ± 0.016 | 0.350 ± 0.005 |

| case | fn_rev | fn_local | fn_dara | fn_rev_true |
|---|---|---|---|---|
| A | nan | nan | nan | nan |
| B | nan | nan | nan | nan |
| C | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| D | 0.002 ± 0.001 | 0.003 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| E | 0.168 ± 0.011 | 0.202 ± 0.038 | 0.040 ± 0.011 | 0.165 ± 0.005 |

- 第一关（Case B）：`abs_rev` 0.605 ± 0.036 vs `abs_local` 0.062 ± 0.007，每个 seed 约 10 倍，与 Case A 同量级 → **通过，5/5**。理想 reverse 在 B 上是 0.468：误报是 score 固有的。
- 第二关（Case D）：`rho_local` 0.965 > `rho_rev` 0.894，逐 seed +0.06 到 +0.08 → **通过，5/5**
- 1b（Case E）：幅度层面的 false negative 在线性高斯家族不可构造（$E|\ell_{\rm rev}|/E|\ell_{\rm dyn}|\ge0.98$ 对所有 target behavior，含漂移型 shift）。排序层面：理想 reverse 漏报 0.165、误报 0.257；corrected 误报降到 0.055、Spearman 0.34 → 0.70 → **成立**；corrected 漏报 0.202 不低于 reverse 的 0.168 → **不成立**，85% 的漏报在 target 支撑之外（可识别性上限）。DARA global 在 E 上 FN 0.040、Spearman 0.883，1 维 toy 里 $s'$ 级外推优于 cluster 残差。
- 首轮单 seed 结果（seed 0）被完整复现。
- 未做：K / N_t / steps 扫描，多维状态。

### ODRL 侧（2026-10-15）：管线已就绪，基准数据被网络策略挡住

- 模拟器：容器是 python 3.13，mujoco-py 不可用；ODRL 的 xml 用 `coordinate="global"`，MuJoCo ≥ 2.3.4 不再支持。gymnasium 自带的 hopper.xml 是同一模型的局部坐标版（RK4，dt 0.002 × 4，obs = qpos[1:] + clip(qvel)），ODRL 的 gravity / friction 变体只改 option gravity / geom friction，已程序化复现并验证：确定性、从 obs 重建状态后单步结果与原 rollout 一致（误差 1e-16）、x 平移不变。
- 数据：D4RL（rail.eecs.berkeley.edu）、ODRL target（drive.google.com）、Hugging Face 镜像在本环境均被拒绝；Google Drive 连接器能按 ID 读到 ODRL 文件夹的元数据但列不出子文件。**需要在环境设置里放行域名，或把 hdf5 放进仓库可达的位置。**
- surrogate 冒烟（见 [results/surrogate/README.md](../results/surrogate/README.md)）给出三个约束：gravity 的一步 $D_{\rm true}$ 几乎常数，Experiment 3 改用 friction；K = 10 时 shift 只让 2–6% 样本换 cluster，cluster 级分辨率是 LoDADA 和 ours 共同的上限；friction 的接触/飞行与 occupancy 强相关，纯 occupancy score 的 Spearman 已有 0.4，真实数据上要同时报接触相内部的排序。

## 待做

- [x] Experiment 0（pytest 通过）
- [x] Experiment 1 多 seed（5 seeds，两关均通过）
- [x] Experiment 1b adversarial Case E（字面判据不可构造；逐样本判据：修误报、修排序成立，修漏报不成立）
- [ ] **待决定**：支撑外样本的处理（corrected score 在 target 支撑外无信息，默认 0 = 保留）；候选：用 $D_{SA}$ 的 occupancy ratio 做置信度门控，或低维时用 $s'$ 级残差（即 DARA 形式）代替 cluster 残差
- [x] ODRL 侧管线：`locomotion/` 数据加载、模拟器真值、评估脚本（surrogate 冒烟通过）
- [ ] **阻塞**：拿到 D4RL hopper medium-replay / expert 与 ODRL hopper_friction_{0.5,2.0}_{medium,expert} hdf5（放行 rail.eecs.berkeley.edu、drive.google.com、docs.google.com、*.googleusercontent.com，或手动放入 `dataset/mujoco/`）
- [ ] Experiment 2：D4RL medium-replay vs expert（同 physics），5 seeds
- [ ] Experiment 3：friction 0.5 / 2.0，扫 K ∈ {5, 10, 20, 50}，报 Spearman、FN/FP、接触 vs 飞行 AUC、接触相内 Spearman
- [ ] `local_occ_iql.py` 目前是 quantile hard filter，downstream 阶段再换 `exp_weights`
- [ ] 环境：ODRL 要 python 3.8 / mujoco-py / d4rl，toy 阶段 python 3.13 + torch 2.x 即可（本次 torch 2.14 cpu）
- [ ] 可选：Experiment 1 扫 K ∈ {5, 10, 20}、N_t ∈ {1k, 5k, 20k}，看 corrected 的优势是否随 target 样本量 / cluster 粗细变化
