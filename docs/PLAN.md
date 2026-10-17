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
  adversarial.py               # Experiment 1b：偏置型动态偏移 + 指数倾斜 occupancy（闭式），LoDADA 的 l_rev 精确为 0
odrl/
  datasets.py                  # 源 = d4rl hopper，目标 = ODRL hdf5 (dataset/mujoco/download.sh) 或 d4rl 同物理集
  ground_truth.py              # probe-state 真值：观测重建 qpos/qvel，两个 simulator 各走一步，gap = |f_t - f_s| / std
  evaluate_ratio.py            # Experiment 2 + 3：null / behavior / g{0.5,2.0}_{expert,medium,random} / paired_<g>（perfect support）
  resolution.py                # Gate R：配对开环 rollout 的 cluster flip rate、gap cv
  detectability.py             # Gate P / S：5 折 cross-fit 条件对数损失增益 + 打乱标签对照 + on-support 行；cluster_on = next | delta
  oracle.py                    # Gate O：配对目标集（paired / pairedheld / pairedbig）、oracle 分数（state gap / Bellman gap / DARA / random）、held-out Bellman 残差
  gate_o_summary.py            # Gate O 汇总：每 run 最后 5 次 target eval 均值（--resid 加 Bellman 残差）
  gate_o_mask_diag.py          # Gate O 诊断：各 mask 的 terminal 比例 / gap / 闭包最近邻距离
  gate_o_rollout_diag.py       # Gate O 诊断：保存的 actor 在 source / target 环境重放（回合长度、速度、最低高度）
  gate_o_qv_diag.py            # Gate O 诊断：held-out 目标集上 Q / V 量级（发散检查）
  run_*.sbatch                 # 每个 gate 一个 sbatch（defq GPU / CPU）
adaptation/normalized_cluster_classifier.py   # Step 1 归一化估计器（g 对 log P̂_s(k|z) 做 logsumexp 归一）+ 两分类器差 sep
algo/offline_offline/local_occ_iql.py   # IQL + 一次性 source 过滤（Experiment 4，未用）
algo/offline_offline/score_iql.py       # Gate O：IQL + 按预计算分数的 source 选择（concat | target_only | <score key>）+ transition 级目标集覆盖
config/mujoco/local_occ_iql/   # 继承 iql 配置 + num_clusters / classifier_steps / keep_fraction
config/mujoco/score_iql/       # 继承 iql 配置 + source / keep_fraction / score_file / target_file
```

## 实验阶梯与 go/no-go

| # | 实验 | 命令 | 通过判据 |
|---|---|---|---|
| 0 | estimator 恒等式：joint − occ = conditional | `python -m pytest toy/test_ratio_identity.py` | Pearson > 0.9（DARA vs 真 $s'$ ratio；ours vs 真 cluster ratio） |
| 1 | toy 四格矩阵 | `python -m toy.evaluate_ratio` | **Case B**：`abs_rev` 明显 > 0 且 `abs_local` ≈ `abs_rev(A)`；**Case D**：`rho_local` > `rho_rev`，多 seed 稳定 |
| 1b | adversarial false negative | `python -m toy.adversarial`（平稳一步采样）：F = 偏置偏移 $c$（KL 同 C/D）+ 源 occupancy；E = 同一偏移 + 倾斜 occupancy $\delta=(c/\sigma^2)\Sigma_{sa}(A,B)^\top$，解析 $\mathbb E|\ell_{\rm rev}|$ = 0 | 全部 seed：`abs_rev(E)` ≤ 2×`abs_rev(A)`、`abs_local(E)` ≥ 0.5×`abs_local(F)`、`rho_local(E)` > 0.9 |
| 2 | 同 physics + behavior shift | `python -m odrl.evaluate_ratio` 的 `null` / `behavior` 行：源 d4rl hopper-medium-replay-v2，目标 5000 条 hopper-expert-v2（ODRL 没有 gravity 1.0 数据，`call_mujoco_env` 也只认 0.1/0.5/2.0/5.0） | `abs_local(behavior) − abs_local(null)` < `abs_rev(behavior) − abs_rev(null)` |
| 3 | ODRL dynamics + behavior shift | 同一脚本的 `g{0.5,2.0}_{expert,medium,random}` 行（`odrl/run_exp23.sbatch`，3 seed） | 每行 `rho_local` > `rho_rev`；$D_{\rm true}$ = held-out 源 (s,a) 在两个 simulator 各走一步的标准化 next-state gap（mujoco 确定性，不需 Monte Carlo），`recon` 列核对源 simulator 复现数据 s' |
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

## 五 seed 结果（2026-10-15，job 202589，node004 CPU，每格 ~300 s）

```
case           true_kl         abs_rev        abs_dara       abs_local         rho_rev       rho_local        rho_dara
A         0.000±0.000    0.189±0.008    0.052±0.014    0.047±0.019             nan             nan             nan
B         0.000±0.000    0.605±0.036    0.122±0.032    0.062±0.006             nan             nan             nan
C         1.413±0.018    1.557±0.084    1.452±0.055    1.229±0.068    0.938±0.008    0.966±0.003    0.984±0.004
D         1.413±0.018    1.767±0.062    1.391±0.048    1.148±0.056    0.894±0.005    0.965±0.004    0.983±0.005
```

- 第一关（Case B）：`abs_rev` 0.56–0.67 vs `abs_local` 0.05–0.07，后者与 Case A 的 0.02–0.08 同量级 → 5/5 seed 成立
- 第二关（Case D）：`rho_local` > `rho_rev` 5/5 seed（C 也 5/5），差距 0.06–0.08，远大于 seed 间 std → 成立
- DARA（全局 D_SAS，用完整 s'）的 `rho_dara` 在 C/D 都高于 `rho_local`（5/5、5/5）。toy 里 s' 是 1 维精确可观测，D_SAS 本来就是 cluster 版的上界；toy 还没有一个让全局 SAS 退化的轴（高维 s'、N_t 小时 D_SAS 过拟合），所以 Exp 1 只证明"比 LoDADA 好"，不证明"该用局部聚类而不是全局 SAS"

## Experiment 1b 的构造说明（2026-10-15）

- $\ell_{\rm rev}=\ell_{\rm dyn}+\ell_{\rm occ}-\ell_{\rm marginal}\equiv0$ 要求 $\ell_{\rm dyn}(s,a,s')=h(s,a)+m(s')$ 可加分离，且 $p_t(s,a)\propto p_s(s,a)e^{-h(s,a)}$（指数倾斜）
- 高斯动态：$\ell_{\rm dyn}\propto(\mu_t-\mu_s)(2s'-\mu_s-\mu_t)$，只有 $\mu_t-\mu_s$ 与 $(s,a)$ 无关（**偏置型**，重力类）时可分；**增益型**（A/B，摩擦、质量、执行器类）有 $(s,a)\times s'$ 交叉项，任何行为都抹不掉。解析网格搜索证实：C 的增益偏移下，$(K_t,b_t,\sigma_{a,t})$ 全网格最小 $\mathbb E|\ell_{\rm rev}|$ 仍 ≈ 源行为的值
- 偏置型下只换策略偏置 $b$ 也不够（$b=-c/B$ 反而把 $|\ell_{\rm rev}|$ 从 1.69 推到 2.35）：反向条件 $p(a\mid s,s')$ 本质是逆动力学，策略只能改先验。要抵消必须**初始状态均值和策略偏置一起**按 $\delta$ 移，这在平稳一步采样下闭式可达（`toy/adversarial.py`：$\delta_s=+0.364$（1.75 个平稳 std）、$b=+0.756$，解析 $|\ell_{\rm rev}|$=0.0000，KL 仍 1.41）
- 直觉：LoDADA 分不开"$s'$ 偏大是因为动态把它推大"和"因为 agent 恰好常去会导向大 $s'$ 的 $(s,a)$"
- 对 ODRL 的可证伪预测：重力偏移属偏置型，是 LoDADA 最可能漏报的轴；摩擦/形态属增益型，漏报风险小

## Experiment 2 / 3 结果（2026-10-15，job 202620，hopper，源 medium-replay，3 seed，steps 3000，K = 10）

```
case                    gap         recon       abs_rev      abs_dara     abs_local       rho_rev     rho_local      rho_dara
null            0.000±0.000   0.000±0.000   1.773±0.042   0.204±0.019   0.149±0.005           nan           nan           nan
behavior        0.000±0.000   0.000±0.000  38.387±1.434   2.406±0.473   3.011±0.126           nan           nan           nan
g0.5_expert     0.030±0.000   0.000±0.000  28.209±0.143   3.245±0.454   4.276±0.384   0.059±0.023   0.053±0.068   0.136±0.060
g0.5_medium     0.030±0.000   0.000±0.000  33.617±0.395   3.602±0.392   5.071±0.110  -0.061±0.013   0.070±0.031   0.090±0.057
g0.5_random     0.113±0.000   0.000±0.000  35.050±0.683   7.843±1.379   3.544±0.595   0.435±0.010   0.356±0.077   0.394±0.072
g2.0_expert     0.042±0.000   0.000±0.000  28.967±1.620   3.187±0.319   5.349±0.311   0.148±0.046   0.313±0.033   0.099±0.087
g2.0_medium     0.046±0.000   0.000±0.000  32.456±0.654   4.028±0.531   5.367±0.256   0.027±0.031   0.042±0.050   0.013±0.063
g2.0_random     0.173±0.000   0.000±0.000  30.529±1.626   7.257±1.628   4.284±0.863   0.460±0.010   0.320±0.050   0.388±0.066
```

- `recon` = 0：源 simulator 从观测重建状态后走一步与数据 s' 一致，gap 真值可信
- Exp 2（behavior 行减 null 行）：LoDADA +37、ours +2.9、DARA +2.2，判据形式上成立，但 ours 的假阳性也是地板的 ~20×：medium-replay 与 expert 的 support 几乎不重叠，所有分类器饱和，logit 差在 support 外没有校准意义
- LoDADA 的 null 地板 1.8 vs ours 0.15 vs DARA 0.2：每簇只有 N_t/K = 500 条目标数据，分簇分类器过拟合——这正是 joint classifier 设计要避开的瓶颈，在真实数据上得到了证实
- Exp 3（`rho_local` > `rho_rev` 按行计数）：{'g0.5_expert': 2, 'g0.5_medium': 3, 'g0.5_random': 1, 'g2.0_expert': 3, 'g2.0_medium': 2, 'g2.0_random': 0}；DARA > ours：{'g0.5_expert': 2, 'g0.5_medium': 2, 'g0.5_random': 2, 'g2.0_expert': 0, 'g2.0_medium': 1, 'g2.0_random': 2}。全部相关 ≤ 0.45：重力 0.5/2.0 的单步 gap 只有 0.03–0.11 个标准差（gravity 只改速度更新里的一个常数加速度项，dt = 0.008 s），三家都看不见；最可辨识的是 random 质量的目标（gap 最大、与 medium-replay 覆盖重叠最多），且在 g0.5_random 上 LoDADA 反超 ours 2/3
- 结论：在 N_t = 5000 的 ODRL hopper-gravity 上，transition 级的动态差异基本不可辨识，三个过滤器实际都在按 occupancy 过滤。这比原问题更根本，但要先排除三个解释：classifier 只训 3000 步（ODRL 用 20000）、kmeans 用了按维标准化的 s'、只测了 hopper

### occupancy 对照列（2026-10-15，job 202638，同设置重跑，加 `occ` = logit D_SA 单独的分数）

```
case                abs_rev      abs_dara     abs_local       abs_occ       rho_rev     rho_local      rho_dara       rho_occ
null            1.773±0.042   0.204±0.019   0.149±0.005   0.225±0.017           nan           nan           nan           nan
behavior       38.387±1.434   2.406±0.473   3.011±0.126   9.373±0.350           nan           nan           nan           nan
g0.5_expert    28.209±0.143   3.245±0.454   4.276±0.384  17.392±0.189   0.059±0.023   0.053±0.068   0.136±0.060  -0.026±0.021
g0.5_medium    33.617±0.395   3.602±0.392   5.071±0.110  18.569±0.947  -0.061±0.013   0.070±0.031   0.090±0.057  -0.071±0.005
g0.5_random    35.050±0.683   7.843±1.379   3.544±0.595  21.214±1.258   0.435±0.010   0.356±0.077   0.394±0.072   0.452±0.011
g2.0_expert    28.967±1.620   3.187±0.319   5.349±0.311  18.473±0.372   0.148±0.046   0.313±0.033   0.099±0.087   0.285±0.007
g2.0_medium    32.456±0.654   4.028±0.531   5.367±0.256  19.940±0.939   0.027±0.031   0.042±0.050   0.013±0.063   0.011±0.011
g2.0_random    30.529±1.626   7.257±1.628   4.284±0.863  17.235±0.999   0.460±0.010   0.320±0.050   0.388±0.066   0.490±0.019
```

- random 两行：`rho_occ` 0.45 / 0.42 与 LoDADA 0.43 / 0.46、DARA 0.39 / 0.39 同量级，ours 0.36 / 0.32 还更低。纯 occupancy 分类器就能排出 gap → 这两行的"信号"是 occupancy 与 gap 真值的相关（随机策略常去的状态正是重力影响大的状态），不是任何估计器在测动态
- expert / medium 四行：`rho_occ` ≈ 0 到 0.3，其他三家也 ≤ 0.31；g2.0_expert 上 ours 0.31 vs occ 0.24，差不出来
- 结论不变且更强：hopper-gravity、N_t = 5000、单步、K = 10 下，没有任何一列能被解释为"检测到了动态偏移"

## Experiment 1b 结果（2026-10-16，job 202626，平稳一步采样，3 seed，`true_rev` = s' 级解析 mean |l_rev|）

```
case           true_kl        true_rev         abs_rev        abs_dara       abs_local         rho_rev       rho_local        rho_dara
A         0.000±0.000    0.000±0.000    0.169±0.005    0.046±0.014    0.038±0.006             nan             nan             nan
C         1.038±0.001    1.332±0.007    1.408±0.042    1.293±0.027    1.152±0.025    0.931±0.003    0.970±0.003    0.988±0.003
F         1.410±0.000    1.484±0.004    1.417±0.029    1.837±0.051    1.580±0.056    0.854±0.008    0.986±0.001    0.997±0.001
E         1.410±0.000    0.000±0.000    3.536±0.495    1.628±0.216    0.739±0.102   -0.951±0.008    0.991±0.002    0.950±0.017
```

- 预注册判据：`abs_rev(E)` ≤ 2×`abs_rev(A)` **0/3**；`abs_local(E)` ≥ 0.5×`abs_local(F)` **1/3**；`rho_local(E)` > 0.9 **3/3**
- 第一条判据写错了层级：抵消是在 s' 级逐点成立的（`true_rev` = 0.000），LoDADA 的估计器在 cluster 级，分箱之后有界的 coarse dynamics 项抵消不了（c/σ² = 16.8 倍的）occupancy 倾斜，剩下的是**符号反转**：`rho_rev(E)` = −0.95（3/3）。解析核查（`exact_cluster_reverse`）：精确 cluster 级 |l_rev_C| = 3.0、与 l_dyn_C 的 Spearman −0.96，估计器忠实。所以 LoDADA 在 E 上不是"看不见"，而是把最可迁移的样本排到最后——比预注册的漏报更坏
- ours：`rho_local(E)` 0.99（3/3，四种方法里最高），幅度判据只过 1/3——E 的精确 |l_dyn_C| 只有 0.11（边界按 E 的目标 s' 分位数定，源样本挤在低位箱），ours 幅度 0.74 高估约 6 倍但排序正确；DARA rho 0.95
- 结论：假阴性构造成立且估计器层面表现为符号反转；修正估计器的排序不受影响。判据应改为 `rho_rev(E)` ≤ 0 且 `rho_local(E)` > 0.9（事后修订，需在论文里注明）

## 评注核对与重构（2026-10-16）

用户给的评注把问题从"给 LoDADA 加 occupancy correction"升级为 identifiability / detectability。逐条对照本仓库的测量：

- **成立（有测量支撑）**：Case B 假阳性（5/5 seed）；Case D 修正后更接近真值但 DARA 仍是上界；1b 的充要结构（可加分离 + 指数倾斜）与"偏置型可抵消、增益型不可"；E 的 s' 级抵消在 cluster 级变成符号反转（解析核查 −0.96）；Exp 2 的饱和来自 support 不重叠；Exp 3 上三家都在按 occupancy 排序。
- **是推断、尚未测量**：评注说"$D(q_s(K\mid z)\,\|\,q_t(K\mid z))\approx0$，shift 全在 cluster 内残差"。Exp 3 只测了估计器的 rho，没有直接测 coarse 通道。mujoco 确定性下这项逐点取值只有 0 或 ∞，正确的量是翻转率 $P_z[g(f_s(z))\neq g(f_t(z))]$（Gate R）与 held-out 条件 log-loss 增益（Gate 2）。
- **需要修正的表述**：(i) 评注把可分离性写成 $\ell_{\rm dyn}(z,k)=h(z)+m(k)$，本仓库的结果是在连续 $s'$ 上 $h(s,a)+m(s')$，cluster 级恰恰不可分离——这正是符号反转的来源；(ii) 评注引用 arXiv 2602.21072 作为 DARC 的出处，核对后该编号是 LoDADA 论文（Xia, Yang, Xu, "Localized Dynamics-Aware Domain Adaption for Off-Dynamics Offline RL"），DARC 是 Eysenbach et al. 2020（arXiv 2006.13916），引用前要分开；(iii) Exp 3 表里 `gap` 跨 quality 行不同（0.030 vs 0.113）只是 `scale` = 目标 s' 标准差不同造成的，$f_t(z)-f_s(z)$ 本身对同一重力是同一组数（held-out 源 z 相同），`odrl/resolution.py` 改按源物理结果的 std 归一；(iv) `gap` 从未被称为 KL，但理论里的 KL 与确定性 simulator 对不上：若两域加同一高斯观测噪声 $\sigma$，KL = gap²/(2σ²)，单调于 gap，Spearman 对 gap 的评估不受影响，但幅度主张（Theorem 3/4 的 bound）要以 gap² 口径重写。
- **不同意或存疑**：(i) 评注的 Gate 1 只比 LoDADA 和 occupancy，漏了 DARA；toy 上 DARA 5/5 高于 ours，存在理由问题（高维 s' / 小 N_t 下全局 D_SAS 退化）必须进 Gate；(ii) 两通道权重 $w_{\rm dyn}w_{\rm ov}w_{\rm unc}$ 引入 α、τ、λ 三个超参，比 LoDADA 复杂，按"修复应比 baseline 简单"的原则，先证明 Gate 2 在某个 observable 上能过，再谈加权；(iii) "occupancy 在 random 行有预测价值"是对的，但机制是接触模式：站立接触态下重力变化被地面反力吸收、飞行态直接进速度，random 策略多在接触态。这是 gap 与 occupancy 的混杂，不是 occupancy 可用的证据。

### 预注册（2026-10-16，启动前写下）

**Gate R（resolution，`odrl/resolution.py`）**：mujoco 确定性 ⇒ $I(D;K\mid Z)\le\ln2\cdot\mathrm{flip}$，$\mathrm{flip}=P_z[g(f_s(z))\neq g(f_t(z))]$；held-out 源 20k 个 20 步窗口，同一动作序列在两种物理下开环 h 步。扫 gravity {0.5, 2.0} × h {1, 5, 20} × feature {last = $s_{t+h}$, delta = $s_{t+h}-s_t$, window} × K {10, 30, 100, 300, 1000} × partition {target-fit（h = 1，对应 Exp 3）, pooled-fit（在待分开的配对结果上拟合，有利）}。
- 预测：h = 1、K = 10、target-fit 的 flip < 2%（两种重力）；flip 随 K、h 上升；flip ≥ 10% 需要 h ≥ 5 或 K ≥ 300；delta ≥ last。
- 判据：flip ≥ 0.10 的配置才算 coarse 通道有可检测信息；若 h = 1 的全部配置 flip < 0.10，Exp 3 的阴性归因为 representation 不可辨识，不是 estimator。

**Gate 2（detectability，`odrl/detectability.py` 的 gain 行）**：目标 5 折 cross-fit，balanced held-out CE（nats）；g_coarse = CE(occ) − CE(joint)，g_full = CE(occ) − CE($D_{SAS}$)；`*_fit` 为训练集上的同一量（过拟合参照）。null、behavior 行真值恰为 0（同物理 + 确定性），是同 N_t 的有限样本地板。
- smoke（20 步、N_t 500）发现 behavior 行 g_coarse = +0.085 而真值为 0：残差分支多一套 (s, a) 输入，拟合 occupancy 比 occ 分支快，增益被容量 / 优化差异污染。因此加 `*_shuf` 对照：K（或 s'）在两域合并后随机置换，$I(D;\cdot\mid Z)=0$ 按构造成立，同模型同数据测出的增益就是该行的容量地板。检验量 = g − g_shuf。
- 判据（逐重力行）：3/3 seed 上 g_coarse − g_coarse_shuf > 0，且大于 null / behavior 行的同一差值；g_full 同理。
- 预测：K = 10、h = 1 下所有重力行 coarse 不过；full 在 expert / medium 行不过，random 行不确定（倾向不过）；null / behavior 行的 g − g_shuf 在 ±0.005 内。

**Gate S（support，`odrl/detectability.py` 的 supp 行）**：$e(z)=\sigma(\operatorname{logit}D_{SA})$，$\Omega_\tau=\{\tau\le e\le1-\tau\}$，τ ∈ {0, 0.02, 0.1, 0.2}。
- 评注的假设：behavior 行的 abs_local 饱和来自 off-support。判据：τ = 0.1 下 abs_local(behavior | Ω) ≤ 2 × abs_local(null | Ω)，3/3 seed。替代结论：若仍 ≫ null，corrected estimator 在共同 support 上有残余假阳性，是方法问题。
- 预测：behavior 行 τ = 0.1 的 frac < 0.2；判据倾向不通过（D_SA 与 D_SAC 在 overlap 区同样欠拟合，残差不为 0）。frac < 0.05 判为低功效，不下结论。

**1b 复现（`toy/run_adversarial_replicate.sbatch`，seed 3–5）**：事后判据 rho_rev(E) ≤ 0 且 rho_local(E) > 0.9 在全新 seed 上 3/3；原判据 abs_local(E) ≥ 0.5·abs_local(F) 预期仍不过（箔位置），一并报告。

## Gate R / 2 / S 与 1b 复现结果（2026-10-16，job 202643 / 202645 / 202644）

### Gate R（resolution，`logs/odrl_resolution-202643.log`，held-out 源 20k 个 20 步窗口，同一动作序列开环）

```
gravity  h   feature  partition         gap     recon  flip@K=10     flip@K=30     flip@K=100    flip@K=300    flip@K=1000
0.5      1   last     pooled         0.033     0.000       0.0029       0.0017       0.0025       0.0017       0.0006
0.5      1   last     target         0.033     0.000       0.0046       0.0073       0.0095       0.0124       0.0193
0.5      1   delta    pooled         0.245     0.000       0.0154       0.0328       0.0371       0.0409       0.0322
0.5      5   last     pooled         0.150     0.000       0.0181       0.0202       0.0258       0.0274       0.0254
0.5      5   delta    pooled         0.287     0.000       0.0308       0.0355       0.0425       0.0559       0.0582
0.5      5   window   pooled         0.227     0.000       0.0100       0.0096       0.0135       0.0102       0.0058
0.5      20  last     pooled         0.650     0.000       0.0780       0.1054       0.1620       0.2323       0.3180
0.5      20  delta    pooled         0.621     0.000       0.0901       0.1161       0.1499       0.1973       0.2711
0.5      20  window   pooled         1.702     0.000       0.0452       0.0488       0.0541       0.0519       0.0469
2.0      1   last     pooled         0.065     0.000       0.0071       0.0058       0.0068       0.0072       0.0052
2.0      1   last     target         0.065     0.000       0.0063       0.0112       0.0174       0.0356       0.0453
2.0      1   delta    pooled         0.486     0.000       0.0309       0.0645       0.0851       0.0907       0.1245
2.0      5   last     pooled         0.291     0.000       0.0361       0.0406       0.0575       0.0709       0.0845
2.0      5   delta    pooled         0.565     0.000       0.0673       0.0819       0.1058       0.1432       0.2233
2.0      5   window   pooled         0.442     0.000       0.0180       0.0224       0.0300       0.0307       0.0216
2.0      20  last     pooled         1.334     0.000       0.1825       0.2348       0.3749       0.5267       0.6570
2.0      20  delta    pooled         1.334     0.000       0.1995       0.2789       0.3514       0.4882       0.5998
2.0      20  window   pooled         3.386     0.000       0.0754       0.0978       0.1173       0.1560       0.1899
```

- 预测全部命中：h = 1、K = 10、target-fit 的 flip = 0.46% / 0.63% ⇒ coarse 通道上限 ln2 × 0.006 ≈ 0.004 nats；K = 1000 也只有 1.9% / 4.5%。Exp 3 的阴性是单步 cluster 表征不可辨识，不是估计器
- flip ≥ 10% 只在 h = 20（g0.5 需 K ≥ 30，g2.0 任意 K）或 g2.0 的 delta 特征 h = 5、K ≥ 300（14–22%）；`recon` = 0 到 h = 20，simulator 开环复现数据可信；`gap` 按源物理结果 std 归一，h = 1 只有 0.033 / 0.065，h = 20 到 0.65 / 1.33
- 未预料：pooled-fit 在大 K 时把每对配对结果聚成一簇，flip 随 K 反降，所以它不是"有利"分区，target-fit 才是真实参照；window（整段拼接）最差，维度稀释
- 但 h = 20 的代价：目标 random 数据只有 1553 / 1352 条（31% / 27%）能构成 20 步连续窗口（expert / medium 为 97–98%）；h = 5 为 4180 / 4132 条

### Gate 2（detectability，`logs/odrl_detectability-202645_*.log`，5 折 cross-fit，nats，3 seed mean ± sd）

```
case             ce_occ        g_coarse   g_coarse_shuf          g_full     g_full_shuf        d_coarse          d_full
null      0.7089±0.0013  -0.0147±0.0012  -0.0301±0.0026  -0.0067±0.0001  -0.0364±0.0005   0.0154±0.0034   0.0297±0.0005
behavior  0.0718±0.0018   0.0054±0.0039  -0.0077±0.0028   0.0062±0.0026  -0.0593±0.0033   0.0131±0.0014   0.0655±0.0046
g0.5_expert  0.0035±0.0005   0.0010±0.0003  -0.0006±0.0003   0.0011±0.0003  -0.0195±0.0008   0.0016±0.0005   0.0206±0.0007
g0.5_medium  0.0038±0.0004   0.0009±0.0003  -0.0001±0.0002   0.0005±0.0010  -0.0165±0.0023   0.0010±0.0004   0.0170±0.0030
g0.5_random  0.1242±0.0014   0.0082±0.0027  -0.0087±0.0003   0.0174±0.0016  -0.0577±0.0013   0.0169±0.0024   0.0751±0.0024
g2.0_expert  0.0010±0.0002   0.0004±0.0000  -0.0001±0.0001   0.0002±0.0002  -0.0133±0.0003   0.0005±0.0001   0.0135±0.0002
g2.0_medium  0.0043±0.0003   0.0015±0.0003   0.0003±0.0004   0.0005±0.0007  -0.0123±0.0006   0.0012±0.0000   0.0128±0.0013
g2.0_random  0.1164±0.0036   0.0105±0.0018  -0.0059±0.0010   0.0198±0.0042  -0.0553±0.0043   0.0164±0.0016   0.0752±0.0014
```

- 预注册判据（d = g − g_shuf 同 seed 大于 null 与 behavior 两行，3/3）：**coarse 全部不过**（expert / medium 0/3，random 1/3）；**full：g2.0_random 3/3 通过，g0.5_random 2/3 不通过**，其余 0/3。原始增益 g_full 在两个 random 行 3/3 高于两条地板（+0.017 / +0.020 vs behavior +0.006、null −0.007）
- 解释：$I(D;\cdot\mid Z)\le H(D\mid Z)$ = ce_occ。expert / medium 行 ce_occ 仅 0.001–0.004，两域几乎完全可分，没有 overlap，任何条件量都无从检测，与估计器无关；random 行 ce_occ 0.12 有空间，连续 s' 通道取回 0.017–0.020（占 14–17%，behavior 地板 8.6%），K = 10 的 coarse 通道 ≤ 地板，与 Gate R 的 0.004 nats 上限一致
- 偏离预测：(i) null / behavior 行的 d_coarse 是 +0.011 到 +0.020，不在预注册的 ±0.005 内——置换标签的对照过拟合更重（g_shuf −0.03 vs 真标签 −0.015），对照把地板压低了，所以只认"大于 null / behavior 行"的判据；(ii) 我预测 random 行 full 不过，g2.0_random 实际 3/3 通过：有 overlap 的地方连续 s'（DARA 的通道）有弱但可检测的信号，cluster 通道没有

### Gate S（support，同一 job 的 supp 行，τ 为 D_SA 倾向分 e(z) 的截断，frac = held-out 源落在 Ω_τ 的比例）

```
case          tau       frac      abs_rev     abs_dara    abs_local      abs_occ      rho_rev    rho_local     rho_dara      rho_occ
null          0.0      1.000        1.773        0.204        0.149        0.225
null          0.1      0.999        1.768        0.204        0.149        0.224
behavior      0.0      1.000       38.387        2.406        3.011        9.373
behavior      0.02     0.153       10.781        1.429        1.056        2.227
behavior      0.1      0.069        7.683        1.317        0.938        1.157
behavior      0.2      0.040        6.698        1.288        0.915        0.703
g0.5_expert   0.1      0.004        6.258        2.030        1.186        1.251        0.225        0.088        0.085       -0.042
g0.5_medium   0.1      0.005       10.868        2.467        1.833        1.197        0.102        0.084        0.029       -0.016
g0.5_random   0.0      1.000       35.050        7.843        3.544       21.214        0.435        0.356        0.394        0.452
g0.5_random   0.1      0.110        5.864        1.642        0.762        1.096       -0.171       -0.026       -0.141       -0.069
g2.0_expert   0.1      0.002       12.176        2.458        1.217        1.248        0.066        0.112       -0.030       -0.088
g2.0_medium   0.1      0.006       11.045        3.008        1.796        1.155        0.125        0.117       -0.005        0.026
g2.0_random   0.0      1.000       30.529        7.257        4.284       17.235        0.460        0.320        0.388        0.490
g2.0_random   0.1      0.106        5.084        1.570        0.981        1.060       -0.209       -0.156       -0.218       -0.121
```

- behavior 行 τ = 0.1：frac 0.069（预测 < 0.2 ✓）；带内 abs_local 0.94 ± 0.07 vs null 0.15 ⇒ 判据 ≤ 2 × null **0/3**。评注的"common support 上 corrected score ≈ 0"被证伪。带内排序 ours 0.94 < occ 1.16 < DARA 1.32 < LoDADA 7.7，ours 仍最干净但离零很远
- 残余假阳性的机制：f_occ 对 joint 分支 detach，occ 分支欠拟合的那部分 occupancy 被残差分支吸收（Gate 2 的 behavior 行 g_coarse = +0.005 而真值 0 是同一现象）。候选修法（未测）：残差按源 cluster 频率归一，$\sum_k P_s(k\mid z)e^{\ell(z,k)}=1$，把只依赖 z 的分量从残差里挤出去；或两阶段先训满 occ
- random 行 τ = 0.1：frac 0.11，带内所有 rho ≤ 0（occ −0.07 / −0.12、local −0.03 / −0.16、rev −0.17 / −0.21、DARA −0.14 / −0.22）。τ = 0 时的 rho 0.45 完全来自带外，即 occupancy 与接触模式的混杂；overlap 区内没有任何一列能排出 gap
- expert / medium 行带内只剩 0.2–0.6%，按预注册规则判低功效

### 1b 复现（seed 3–5，`logs/toy_adversarial_replicate-202644.log`）

```
case      true_kl   true_rev    abs_rev   abs_dara  abs_local    rho_rev  rho_local   rho_dara
E (s3)      1.410      0.000      3.813      1.304      0.590     -0.956      0.997      0.922
E (s4)      1.410      0.000      4.258      1.553      0.808     -0.959      0.996      0.952
E (s5)      1.410      0.000      3.475      1.710      0.659     -0.953      0.992      0.975
```

- 事后判据 rho_rev(E) ≤ 0 且 rho_local(E) > 0.9：全新 seed **3/3**，可作 confirmatory；原幅度判据 abs_local(E) ≥ 0.5 × abs_local(F)：0.590 / 0.764 ✗、0.808 / 0.714 ✓、0.659 / 0.696 ✗ ⇒ 1/3，与首轮一致，原因不变

### 总判断（2026-10-16）

- 三个条件都量化了：overlap（ce_occ）、resolution（flip）、power（g − 地板）。hopper-gravity、N_t = 5000 上 cluster 级分数无法同时满足前两条：有 overlap 的只有 random 行，有 resolution 的只有 h = 20，而 random 目标只有 27–31% 的数据能构成 20 步窗口
- 唯一检测到的动态信号在连续 s'（DARA 的通道）的 random 行，0.017–0.020 nats，幅度小且 g0.5 行只 2/3。这对"为什么要 cluster 而不是全局 D_SAS"是负面证据，与 toy 里 DARA 5/5 高于 ours 一致
- Exp 4（下游 IQL）继续关闭
- 下一步候选（按信息量）：(1) halfcheetah：无提前终止，random 目标也是整段 1000 步，h = 20 窗口充足，是唯一可能同时满足 overlap 与 resolution 的 ODRL 环境，先跑 Gate R + Gate 2（h ∈ {1, 5, 20}）；(2) N_t 扫描看 full 通道的增益是否随 N_t 增长（区分 power 与 representation）；(3) 残差归一化修 on-support 假阳性，用 behavior 行 τ = 0.1 的 abs_local 作判据；(4) friction（增益型）轴

## 第二轮评注后的计划（2026-10-16）：先修估计器，再换环境

评注给的顺序：(1) 归一化条件估计器修 on-support 泄漏 → (2) halfcheetah Gate R / S / P → (3) N_t 扫描 → (4) 固定 probe 分布 → (5) friction → (6) oracle IQL → (7) learned IQL。采纳，补两点：Gate R 不涉及估计器，和 (1) 并行跑；评注把我的 Gate 2 改名 Gate P，以下沿用 S / R / P / C / O。

### 预注册：Step 1 归一化估计器（启动前写下）

`adaptation/normalized_cluster_classifier.py`：$P_t(k\mid z)=P_s(k\mid z)e^{g(z,k)}/\sum_j P_s(j\mid z)e^{g(z,j)}$，$\ell=g(z,k)-\log\sum_j\hat P_s(j\mid z)e^{g(z,j)}$，$\hat P_s(k\mid z)$ 是只在源数据上训练并冻结的 K 路分类器；occ 分支仍 detach。同时加对照 `sep` = 两个独立分类器之差 logit $D_{ZK}$ − logit $D_Z$（用户笔记里的原始写法，无结构约束）。三者共用 `fit_estimators`、同数据同步数。

- ODRL（`odrl/run_gate_s.sbatch`，null / behavior / g2.0_random，3 seed，只跑 supp 行）：判据 behavior 行 τ = 0.1 的 abs_norm ≤ 2 × abs_norm(null | τ = 0.1)，3/3。g2.0_random 的 rho_norm(τ = 0) 与 rho_local 差 ≤ 0.1（排序不被归一化破坏）
- toy（`toy/run_toy_seeds.sbatch` seed 0–2，`toy/run_adversarial.sbatch` seed 0–2）：Case B abs_norm ≤ 2 × abs_norm(A)；Case D、F、E 的 rho_norm > 0.9 且 ≥ rho_local − 0.02
- 预测：abs_norm(behavior | Ω_0.1) 从 0.94 降到 0.5 以下但到不了 null 的 0.15——泄漏并没有被完全封死，$\ell(z,k_s)$ 的上界是 $-\log\hat P_s(k_s\mid z)$，向负方向仍无界，只是不再有"欠拟合 occupancy 直接写进残差"的通道；abs_sep 与 abs_local 同量级；toy 上 norm 与 local 的 rho 差 < 0.01
- 门：ODRL 判据不过 → 不进 halfcheetah 的 Gate S / P，先找泄漏来源

### 预注册：Step 2.1 halfcheetah Gate R（`odrl/run_resolution_halfcheetah.sbatch`）

源 d4rl halfcheetah-medium-replay-v2（dt = 0.05，是 hopper 的 6 倍），目标 ODRL halfcheetah-gravity 0.5 / 2.0 的 medium（target-fit 分区）；h ∈ {1, 5, 10, 20}，K ∈ {10, 30, 100, 300, 1000}，feature {last, delta, window}。无提前终止，random 目标 5000 条里 20 步窗口应 ≥ 4900。
- 预测：h = 1、K = 10、target-fit 的 flip 在 1–5%（单步效应比 hopper 大 6 倍但仍小于簇尺度）；h = 20 的 flip > 30%；delta ≥ last
- 判据：同 Gate R，flip ≥ 10% 的配置才进 Gate P

### Step 1 结果（2026-10-16，job 202658 toy / 202659 1b / 202660 ODRL，3 seed）

```
toy ABCD           abs_rev   abs_dara  abs_local   abs_norm    abs_sep    rho_rev  rho_local   rho_norm    rho_sep   rho_dara
A                    0.189      0.052      0.038      0.072      0.064
B                    0.598      0.106      0.060      0.095      0.232
C                    1.615      1.468      1.271      1.449      1.261      0.942      0.968      0.988      0.965      0.986
D                    1.784      1.380      1.179      1.420      1.195      0.898      0.963      0.984      0.951      0.987
toy ACFE
A                    0.169      0.046      0.038      0.075      0.056
C                    1.408      1.293      1.152      1.235      1.163      0.931      0.970      0.985      0.965      0.988
F                    1.417      1.837      1.580      1.580      1.520      0.854      0.986      0.990      0.989      0.997
E                    3.536      1.628      0.739      0.126      0.711     -0.951      0.991      0.974      0.928      0.950
ODRL (supp)   tau   frac   abs_rev  abs_dara abs_local  abs_norm   abs_sep   abs_occ   rho_rev rho_local  rho_norm   rho_sep  rho_dara   rho_occ
null          0.1  0.999     1.768     0.203     0.150     0.132     0.222     0.226
behavior      0.0  1.000    38.424     2.407     3.011     2.067     3.800     9.375
behavior      0.1  0.069     7.732     1.317     0.938     0.890     1.464     1.155
g2.0_random   0.0  1.000    30.546     7.269     4.285     5.550     3.953    17.243     0.460     0.319     0.134    -0.061     0.389     0.489
g2.0_random   0.1  0.106     5.080     1.573     0.981     1.405     1.269     1.059    -0.209    -0.154    -0.080    -0.063    -0.216    -0.120
```

- **toy 判据全部通过**：B abs_norm 0.095 ≤ 2 × A 0.072；C / D / F / E 的 rho_norm 0.988 / 0.984 / 0.990 / 0.974，都 > 0.9 且 ≥ rho_local − 0.02。两个预期之外的好处：(i) E 的幅度 abs_norm = 0.126，与精确 cluster 级 |l_dyn_C| = 0.11 对上了（local 高估到 0.74），1b 原来过不了的幅度判据在 norm 上是过的；(ii) C / D 上 rho_norm 追平 DARA（0.988 vs 0.986、0.984 vs 0.987），local 是 0.968 / 0.963。`sep`（两个独立分类器之差）B 行假阳性 0.232，是 local 的 4 倍：结构确实重要
- **ODRL 判据不过（0/3）**：behavior 行 τ = 0.1 的 abs_norm 0.89 ± 0.17（1.05 / 0.65 / 0.97）vs null 0.13，与 local 的 0.94 没有区别；sep 1.46 更差。预测（降到 0.5 以下）错了，评注的归因（detach 的 occ 分支欠拟合被残差吸收）在真实数据上不成立——归一化把这条通道封了，带内残余不动
- g2.0_random τ = 0 的 rho_norm 0.13 vs rho_local 0.32，预注册的 ±0.1 判据形式上不过，但这条判据写错了对象：那一行 τ = 0 的 rho 是 occupancy–接触模式混杂（rho_occ 0.49），去掉 occupancy 成分的估计器理应更低；带内（τ = 0.1）所有列都 ≈ 0，无差别
- 残余的两个候选来源：(a) 归一化只封正向，$\ell(z,k_s)\le-\log\hat P_s(k_s\mid z)$，负向无界，欠拟合 occupancy 仍可写成"源侧 z 的负 ℓ"；(b) 带内数据稀薄（源只有 7% 在带内）的估计方差，与泄漏无关。区分实验（job 202708）：同数据、同 cluster、两次独立训练，报带内 signed mean 与两次拟合之间的 Spearman——系统泄漏跨拟合可复现、方差不可复现
- 门：按预注册，halfcheetah 的 Gate S / P 暂不进，等诊断

### halfcheetah Gate R 结果（2026-10-16，job 202661，源 halfcheetah-medium-replay-v2，held-out 20k 个 20 步窗口）

```
gravity  h   feature  partition         gap     recon  flip@K=10     flip@K=30     flip@K=100    flip@K=300    flip@K=1000
0.5      1   last     pooled         0.314     0.000       0.0249       0.0309       0.0553       0.0661       0.0611
0.5      1   last     target         0.314     0.000       0.0175       0.0182       0.0262       0.0721       0.0864
0.5      1   delta    pooled         0.345     0.000       0.0277       0.0359       0.0447       0.0477       0.0368
0.5      5   last     pooled         1.703     0.000       0.1646       0.3117       0.4718       0.5637       0.6232
0.5      5   delta    pooled         1.562     0.000       0.1847       0.2709       0.3748       0.4395       0.4698
0.5      5   window   pooled         2.679     0.000       0.0246       0.1218       0.2079       0.2816       0.3314
0.5      10  last     pooled         2.628     0.000       0.2135       0.4045       0.5994       0.6962       0.7466
0.5      10  delta    pooled         2.599     0.000       0.2334       0.4368       0.5788       0.6423       0.6756
0.5      10  window   pooled         6.240     0.000       0.0398       0.2507       0.4029       0.5049       0.5716
0.5      20  last     pooled         3.208     0.000       0.2077       0.4778       0.6481       0.7309       0.7764
0.5      20  delta    pooled         3.821     0.000       0.2946       0.5480       0.6338       0.6919       0.7229
0.5      20  window   pooled        11.866     0.000       0.0595       0.2781       0.5205       0.6294       0.6564
2.0      1   last     pooled         0.621     0.000       0.0505       0.0628       0.1162       0.1524       0.1731
2.0      1   last     target         0.621     0.000       0.1560       0.1852       0.2459       0.2797       0.3240
2.0      1   delta    pooled         0.679     0.000       0.0530       0.0756       0.1000       0.1185       0.1232
2.0      5   last     pooled         2.453     0.000       0.1681       0.4384       0.6268       0.7006       0.7528
2.0      5   delta    pooled         2.278     0.000       0.2440       0.4415       0.5722       0.6547       0.6906
2.0      5   window   pooled         4.241     0.000       0.0765       0.2130       0.3491       0.5260       0.5911
2.0      10  last     pooled         2.766     0.000       0.3765       0.5691       0.6794       0.7482       0.7873
2.0      10  delta    pooled         2.760     0.000       0.3936       0.5667       0.6744       0.7192       0.7509
2.0      10  window   pooled         7.639     0.000       0.0698       0.3075       0.6169       0.6788       0.7167
2.0      20  last     pooled         2.867     0.000       0.4037       0.6180       0.7450       0.7984       0.8269
2.0      20  delta    pooled         3.072     0.000       0.4567       0.6040       0.7022       0.7511       0.7683
2.0      20  window   pooled        12.072     0.000       0.0751       0.5566       0.7229       0.7559       0.7698
```

- **g2.0 在 h = 1、K = 10、target-fit 就过 Gate R：flip 15.6%**（hopper 是 0.6%）；g0.5 的 h = 1 只有 1.8%（K = 1000 也只 8.6%），要 h = 5 才到 16%（K = 10）。单步 gap 0.31 / 0.62，是 hopper 的 10 倍（dt 0.05 vs 0.008）
- 预测核对：g0.5 的 1–5% ✓；g2.0 预测偏低（实际 15.6%）；h = 20 > 30% ✓；delta ≥ last 在 pooled 上 ✓（差别小）；recon = 0 ✓
- 可进 Gate P 的配置：g2.0 h = 1（任意 K）、两种重力 h ≥ 5。halfcheetah-gravity 是第一个 overlap（待测）与 resolution 有可能同时满足的 ODRL 环境

### 带内残余的诊断（2026-10-16，job 202708，null / behavior，3 seed，同数据同 cluster 两次独立训练）

```
behavior tau=0.1   seed   abs     mean     rep        null tau=0.1   abs    mean    rep
local               0    1.034   -0.048   0.837                     0.146  -0.059  0.513
                    1    0.880   -0.642   0.774                     0.159  -0.057  0.676
                    2    0.900   -0.533   0.675                     0.145  -0.040  0.620
norm                0    1.045   -1.003   0.235                     0.129  -0.060  0.582
                    1    0.650   -0.596  -0.101                     0.132  -0.067  0.668
                    2    0.974   -0.912   0.570                     0.135  -0.073  0.651
sep                 0    1.498   -0.850   0.535                     0.250   0.030  0.496
                    1    1.523   -0.430   0.553                     0.216  -0.001  0.430
                    2    1.372   -0.065   0.328                     0.200   0.029  0.081
dara                0    1.595   -1.213   0.386                     0.229   0.033  0.462
                    1    1.119   -0.323   0.331                     0.201   0.042  0.007
                    2    1.238   -0.743   0.495                     0.178  -0.018  0.186
rev                 0    8.510   -8.055   0.930                     1.808  -1.545  0.825
                    1    7.017   -6.483   0.928                     1.781  -1.469  0.795
                    2    7.668   -7.033   0.922                     1.715  -1.434  0.821
occ                 0    1.141   -0.795   0.685
```

- `rep` = 两次独立拟合在同一批带内 held-out 点上的 Spearman。**local 的带内残余是可复现的逐点结构**（rep 0.68–0.84），signed mean 在 2/3 seed 里明显为负：这是数据驱动的系统泄漏。**norm 的残余逐点不可复现**（rep 0.24 / −0.10 / 0.57），但 |mean| ≈ abs（−1.00 / 1.05、−0.60 / 0.65、−0.91 / 0.97）：几乎是一个常数负偏移加噪声。对分位数或 exp 权重过滤来说常数偏移不改变排序；预注册的 abs 判据把它算成失败
- 偏移的方向是预注册里写的负向通道：$\ell(z,k_s)=-\log\big(\hat p_s+\sum_{j\ne s}\hat P_s(j\mid z)e^{g_j-g_s}\big)$，正向有界、负向无界，剩余的 occupancy 以"目标把质量放到别的簇"的形式进来，带内近似常数（τ = 0 全体的 mean 是 −1.8 到 −2.3，随 occupancy 区域变，不随点变）
- LoDADA 带内 −7 到 −8、rep 0.93：假阳性巨大、系统、完全可复现。DARA 带内 1.1–1.6，rep 0.33–0.50，也有一半是偏移
- **事后读法**：带内去均值后的 |ℓ_norm| 约 0.1–0.3，与 null 的 0.13 同量级。这是事后判据，需要全新 seed 确认

### 预注册：Step 1 确认 + halfcheetah Gate S / P（启动前写下）

- **确认（`odrl/run_gate_s_confirm.sbatch`，seed 3–5，null / behavior）**：新列 `cabs` = 带内去均值 |ℓ|。判据：τ = 0.1 的 cabs_norm(behavior) ≤ 2 × cabs_norm(null)，3/3；预测 cabs_local(behavior) 不过（逐点泄漏），rep_local > 0.6、rep_norm < 0.6 复现
- **halfcheetah Gate S / P（`odrl/run_detectability_halfcheetah.sbatch`，h = 1，K = 10，3 seed）**：源 halfcheetah-medium-replay-v2；行 null、behavior（expert）、behavior_m（d4rl medium-v2，第二个同物理对照，overlap 应比 expert 多）、g0.5 / g2.0 × medium / random。判据同 Gate P（d = g − g_shuf 同 seed 大于 null 与两个 behavior 行，3/3）和 Gate S（τ = 0.1 带内 cabs ≤ 2 × null）
- 预测：g2.0 两行 coarse 与 full 都过（Gate R 15.6%，gap 0.62）；g0.5 两行 coarse 不过（1.8%）、full 过；random 行 ce_occ 高于 medium 行；带内 rho_norm 在 g2.0 行 > 0.3 且 > rho_occ（第一次有机会看到"估计器在带内排出 gap"）
- 若 g2.0 行 Gate P 过而带内 rho 不过 occupancy 对照，结论是"可检测但估计器未校准"（Gate C 不过），下一步是 N_t 扫描而不是下游

### Step 1 确认结果（2026-10-16，job 202716，seed 3–5）

```
tau=0.1        frac   cabs_local  cabs_norm  cabs_sep  cabs_dara  cabs_rev   abs_norm  mean_norm  rep_local  rep_norm
null          1.000      0.131      0.167     0.207      0.202     1.897      0.123     -0.068      0.588     0.651
behavior      0.089      0.969      1.334     1.870      1.199     6.372      0.934     -0.869      0.684     0.487
```

- **去均值判据也不过（0/3）**：cabs_norm(behavior) 1.33 vs 2 × null 0.33。上一节"去均值后约 0.1–0.3"的事后读法是错的——abs ≈ |mean| 只说明几乎全负，不说明集中；实际是重尾的负分布。预注册的确认跑否定了事后读法，这正是它的用处
- 复现的部分：rep_local 0.68 > 0.6、rep_norm 0.49 < 0.6（预测方向对，差距比 seed 0–2 小）
- **Step 1 结论**：归一化估计器在 toy 上全面更好（校准 + 追平 DARA），但在 hopper 同物理行的 overlap 带内没有去掉假阳性，detach 泄漏不是主因。剩下的解释是带内数据稀薄的有限样本方差——halfcheetah 的第二个同物理对照 behavior_m（d4rl medium-v2，带内占 62%）支持这个解释：那一行 cabs_local 0.435 / cabs_norm 0.672 vs null 0.258 / 0.369，比值 1.7 / 1.8，**过 2× 判据**；expert 对照（带内 1%）比值 3.6 / 3.8 不过。假阳性幅度随 overlap 增大而下降，符合方差解释，不符合结构泄漏

### 预注册：perfect-support 目标（`odrl/run_paired.sbatch`，启动前写下）

halfcheetah Gate P 前 4 行已出：null ce_occ 0.74、behavior（expert）0.046、behavior_m 0.48、**g0.5_medium 0.006**——ODRL 的 medium 目标与 d4rl medium-replay 源几乎完全可分，与 hopper 一样。support 条件在 ODRL 现成目标上基本只有 random 质量能部分满足，这是 benchmark 的性质，不是估计器的。按评注 Step 4 / 2.2，构造 support 完美的目标：`paired_<g>` = null 块的 (s, a) 在目标物理里走一步（源 occupancy、纯动态偏移，ce_occ 理论值 ln 2），hopper 与 halfcheetah 各 g ∈ {0.5, 2.0}，3 seed，Gate P + Gate C（带内 rho vs gap，带 = 全体）。

- 判据：Gate P 同前（d = g − g_shuf 同 seed 大于 null 行，3/3）；Gate C：rho_norm 与 rho_local > rho_occ 且 > 0.3，3/3；与 rho_dara 比较报告
- 预测：hopper coarse 两行都不过（flip < 1%），full g0.5 不过、g2.0 边缘；halfcheetah coarse g2.0 过、g0.5 不过，full 两行都过；Gate C 在 halfcheetah g2.0 行 rho_norm > 0.3，hopper 全部 < 0.3
- 这组实验把 support 从问题里拿掉，只剩 resolution 与 power，是 N_t 扫描的基座

### halfcheetah Gate S / P 结果（2026-10-16，job 202719，h = 1，K = 10，N_t = 5000，3 seed）

```
case             ce_occ   g_coarse  g_coarse_shuf   g_full  g_full_shuf   d_coarse   d_full   >floors(dc, df)
null              0.744     -0.053     -0.083       -0.105    -0.168       0.030     0.063
behavior          0.046     -0.009     -0.016       -0.006    -0.071       0.007     0.065
behavior_m        0.479     -0.019     -0.042       -0.008    -0.137       0.023     0.129
g0.5_medium       0.006     -0.001     -0.001       -0.000    -0.021       0.001     0.021     0/3  0/3
g0.5_random       0.160     -0.008     -0.023       -0.033    -0.071       0.015     0.039     0/3  0/3
g2.0_medium       0.009     -0.003     -0.003        0.003    -0.017       0.001     0.019     0/3  0/3
g2.0_random       0.085     -0.005     -0.008       -0.004    -0.034       0.003     0.030     0/3  0/3

tau=0.1         frac   cabs_local  cabs_norm  cabs_dara  cabs_rev   rho_rev rho_local rho_norm rho_dara rho_occ
null           0.997      0.258      0.369     0.606      3.225
behavior       0.010      0.931      1.398     3.341      4.618
behavior_m     0.617      0.435      0.672     1.010      4.358
g0.5_medium    0.003      1.118      3.020     5.131      6.951    -0.054   -0.059   -0.096   -0.051   -0.003
g0.5_random    0.120      0.976      0.785     1.594      4.465     0.054    0.054    0.030    0.015   -0.026
g2.0_medium    0.002      1.152      1.447     5.812      6.443    -0.075   -0.116   -0.142   -0.087    0.022
g2.0_random    0.036      0.788      1.152     2.462      5.309     0.055    0.023    0.026    0.007    0.025
```

- **Gate P 四行全不过（coarse 与 full 都 0/3）**，预测（g2.0 两行过）错了。原因在 Gate S：medium 目标 ce_occ 0.006–0.009，random 目标 0.16 / 0.085，都远低于 behavior_m 的 0.48。Gate R 的 15.6% 是在源 occupancy 的探针上测的；真正的目标数据与源几乎不重叠，重叠的那一小块（random 的近初始态）又不是重力翻转多的地方。resolution 在探针上成立，不等于在目标数据的 support 上成立
- Gate C：带内所有 rho ≈ 0，expert / medium 行带内只剩 0.2–0.3%，random 行 3.6–12%
- behavior_m（同物理、带内 62%）是这轮最有信息的一行：cabs_local / cabs_norm 比 null 只高 1.7 / 1.8 倍，过 2× 判据，而 hopper 与 halfcheetah 的 expert 对照（带内 1–9%）是 3.6–7 倍。带内假阳性随 overlap 增大而消退 → 有限样本方差，不是结构泄漏
- 跨两个环境的结论：ODRL 现成目标数据（ODRL 自己的策略在偏移物理里收集）与 d4rl 源在 (s, a) 上几乎完全可分，support 条件在 benchmark 层面就不成立；transition 级条件估计在这些数据上不可辨识，与估计器无关。能检验 resolution 与 power 的只有 perfect-support 构造（`paired_<g>`，job 202726 在跑）

### perfect-support 结果（2026-10-16，job 202726，`paired_<g>`，h = 1，K = 10，N_t = 5000，3 seed）

```
hopper           ce_occ   g_coarse  g_coarse_shuf   g_full  g_full_shuf   d_coarse   d_full   >null(dc, df)
null              0.709     -0.014     -0.030       -0.007    -0.036       0.015     0.029
paired_0.5        0.709     -0.013     -0.032        0.012    -0.036       0.020     0.049     3/3  3/3
paired_2.0        0.709     -0.013     -0.032        0.227    -0.036       0.019     0.264     3/3  3/3
halfcheetah
null              0.744     -0.053     -0.083       -0.105    -0.168       0.030     0.063
paired_0.5        0.744     -0.052     -0.084        0.026    -0.168       0.032     0.194     3/3  3/3
paired_2.0        0.744     -0.049     -0.082        0.287    -0.167       0.033     0.454     3/3  3/3

gate c (tau = 0, band = all)   rho_rev  rho_local  rho_norm  rho_sep  rho_dara  rho_occ
hopper      paired_0.5           0.115     0.011    -0.024    0.111     0.245   -0.092
hopper      paired_2.0           0.120    -0.003    -0.037    0.113    -0.007   -0.094
halfcheetah paired_0.5          -0.047     0.068     0.059    0.041     0.119   -0.021
halfcheetah paired_2.0          -0.010     0.012     0.007    0.033     0.087   -0.027
```

- support 按构造完美（ce_occ = null 的 0.709 / 0.744）。**full 通道（连续 s'，DARA 的通道）清楚地检测到动态偏移**：hopper g2.0 原始增益 +0.227 nats、halfcheetah g2.0 +0.287、halfcheetah g0.5 +0.026、hopper g0.5 +0.012（null −0.007 / −0.105），d_full 3/3 全过
- **coarse 通道（K = 10，cluster 级，LoDADA 与 ours 的通道）没有信号**：原始 g_coarse 与 null 相同（hopper −0.013 vs −0.014，halfcheetah −0.05 vs −0.05）。预注册判据 d > null 形式上 3/3，但差 0.003–0.005 nats，在噪声内；判据缺了幅度门槛，这里按原始增益读。hopper 与 Gate R 的 0.004 nats 上限一致；**halfcheetah g2.0 的 Gate R 翻转率 15.6%（上限 0.108 nats）却一点没提出来**——resolution 存在但估计器在 3000 步、N_t 5000 下学不出 z → k_s(z) 的分区，是 power / 估计器侧的失败
- **Gate C 全部不过**：perfect support 下 rho_occ ≈ −0.02 到 −0.09（无 occupancy 混杂，评估终于干净），但没有任何估计器的逐样本分数与配对 gap 相关（最高 rho_dara 0.245，hopper g0.5；其余 ≤ 0.12）。DARA 能检测偏移（CE 增益）却排不出 gap
- 预测核对：hopper coarse 不过 ✓、full g2.0 过 ✓（比"边缘"强得多）、g0.5 过（预测不过 ✗）；halfcheetah coarse g2.0 过 ✗（没过）、full 两行过 ✓；Gate C halfcheetah g2.0 > 0.3 ✗

### 配对 gap 的异质性（2026-10-16，5000 个探针，h = 1，按源 s' std 归一）

```
                    mean    std    cv   q10 / q50 / q90        主导维度（平方 gap 份额）
hopper      g0.5   0.030  0.005  0.17  0.023 / 0.033 / 0.033   dim 6 = 竖直速度 95%
hopper      g2.0   0.059  0.011  0.18  0.045 / 0.066 / 0.066   dim 6 95%
halfcheetah g0.5   0.323  0.194  0.60  0.241 / 0.312 / 0.312   dim 9 = 竖直速度 61%
halfcheetah g2.0   0.632  0.295  0.47  0.464 / 0.624 / 0.624   dim 9 63%
```

- **重力偏移的单步 gap 在状态间几乎是常数**：hopper q50 = q90，CV 0.17，95% 的 gap 在竖直速度一维上（Δv_z = Δg·dt 与状态无关，只有接触态被地面反力吸收一部分）；halfcheetah 多数状态同一个值，CV 0.5 来自尾部
- 这解释了 Gate C：排序目标本身没有结构，Spearman 对近常数目标无意义。更根本地，**偏置型（重力）偏移的"局部可迁移性"本来就近似均匀**，逐 transition 过滤在这个轴上没有杠杆——与 1b 理论一致（偏置型 ℓ_dyn 与 (s, a) 可分离）。能让 localization 有意义的是增益型轴（friction / mass，gap 随状态变），这是评注 Step 5 的理论动机，现在也有实验动机

### 预注册：power 消融（`odrl/run_paired_steps.sbatch`，job 202735，2026-10-16 提交前写下）

halfcheetah null / paired_0.5 / paired_2.0，steps 3000 vs 20000，加 `g_norm` 列（归一化模型自己的 coarse 增益，它把 $\hat P_s(k\mid z)$ 显式学出来）。
- 判据：20000 步下 paired_2.0 的原始 g_coarse − g_coarse(null) ≥ 0.03 nats（上限 0.108 的 30%），3/3；g_norm 同理
- 预测：g_norm 在 3000 步就高于 g_coarse（显式分区）；20000 步 g_coarse 到 0.03–0.06；Gate C 在 20000 步仍 < 0.3（gap 近常数，与步数无关）

### 预注册：friction 轴（增益型，评注 Step 5；`odrl/run_resolution_friction.sbatch` + `odrl/run_paired_friction.sbatch`，启动前写下）

hopper / halfcheetah 的 ODRL friction xml，shift 0.5 / 2.0；Gate R（h ∈ {1, 5, 20}，K 扫描，新增 `cv` 列 = 配对 gap 的变异系数）；perfect-support `paired_<g>` 的 Gate P + Gate C（h = 1，K = 10，N_t 5000，3000 步，3 seed）。
- 理论预期（1b）：friction 是增益型，ℓ_dyn 不可与 (s, a) 分离，gap 随状态变（只有接触态受影响）
- 预测：gap 的 cv > 1（与重力的 0.17–0.6 对比），q50 接近 0 而尾部大；Gate R 的 h = 1 翻转率低于重力（多数 transition 无接触、不翻），但翻的那部分集中；paired Gate P：full 两个环境都过，coarse 仍不过（3000 步）；**Gate C：rho_dara > 0.3 至少一行**（排序目标第一次有结构），rho_local / rho_norm 仍 < 0.3
- 这一轮回答的问题：localization 的前提——逐 transition 的可迁移性是否异质——在增益型轴上是否成立；成立则 Gate C 的失败归因于估计器，不成立则归因于轴

### friction Gate R 结果（2026-10-16，job 202741，held-out 源 20k 个 20 步窗口，`cv` = 配对 gap 变异系数）

```
hopper   h   feature  partition     gap    cv   flip@K=10   K=100   K=1000      halfcheetah  h  feature partition   gap    cv   K=10    K=100   K=1000
0.5      1   last     target      0.079  1.83     0.0112   0.0316   0.0582      0.5   1  last   target        0.101  2.06  0.0128  0.0203  0.0327
0.5      1   delta    pooled      0.498  1.76     0.0537   0.0905   0.1658      0.5   1  delta  pooled        0.103  1.94  0.0103  0.0148  0.0095
0.5      5   delta    pooled      0.238  1.47     0.0339   0.0503   0.0688      0.5   5  last   pooled        0.438  1.24  0.0374  0.0907  0.1110
0.5      20  last     pooled      0.355  1.44     0.0396   0.0920   0.1502      0.5   20 last   pooled        2.093  0.68  0.1902  0.4787  0.6177
2.0      1   last     target      0.080  1.86     0.0143   0.0662   0.1078      2.0   1  last   target        0.131  1.87  0.0101  0.0252  0.0369
2.0      1   delta    pooled      0.503  1.80     0.0791   0.1300   0.2617      2.0   1  delta  pooled        0.134  1.79  0.0160  0.0200  0.0145
2.0      5   delta    pooled      0.625  1.52     0.0874   0.2656   0.3805      2.0   5  last   pooled        0.541  1.20  0.0553  0.1112  0.1463
2.0      20  last     pooled      1.102  1.34     0.1446   0.2765   0.4092      2.0   20 last   pooled        2.398  0.67  0.2483  0.5373  0.6498
```
（完整 36 行在 `logs/fric_resolution-202741.log`）

- **gap 异质性预测命中**：h = 1 的 cv 1.8–2.1（重力 0.17–0.6），gap 集中在少数接触态——排序目标第一次有结构
- h = 1、K = 10、target-fit 的翻转率 1.0–1.4%，比重力低（预测 ✓）：多数 transition 无接触不翻。hopper 的 delta 特征明显更好（h = 1 K = 10 5.4% / 7.9%，K = 1000 17% / 26%）——接触冲量在 Δs 上可见；halfcheetah 的 delta 没有优势。h = 20 到 14–25%（K = 10）、41–65%（K = 1000）
- 对 paired friction 的含义：s' 级 coarse 通道上限仍只有 ln2 × 1.3% ≈ 0.01 nats，coarse 不过的预测维持；full 通道与 Gate C 看 job 202738

### friction perfect-support 结果（2026-10-16，job 202738，h = 1，K = 10 on s'，N_t 5000，3 seed）

```
                     ce_occ  g_coarse  g_full   d_full  >null   rho_rev  rho_local  rho_norm  rho_sep  rho_dara  rho_occ
hopper null           0.709   -0.014   -0.007   0.029
hopper paired_0.5     0.709   -0.013    0.066   0.103   3/3     -0.054    -0.010     0.019   -0.088    0.201    0.107
hopper paired_2.0     0.709   -0.014    0.136   0.172   3/3     -0.016    -0.008     0.041   -0.092   *0.677*  0.109
halfcheetah null      0.744   -0.053   -0.105   0.063
halfcheetah paired_0.5 0.744  -0.054   -0.096   0.071   3/3      0.117    -0.036    -0.037   -0.016    0.084    0.012
halfcheetah paired_2.0 0.744  -0.053   -0.096   0.072   3/3      0.098    -0.005     0.005   -0.011    0.119    0.011
```

- **Gate C 第一次有估计器通过：hopper friction 2.0 的 rho_dara = 0.68 ± 0.02（3/3 > 0.3）**，friction 0.5 为 0.20；rho_occ 0.11，无混杂。预测（rho_dara > 0.3 至少一行）命中。增益型轴上排序目标有结构（cv 1.9），连续 s' 的估计器就能排出逐 transition 的 gap
- cluster 级估计器（local / norm / LoDADA）在同一行全部 ≈ 0：K = 10 on s' 的翻转率只有 1.4%，coarse 通道上限 0.01 nats，g_coarse 与 null 相同。不是估计器排错，是它拿到的表征里没有信息
- halfcheetah friction：full 通道原始增益 −0.096 vs null −0.105，形式上 3/3 但只差 0.01 nats；rho_dara 0.08 / 0.12。单步摩擦效应太小（gap 均值 0.10–0.13、cv 2），h = 1 不够
- 到此为止整条线上**唯一同时过 Gate S、P、C 的配置是 DARA（全局连续 D_SAS）在 hopper friction 2.0 perfect-support 上**。cluster 级方法还没有在任何真实配置上过 Gate P

### 预注册：给 cluster 通道一个有分辨率的表征（`odrl/run_paired_friction_delta.sbatch`，启动前写下）

hopper friction 2.0 perfect-support，null + paired_2.0，3 seed，三种聚类：delta K = 10（Gate R pooled 翻转 7.9%，上限 0.055 nats）、delta K = 30（10%，0.07）、next K = 30（对照，3.1%）。`cluster_on = 'delta'` 只改聚类特征，gap 归一化与其余路径不变。
- 判据：Gate P 原始 g_coarse − g_coarse(null) ≥ 0.02 nats（3/3）；Gate C rho_norm 或 rho_local > 0.3（3/3），与 rho_dara 0.68 对照
- 预测：delta K = 30 的 g_coarse 提到 0.02–0.04；rho_norm 到 0.2–0.4，仍明显低于 DARA；next K = 30 不动。若 delta 下 cluster 通道仍为 0，结论是单步 cluster 估计器在 N_t 5000 下无法利用 ≤ 10% 的翻转，localization 的存在理由只能靠 N_t 扫描在更小 N_t 下找

### 中途记录（2026-10-16）：20000 步是过拟合区；delta 聚类让 coarse 通道第一次过 Gate P

- **power 消融（job 202735）的 20000 步段**：null 行 held-out ce_occ 1.65–1.68 nats（3000 步 0.74，chance ln2 = 0.693），g_coarse / g_full 都到 −1.0，带内 abs_norm 1.88（3000 步 0.27）。N_t = 5000、无正则无早停的 balanced classifier 在 20000 步（ODRL 的默认预算）已经深度记忆目标样本。两个含义：(i) 预注册的"20000 步让 coarse 提到 0.03"无法在这种过拟合下检验；(ii) 带内假阳性随训练步数上涨，再次支持方差 / 过拟合解释
- **delta 聚类 K = 10（job 202751 前半）**：hopper friction 2.0 perfect-support 的原始 g_coarse +0.012 / +0.017 / +0.019 vs null −0.019（3 seed），高出 0.032–0.038 nats ≥ 0.02 ⇒ **Gate P 3/3 通过**，是 cluster 通道第一次检测到动态偏移（上限 ln2 × 7.9% = 0.055，拿到约 2/3）；g_norm 高出 null 0.04–0.06。Gate C：rho_norm 0.24 ± 0.03、rho_local 0.20，从 0 起来了但 < 0.3，DARA 同行 0.68。K = 30 与 next K = 30 对照待出

### 预注册：N_t 扫描（`odrl/run_paired_nt.sbatch`，job 见 `logs/paired_nt_job.txt`，启动前写下）

hopper friction 2.0 perfect-support，delta K = 10，N_t ∈ {500, 1000, 2500, 5000, 10000, 20000}，null + paired_2.0，3 seed，3000 步。这是评注 Step 3，直接回答 localization 的存在理由：
- 判据：存在某个 N_t 使 rho_norm 或 rho_local > rho_dara 在 3/3 seed 成立 ⇒ cluster 估计器有样本效率优势；否则没有
- 预测：rho_dara 随 N_t 下降（5000 时 0.68，500 时可能 0.3 左右），cluster 估计器从 0.2 起随 N_t 缓慢变化；**我预测不存在交叉点**，DARA 在所有 N_t 上更高。若预测成立，cluster 方法在这条线上没有存在理由，论文定位为诊断 / 识别性研究
- 同时报告 Gate P 的 g_coarse、g_full 随 N_t 的曲线（null 行作每个 N_t 的地板）

### power 消融结果（2026-10-16，job 202735，halfcheetah gravity perfect-support，3000 vs 20000 步，3 seed）

```
steps 20000        ce_occ   g_coarse  g_coarse_shuf   g_full  g_full_shuf   g_norm   g_norm_shuf   rho_dara  rho_norm
null                1.64     -1.04      -1.56        -0.99     -1.83        -1.03     -1.74
paired_0.5          1.64     -1.03      -1.55        -0.25     -1.84        -0.99     -1.75         0.07-0.21   ≈0
paired_2.0          1.64     -1.02      -1.59        +0.29     -1.84        -0.93     -1.74         0.06-0.16   ≈0
(steps 3000, 同数据：null ce_occ 0.744，paired_2.0 g_coarse −0.049 / g_full +0.287，见上节)
```

- 20000 步下所有分类器深度过拟合（held-out ce_occ 1.64 vs chance 0.693），coarse 增益与 null 完全相同（−1.02 vs −1.04），full 通道仍能把 paired_2.0 拉到 +0.29（信号强到过拟合压不住），Gate C 全部 ≈ 0.1 以下
- 预注册判据（20000 步 coarse ≥ 0.03 nats）**不过**，但这个结果说的是训练预算而不是 resolution：要检验"更长训练能否学出 z → k_s(z)"需要早停或正则，不在本轮范围。ODRL 默认的 20000 步在 N_t = 5000 时是过拟合区，这一点对 LoDADA / DARA 的原始实验也成立，值得单独写
- g_norm 在 3000 步的 halfcheetah gravity 行也没有高于 g_coarse（−0.08 vs −0.05），"显式分区更容易学"的预测在 s' 聚类上不成立；在 hopper friction 的 delta 聚类上成立（g_norm 高出 null 0.04–0.06 vs g_coarse 0.03–0.04）

### delta 聚类结果（2026-10-16，job 202751，hopper friction 2.0 perfect-support，N_t 5000，3000 步，3 seed）

```
cluster_on / K     g_coarse − null (seed 0/1/2)     g_norm − null          rho_norm            rho_local           rho_rev    rho_dara  rho_occ
delta  K=10        +0.032 +0.036 +0.038  3/3        +0.042 +0.059 +0.056   0.20 0.27 0.25 (0/3)  0.17 0.19 0.23      0.15-0.16  0.66-0.70  0.22/-0.03/0.13
delta  K=30        +0.047 +0.042 +0.041  3/3        +0.060 +0.062 +0.063   0.30 0.31 0.34 (2/3)  0.24 0.19 0.37 (1/3) 0.14-0.21  0.66-0.70  同上
next   K=30        +0.001 +0.001 +0.000  0/3        +0.005 +0.007 +0.002   -0.01 -0.01 0.05      -0.02 -0.10 0.15    ≈0         0.66-0.70  同上
```

- **Gate P：delta 聚类 K = 10 / 30 都 3/3 通过**（≥ 0.02 nats），K = 30 拿到约 0.045 nats；s' 聚类 K = 30 对照为 0 ⇒ 起作用的是表征（接触冲量在 Δs 上可见），不是簇数
- **Gate C：rho_norm 在 K = 30 到 0.30 / 0.31 / 0.34（判据 > 0.3 为 2/3，形式上不过）**，且每个 seed 都高于 rho_occ；rho_local 0.19–0.37；LoDADA 0.14–0.21；DARA 0.66–0.70。cluster 估计器从 0 升到 0.3，但仍不到 DARA 的一半
- 结论：给对表征后 cluster 通道能检测、能部分排序；全局连续估计器在同一数据上远好于它。localization 的存在理由只剩 N_t 扫描（job 202755）这一条路

### N_t 扫描结果（2026-10-16，job 202755，hopper friction 2.0 perfect-support，delta K = 10，3000 步，3 seed）

```
N_t      g_coarse − null (nats)     g_full − null       rho_dara            rho_norm            rho_local           rho_rev
500      +0.08 +0.15 +0.08 (过拟合)  +0.22 +0.19 +0.17   0.51 0.52 0.49      0.16 0.21 0.19      0.14 0.21 0.06      0.14 0.10 0.08
1000     +0.08 +0.12 +0.01           +0.16 +0.20 +0.14   0.61 0.62 0.61      0.22 0.23 0.24      0.18 0.28 0.21      0.18 0.13 0.06
2500     +0.04 +0.05 +0.05           +0.14 +0.13 +0.14   0.71 0.64 0.68      0.16 0.29 0.28      0.25 0.27 0.33      0.13 0.14 0.13
5000     +0.03 +0.04 +0.04           +0.15 +0.14 +0.14   0.67 0.66 0.70      0.20 0.27 0.25      0.17 0.19 0.23      0.15 0.16 0.16
10000    +0.03 +0.03 +0.03           +0.15 +0.16 +0.16   0.73 0.68 0.75      0.25 0.28 0.27      0.20 0.25 0.24      0.12 0.13 0.15
20000    +0.03 +0.04 +0.03           +0.16 +0.17 +0.16   0.70 0.72 0.75      0.37 0.33 0.23      0.33 0.29 0.32      0.18 0.15 0.10
```

- **预注册判据（存在 N_t 使 cluster 估计器 rho > rho_dara，3/3）不成立，预测命中**：六档里每一档 3/3 DARA 更高；N_t = 500 时 DARA 0.49–0.52 vs cluster ≤ 0.21，N_t = 20000 时 0.70–0.75 vs ≤ 0.37。cluster 估计器随 N_t 缓慢上升但从未接近
- N_t ≤ 1000 的 Gate P 绝对值是过拟合区（null 行 g_coarse −0.3 到 −0.96），只能看差值；full 通道在所有 N_t 上高出 null 0.14–0.22
- 结论：在唯一能排序的配置上、给对表征之后、任何目标样本量下，cluster 级估计器都没有样本效率优势。"为什么要 cluster 而不是全局 D_SAS"在这条线上没有答案

## 总判断 v2（2026-10-16，全部 gate 跑完）

**方法线（cluster 级 occupancy-corrected 过滤作为 LoDADA 的改进）：没有存在理由。** 证据链：
1. ODRL 现成目标数据在两个环境上都不满足 support（ce_occ ≤ 0.16），任何 transition 级条件估计都不可辨识，三家都在按 occupancy 过滤（Exp 3、halfcheetah Gate S/P）
2. perfect support 下，s' 聚类的 coarse 通道在重力、摩擦、两个环境、K 10–1000、3000–20000 步上从未检测到偏移；Δs 聚类在 hopper friction 上检测到（0.03–0.045 nats）并部分排序（rho 0.2–0.37）
3. 同一数据上全局连续 D_SAS（DARA）：检测 0.07–0.29 nats、排序 0.49–0.75，N_t 500–20000 全程领先
4. 带内假阳性（behavior 行 0.9 vs null 0.15）是 overlap 稀薄处的有限样本方差（behavior_m 过 2× 判据），归一化不能消除，训练越久越大

**诊断线（what does local domain discrimination identify）：证据完整。**
1. LoDADA 的 reverse score 不是动态差异：toy 假阳性（5/5）、精确抵消结构 + cluster 级符号反转（rho −0.95，6/6 seed）、真实数据带内假阳性 7–8 nats 且跨拟合完全可复现（rep 0.93）
2. 三个可测的可辨识条件及其测量工具：support = H(D|Z)（ce_occ）、resolution = 配对翻转率（I(D;K|Z) ≤ ln2·flip）、power = cross-fit 增益减置换对照；null / behavior 行作精确零真值
3. 偏置型偏移（重力）的逐 transition gap 在状态间近常数（cv 0.17），"局部可迁移性"无结构可局部化，与 1b 的可分离性理论一致；增益型（摩擦）cv 1.9，才有东西可排
4. ODRL 默认 20000 步分类器在 N_t = 5000 时深度过拟合（held-out CE 1.64 vs 0.69），LoDADA / DARA 原始实验共享这个问题

**下一步由用户决定**，候选按价值：(a) Gate O：用 hopper friction 2.0 的配对真值做 oracle 过滤 / 加权的 IQL 对照（target-only、concat、oracle gap、occupancy-only），回答"即使完美知道动态差异，过滤能否提高 target return"；(b) 以诊断线写作，题目方向 "When does local domain discrimination identify dynamics shift"；(c) 若要救方法线，只剩多步结果（h = 5–20）+ Δs 表征 + 早停的组合，且要在 N_t 扫描上重新对 DARA 打一次

## Gate O：oracle 过滤的下游 IQL pilot（2026-10-16，第三份评注；预注册，启动前写下）

**问题**：即使完美知道每条 source transition 的动态差异，按它过滤 source data 是否真的提高 target offline RL 的 return？这是整条线上唯一还没测的层（decision utility）。

**设定**（`odrl/oracle.py`、`algo/offline_offline/score_iql.py`、`odrl/run_gate_o.sbatch`）：hopper friction 2.0；源 d4rl hopper-medium-replay-v2（402k）；目标 = `paired`（split seed 0 的 null 块 5000 条 (s, a) 在摩擦 2.0 物理里走一步，带目标奖励与终止；perfect support，transition 级，绕过 `call_tar_dataset` 的轨迹重建）；IQL（ODRL 的实现，每步 batch = 128 源 + 128 目标）500k 步，eval 每 10k 步 10 回合；所有 arm 共享数据、超参、seed，只改 source 选择：
1. target_only（batch 两半都来自目标）
2. concat（全部 source）
3. random（随机保留 50%）
4. oracle_state（保留配对 state gap 最小的 50%）
5. oracle_bellman（保留 |r_s + γ(1−d_s)V_t(s'_s) − r_t − γ(1−d_t)V_t(s'_t)| 最小的 50%；V_t = 在 `pairedbig`（全部 402k 源 (s, a) 在目标物理里走一步）上 target-only 训练的 IQL 的 V，只作评估 oracle）
6. dara（保留 logit D_SAS − logit D_SA 最高的 50%，分类器与识别 gate 同设置）

主终点 target return（原始与 ODRL 归一化分）；次终点 held-out 配对目标集（`pairedheld`，50k）上的 Bellman 残差 |r + γ(1−d)V(s') − min Q(s,a)|。3 seed；keep 0.5；之后只做 0.25 / 0.75 作敏感性。

**判据**：取每个 run 最后 5 次 eval 的均值为该 run 的 return；一个 arm "赢" = 其 3 seed 均值高于 concat 与 random 两者，且差距大于两者 seed 间 std 的较大者，3/3 seed 逐一高于 random。

**预测**（我的）：concat ≫ target_only（5000 条目标数据太少，源数据同形态近物理）；oracle_state ≈ random ≈ concat（差在 seed 噪声内）；oracle_bellman 与 oracle_state 差别不大；dara ≈ random。即倾向评注的结果 A：在这个任务上按局部动态差异过滤本身没有下游价值。若 oracle_bellman 赢而 oracle_state 不赢 → 结果 B（task-aware mismatch）；若 oracle 赢且 dara 接近 → 结果 C；oracle 赢且 dara 明显落后 → 结果 D。

**构建结果（job 202758 / smoke 202759）**：402k 源 transition 的配对 state gap 均值 0.119、cv 4.0（全数据集，比 5000 探针的 1.9 更偏态）；DARA 分数（train 块 vs paired 5000，3000 步）对 −gap 的 Spearman 0.60；IQL 吞吐 ~210 it/s，500k 步 ≈ 50 min / run；smoke 模型的 held-out Bellman 残差函数可用（1.27）。作业：5 arm × 3 seed = 202760，oracle V_t = 202761，oracle_bellman arm 在 V_t 结束后自动提交。

**对评注的两点核对**：(i) 评注说 DARA "参数共享的连续 sample-level estimator"、"clustered 不等于更高局部分辨率"——与本仓库全部测量一致；(ii) 评注的 Bellman oracle 用 V_t 不用 Q_t，对 IQL 的 expectile V 是对的，但 V_t 是 oracle IQL 隐式策略的价值，不是最优价值，写作时要说明

**结果（2026-10-16，jobs 202760 / 202761 / 202776，18 run 全部 EXIT=0；汇总脚本取每 run 最后 5 次 target eval 的均值）**

| arm | last-5 target return（seed 0/1/2） | mean ± sd | ODRL 归一化 | held-out Bellman 残差（seed 0/1/2） | held-out Q 均值 |
|---|---|---|---|---|---|
| concat | 283.5 / 294.0 / 289.3 | 289.0 ± 4.3 | 9.7 | 2.97 / 2.82 / 3.11 | 177 |
| random | 189.2 / 283.5 / 290.0 | 254.2 ± 46.1 | 8.6 | 3.03 / 3.02 / 3.19 | 179 |
| oracle_bellman | 264.8 / 345.1 / 289.4 | 299.8 ± 33.6 | 10.0 | 1.81 / 2.19 / 2.24 | 94 |
| oracle_state | 81.4 / 14.3 / 107.9 | 67.9 ± 39.4 | 2.9 | 2989 / 5414 / 245 | −10⁴（发散） |
| dara | 17.0 / 15.4 / 106.1 | 46.1 ± 42.4 | 2.2 | 331 / 482 / 109 | −10³（发散） |
| target_only | 161.1 / 43.7 / 100.2 | 101.7 ± 47.9 | 3.9 | 2.5e5 / 6.9e4 / 2.0e5 | 10⁶（发散） |

**预注册判据**：没有 arm 赢。oracle_bellman 均值高于 concat 与 random ✓，但对 concat 的差距 10.8 < 两者 seed sd 的较大者 46.1 ✗，逐 seed 高于 random 只有 2/3 ✗。oracle_state / dara / target_only 远低于 random。按预注册 → **结果 A**：即使完美知道每条 transition 的动态差异，按它硬过滤 50% 的 source 在这个任务上没有下游价值。

**次终点**：oracle_bellman 的 held-out Bellman 残差 3/3 seed 低于 concat 与 random（约 −30%），但 return 没有跟着动，而且它的 source return 不稳（seed 1 source 只有 514 vs concat 2200–3100）：按目标 Bellman 一致性选数据让 Q 更贴目标，策略不更好。

**两项事后诊断（写判断前必须做，因为表里有两个异常）**

(a) *~290 是什么*：把保存的 actor（确定性均值动作，`odrl/gate_o_rollout_diag.py`）在 source（Hopper-v3）和 target（friction 2.0 xml）各跑 5 回合。concat/0 在 target 每回合都在第 117–118 步、concat/1 在 124–127 步终止，最低高度 0.69 < healthy 阈值 0.7，前进速度 1.35–1.41；random/0 在 94 步终止；oracle_bellman/1 走到 181–191 步但速度只有 0.67（慢步态），同时 source 只有 202–336 步。所以**主终点 = 摔倒前的步数**，所有不发散 arm 的动态范围只有 94–191 步；source 侧 concat / random 都健康（2200–3100）。换句话说，5000 条 transition 级目标数据 + 500k 步在 friction 2.0 上没有任何 arm 学会持续跳跃；这个 pilot 对"过滤有益"的检出力很低，**A 的含义是"没有显示出有益"，不是"显示了无益"**。

(b) *三个 arm 为什么发散*：held-out 目标集上 Q/V 量级（`odrl/gate_o_qv_diag.py`）oracle_state 10⁴–10⁵、dara 10³–10⁵、target_only 10⁶–10⁷（concat / random 177–180、oracle_bellman 85–99，合理）；发散 arm 的 source return 也掉到 1–15（策略倒退、22–49 步摔倒）。kept 半集的组成（`odrl/gate_o_mask_diag.py`）：

| mask | 保留 terminal 比例 | gap 均值 | reward | vx | kept s′ 到 kept s 的最近邻距离 p90（标准化坐标） |
|---|---|---|---|---|---|
| 全集 | 0.0041（1639 条） | 0.119 | 2.37 | 1.37 | 0.004 |
| random | 0.0040 | 0.119 | 2.37 | 1.37 | 0.514 |
| oracle_state | **0.0000** | 0.006 | 2.55 | 1.55 | 0.224 |
| dara | 0.0013 | 0.041 | 2.60 | 1.61 | 0.365 |
| oracle_bellman | 0.0081 | 0.069 | 2.54 | 1.54 | 0.214 |
| target_only（paired 5000） | 0.0056 | – | 2.39 | 1.40 | **1.118**（14% > 1） |

oracle_state 与 oracle_bellman 的 mask Jaccard 0.81，差别最显眼的是 terminal：state gap 过滤把全部 1639 条终止 transition 都丢了（摔倒是重接触、gap 大），oracle_bellman 反而保留 2 倍。闭包（kept s′ 是否落在 kept s 附近）对 source 过滤不是机制（oracle_state 的 p90 比 random 还好）；target_only 的闭包确实坏（transition 级集合，s′ 从不在样本内），它的发散符合 off-support bootstrapping。**INFERENCE**：按动态 gap 硬过滤系统性删除终止 transition，IQL 在这份数据上随之发散。机制检验已提交（job 202812，3 seed）：`oracle_state_term` = oracle_state 分数但 terminal 强制保留（多 0.4% 数据，与 oracle_state mask 重叠 199160/200799）。预注册：若不再发散（Q 均值 0–300、source return > 1000）且 target return ≈ concat / random → state oracle 的失败是 terminal 删除的人为效应，结果仍是 A；若仍发散 → 机制在 kept 半集的其它组成差异，结果也仍是 A。

**机制检验结果（2026-10-16，job 202812，`oracle_state_term` = oracle_state 分数 + 1639 条 terminal 强制保留，3 seed）**

| seed | target last-5 | source last-5 范围 | held-out Bellman 残差 | held-out Q min / mean / max |
|---|---|---|---|---|
| 0 | 60.1 | 24–68 | 3.20 | −232 / 60 / 234 |
| 1 | 174.8 | 814–1001 | 1.74 | −114 / 100 / 243 |
| 2 | 334.0 | 699–965 | 26.3 | −10271 / −27 / 710 |

target 均值 189.6 ± 112.3（oracle_state 67.9 ± 39.4、concat 289.0、random 254.2）。对照预注册判据：Q 均值落在 0–300 的 2/3 seed（seed 2 均值 −27、min −10⁴，仍有局部发散）；source return > 1000 的 0/3（seed 1、2 在 700–1000，seed 0 塌到 24–68）；target ≈ concat / random 只有 seed 2。读法：放回 0.4% 的 terminal 让 Q/V 量级从 10⁴–10⁵ 回到 10²、让 2/3 seed 的 source 策略恢复到 medium-replay 水平的三分之一，所以 **terminal 删除是发散量级的主因，但不是全部**：state-gap 过滤剩下的那一半（gap 均值 0.006，偏向无接触 / 高速相位）即使补回 terminal，对 IQL 仍是比 random 半集差得多的训练集（source 46–915 vs random 2200–3100）。判断不变：结果 A；"gap 硬过滤有害"的机制 = terminal 系统性删除 + kept 半集组成偏斜，两者来自同一个事实——摩擦敏感的 transition 正是接触 / 摔倒相位，按动态差异过滤等于把落地和摔倒从数据里删掉。

**判断**：方法线维持关闭。对诊断线新增第四条：在 ODRL hopper friction 2.0 perfect-support 设定上，oracle 级别的动态 gap 过滤没有决策价值（Bellman oracle ≈ concat ≈ random），而且 gap 硬过滤（oracle 或 DARA）在 keep 0.5 下不是无益而是有害（Q 发散），机制候选是终止 transition 被系统性删除。限制：主终点动态范围 94–191 步；要做一个有检出力的 Gate O 需要换设定（轨迹级目标数据、更多目标数据或更轻的 shift），这是范围变更，由用户决定。keep 0.25 / 0.75 敏感性按评注只在出现趋势时做，没有趋势，不做。

## 第四份评注后的审计与正对照（2026-10-16；先查代码、评估已有模型，再起小训练）

评注的核心：Gate O 混合了数据构造、筛选导致的覆盖变化和训练稳定性，不足以否定 dynamics-aware transfer；最该先查的是 terminal 行的 `s'` 语义。

**代码审计（读 `odrl/oracle.py`、`algo/offline_offline/score_iql.py`、ODRL `train.py` / `utils.py` / `iql.py` / `dataset/call_dataset.py`，FACT）**

| 评注的疑点 | 本仓库实际情况 |
|---|---|
| state gap 的 source 一端是数据数组还是重模拟 | **数据数组**：`gap = ‖ns_t − d['next_observations']‖/scale`，`d` 来自 `d4rl.qlearning_dataset`，它用 `observations[i+1]` 当 `s'`；terminal 行的 `s'` 是下一回合的 reset 状态。评注的嫌疑成立，下面用全量重模拟审计量化 |
| `TimeLimit` 计数 | 排除：`step_full` 走 `env.unwrapped.step`，不经过 wrapper |
| paired 目标集的 reward / terminal 来源 | 目标 simulator 的 `step` 返回值，不沿用 source |
| source mask / score / buffer 行号对齐 | 由构造保证：ODRL `train.py` 的源 buffer = `convert_D4RL(d4rl.qlearning_dataset(src_eval_env))`，与 `source_dataset` 是同一调用；`score_iql` 另有长度断言 |
| `reward` / `not_done` 形状 | 都是 `[N, 1]`，`set_buffer` 与 `convert_D4RL` 同键同形 |
| d = 1 ⇒ y = r | ODRL `iql.py`：`value_target = reward + not_done * discount * v_t` |
| 未筛选的新 learner 与原 IQL 一致 | `ScoreIQL` 只重写 `prepare`，`train` 调 `super().train`；`source = concat` 且无 `target_file` 时 buffer 原样不动 |
| ODRL IQL 的 actor | advantage-weighted **MSE 到均值动作**（`exp_adv * (tanh(mu) − a)²`，clamp 100），不是官方 IQL 的加权 log-likelihood；`train.py` 没有 locomotion reward normalization（只有 antmaze 的 −1）；lam 0.7、temp 3.0、batch 128 + 128 |
| Gate O 的 `dara` arm | 是 **DARA-score hard filtering**（按 logit D_SAS − logit D_SA 硬保留 50%），不是 ODRL 的 DARA（reward correction）；下文一律改称 DARA-score hard filter，它的失败不能报告为 DARA 失败 |
| Gate O 与 LoDADA | 全局 50% 硬删 ≠ LoDADA 的按 cluster 保留 + critic weighting + target 行为正则（评注记载保留比例 90/80/70%，arXiv HTML 正文未见数值，**未核对**） |
| ODRL 公开 `dara.py` 的 classifier（Softmax 后再 cross_entropy、输入噪声 std 1.0） | 与本仓库无关：识别 gate 和 Gate O 的分类器都是 `adaptation/` 自实现（logit 输出 + BCE），没有用 ODRL 的 classifier |

**正对照 1：oracle V_t 自己的 actor（`odrl/gate_o_vt_eval.py`，FACT）**。V_t 是在全部 402k 条 paired 目标 transition 上 target-only 训练的 IQL（job 202761）。它的 actor 在目标环境 10 回合全部 104–116 步摔倒（return 254、vx 1.31、最低 z 0.71–0.92），在 **source** 环境却跑满 1000 步（return 2691、vx 1.69）。V(s₀) ≈ 80，而它自己策略在目标里实现的折现回报 G₀ = 133–146，source 里 215–233；轨迹上 corr(V, G) 目标 0.35–0.74、source ≈ 0。读法：**拿到最好的一步目标数据（perfect support、402k、目标物理）也学不出目标行为**；AWR 提取只能在 source 的 (s, a) 上重加权，paired 数据没有后继状态处的动作覆盖（评注第五节），V_t 也不是校准的参考价值。这个装置没有正对照。

**正对照 2：benchmark 这一格本身（FACT）**。ODRL 自带的 `hopper_friction_2.0_medium.hdf5`：5000 行、36 个完整回合、**每回合 137–139 步、return 332–339、最低 z 0.71**、vx 1.41——采数据的 "medium" SAC 策略自己在 2 倍摩擦下 138 步必摔，数据里没有一步持续跳跃的示范。LoDADA Table 1（arXiv 2602.21072v1，2026-10-16 核对）hopper friction 2.0：BOSA 10.32 ± 0.06、IQL 10.15 ± 0.03、DARA 10.15 ± 0.06、IGDF 10.35 ± 0.60、OTDF 10.52 ± 0.29、LoDADA 10.44 ± 0.13，折回原始 return 305–317，**都低于数据本身的 335**；ODRL 论文（2410.20750）locomotion 目标数据 = 5000 transitions（FACT）。所以"所有策略 ~290 摔倒"是这一格的性质：选它做 Gate O 是因为一步 gap 好排序，但它没有适配收益的空间，整条方法线的生死不能押在这一格上。

**预注册（写于读 job 202816 / 202815 / 202817 / 202822 / 202823 结果之前）**

- *A 审计（`odrl/gate_o_audit.py`，job 202816，全部 401,598 条 source 在 source 物理里重模拟）*：预测 nonterminal 行 ‖s'_data − s'_sim‖ 中位 < 10⁻³（标准化单位），只有速度触到 ±10 裁剪的行例外；**terminal 行中位 ≫ 1**（reset 状态 vs 摔倒后状态）；D4RL 原始 `next_observations` 键在 terminal 行与重模拟一致（误差小）；gap_data 与 gap_sim 在 nonterminal 行上 Spearman > 0.99；用 gap_sim 以 50% 保留时 terminal 保留 20–60%（接触大 gap 但不会全删），而 gap_data 下是 0。
- *C 等数量随机加回（`oracle_state_rand`，job 202815，oracle_state mask + 1639 条随机被删 nonterminal）*：预测仍像 oracle_state 一样发散（≥ 2/3 seed Q 量级 ≥ 10³、target < 100），即补回 terminal 的作用是特异的，不是"多补一点数据"。
- *修正 arm（`oracle_state_sim` = −gap_sim，job 202817，afterok 审计）*：预测 3/3 seed 不再灾难性发散（held-out Q 在 ±300 内）、source return 1000–3000、target 200–300 ≈ concat / random，仍不满足原判据。
- *参考跑（`odrl/run_gate_o_reference.sbatch`，ODRL 真实目标 5000 条轨迹数据；job 202822 concat = 原版 IQL，job 202823 target_only）*：预测两者 target return 都在 250–340（120–140 步摔倒），都不会持续跳跃，因为数据本身只示范摔倒。
- 若 A 证实 terminal 语义问题而修正 arm 不发散，则"gap 硬过滤有害"的机制改写为：**oracle_state 的 terminal 全删是 D4RL 数据语义造成的人为效应**，不是物理 gap 的性质；DARA-score hard filter 的 terminal 偏低（0.0013）另有解释（分类器学到 terminal 行的 s' 像 reset 状态 → 判为 source）需单独说明。

**A 审计结果（job 202816，FACT）**

| 类别 | n | ‖s'_data − s'_sim‖ 中位 / p90 / max | ‖raw next_obs − s'_sim‖ 中位 / max | gap_data 中位 | gap_sim 中位 | 50% 保留率 data | 50% 保留率 sim |
|---|---|---|---|---|---|---|---|
| nonterminal | 336,247 | 0.000 / 0.000 / 0.018 | 0.000 / 0.018 | 0.028 | 0.028 | 0.55 | 0.46 |
| nonterminal，速度触 ±10 裁剪 | 63,712 | 0.088 / 0.300 / 2.108 | 0.088 / 2.108 | 0.122 | 0.000 | 0.24 | 0.70 |
| terminal | 1,448 | **6.911 / 8.647 / 12.732** | **0.000 / 0.000** | 6.903 | 0.000 | **0.00** | 0.59 |
| terminal，裁剪 | 191 | 7.620 / 10.105 / 12.527 | 0.119 / 1.010 | 7.620 | 0.000 | 0.00 | 0.87 |

reward 与 terminal 一致性：|r − r_sim| 中位 0（max 0.19，只在裁剪行），d ≠ d_sim 比例 0.0000–0.0052。原始 D4RL 有 402,000 行、402 个 timeout、1639 个 terminal；`qlearning_dataset` 丢 timeout 行得 401,598 行，`observations[i+1]` 与原始 `next_observations` 键在 99.59% 的行一致，不一致的正是 1639 个 terminal 行。

读法：
1. **评注的嫌疑完全成立**：terminal 行的 `s'_data` 是下一回合的 reset 状态，与重模拟差 7–13 个标准差；D4RL 原始 `next_observations` 键在这些行与重模拟完全一致（0.000），是 `qlearning_dataset` 把它丢了。oracle_state 删光 terminal 是数据语义造成的人为效应。
2. **我之前的推断是错的**：我写过"摔倒 = 重接触 = gap 大"。真实物理 gap 在 terminal 行的中位是 **0.000**（63% 的 terminal 行 gap_sim < 0.01）；进入终止的那一步大多没有摩擦敏感的接触。修正后的 50% 保留会留下 1015 / 1639 条 terminal（62%，高于基础率）。预注册预测 20–60%，实际 62%，方向对、量偏高。
3. **新发现一个更大的测量限制**：15.9% 的 source 行有速度分量触到 Hopper-v3 的 ±10 观测裁剪（63,903 行），这些状态从观测重建不回去，重建误差中位 0.088、p90 0.30、max 2.1 标准差。对它们 gap_data 被重建误差污染（中位 0.122 vs 像物理的 0.000），gap_data 下只保留 24%、gap_sim 下保留 70%。这同样影响前面所有用 `ground_truth.py` 从观测重建再走一步的"真值 gap"（Gate R 的 flip rate、Gate C 的 rho）：对裁剪行真值是在错误状态上算的。它不偏向任何估计器，但 rho 的上限被它压低，写作时要作为限制说明，或把裁剪行从探针里剔除后重算。
4. gap_data 与 gap_sim 的整体 Spearman 只有 0.73，但在未裁剪、非 terminal、两者都 > 0.01 的行上是 1.000（n 179861）；低值来自 48% 的行 gap_sim 恰为 0（空中相位，并列名次随机打破）和裁剪行。两套 50% mask 的 Jaccard 0.736；keep_sim 的 terminal 比例 0.0051（基础率 0.0041）、裁剪行比例 0.222（基础率 0.159）。
5. 对 DARA-score hard filter 的 terminal 偏低（0.0013）现在有了更直接的解释：D_SAS 在 source 看到 1639 条 `s'` = reset 状态的行，目标 paired 集里没有这种 `s'`，所以这些行被判为强 source → 分数低 → 被删。

**预注册对照结果（2026-10-16，jobs 202815 / 202817 / 202822 / 202823，全部 EXIT=0；最后 5 次 target eval 均值，held-out = `pairedheld`，目标回合长度 / 速度 = 保存的 actor 在目标环境 5 回合）**

| arm | target（seed 0/1/2） | mean ± sd | source last-5 | held-out Q 均值（seed 0/1/2） | 目标回合长度 | vx | 预测 | 结果 |
|---|---|---|---|---|---|---|---|---|
| C `oracle_state_rand`（oracle_state + 1639 条随机被删 nonterminal） | 15.5 / 6.5 / 14.8 | 12.3 ± 4.1 | −0.5 / 8.1 / 3.3 | −3.2e4 / −4.9e4 / −1.0e4 | 12–53 | −0.47…−0.70 | 仍发散 | **✓** |
| 修正 arm `oracle_state_sim`（−gap_sim，terminal 保留 62%） | 67.9 / 49.9 / 129.3 | 82.4 ± 34.0 | 161 / 267 / 529 | 1.0e4 / 7.4e2 / 3.7e4（min −10⁶） | 69–151 | −0.40…−0.02 | 不发散、≈ concat | **✗ 仍发散** |
| 参考 concat（ODRL 真实目标 5000 条，原版 `iql`） | 294.3 / 294.6 / 302.5 | 297.1 ± 3.8 | 1826 / 2220 / 1684 | 203 / 203 / 203 | 127–129 | 1.32–1.35 | 250–340 | ✓ |
| 参考 target_only（真实目标轨迹） | 335.0 / 334.9 / 334.9 | **334.9 ± 0.1** | 273 / 333 / 260 | 140 / 135 / 138 | 137–138 | 1.44 | 250–340 | ✓ |

读法：

1. **C 对照成立**：随机补回等量数据不改变任何东西（Q 仍 10⁴–10⁶、策略倒退 12–53 步摔倒），补 terminal 的部分恢复（job 202812）是特异的。
2. **修正 arm 推翻了我上一段的机制解释**。用重模拟的真实 gap 以后 terminal 保留 62%（高于基础率），IQL 仍然发散（3/3 seed held-out Q 量级 10³–10⁵、min −10⁶，策略向后移动）。所以 **"terminal 系统性删除是发散主因"不成立**：terminal 被删是 D4RL 语义的人为效应，但不是发散的原因；强制补回全部 terminal（202812）只部分恢复，也与此一致。目前能稳住的说法只有：按一步动态 gap 硬保留最小的 50%（无论 gap 来自数据数组还是重模拟）在 hopper medium-replay 上 3 个变体 × 3 seed 全部让 IQL 发散，而随机 50% 和 Bellman-gap 50% 不发散。48.2% 的行真实 gap 恰为 0（空中相位），"保留最小 50%" 事实上 = "只保留摩擦无关的 transition"，被删的是全部接触相位。INFERENCE（未检验）：缺少接触相位的 transition 后，bootstrap 经过的落地状态没有自己的 transition，之前用最近邻距离做的闭包检查（p90 0.22）分辨率不够（没有分开位置和速度）。机制留作开放问题，不再在这一格上追。
3. **参考跑给出这一格的天花板**：真实目标轨迹 target_only 的 return 334.9 ± 0.1 = 数据本身的 335（学回了行为策略，138 步摔倒，vx 1.44）；参考 concat 297 ≈ paired concat 289。没有任何配置超过行为数据，与文献（全部方法 10.15–10.52）一致。正对照不存在 → 按评注，这个装置不能用来裁决细微的 filtering 收益，第三阶段（DARA 原形式 vs hard filter）不在这一格上做。
4. 次终点 Bellman 残差：参考 concat 3.1–3.3、参考 target_only 15–18（在 paired held-out 上，它只见过 5000 条真实轨迹）；发散 arm 10²–10³，无信息。

**换格的依据（FACT，2026-10-16 核对）**。LoDADA Table 1 全表（arXiv 2602.21072v1，medium 源 1M + 目标 5000）：32 格里 **hopper friction 2.0 是唯一一格六方法差距 < 1 分（0.37）**；21 格差距 ≥ 10 分；LoDADA 大幅领先的格有 halfcheetah gravity 5.0（71.1 vs IQL 20.9）、ant friction 2.0（62.3 vs 17.5）、walker2d gravity 0.1（71.4 vs 20.1）、hopper gravity 0.5（43.7 vs IQL 17.8 / DARA 14.7）；论文不报各格目标数据的回报。本地已有的 ODRL 目标 hdf5 的回合结构（5000 行；halfcheetah 全是 timeout、无 terminal）：

| 目标数据 | 回合数 | 回合长度 中位 / 范围 | 回合 return 中位 / 范围 | vx | LoDADA 表差距 |
|---|---|---|---|---|---|
| hopper friction 2.0 medium | 36 | 138 / 137–139 | 335 / 332–339 | 1.41 | 0.37 |
| hopper gravity 2.0 medium | 20 | 233 / 228–285 | 752 / 744–907 | 2.23 | 6.6 |
| hopper friction 0.5 medium | 8 | 336 / 320–1321 | 1117 / 1038–4277 | 2.26 | 8.8 |
| **hopper gravity 0.5 medium** | 7 | 660 / 505–815 | 2397 / 1785–2885 | 2.57 | **29.0** |
| hopper gravity 0.5 expert | 5 | 988 / 896–999 | 4126 / 3746–4186 | 3.12 | – |

若要再做一次有正对照的 Gate O，首选 **hopper gravity 0.5 medium**：同环境只换轴与强度（本仓库 gravity 轴的 Gate R / S / P 已有），目标数据示范了 500–815 步的持续跳跃，文献里 IQL 17.8 / DARA 14.7 / LoDADA 43.7 说明方法间有可测差距。代价：gravity 是偏置型 shift（gap cv 0.17，近常数），一步 gap 排序本身区分度低，这正好是检验"dynamics-aware 数据复用的收益是否需要局部分辨率"的格。先做的事：参考 target_only（真实目标轨迹）是否 ≥ 数据本身、concat 是否 < target_only（有适配空间），再上任何 score。

**Gate O 最终表述**：未达预注册收益判据；本装置（paired 目标、全局 50% hard-selection、ODRL-IQL、hopper friction 2.0）没有正对照，天花板 = 行为策略，不能作为 dynamics-aware data reuse 无效的证据。可保留的、被复现的负结果只有一条：**按一步动态 gap 硬保留低 gap 半集会毁掉 IQL**（3 变体 × 3 seed），而且一步 gap 对数据选择的分辨率有限（48% 恰为 0）。要测 dynamics-aware data reuse 的决策价值，得换一个有正对照的 benchmark 格（目标数据示范成功、文献方法间有差距），这是用户的决定。

**结论改写（按评注第十节，替代 "Gate O 结果 A → 方法线关闭" 的强版本）**

> Gate O 未达到预注册的收益判据。该结果只适用于当前 paired-target、全局 50% hard-selection、ODRL-IQL（MSE actor、无 reward normalization）协议，且这一 benchmark 格的目标数据本身只示范摔倒、文献中所有方法都在 10.2–10.5。由于 terminal 行的 `s'` 语义问题、参考价值 V_t 未校准、所有稳定组目标控制均差、以及没有任何正对照，**本实验不能作为 dynamics-aware data reuse 无效的证据**。

保留两个独立判断：(i) cluster 方法：estimator 层没有证据优于连续 conditional estimator，暂停扩展合理；(ii) occupancy-confounding 诊断：核心问题成立，下一步是验证"识别修正如何进入一个不破坏学习数据结构的训练目标"（reward correction 而非硬删除），不是继续假定正确 score 必须通过硬删除体现收益。

同时降调三处旧表述：次终点 Bellman 残差是模型自身的内部一致性 |r + γ(1−d)V_θ(s') − Q_θ|，不是对真值的误差，不能写成"Q 更贴近目标"；"290 只是生存步数"改为"return 主要受生存时间限制，同时受速度与控制代价影响，更晚摔倒是合法改善，必须同时报速度"；诊断线的 held-out CE 只是 H(D|Z) 的上界（CE = H + E KL），两个分类器的 CE 差还含各自拟合误差，报告时把"统计诊断"与"精确识别了信息量"分开。

## 第五份评注：范围决定（2026-10-16）

**决定**：封存 hopper friction 2.0 上的继续排障；先做已有 score 的干净子集复核（不加新训练的方法实验）；gravity 0.5 medium 只跑 `target_only` 与 `concat` 两个参考组 × 3 seed，不启动任何 score / 过滤器 / 方法实验。

**三条线分开固定（替代任何"idea 成功 / 失败"总判决）**
- *识别问题*：reverse conditional ≠ 纯 dynamics discrepancy，解析恒等式与完整状态 toy 反例成立。真实 Hopper 上的量化归因要加一条限制：观测把速度裁剪到 ±10，完整状态 x 没被观测 o 完整保留，p_d(o'|o,a) = ∫ p(o'|x,a) p_d(x|o,a) dx，同物理下不同采集行为也可能让观测条件分布不同；所以真实数据上"同物理假阳性"里可能同时混着 reverse conditioning 的 occupancy 差异、裁剪造成的隐藏状态混合差异、有限样本误差，不能全部归到 occupancy confounding。
- *cluster 修正方法*：在已测设置中没有优于连续 conditional estimator 的证据，暂停扩展；N_t 扫描的负面证据要在干净子集上复核确认，不靠新的 RL 结果绕过。
- *下游数据使用*：低一步 gap 的全局 50% 硬筛选在 friction 2.0 上严重破坏 IQL；这不是"正确识别 dynamics 没有价值"的证明，而是这种选择规则没有同时保护学习所需的数据结构（48% 行 gap 恰为 0 → 规则 ≈ 保留空中阶段、移除接触阶段；"减小单步模型偏差"与"保留完成任务所需的覆盖"冲突，这是对规则的解释，不是已证实的发散机制）。
- 降调一处：真实目标轨迹 target_only = 334.9 ≈ 数据的 335 支持"当前数据与实现没有展示出超过行为水平的策略改善"，不能写成"这个数据集理论上只能到 335"。停止这一格的理由是资源分配（多个参考组都停在差的目标行为水平、文献无方法差距），不是上界证明。

**干净子集复核（预注册，不加新 RL 训练；`odrl/detectability.py` 加评估行有效性掩码）**。有效行 = held-out source 行中，s 与 s' 的速度分量都没触到 ±10 裁剪边界，且不是 terminal（有有效物理后继）。分类器训练数据不变（同全样本），只筛评估行；所有方法用同一批有效行、同一归一化尺度；同时报全样本结果和有效行占比（预期约 84%）。这是评估样本的有效性筛查，不把这些行从任何训练集删除。复核两个结果：
1. 同物理 behavior 对照（源 medium-replay vs 目标 expert-v2）的假阳性：LoDADA `rev` 带内 abs 7–8 nats、rep 0.93；`norm` / `local` / `dara` 的带内 cabs。预测：裁剪行占假阳性的一部分但不是全部——有效行上 abs_rev 仍 > 3 nats、rep_rev > 0.8；cabs_norm 仍高于 null 行（之前确认跑 1.33 vs 0.17）。若有效行上 abs_rev 掉到 < 1 nat，则真实数据上的 LoDADA 假阳性主要是裁剪造成的隐藏状态混合，归因要改写。
2. friction 2.0 perfect-support、delta 聚类 K = 10 下 DARA 与 cluster score 的排序差距：rho_dara 0.66–0.70 vs rho_norm 0.20–0.27 / rho_local 0.17–0.23。有效行上真值 gap 才是原始样本的真值。预测：两者都上升（真值噪声减少），差距保持（rho_dara − rho_norm ≥ 0.3）；若差距收窄到 < 0.1，cluster 线的负面证据要重新评估。

**干净子集复核结果（job 202843，2026-10-17；有效行占 held-out 的 83.5–83.7%；带内 τ = 0.1，全样本 → 有效行，seed 0/1/2）**

| 量 | 全样本 | 有效行 | 预测 | 结果 |
|---|---|---|---|---|
| behavior abs_rev（LoDADA 同物理假阳性，nats） | 8.51 / 7.02 / 7.67 | 8.32 / 6.73 / 7.05 | > 3 | ✓ |
| behavior rep_rev（两次独立拟合的 Spearman） | 0.930 / 0.928 / 0.922 | 0.934 / 0.922 / 0.920 | > 0.8 | ✓ |
| behavior cabs_norm | 1.50 / 1.06 / 1.51 | 1.49 / 1.06 / 1.29 | > null 0.17 | ✓ |
| behavior cabs_dara / cabs_local | 1.44 / 1.09 / 1.22 ；1.05 / 0.86 / 0.98 | 1.35 / 1.04 / 1.14 ；1.03 / 0.82 / 0.91 | – | 几乎不变 |
| paired friction 2.0 rho_dara（delta K = 10） | 0.666 / 0.663 / 0.703 | **0.727 / 0.707 / 0.759** | 上升 | ✓ |
| paired friction 2.0 rho_norm | 0.203 / 0.269 / 0.254 | 0.216 / 0.285 / 0.278 | 上升 | ✓ |
| paired friction 2.0 rho_local | 0.172 / 0.194 / 0.232 | 0.198 / 0.211 / 0.255 | 上升 | ✓ |
| rho_dara − rho_norm | 0.46 / 0.39 / 0.45 | 0.51 / 0.42 / 0.48 | ≥ 0.3 | ✓ |

读法：(1) 观测裁剪行不是真实数据上 LoDADA 假阳性的来源——去掉它们后 abs_rev 只降 2–8%，可复现性不变；真实 Hopper 上同物理假阳性的归因限制（隐藏状态混合）仍然要写，但量上不改变结论。(2) 在真值有效的行上，DARA 的排序更好、cluster 估计器略好，差距反而略增；cluster 线"不优于连续 conditional estimator"的负面证据在干净子集上成立。全样本结果同时保留。(3) 这是评估行筛查，没有删除任何训练行。

## Gate O2：hopper gravity 0.5 medium 基线（预注册，2026-10-16，启动前写下；新协议版本，不是"只换轴"）

**目的**：确认在一个目标行为较好、文献中存在方法差距的任务上，当前 learner 能否稳定学习，以及加入 source 数据是否带来可重复的影响。不是再找一格让 cluster 方法赢。

**协议**（`odrl/run_gate_o_reference.sbatch`，ENV / SHIFT / SRCTYPE 参数化）：源 = d4rl `hopper-medium-v2`（1M，原始物理；此前 Gate O 用的是 402k medium-replay + paired 目标，本协议单独标版本）；目标 = ODRL `hopper_gravity_0.5_medium.hdf5`（5000 行，7 个回合，回合长 505–815、return 1785–2885 中位 2397，vx 2.57；归一化参考 −26.3 / 3234.3，数据中位 ≈ 74 分）；ODRL IQL 原实现（MSE actor、lam 0.7、temp 3.0、无 reward normalization）500k 步，eval 每 10k 步 10 回合，不改任何超参。两组各 3 seed，共 6 run：
- `target_only`：`score_iql` 的 `source = target_only`，每步 128 + 128 = **256 条目标样本**；
- `concat`：原版 `iql`，每步 128 源 + 128 目标。
公平性记录：总更新次数相同 ≠ 目标数据训练强度相同；若 concat 明显更差，先核对目标项采样次数与 loss 系数，再谈 negative transfer。

**终点**：主 = 最后 5 次 target eval 均值（不事后挑 checkpoint），报逐 seed 差值，不把"均值差 > 某组 std"当显著性检验；辅 = 目标回合长度、前进速度、held-out Q/V 量级。文献参照：LoDADA Table 1 该格 IQL（**混合数据**）17.78 ± 6.85、DARA 14.68 ± 5.24、OTDF 35.22 ± 5.94、LoDADA 43.72 ± 9.72；17.78 是 concat 的参照，不是 target_only 的预期。

**预测**（我的）：target_only 学到接近行为数据的策略，target return 1200–2600（归一化 38–81），seed 间 sd > 300，回合长度 400–1000；concat 被 1M 源数据主导，在半重力下 400–1200（13–38，与文献 17.78 ≈ 553 同量级），逐 seed 低于 target_only → 落在评注表的第一行（存在值得研究的混合数据问题，先排除训练强度差异）。Q/V 量级两组都有界（< 500）。

**决策表（评注）**：target_only 稳、concat 退化 → 排除训练强度差异后才考虑一个已知方法或受控数据使用方式；两组稳且相近 → learner 正常、无明显 source 使用问题、不急于加过滤器；concat 明显优于 target_only → source 已有帮助，后续方法必须保住这部分收益；两组都远低于目标示范 → 先定位基线问题，不训练 score。跑出基线差距也不恢复 cluster 方法的地位。

**Gate O2 结果（2026-10-17，jobs 202844 concat / 202845 target_only，6/6 EXIT=0；首发 202837/202838 因 sbatch 变量 `ENV` 与 module 系统冲突即时失败、无训练）**

| 组 | seed | target last-5 | 归一化 | source last-5 | 目标回合长度 | 回放 return | vx | Q 均值 / max（目标数据） |
|---|---|---|---|---|---|---|---|---|
| target_only | 0 | 2095.4 | 65.1 | 618 | 616 | 2246 | 2.64 | 330 / 409 |
| target_only | 1 | 1795.2 | 55.9 | 814 | 526 | 1773 | 2.36 | 329 / 408 |
| target_only | 2 | 1855.7 | 57.7 | 549 | 576 | 2043 | 2.55 | 330 / 409 |
| concat | 0 | 567.9 | 18.2 | 1422 | 180 | 364 | 1.02 | 332 / 408 |
| concat | 1 | 414.1 | 13.5 | 1865 | 239 | 575 | 1.30 | 331 / 407 |
| concat | 2 | 526.8 | 17.0 | 1392 | 282 | 759 | 1.51 | 332 / 409 |

target_only 均值 1915.4 ± 129.6（归一化 59.6 ± 4.0），concat 502.9 ± 65.0（16.2 ± 2.0）；逐 seed 差 target_only − concat = +1527 / +1381 / +1329。Q / V 两组都有界（330 ± 1，max 409），没有发散。

对预注册：target_only 预测 1200–2600、sd > 300 → 实际 1915 ± 130（范围对、方差比预测小）；concat 预测 400–1200、与文献 17.78 同量级 → 实际 503 ± 65、归一化 16.2 ± 2.0，**与 LoDADA Table 1 的 IQL（混合数据）17.78 ± 6.85 在噪声内一致**；逐 seed target_only > concat → 落在评注决策表**第一行**。

读法：
1. **装置有正对照了**：这一格的 learner 能从 5000 条真实目标轨迹学到持续跳跃（526–616 步、vx 2.4–2.6、回放 return 1773–2246，接近数据的 1785–2885），且 concat 复现了文献的 IQL 基线。
2. **加入 1M 条源 medium 数据让 target return 掉 1300–1500（−72%）**，回合长度从 ~570 步掉到 180–280 步，速度从 2.5 掉到 1.0–1.5；源侧 return 反而 1392–1865。这是真实、可复现（3/3 seed）的混合数据问题。
3. **INFERENCE（待核）**：target_only 的 59.6 高于 LoDADA 表里这一格的所有方法（LoDADA 43.7、OTDF 35.2、IGDF 19.8、DARA 14.7、IQL 17.8）。如果原文协议与本协议可比（训练步数、batch、目标数据同为 5000 条 medium），那么这一格文献中所有"dynamics-aware"方法都没有恢复到纯目标数据的水平，"方法差距"实际上是"避免了多少 negative transfer"。要先核对：(a) 原文是否报 target-only 基线，(b) 训练步数 / batch 是否相同，(c) 本组 target_only 每步 256 条目标样本 vs concat 128 条的训练强度差异。
4. 按评注，这一行的后续是先排除训练强度差异，再考虑一个已知方法或受控数据使用方式；不启动 score / 过滤器。

**预注册：训练强度对照（启动前写下，job 见 `logs/gate_o_jobs.txt`）**。`target_only` 同协议只跑 250k 步（目标样本总数 = concat 500k 步的 128 × 500k，更新次数减半，eval 25 次，仍取最后 5 次均值）。这是保守对照：它只会让 target_only 变差。预测：last-5 target 1500–2100、仍逐 seed 高于 concat 的 414–568 ≥ 900；若 ≤ 700，则 concat 的劣势可能主要是目标项训练强度，negative transfer 的结论要收回。loss 系数差异（concat 里目标行只占梯度的一半）这个对照不覆盖，要单独说明。

## 待做

- [x] Experiment 1 多 seed（`toy/run_toy_seeds.sbatch`，按 seed 拆 array，每格 ~300 s CPU）
- [x] Experiment 1b adversarial case E（`toy/adversarial.py`）
- [x] ODRL 侧：`odrl/`（数据、probe-state 真值、Exp 2/3 评估），hopper 数据已在 `dataset/mujoco/`
- [x] Gate R（`odrl/resolution.py`，K / h / feature 扫描）、Gate 2 + Gate S（`odrl/detectability.py`）、1b 复现 seed 3–5
- [x] halfcheetah Gate R / S / P（h = 1）；perfect-support 目标（`paired_<g>`）；friction 轴 Gate R + paired；Δs 聚类；N_t 扫描；归一化估计器 + 确认跑；20000 步消融
- [x] Gate O（oracle 下游 IQL，hopper friction 2.0）：未达判据；oracle_state 的 terminal 全删是 D4RL `qlearning_dataset` 语义的人为效应（第四份评注审计），不是物理 gap 的性质
- [x] 第四份评注：A 审计 / C 随机加回 / 修正 arm / 参考跑已读并写回；正对照失败 → 第三阶段不在 hopper friction 2.0 上做
- [x] 干净子集复核（job 202843）：两项预测成立
- [x] Gate O2 六个基线 run：决策表第一行（target_only 1915 ± 130 vs concat 503 ± 65，concat = 文献 IQL）
- [ ] 训练强度对照 target_only 250k 步；核对 LoDADA / ODRL 原文协议（步数、batch、是否报 target-only）；用 D4RL 原始 `next_observations` 键或重模拟替代 `qlearning_dataset` 的 `observations[i+1]`；把速度裁剪行从真值探针里剔除后重算 Gate R / C 的 rho
- [ ] 多步结果（h = 5–20）的 Gate P / C；早停 / 正则下的分类器预算；写作
- [ ] `odrl/evaluate_ratio.py` 的 kmeans 用目标 s' 按维标准化，`local_occ_iql.py` 还是原始尺度，下游阶段要统一
- [ ] `local_occ_iql.py` 目前是 quantile hard filter，downstream 阶段再换 `exp_weights`
- [x] 环境：drcc `~/envs/odrl`（python 3.8 + mujoco-py + d4rl + torch 2.4.1 cu121），一律 sbatch 到 defq，登录节点不要跑 torch
