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
  evaluate_ratio.py            # Experiment 2 + 3：null / behavior / g{0.5,2.0}_{expert,medium,random}
algo/offline_offline/local_occ_iql.py   # IQL + 一次性 source 过滤（Experiment 4）
config/mujoco/local_occ_iql/   # 继承 iql 配置 + num_clusters / classifier_steps / keep_fraction
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

## 待做

- [x] Experiment 1 多 seed（`toy/run_toy_seeds.sbatch`，按 seed 拆 array，每格 ~300 s CPU）
- [x] Experiment 1b adversarial case E（`toy/adversarial.py`）
- [x] ODRL 侧：`odrl/`（数据、probe-state 真值、Exp 2/3 评估），hopper 数据已在 `dataset/mujoco/`
- [x] Gate R（`odrl/resolution.py`，K / h / feature 扫描）、Gate 2 + Gate S（`odrl/detectability.py`）、1b 复现 seed 3–5
- [ ] halfcheetah 上的 Gate R + Gate 2（h ∈ {1, 5, 20}，random 目标窗口充足）；N_t 扫描；残差归一化修 on-support 假阳性；friction 轴；classifier_steps 20000
- [ ] `odrl/evaluate_ratio.py` 的 kmeans 用目标 s' 按维标准化，`local_occ_iql.py` 还是原始尺度，下游阶段要统一
- [ ] `local_occ_iql.py` 目前是 quantile hard filter，downstream 阶段再换 `exp_weights`
- [x] 环境：drcc `~/envs/odrl`（python 3.8 + mujoco-py + d4rl + torch 2.4.1 cu121），一律 sbatch 到 defq，登录节点不要跑 torch
