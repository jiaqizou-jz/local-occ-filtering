# Experiment 1: toy 四格矩阵，5 seeds

日期 2026-10-15。设置：seed 0–4，steps 3000，K = 10 quantile bins，source 50k / target 5k，评估用 held-out source rollouts（seed+100，50k）。
命令：`python -m toy.evaluate_ratio --seeds 5 --out results/exp1_toy_matrix_seeds5.csv`（本次为 5 个 `--seeds 1 --first_seed k` 并行进程，结果等价）。
逐 seed 原始数据见 `exp1_toy_matrix_seeds5.csv`。

## mean ± std（5 seeds）

| case | true_kl | abs_rev | abs_dara | abs_local | rho_rev | rho_local | rho_dara |
|---|---|---|---|---|---|---|---|
| A | 0.000 ± 0.000 | 0.189 ± 0.008 | 0.052 ± 0.015 | 0.048 ± 0.018 | nan | nan | nan |
| B | 0.000 ± 0.000 | 0.605 ± 0.036 | 0.123 ± 0.033 | 0.062 ± 0.007 | nan | nan | nan |
| C | 1.413 ± 0.018 | 1.556 ± 0.085 | 1.455 ± 0.056 | 1.231 ± 0.067 | 0.938 ± 0.008 | 0.966 ± 0.003 | 0.985 ± 0.004 |
| D | 1.413 ± 0.018 | 1.765 ± 0.060 | 1.391 ± 0.049 | 1.147 ± 0.056 | 0.894 ± 0.005 | 0.965 ± 0.005 | 0.983 ± 0.006 |

## 逐 seed

| seed | case | true_kl | abs_rev | abs_dara | abs_local | rho_rev | rho_local | rho_dara |
|---|---|---|---|---|---|---|---|---|
| 0 | A | 0.000 | 0.203 | 0.071 | 0.020 | nan | nan | nan |
| 0 | B | 0.000 | 0.579 | 0.154 | 0.050 | nan | nan | nan |
| 0 | C | 1.410 | 1.561 | 1.414 | 1.194 | 0.946 | 0.966 | 0.980 |
| 0 | D | 1.410 | 1.836 | 1.352 | 1.118 | 0.898 | 0.958 | 0.986 |
| 1 | A | 0.000 | 0.183 | 0.034 | 0.056 | nan | nan | nan |
| 1 | B | 0.000 | 0.620 | 0.079 | 0.063 | nan | nan | nan |
| 1 | C | 1.441 | 1.702 | 1.545 | 1.320 | 0.949 | 0.966 | 0.989 |
| 1 | D | 1.441 | 1.740 | 1.418 | 1.210 | 0.900 | 0.967 | 0.988 |
| 2 | A | 0.000 | 0.181 | 0.048 | 0.040 | nan | nan | nan |
| 2 | B | 0.000 | 0.598 | 0.086 | 0.064 | nan | nan | nan |
| 2 | C | 1.402 | 1.575 | 1.447 | 1.295 | 0.931 | 0.970 | 0.989 |
| 2 | D | 1.402 | 1.775 | 1.373 | 1.209 | 0.896 | 0.964 | 0.988 |
| 3 | A | 0.000 | 0.185 | 0.038 | 0.047 | nan | nan | nan |
| 3 | B | 0.000 | 0.562 | 0.151 | 0.060 | nan | nan | nan |
| 3 | C | 1.387 | 1.462 | 1.483 | 1.203 | 0.931 | 0.966 | 0.980 |
| 3 | D | 1.387 | 1.812 | 1.474 | 1.137 | 0.888 | 0.964 | 0.975 |
| 4 | A | 0.000 | 0.192 | 0.067 | 0.076 | nan | nan | nan |
| 4 | B | 0.000 | 0.666 | 0.146 | 0.071 | nan | nan | nan |
| 4 | C | 1.424 | 1.482 | 1.383 | 1.141 | 0.931 | 0.961 | 0.985 |
| 4 | D | 1.424 | 1.664 | 1.340 | 1.064 | 0.887 | 0.972 | 0.977 |

## go/no-go 逐 seed 检查

```
seed 0: B abs_rev/abs_local = 11.5x, B abs_local vs A abs_local = 0.050 vs 0.020, D rho_local-rho_rev = +0.059
seed 1: B abs_rev/abs_local = 9.8x, B abs_local vs A abs_local = 0.063 vs 0.056, D rho_local-rho_rev = +0.067
seed 2: B abs_rev/abs_local = 9.4x, B abs_local vs A abs_local = 0.064 vs 0.040, D rho_local-rho_rev = +0.068
seed 3: B abs_rev/abs_local = 9.4x, B abs_local vs A abs_local = 0.060 vs 0.047, D rho_local-rho_rev = +0.076
seed 4: B abs_rev/abs_local = 9.4x, B abs_local vs A abs_local = 0.071 vs 0.076, D rho_local-rho_rev = +0.084

```

## 结论

- **第一关 Case B（只有 occupancy shift）通过**：reverse 的 `abs_rev` = 0.605 ± 0.036，corrected 的 `abs_local` = 0.062 ± 0.007，每个 seed 都相差约 10 倍；`abs_local` 与 Case A 的 0.048 ± 0.018 同量级。LoDADA reverse score 稳定地把 occupancy shift 报成 dynamics shift，corrected score 没有。
- **第二关 Case D（dynamics + occupancy 同时 shift）通过**：`rho_local` = 0.965 ± 0.005 > `rho_rev` = 0.894 ± 0.005，5 个 seed 全部成立，差值 +0.06 到 +0.08，两组分布无重叠。
- Case C（只有 dynamics shift）corrected 也略好于 reverse（0.966 vs 0.938），说明 reverse 的问题不只在 occupancy shift 时出现。
- DARA global 在 Case B 的 `abs_dara` = 0.123，比 corrected 大 2 倍，但比 reverse 小 5 倍；它的 `rho_dara` 是对 $s'$-level 真值算的，与 cluster-level 的 `rho_local` 不是同一个真值，不能直接比大小。

## 未覆盖

- 仍只有 1 维 linear-gaussian、1 组 (A, B, K) 参数、K = 10 一个设置；没有扫 K、N_target、steps。
- 没有 adversarial false negative（Experiment 1b）。
