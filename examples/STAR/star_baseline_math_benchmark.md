# STAR baseline econometric benchmark and HCM family benchmark

- outcome baseline économétrique: `gktmathss`

## Econometric baseline

| Model | Coef small | SE small | p small | Coef regular+aide | SE regular+aide | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ols_minimal` | 8.0962 | 1.5974 | 4.014e-07 | -0.3865 | 1.4817 | 0.0064 | 0.017 |
| `ols_school_fe` | 9.4988 | 1.4633 | 8.516e-11 | 0.8076 | 1.3328 | 0.2193 | 0.655 |
| `ols_with_student_controls` | 9.3797 | 1.4149 | 3.38e-11 | 1.0486 | 1.2918 | 0.2676 | 1.014 |
| `ols_full_like_startenesse` | 9.4316 | 1.4192 | 3.014e-11 | 0.8313 | 1.3097 | 0.2695 | 1.142 |

| IV model | Coef class size | SE | p-value | Time (s) |
|---|---:|---:|---:|---:|
| `2SLS small/reg_aide -> class size` | -1.2186 | 0.1632 | 9.457e-14 | 1.507 |

## HCM family benchmark

### all_gaussian_except_A

- note: Near the original first-pass convention.
- total time: `31.818` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 0.144 |
| `FCI` | `ok` | 0.0000 | 0.106 |
| `DirectLiNGAM` | `ok` | 27.6027 | 21.254 |
| `ExactBIC` | `ok` | -47.0433 | 10.169 |
| `ConsensusMean` | `ok` | 0.0000 | 0.145 |

### mixed_binary_gmm2_ym

- note: Aligné sur star_hcm v2 : Y,M = mélange gaussien K=2 par école.
- total time: `96.990` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 9.551 |
| `FCI` | `ok` | 0.0000 | 8.708 |
| `DirectLiNGAM` | `ok` | 4308.5293 | 46.086 |
| `ExactBIC` | `ok` | -18.4645 | 25.212 |
| `ConsensusMean` | `ok` | 0.0000 | 7.433 |

### mixed_with_poisson_count_proxy

- note: Stress-test Poisson sur S ; Y,M en GMM K=2.
- total time: `88.831` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 8.267 |
| `FCI` | `ok` | 0.0000 | 8.112 |
| `DirectLiNGAM` | `ok` | 3802.7458 | 41.315 |
| `ExactBIC` | `ok` | -17.0244 | 23.354 |
| `ConsensusMean` | `ok` | 0.0000 | 7.783 |

## Best variant

- selected variant: `all_gaussian_except_A`

| Variant | OK graphs | Error graphs | Median abs deviation from 5.34 | Mean abs ATE | Total time (s) |
|---|---:|---:|---:|---:|---:|
| `all_gaussian_except_A` | 5 | 0 | 5.3400 | 14.9292 | 31.818 |
| `mixed_with_poisson_count_proxy` | 5 | 0 | 5.3400 | 763.9540 | 88.831 |
| `mixed_binary_gmm2_ym` | 5 | 0 | 5.3400 | 865.3988 | 96.990 |
