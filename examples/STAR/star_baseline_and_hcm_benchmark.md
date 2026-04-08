# STAR baseline econometric benchmark and HCM family benchmark

## Econometric baseline

| Model | Coef small | SE small | p small | Coef regular+aide | SE regular+aide | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ols_minimal` | 5.7362 | 1.0478 | 4.388e-08 | 0.6503 | 0.9918 | 0.0062 | 0.004 |
| `ols_school_fe` | 6.5843 | 0.9712 | 1.206e-11 | 1.0936 | 0.8967 | 0.2109 | 0.400 |
| `ols_with_student_controls` | 6.5084 | 0.9353 | 3.441e-12 | 1.2497 | 0.8645 | 0.2632 | 0.786 |
| `ols_full_like_startenesse` | 6.5521 | 0.9362 | 2.588e-12 | 0.9643 | 0.8756 | 0.2655 | 0.886 |

| IV model | Coef class size | SE | p-value | Time (s) |
|---|---:|---:|---:|---:|
| `2SLS small/reg_aide -> class size` | -0.8182 | 0.1089 | 6.752e-14 | 1.179 |

## HCM family benchmark

### all_gaussian_except_A

- note: Near the original first-pass convention.
- total time: `0.507` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 0.100 |
| `FCI` | `ok` | 0.0000 | 0.093 |
| `DirectLiNGAM` | `error` |  | 0.189 |
| `ExactBIC` | `error` |  | 0.045 |
| `ConsensusMean` | `ok` | 0.0000 | 0.080 |

### mixed_binary_gaussian

- note: Binary student covariates use Bernoulli/logistic; continuous scores stay Gaussian.
- total time: `0.820` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 0.071 |
| `FCI` | `ok` | 0.0000 | 0.070 |
| `DirectLiNGAM` | `error` |  | 0.481 |
| `ExactBIC` | `error` |  | 0.059 |
| `ConsensusMean` | `ok` | 0.0000 | 0.139 |

### mixed_with_poisson_count_proxy

- note: Stress-test with Poisson on the discrete urbanicity code; mainly diagnostic.
- total time: `1.065` s

| Graph | Status | ATE | Time (s) |
|---|---|---:|---:|
| `PC` | `ok` | 0.0000 | 0.073 |
| `FCI` | `ok` | 0.0000 | 0.073 |
| `DirectLiNGAM` | `error` |  | 0.780 |
| `ExactBIC` | `error` |  | 0.055 |
| `ConsensusMean` | `ok` | 0.0000 | 0.083 |

## Best variant

- selected variant: `all_gaussian_except_A`

| Variant | OK graphs | Error graphs | Median abs deviation from 5.34 | Mean abs ATE | Total time (s) |
|---|---:|---:|---:|---:|---:|
| `all_gaussian_except_A` | 3 | 2 | 5.3400 | 0.0000 | 0.507 |
| `mixed_binary_gaussian` | 3 | 2 | 5.3400 | 0.0000 | 0.820 |
| `mixed_with_poisson_count_proxy` | 3 | 2 | 5.3400 | 0.0000 | 1.065 |
