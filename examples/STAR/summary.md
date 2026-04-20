# STAR Summary (Math Outcome)

## Outcome convention

Cette version est strictement alignée sur l'outcome maths:

- `M = gktmathss`

## Baseline econometric results (math)

Source:

- `examples/STAR/results/star_baseline_math_benchmark.json`

Main estimates:

- OLS `small`: `8.0962` to `9.4988` across specs.
- IV/2SLS `class size`: `-1.2186` (math score points per additional student).

## HCM notes

- HCM runs should be compared only with math-focused baselines.
- Historical runs with missing factors replaced by `1` are diagnostic and may under-represent full causal structure.

## Parallel runtime notes

Source:

- `examples/STAR/results/ate_10_40_parallel_speed_test.json`

Takeaway:

- process-based parallelization (`proc4`) is the most effective on tested workloads.
