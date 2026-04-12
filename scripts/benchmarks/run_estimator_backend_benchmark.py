#!/usr/bin/env python3
"""Compare runtime: Torch CPU vs CUDA, parallel vs sequential sklearn, Numba on vs off.

Run from repo root::

    uv run python scripts/benchmarks/run_estimator_backend_benchmark.py

Uses fresh subprocesses for Numba A/B (env is read at import of ``numba_kernels``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from sklearn.linear_model import LinearRegression  # noqa: E402

from hierarchicalcausalmodels.estimation import (  # noqa: E402
    NUMBA_AVAILABLE,
    estimate_ate_confounder_torch_batched,
    fit_regressors_per_unit,
)


def _median_seconds(fn: Callable[[], object], *, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def _bench_sklearn_parallel(
    n_units: int,
    n_obs: int,
    *,
    repeats: int = 5,
) -> tuple[float, float, int, int]:
    rng = np.random.default_rng(2026)
    a = rng.binomial(1, 0.5, size=(n_units, n_obs)).astype(np.float64)
    y = rng.normal(0.0, 1.0, size=(n_units, n_obs)) + 0.4 * a

    def seq():
        fit_regressors_per_unit(
            a,
            y,
            LinearRegression,
            n_jobs=1,
            parallel_backend="threads",
        )

    n_workers = min(8, max(2, os.cpu_count() or 4))

    def par():
        fit_regressors_per_unit(
            a,
            y,
            LinearRegression,
            n_jobs=n_workers,
            parallel_backend="threads",
        )

    t_seq = _median_seconds(seq, repeats=repeats)
    t_par = _median_seconds(par, repeats=repeats)
    return t_seq, t_par, n_workers, n_units * n_obs


def _bench_torch_cpu_gpu(
    n_units: int,
    n_obs: int,
    *,
    repeats: int = 5,
) -> tuple[float | None, float | None]:
    try:
        import torch
    except ImportError:
        return None, None
    rng = np.random.default_rng(42)
    a = rng.binomial(1, 0.5, size=(n_units, n_obs)).astype(np.float32)
    y = (1.2 + 1.8 * a + rng.normal(0.0, 0.15, size=(n_units, n_obs))).astype(np.float32)

    def run_cpu():
        estimate_ate_confounder_torch_batched(
            a,
            y,
            family="gaussian",
            device="cpu",
            ridge=1e-4,
            max_iter=60,
        )

    t_cpu = _median_seconds(run_cpu, repeats=repeats)
    if not torch.cuda.is_available():
        return t_cpu, None

    def run_gpu():
        estimate_ate_confounder_torch_batched(
            a,
            y,
            family="gaussian",
            device="cuda:0",
            ridge=1e-4,
            max_iter=60,
        )

    t_gpu = _median_seconds(run_gpu, repeats=repeats)
    return t_cpu, t_gpu


def _bench_numba_subprocess(*, repeats: int = 1) -> tuple[float | None, float | None, bool, bool]:
    """Return (time_disable, time_enable, numba_reported_off, numba_reported_on)."""
    n_rep = max(3, repeats)
    code_template = """
import sys, time, statistics
sys.path.insert(0, {src!r})
import numpy as np
from hierarchicalcausalmodels.estimation import numba_kernels as nk
z = np.random.default_rng(0).standard_normal(5_000_000).astype("float64")
for _ in range(12):
    nk.expit_array(z)
times = []
for _ in range({n_rep}):
    t0 = time.perf_counter()
    nk.expit_array(z)
    times.append(time.perf_counter() - t0)
print(statistics.median(times))
print(int(nk.NUMBA_AVAILABLE))
""".format(
        src=SRC,
        n_rep=n_rep,
    )

    def run_child(disable: bool) -> tuple[float, bool]:
        env = os.environ.copy()
        if disable:
            env["HCM_DISABLE_NUMBA"] = "1"
        else:
            env.pop("HCM_DISABLE_NUMBA", None)
        cp = subprocess.run(
            [sys.executable, "-c", code_template],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            check=False,
        )
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr or cp.stdout or "child failed")
        lines = (cp.stdout or "").strip().splitlines()
        t = float(lines[0])
        nb = bool(int(lines[1]))
        return t, nb

    try:
        t_off, nb_off = run_child(True)
        t_on, nb_on = run_child(False)
    except Exception:
        return None, None, False, False
    return t_off, t_on, nb_off, nb_on


def main() -> None:
    print("=== HCM estimator backend benchmark ===\n")

    print("(1) Sklearn LinearRegression per-unit: n_jobs=1 vs parallel threads")
    t_seq, t_par, nw, nfit = _bench_sklearn_parallel(2000, 64, repeats=5)
    print(f"    n_units=2000 n_obs=64  n_jobs_parallel={nw}  (~{nfit} subfits)")
    print(f"    sequential: {t_seq:.4f}s   parallel: {t_par:.4f}s   speedup: {t_seq / t_par:.2f}x")
    if t_seq < t_par:
        print("    (parallel slower → joblib overhead dominates for very cheap fits per unit)\n")
    else:
        print()

    print("(2) Torch batched ATE (gaussian): device=cpu vs cuda:0")
    tc, tg = _bench_torch_cpu_gpu(2048, 96, repeats=5)
    print(f"    n_units=2048 n_obs=96")
    print(f"    cpu: {tc:.4f}s")
    if tg is None:
        print("    cuda: (skipped — CUDA not available)\n")
    else:
        sp = tc / tg
        print(f"    cuda:0: {tg:.4f}s   speedup vs cpu: {sp:.2f}x")
        if sp < 1.0:
            print("    (GPU slower → batch still small vs kernel/transfer latency)\n")
        else:
            print()

    print("(3) numba_kernels.expit_array (~5e6 elements, median of 3 runs in subprocess)")
    print(f"    NUMBA_AVAILABLE in this process: {NUMBA_AVAILABLE}")
    t_off, t_on, nb_off, nb_on = _bench_numba_subprocess(repeats=3)
    if t_off is None:
        print("    subprocess benchmark failed\n")
    else:
        print(f"    HCM_DISABLE_NUMBA=1 → time={t_off:.4f}s  NUMBA_AVAILABLE={nb_off}")
        print(f"    numba allowed     → time={t_on:.4f}s  NUMBA_AVAILABLE={nb_on}")
        if t_on > 0:
            r = t_off / t_on
        print(f"    ratio scipy/numba time: {r:.2f}x  (>1 means numba faster)")
        if r < 1.0:
            print("    (SciPy expit vectorized can beat numba here; numba helps inside MC hot paths)\n")
        else:
            print()

    print("Done.")


if __name__ == "__main__":
    main()
