from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hierarchicalcausalmodels.estimation import estimate_ate_confounder_torch_batched


def make_data(
    rng: np.random.Generator,
    n_units: int,
    n_subunits: int,
) -> tuple[np.ndarray, np.ndarray]:
    unit_effect = rng.normal(0.0, 1.0, size=(n_units, 1)).astype(np.float32)
    treatment_prob = 1.0 / (1.0 + np.exp(-unit_effect))
    treatment = rng.binomial(1, treatment_prob, size=(n_units, n_subunits)).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=(n_units, n_subunits)).astype(np.float32)
    outcome = (1.5 * treatment + 0.5 * unit_effect + noise).astype(np.float32)
    return treatment, outcome


def run_gpu_batch(
    treatment: np.ndarray,
    outcome: np.ndarray,
    device: str,
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    ate = estimate_ate_confounder_torch_batched(
        treatment,
        outcome,
        family="gaussian",
        device=device,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "gpu_seconds": elapsed,
        "gpu_ate": float(ate),
        "gpu_peak_memory_gb": peak_memory,
        "tasks_per_second": float(treatment.shape[0] / elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local CUDA batch ceiling for a batched HCM kernel.")
    parser.add_argument("--n-subunits", type=int, default=80)
    parser.add_argument("--sizes", type=int, nargs="+", default=[800_000, 1_200_000, 1_600_000, 2_000_000, 2_400_000, 2_800_000, 3_000_000])
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out", type=Path, default=Path("refs/paperjrss/gpu_hcm_batched_benchmark.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for this benchmark.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    best_gpu_batch: dict[str, float | int | str] | None = None

    for n_units in args.sizes:
        treatment, outcome = make_data(rng, int(n_units), int(args.n_subunits))
        row: dict[str, float | int | str] = {
            "n_units": int(n_units),
            "n_subunits": int(args.n_subunits),
            "tasks": int(n_units),
        }
        try:
            row.update(run_gpu_batch(treatment, outcome, "cuda:0"))
            best_gpu_batch = dict(row)
        except RuntimeError as exc:
            row["gpu_error"] = str(exc).split("\n", maxsplit=1)[0]
            rows.append(row)
            break
        rows.append(row)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "n_subunits": int(args.n_subunits),
        "rows": rows,
        "best_gpu_batch": best_gpu_batch,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
