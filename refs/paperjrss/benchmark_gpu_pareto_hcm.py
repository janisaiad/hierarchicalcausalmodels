from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hierarchicalcausalmodels.estimation import estimate_ate_confounder_torch_batched


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


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


def synchronize_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def reset_cuda_peak_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return float(torch.cuda.max_memory_allocated() / 1024**3)


def benchmark_once(
    treatment: np.ndarray,
    outcome: np.ndarray,
    device: str,
) -> dict[str, float | str | None]:
    reset_cuda_peak_if_needed(device)
    started_at = time.perf_counter()
    ate = estimate_ate_confounder_torch_batched(
        treatment,
        outcome,
        family="gaussian",
        device=device,
    )
    synchronize_if_cuda(device)
    elapsed = time.perf_counter() - started_at
    return {
        "seconds": float(elapsed),
        "ate": float(ate),
        "peak_memory_gb": peak_memory_gb(device),
    }


def median_benchmark(
    treatment: np.ndarray,
    outcome: np.ndarray,
    device: str,
    repeats: int,
) -> dict[str, float | str | None]:
    rows = [benchmark_once(treatment, outcome, device) for _ in range(max(1, repeats))]
    seconds = np.array([float(row["seconds"]) for row in rows], dtype=float)
    best_idx = int(np.argsort(seconds)[len(seconds) // 2])
    return rows[best_idx]


def is_pareto_efficient(memory: np.ndarray, throughput: np.ndarray) -> np.ndarray:
    efficient = np.ones(memory.shape[0], dtype=bool)
    for idx in range(memory.shape[0]):
        if not efficient[idx]:
            continue
        dominated = (memory <= memory[idx]) & (throughput >= throughput[idx])
        strictly_better = (memory < memory[idx]) | (throughput > throughput[idx])
        if np.any(dominated & strictly_better):
            efficient[idx] = False
    return efficient


def add_pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pareto"] = False
    gpu_success = result[
        (result["backend"] == "cuda")
        & result["seconds"].notna()
        & result["peak_memory_gb"].notna()
    ].copy()
    if gpu_success.empty:
        return result
    mask = is_pareto_efficient(
        gpu_success["peak_memory_gb"].to_numpy(dtype=float),
        gpu_success["tasks_per_second"].to_numpy(dtype=float),
    )
    result.loc[gpu_success.index[mask], "pareto"] = True
    return result


def plot_pareto(frame: pd.DataFrame, out_prefix: Path) -> None:
    style()
    gpu = frame[(frame["backend"] == "cuda") & frame["seconds"].notna()].copy()
    if gpu.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for n_subunits, group in gpu.groupby("n_subunits"):
        axes[0].scatter(
            group["peak_memory_gb"],
            group["tasks_per_second"] / 1_000_000.0,
            s=34,
            label=f"$m={int(n_subunits)}$",
            alpha=0.82,
        )
    pareto = gpu[gpu["pareto"]].sort_values("peak_memory_gb")
    if not pareto.empty:
        axes[0].plot(
            pareto["peak_memory_gb"],
            pareto["tasks_per_second"] / 1_000_000.0,
            color="#222222",
            linewidth=1.4,
            label="Pareto frontier",
        )
        for _, row in pareto.iterrows():
            label = f"{int(row['n_units'])/1_000_000:.1f}M x {int(row['n_subunits'])}"
            axes[0].annotate(
                label,
                (row["peak_memory_gb"], row["tasks_per_second"] / 1_000_000.0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("Peak CUDA memory (GB)")
    axes[0].set_ylabel("Unit tasks per second (millions)")
    axes[0].set_title("GPU throughput--memory frontier")
    axes[0].legend(frameon=True)

    speed = gpu[gpu["speedup_vs_cpu"].notna()].copy()
    if speed.empty:
        axes[1].text(0.5, 0.5, "CPU baseline skipped for large batches", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        for n_subunits, group in speed.groupby("n_subunits"):
            axes[1].plot(
                group["total_observations"],
                group["speedup_vs_cpu"],
                marker="o",
                linewidth=1.4,
                label=f"$m={int(n_subunits)}$",
            )
        axes[1].axhline(
            1.0,
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            label="GPU = CPU",
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"Total observations $n\times m$")
        axes[1].set_ylabel("Speedup vs CPU batched torch")
        axes[1].set_title("Observed GPU speedup regime")
        axes[1].legend(frameon=True)

    for suffix in (".pdf", ".png"):
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def benchmark_grid(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(int(args.seed))
    devices = ["cuda:0"] if torch.cuda.is_available() else []
    if bool(args.include_cpu):
        devices.insert(0, "cpu")
    if not devices:
        raise RuntimeError("No benchmark device available.")

    rows: list[dict[str, float | int | str | bool | None]] = []
    cpu_times: dict[tuple[int, int], float] = {}

    for n_subunits in args.n_subunits:
        for n_units in args.n_units:
            treatment, outcome = make_data(rng, int(n_units), int(n_subunits))
            total_observations = int(n_units) * int(n_subunits)
            for device in devices:
                backend = "cuda" if device.startswith("cuda") else "cpu"
                if backend == "cpu" and int(n_units) > int(args.cpu_max_units):
                    continue
                row: dict[str, float | int | str | bool | None] = {
                    "backend": backend,
                    "device": torch.cuda.get_device_name(0) if backend == "cuda" else "cpu",
                    "n_units": int(n_units),
                    "n_subunits": int(n_subunits),
                    "total_observations": total_observations,
                    "tasks": int(n_units),
                    "success": False,
                    "seconds": None,
                    "ate": None,
                    "peak_memory_gb": None,
                    "tasks_per_second": None,
                    "speedup_vs_cpu": None,
                    "error": None,
                }
                try:
                    result = median_benchmark(
                        treatment,
                        outcome,
                        device,
                        repeats=int(args.repeats),
                    )
                    seconds = float(result["seconds"])
                    row.update(
                        {
                            "success": True,
                            "seconds": seconds,
                            "ate": result["ate"],
                            "peak_memory_gb": result["peak_memory_gb"],
                            "tasks_per_second": float(int(n_units) / seconds),
                        }
                    )
                    if backend == "cpu":
                        cpu_times[(int(n_units), int(n_subunits))] = seconds
                    if backend == "cuda" and (int(n_units), int(n_subunits)) in cpu_times:
                        row["speedup_vs_cpu"] = float(cpu_times[(int(n_units), int(n_subunits))] / seconds)
                except RuntimeError as exc:
                    row["error"] = str(exc).split("\n", maxsplit=1)[0]
                rows.append(row)
                if backend == "cuda" and not bool(row["success"]) and bool(args.stop_on_oom):
                    break
            del treatment, outcome
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return add_pareto_flags(pd.DataFrame(rows))


def write_outputs(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_prefix = Path(args.out_figure)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_units_grid": [int(value) for value in args.n_units],
        "n_subunits_grid": [int(value) for value in args.n_subunits],
        "cpu_max_units": int(args.cpu_max_units),
        "repeats": int(args.repeats),
        "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    frame.to_csv(out_csv, index=False)
    plot_pareto(frame, out_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a GPU/CPU Pareto frontier for batched HCM unit tasks.")
    parser.add_argument("--n-units", type=int, nargs="+", default=[1_000, 10_000, 50_000, 100_000, 400_000, 800_000, 1_200_000])
    parser.add_argument("--n-subunits", type=int, nargs="+", default=[20, 80, 160])
    parser.add_argument("--cpu-max-units", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--include-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-oom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-json", type=Path, default=Path("refs/paperjrss/gpu_hcm_pareto_benchmark.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("refs/paperjrss/gpu_hcm_pareto_benchmark.csv"))
    parser.add_argument("--out-figure", type=Path, default=Path("refs/paperjrss/Fig/gpu_hcm_pareto_frontier"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = benchmark_grid(args)
    write_outputs(frame, args)
    successful = frame[(frame["backend"] == "cuda") & (frame["success"])]
    if not successful.empty:
        best = successful.sort_values("tasks_per_second", ascending=False).iloc[0]
        print(
            "Best CUDA throughput: "
            f"{best['tasks_per_second'] / 1_000_000:.2f}M tasks/s at "
            f"n={int(best['n_units'])}, m={int(best['n_subunits'])}."
        )
    print(f"Wrote {args.out_json}, {args.out_csv}, and {args.out_figure}.pdf/.png")


if __name__ == "__main__":
    main()
