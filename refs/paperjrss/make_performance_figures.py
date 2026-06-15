from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "refs" / "paperjrss" / "Fig"
STAR_RESULTS = ROOT / "examples" / "STAR" / "results"


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(FIG / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def hierarchical_confounded_sample(
    rng: np.random.Generator,
    n_units: int,
    n_sub: int,
    *,
    tau: float,
    confounding: float,
) -> tuple[np.ndarray, np.ndarray]:
    unit_effect = rng.normal(0.0, 1.0, size=n_units)
    treatment_prob = 1.0 / (1.0 + np.exp(-confounding * unit_effect))
    treatment = rng.binomial(1, treatment_prob[:, None], size=(n_units, n_sub)).astype(float)
    noise = rng.normal(0.0, 1.0, size=(n_units, n_sub))
    outcome = tau * treatment + 1.5 * unit_effect[:, None] + noise
    return treatment, outcome


def flat_ols_slope(treatment: np.ndarray, outcome: np.ndarray) -> float:
    a = treatment.reshape(-1)
    y = outcome.reshape(-1)
    centered_a = a - a.mean()
    denom = float(np.dot(centered_a, centered_a))
    return float(np.dot(centered_a, y - y.mean()) / denom)


def within_unit_slope(treatment: np.ndarray, outcome: np.ndarray) -> float:
    slopes: list[float] = []
    for a, y in zip(treatment, outcome, strict=True):
        centered_a = a - a.mean()
        denom = float(np.dot(centered_a, centered_a))
        if denom > 1e-12:
            slopes.append(float(np.dot(centered_a, y - y.mean()) / denom))
    return float(np.mean(slopes))


def convergence_points() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260615)
    tau = 1.5
    n_units_grid = np.array([25, 50, 100, 200, 400, 800])
    n_sub = 40
    n_rep = 120
    rmse_flat: list[float] = []
    rmse_hcm: list[float] = []
    runtimes: list[float] = []
    for n_units in n_units_grid:
        start = time.perf_counter()
        flat_err: list[float] = []
        hcm_err: list[float] = []
        for _ in range(n_rep):
            treatment, outcome = hierarchical_confounded_sample(
                rng,
                int(n_units),
                n_sub,
                tau=tau,
                confounding=1.4,
            )
            flat_err.append(flat_ols_slope(treatment, outcome) - tau)
            hcm_err.append(within_unit_slope(treatment, outcome) - tau)
        rmse_flat.append(float(np.sqrt(np.mean(np.square(flat_err)))))
        rmse_hcm.append(float(np.sqrt(np.mean(np.square(hcm_err)))))
        runtimes.append(float(time.perf_counter() - start))
    total_sample = n_units_grid * n_sub
    return {
        "total_sample": total_sample.astype(float),
        "flat_rmse": np.asarray(rmse_flat),
        "hcm_rmse": np.asarray(rmse_hcm),
        "reference": 1.0 / np.sqrt(total_sample / total_sample[0]) * rmse_hcm[0],
        "runtime_seconds": np.asarray(runtimes),
    }


def plot_convergence(data: dict[str, np.ndarray] | None = None) -> None:
    if data is None:
        data = convergence_points()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        data["total_sample"],
        data["flat_rmse"],
        marker="o",
        color="#D64B4B",
        linewidth=2.0,
        label="Flat pooled estimator",
    )
    ax.plot(
        data["total_sample"],
        data["hcm_rmse"],
        marker="o",
        color="#4C78A8",
        linewidth=2.0,
        label="Hierarchical within-unit estimator",
    )
    ax.plot(
        data["total_sample"],
        data["reference"],
        linestyle=":",
        color="#888888",
        linewidth=1.5,
        label=r"Reference $1/\sqrt{nm}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Total sample size $n\times m$")
    ax.set_ylabel("RMSE")
    ax.set_title("Convergence under hierarchical confounding")
    for x_value, y_value, elapsed in zip(
        data["total_sample"],
        data["hcm_rmse"],
        data["runtime_seconds"],
        strict=True,
    ):
        ax.annotate(
            f"{elapsed:.2f}s",
            xy=(x_value, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#2F5E8E",
        )
    ax.legend(frameon=True, loc="lower left")
    save_figure(fig, "synthetic_hcm_convergence_rmse")


def plot_runtime_comparison() -> None:
    speed = json.loads((STAR_RESULTS / "ate_10_40_parallel_speed_test.json").read_text())
    bench = json.loads((STAR_RESULTS / "parallel_benchmark_results.json").read_text())
    modes = ["seq", "threads4", "proc4"]
    labels = ["Sequential", "4 threads", "4 processes"]
    exactbic = [
        speed["graphs"]["ExactBIC"]["modes"][mode]["elapsed_seconds_total_do1_do0"]
        for mode in modes
    ]
    ols = [bench[mode]["ols_seconds"] for mode in modes]
    iv = [bench[mode]["iv_seconds"] for mode in modes]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    x = np.arange(len(modes))
    axes[0].bar(x, exactbic, color=["#9E9E9E", "#F58518", "#4C78A8"], edgecolor="white")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("ExactBIC HCM evaluation")
    axes[0].text(
        2,
        exactbic[2] + 0.08,
        f"{exactbic[0] / exactbic[2]:.2f}x faster",
        ha="center",
        fontsize=8,
    )

    width = 0.36
    axes[1].bar(x - width / 2, ols, width=width, color="#72B7B2", label="OLS batch", edgecolor="white")
    axes[1].bar(x + width / 2, iv, width=width, color="#B279A2", label="IV batch", edgecolor="white")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Baseline batch workloads")
    axes[1].legend(frameon=True, loc="upper right")
    save_figure(fig, "parallel_runtime_comparison")


def main() -> None:
    style()
    plot_convergence()
    print(f"Wrote performance figure to {FIG}")


if __name__ == "__main__":
    main()
