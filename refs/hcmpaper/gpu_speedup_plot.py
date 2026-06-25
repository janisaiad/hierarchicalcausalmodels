import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


MOTIFS = ("Confounder", "Mediator", "Instrument")
DEFAULT_SIZES = "20000x32,40000x48,60000x64,90000x80,120000x96,180000x128,240000x160"


def parse_sizes(text):
    sizes = []
    for item in text.split(","):
        n_text, m_text = item.lower().split("x")
        sizes.append((int(n_text), int(m_text)))
    return sizes


def make_data(n, m, motif, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(17 + n + 13 * m + 101 * MOTIFS.index(motif))
    x = torch.randn((n, m), generator=generator, device=device)
    z = torch.randn((n, 1), generator=generator, device=device)
    u = torch.randn((n, 1), generator=generator, device=device)
    alpha = 0.30 + 0.10 * torch.randn((n, 1), generator=generator, device=device)
    beta = 0.20 + 0.05 * torch.randn((n, 1), generator=generator, device=device)
    gamma = 0.40 * torch.randn((n, 1), generator=generator, device=device)
    eps = 0.05 * torch.randn((n, m), generator=generator, device=device)
    a_obs = torch.bernoulli(torch.sigmoid(0.35 * z + 0.15 * x))
    return x, z, u, alpha, beta, gamma, eps, a_obs


def confounder_batched(data):
    x, _, u, alpha, beta, gamma, eps, _ = data
    mu1 = alpha + beta * x + 0.10 * x.square() + gamma + u + eps
    mu0 = beta * x + 0.10 * x.square() + gamma + u + eps
    return (mu1.mean(dim=1) - mu0.mean(dim=1)).mean()


def mediator_batched(data):
    x, z, u, alpha, beta, gamma, eps, _ = data
    med1 = torch.sigmoid(0.75 + 0.20 * z + 0.10 * x + 0.10 * u)
    med0 = torch.sigmoid(0.20 * z + 0.10 * x + 0.10 * u)
    y1 = alpha + 0.65 * med1 + beta * x + 0.10 * med1 * x + gamma + eps
    y0 = 0.65 * med0 + beta * x + 0.10 * med0 * x + gamma + eps
    return (y1.mean(dim=1) - y0.mean(dim=1)).mean()


def instrument_batched(data):
    x, z, u, alpha, beta, gamma, eps, a_obs = data
    prob = torch.sigmoid(0.15 + 0.80 * z + 0.10 * x + 0.10 * u).clamp(0.03, 0.97)
    y = alpha * a_obs + beta * x + 0.20 * z + gamma + eps
    q1 = torch.tensor(0.75, device=x.device)
    q0 = torch.tensor(0.25, device=x.device)
    w1 = q1 * a_obs / prob + (1 - q1) * (1 - a_obs) / (1 - prob)
    w0 = q0 * a_obs / prob + (1 - q0) * (1 - a_obs) / (1 - prob)
    return ((w1 - w0) * y).mean()


def sequential_eval(data, motif):
    x, z, u, alpha, beta, gamma, eps, a_obs = data
    total = torch.tensor(0.0)
    for i in range(x.shape[0]):
        unit = (
            x[i : i + 1],
            z[i : i + 1],
            u[i : i + 1],
            alpha[i : i + 1],
            beta[i : i + 1],
            gamma[i : i + 1],
            eps[i : i + 1],
            a_obs[i : i + 1],
        )
        total = total + EVALUATORS[motif](unit).cpu()
    return total / x.shape[0]


EVALUATORS = {
    "Confounder": confounder_batched,
    "Mediator": mediator_batched,
    "Instrument": instrument_batched,
}


def time_call(fn, repeats, device):
    times = []
    value = None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times), float(value)


def cuda_device_summary():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1024**3,
        "multiprocessors": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this Python process. Run from a shell where "
            "`python -c \"import torch; print(torch.cuda.is_available())\"` prints True."
        )

    sizes = parse_sizes(args.sizes)
    rows = []
    device = cuda_device_summary()
    print(
        f"CUDA device: {device['name']} "
        f"({device['multiprocessors']} SMs, "
        f"{device['total_memory_gb']:.1f} GB, CUDA {device['cuda_version']})"
    )
    for motif in MOTIFS:
        for n, m in sizes:
            cpu_data = make_data(n, m, motif, torch.device("cpu"))
            gpu_data = tuple(t.to("cuda") for t in cpu_data)
            EVALUATORS[motif](gpu_data)
            torch.cuda.synchronize()

            cpu_time, cpu_value = time_call(
                lambda d=cpu_data, name=motif: sequential_eval(d, name),
                args.cpu_repeats,
                torch.device("cpu"),
            )
            gpu_time, gpu_value = time_call(
                lambda d=gpu_data, name=motif: EVALUATORS[name](d),
                args.gpu_repeats,
                torch.device("cuda"),
            )
            rows.append(
                {
                    "motif": motif,
                    "n": n,
                    "m": m,
                    "observations": n * m,
                    "cpu_seconds": cpu_time,
                    "cuda_seconds": gpu_time,
                    "speedup": cpu_time / gpu_time,
                    "cpu_value": cpu_value,
                    "cuda_value": gpu_value,
                }
            )
            print(
                f"{motif:10s} n={n:7d} m={m:4d} "
                f"CPU={cpu_time:8.3f}s CUDA={gpu_time:8.4f}s "
                f"speedup={cpu_time / gpu_time:7.1f}x"
            )
    return rows


def fit_slopes(rows):
    slopes = []
    for motif in MOTIFS:
        motif_rows = [row for row in rows if row["motif"] == motif]
        x = np.log(np.array([row["observations"] for row in motif_rows], dtype=float))
        for key, label in (
            ("cpu_seconds", "Sequential CPU"),
            ("cuda_seconds", "Batched CUDA"),
            ("speedup", "Speedup"),
        ):
            y = np.log(np.array([row[key] for row in motif_rows], dtype=float))
            slope, intercept = np.polyfit(x, y, deg=1)
            fitted = intercept + slope * x
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            slopes.append(
                {
                    "motif": motif,
                    "quantity": label,
                    "slope": float(slope),
                    "r2": r2,
                }
            )
    return slopes


def write_slope_outputs(slopes, device, outdir):
    csv_path = outdir / "gpu_speedup_cuda_slopes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slopes[0]))
        writer.writeheader()
        writer.writerows(slopes)

    table_path = outdir / "gpu_speedup_cuda_slopes.tex"
    with table_path.open("w") as handle:
        handle.write("\\begin{tabular}{llrr}\n")
        handle.write("\\toprule\n")
        handle.write("Motif & Quantity & Log--log slope & $R^2$ \\\\\n")
        handle.write("\\midrule\n")
        for row in slopes:
            handle.write(
                f"{row['motif']} & {row['quantity']} & "
                f"{row['slope']:.2f} & {row['r2']:.3f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

    speedup_slopes = [row["slope"] for row in slopes if row["quantity"] == "Speedup"]
    mean_speedup_slope = float(np.mean(speedup_slopes))
    interp_path = outdir / "gpu_speedup_cuda_interpretation.tex"
    with interp_path.open("w") as handle:
        handle.write(
            "The benchmark was run on an "
            f"{device['name']} GPU with {device['multiprocessors']} streaming "
            f"multiprocessors and {device['total_memory_gb']:.1f} GB of memory. "
            "On log--log axes, the fitted slope for speedup is "
            f"{mean_speedup_slope:.2f} on average across motifs. "
            "This slope should not be read as an asymptotic law; it summarizes "
            "the measured regime. A positive slope would indicate that larger "
            "synthetic HSCM workloads expose more independent branches than the "
            "GPU can initially fill. A slope near zero indicates a saturated but "
            "stable acceleration regime. A negative slope means that the GPU is "
            "still much faster in absolute time, but its runtime grows faster "
            "over the measured range as memory traffic, kernel overheads, or "
            "occupancy limits become visible. "
        )


def write_outputs(rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    slopes = fit_slopes(rows)
    device = cuda_device_summary()

    with (outdir / "gpu_speedup_cuda_device.json").open("w") as handle:
        json.dump(device, handle, indent=2)

    csv_path = outdir / "gpu_speedup_cuda_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_path = outdir / "gpu_speedup_cuda_table.tex"
    with table_path.open("w") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(
            "Motif & Units $n$ & Subunits $m$ & Sequential CPU (s) & Batched CUDA (ms) & Speedup \\\\\n"
        )
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                f"{row['motif']} & {row['n']:,} & {row['m']:,} & "
                f"{row['cpu_seconds']:.2f} & {1000 * row['cuda_seconds']:.2f} & "
                f"{row['speedup']:.1f}$\\times$ \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colors = {"Confounder": "#4477AA", "Mediator": "#66AA55", "Instrument": "#CC6677"}
    for motif, color in colors.items():
        motif_rows = [row for row in rows if row["motif"] == motif]
        xs = np.array([row["observations"] for row in motif_rows])
        cpu = np.array([row["cpu_seconds"] for row in motif_rows])
        cuda = np.array([row["cuda_seconds"] for row in motif_rows])
        speedup = np.array([row["speedup"] for row in motif_rows])
        speedup_slope = next(
            row["slope"]
            for row in slopes
            if row["motif"] == motif and row["quantity"] == "Speedup"
        )
        axes[0].plot(xs, cpu, marker="o", color=color, linestyle="--",
                     linewidth=1.5, label=f"{motif} CPU")
        axes[0].plot(xs, cuda, marker="s", color=color,
                     linewidth=1.8, label=f"{motif} CUDA")
        axes[1].plot(xs, speedup, marker="o", color=color,
                     linewidth=1.8, label=f"{motif} slope {speedup_slope:.2f}")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Total observations $n \times m$")
    axes[0].set_ylabel("Wall time (seconds)")
    axes[0].set_title("Two-intervention evaluation time")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Total observations $n \times m$")
    axes[1].set_ylabel("Speedup over sequential CPU")
    axes[1].set_title("Batched CUDA speedup")
    axes[1].axhline(1.0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "gpu_speedup_cuda.pdf", bbox_inches="tight")
    fig.savefig(outdir / "gpu_speedup_cuda.png", dpi=240, bbox_inches="tight")
    write_slope_outputs(slopes, device, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=DEFAULT_SIZES,
        help="Comma-separated n x m sizes, for example 20000x32,60000x64.",
    )
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--gpu-repeats", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()
    rows = benchmark(args)
    write_outputs(rows, args.outdir)


if __name__ == "__main__":
    main()
