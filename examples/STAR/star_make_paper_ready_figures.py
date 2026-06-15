from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import genpareto, norm

from star_hcm_v2_teacher_student import N_SUB_PER_CLASS, RANDOM_STATE, RAW_TAB, REPO_ROOT


PAPER_FIG_DIR = REPO_ROOT / "refs" / "paperjrss" / "Fig"
RESULTS_DIR = REPO_ROOT / "examples" / "STAR" / "results"
FACTOR_SUMMARY = (
    REPO_ROOT
    / "examples"
    / "STAR"
    / "figures"
    / "factor_missing_graphs"
    / "factor1_effective_graphs_summary.json"
)
GENDER_HETEROGENEITY_CSV = (
    REPO_ROOT
    / "examples"
    / "STAR"
    / "figures"
    / "score_distribution_diagnostics"
    / "class_gender_heterogeneity_values.csv"
)


def apply_publication_style() -> None:
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "normal"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 22
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True
    mpl.rcParams["axes.titlesize"] = 18
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["xtick.labelsize"] = 13
    mpl.rcParams["ytick.labelsize"] = 13
    mpl.rcParams["legend.fontsize"] = 11
    mpl.rcParams["figure.titlesize"] = 20
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def save_figure(fig: plt.Figure, name: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(PAPER_FIG_DIR / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_balanced_sample() -> pd.DataFrame:
    raw = pd.read_csv(RAW_TAB, sep="\t")
    cols = [
        "stdntid",
        "gender",
        "race",
        "gkclasstype",
        "gksurban",
        "gktchid",
        "gkfreelunch",
        "gktreadss",
        "gktmathss",
    ]
    df = raw[cols].dropna().copy()
    df["A_small"] = (df["gkclasstype"] == 1.0).astype(float)
    df["Y_read"] = df["gktreadss"].astype(float)
    df["M_math"] = df["gktmathss"].astype(float)
    df["G_female"] = (df["gender"] == 1.0).astype(float)
    df["E_white_asian"] = df["race"].isin([1.0, 3.0]).astype(float)
    df["L_free"] = (df["gkfreelunch"] == 1.0).astype(float)
    df["S_urbanicity"] = df["gksurban"].astype(float)
    class_sizes = df.groupby("gktchid").size()
    valid_classes = class_sizes[class_sizes >= N_SUB_PER_CLASS].index
    df = df[df["gktchid"].isin(valid_classes)].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for _, group in df.groupby("gktchid"):
        parts.append(group.sample(n=N_SUB_PER_CLASS, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(parts, axis=0).sort_values(["gktchid", "stdntid"]).reset_index(drop=True)


def female_indicator(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(float)
        return (values == 2).astype(int) if set(values.dropna().unique()).issubset({1.0, 2.0}) else values.astype(int)
    lowered = series.astype(str).str.lower()
    return lowered.str.contains("female|girl|f").astype(int)


def plot_gender_heterogeneity(sampled: pd.DataFrame) -> None:
    if GENDER_HETEROGENEITY_CSV.exists():
        heterogeneity = pd.read_csv(GENDER_HETEROGENEITY_CSV)["heterogeneity_male_female"].to_numpy(dtype=float)
    else:
        tmp = sampled.copy()
        tmp["female"] = female_indicator(tmp["gender"])
        grouped = tmp.groupby("gktchid")["female"].agg(["sum", "count"])
        heterogeneity = ((grouped["count"] - 2 * grouped["sum"]) / grouped["count"]).to_numpy(dtype=float)
    mean_val = float(heterogeneity.mean())
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.hist(heterogeneity, bins=34, color="#4C78A8", alpha=0.78, edgecolor="white", linewidth=0.9)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=2.0, label="Gender balance")
    ax.axvline(mean_val, color="#E45756", linewidth=2.3, label=f"Mean = {mean_val:.3f}")
    ax.set_title("Class-level gender composition")
    ax.set_xlabel(r"$(n_{\mathrm{male}} - n_{\mathrm{female}}) / n_{\mathrm{class}}$")
    ax.set_ylabel("Number of classes")
    ax.legend(frameon=True, loc="upper left")
    save_figure(fig, "class_gender_heterogeneity_distribution")


def normal_pdf_grid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu = float(np.mean(values))
    sigma = max(float(np.std(values, ddof=1)), 1e-9)
    xs = np.linspace(float(values.min()), float(values.max()), 400)
    return xs, norm.pdf(xs, mu, sigma), mu, sigma


def standardized_normal_qq_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    mu = float(np.mean(v))
    sigma = max(float(np.std(v, ddof=1)), 1e-9)
    ordered = np.sort(v)
    p = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
    xq = norm.ppf(p)
    yz = (ordered - mu) / sigma
    return xq, yz, mu, sigma


def identity_line_crossings_x(xq: np.ndarray, yz: np.ndarray) -> list[float]:
    diff = yz - xq
    crossings: list[float] = []
    for idx in range(len(diff) - 1):
        if diff[idx] == 0:
            crossings.append(float(xq[idx]))
        elif diff[idx] * diff[idx + 1] < 0:
            x0, x1 = float(xq[idx]), float(xq[idx + 1])
            d0, d1 = float(diff[idx]), float(diff[idx + 1])
            crossings.append(x0 - d0 * (x1 - x0) / (d1 - d0))
    return crossings


def body_window_from_crossings(xq: np.ndarray, crossings: list[float]) -> tuple[float, float, bool]:
    finite_crossings = [x for x in crossings if np.isfinite(x)]
    if len(finite_crossings) >= 2:
        return float(finite_crossings[0]), float(finite_crossings[-1]), False
    return float(np.quantile(xq, 0.15)), float(np.quantile(xq, 0.85)), True


def fit_gpd_excess(excess: np.ndarray) -> tuple[float, float]:
    ex = np.asarray(excess, dtype=float)
    ex = ex[np.isfinite(ex) & (ex >= 0)]
    if len(ex) < 8 or np.allclose(ex, 0.0):
        return 0.0, max(float(np.std(ex)), 1e-6)
    shape, _, scale = genpareto.fit(ex, floc=0.0)
    return float(shape), max(float(scale), 1e-9)


def tail_gpd_qq_inset(
    ax_parent: plt.Axes,
    tail_values: np.ndarray,
    cutoff: float,
    loc: str,
    subtitle: str,
    *,
    side: str,
    display_lower: float | None = None,
    display_upper: float | None = None,
) -> None:
    tv = np.asarray(tail_values, dtype=float)
    tv = tv[np.isfinite(tv)]
    if len(tv) < 8:
        return
    ax_in = inset_axes(ax_parent, width="29%", height="24%", loc=loc, borderpad=1.05)
    m = len(tv)
    p = (np.arange(1, m + 1) - 0.5) / m
    if side == "right":
        excess = tv - cutoff
        shape, scale = fit_gpd_excess(excess)
        theo = cutoff + genpareto.ppf(p, c=shape, loc=0.0, scale=scale)
        color = "#1E3A8A"
        xlabel = "GPD quantiles"
    else:
        excess = cutoff - tv
        shape, scale = fit_gpd_excess(excess)
        theo = cutoff - genpareto.ppf(1.0 - p, c=shape, loc=0.0, scale=scale)
        color = "#B22222"
        xlabel = "GPD quantiles"
    ordered_tail = np.sort(tv)
    display_mask = np.ones_like(ordered_tail, dtype=bool)
    if display_lower is not None:
        display_mask &= (theo >= display_lower) & (ordered_tail >= display_lower)
    if display_upper is not None:
        display_mask &= (theo <= display_upper) & (ordered_tail <= display_upper)
    displayed_theo = theo[display_mask]
    displayed_tail = ordered_tail[display_mask]
    if len(displayed_tail) < 3:
        displayed_theo = theo
        displayed_tail = ordered_tail
    ax_in.scatter(displayed_theo, displayed_tail, s=8, alpha=0.55, color=color, edgecolors="none")
    lo = float(min(np.min(displayed_theo), np.min(displayed_tail)))
    hi = float(max(np.max(displayed_theo), np.max(displayed_tail)))
    ax_in.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ax_in.set_title(fr"{subtitle}, $\hat{{\xi}}={shape:.2f}$", fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.set_xlabel(xlabel, fontsize=5.5, labelpad=0.5)
    ax_in.set_ylabel("Sorted score", fontsize=5.5, labelpad=0.5)


def plot_class_means(sampled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.3), constrained_layout=True)
    specs = [("Y_read", "Reading", "#7B6FD6"), ("M_math", "Mathematics", "#4C78A8")]
    for row, (column, label, color) in enumerate(specs):
        means = sampled.groupby("gktchid")[column].mean().to_numpy(dtype=float)
        ax = axes[row, 0]
        ax.hist(means, bins=32, density=True, color=color, alpha=0.72, edgecolor="white")
        xs, pdf, mu, sigma = normal_pdf_grid(means)
        ax.plot(xs, pdf, color="#8B0000", linestyle="--", linewidth=2.2, label=f"Normal fit\n$\\mu={mu:.1f}$, $\\sigma={sigma:.1f}$")
        ax.set_title(f"Class means: {label}")
        ax.set_xlabel("Class mean score")
        ax.set_ylabel("Density")
        ax.legend(frameon=True)
        ordered = np.sort(means)
        p = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
        z = norm.ppf(p)
        ax = axes[row, 1]
        ax.scatter(z, ordered, s=18, alpha=0.65, color="#F58518", edgecolors="none")
        line_x = np.linspace(float(z.min()), float(z.max()), 100)
        ax.plot(line_x, mu + sigma * line_x, color="#8B0000", linewidth=2.0)
        ax.set_title(f"Normal Q-Q: {label} class means")
        ax.set_xlabel("Theoretical quantiles, $N(0,1)$")
        ax.set_ylabel("Empirical quantiles")
    save_figure(fig, "class_level_means_Y_M_gaussian_lognorm")


def plot_standardized_qq_with_tail_insets(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    *,
    left_cut_score: float | None = None,
    fit_left_tail: bool = True,
    fit_right_tail: bool = True,
) -> tuple[int, int, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    xq, yz, mu, sigma = standardized_normal_qq_arrays(values)
    crossings = identity_line_crossings_x(xq, yz)
    x_lo, x_hi, fallback = body_window_from_crossings(xq, crossings)
    if left_cut_score is not None:
        x_lo = float((float(left_cut_score) - mu) / sigma)
    lim = float(max(3.2, np.max(np.abs(xq)) * 1.05, np.max(np.abs(yz)) * 1.05))
    ax.axvspan(-lim, x_lo, color="#FDE2E2", alpha=0.55, zorder=0)
    ax.axvspan(x_lo, x_hi, color="#E3F4E4", alpha=0.65, zorder=0)
    ax.axvspan(x_hi, lim, color="#FDE2E2", alpha=0.55, zorder=0)
    ax.axvline(x_lo, color="#B12A90", linewidth=1.6, zorder=1)
    ax.axvline(x_hi, color="#B12A90", linewidth=1.6, zorder=1)
    ax.scatter(xq, yz, s=12, color="#1F77B4", alpha=0.8, edgecolors="none", label="Empirical quantiles", zorder=2)
    ax.plot([-lim, lim], [-lim, lim], color="black", linestyle="--", linewidth=1.5, label="Normal reference")
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles, $N(0,1)$")
    ax.set_ylabel("Standardized empirical quantiles")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True)
    n = len(xq)
    il = int(np.searchsorted(xq, x_lo, side="left"))
    ih = int(np.searchsorted(xq, x_hi, side="right")) - 1
    il = max(0, min(il, n - 1))
    ih = max(0, min(ih, n - 1))
    if ih < il:
        il, ih = 0, n - 1
    ordered_values = np.sort(values)
    cutoff_left = float(ordered_values[il])
    cutoff_right = float(ordered_values[ih])
    if fit_left_tail and il >= 8:
        tail_gpd_qq_inset(
            ax,
            ordered_values[:il],
            cutoff_left,
            "upper left",
            "Left tail",
            side="left",
            display_lower=mu - 3.0 * sigma,
        )
    if fit_right_tail and n - 1 - ih >= 8:
        tail_gpd_qq_inset(
            ax,
            ordered_values[ih + 1 :],
            cutoff_right,
            "center right",
            "Right tail",
            side="right",
            display_upper=mu + 3.0 * sigma,
        )
    return il, ih, mu, sigma, x_lo, x_hi


def plot_density_panel(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    il: int,
    ih: int,
    mu: float,
    sigma: float,
    *,
    fit_left_tail: bool = True,
    fit_right_tail: bool = True,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_total = max(len(values), 1)
    hist_density, _, _ = ax.hist(values, bins=40, density=True, color="lightgray", alpha=0.68, edgecolor="white", label="Empirical")
    ordered = np.sort(values)
    il = max(0, min(il, len(ordered) - 1))
    ih = max(il, min(ih, len(ordered) - 1))
    body = ordered[il : ih + 1]
    left_tail = ordered[:il]
    right_tail = ordered[ih + 1 :]
    p_left = float(len(left_tail)) / float(n_total)
    p_right = float(len(right_tail)) / float(n_total)
    x_left = float(ordered[il])
    x_right = float(ordered[ih])
    xs = np.linspace(float(values.min()), float(values.max()), 360)
    y_cap = 1.25 * float(np.nanmax(hist_density))
    if fit_left_tail and len(left_tail) >= 8:
        shape_l, scale_l = fit_gpd_excess(x_left - left_tail)
        xs_left = xs[xs <= x_left]
        left_pdf = p_left * genpareto.pdf(x_left - xs_left, c=shape_l, loc=0.0, scale=scale_l)
        left_pdf = np.where(left_pdf <= y_cap, left_pdf, np.nan)
        ax.plot(xs_left, left_pdf, color="#B22222", linestyle="--", linewidth=2.1, label=fr"Left GPD tail, $\xi={shape_l:.2f}$")
    if fit_right_tail and len(right_tail) >= 8:
        shape_r, scale_r = fit_gpd_excess(right_tail - x_right)
        xs_right = xs[xs >= x_right]
        right_pdf = p_right * genpareto.pdf(xs_right - x_right, c=shape_r, loc=0.0, scale=scale_r)
        right_pdf = np.where(right_pdf <= y_cap, right_pdf, np.nan)
        ax.plot(xs_right, right_pdf, color="#1E3A8A", linestyle="--", linewidth=2.1, label=fr"Right GPD tail, $\xi={shape_r:.2f}$")
    ax.axvline(x_left, color="#B12A90", linestyle=":", linewidth=1.5)
    ax.axvline(x_right, color="#B12A90", linestyle=":", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_ylim(0.0, y_cap)
    ax.legend(loc="upper right", frameon=True, fontsize=9)


def plot_global_score_diagnostics(sampled: pd.DataFrame) -> None:
    y_values = sampled["Y_read"].to_numpy(dtype=float)
    m_values = sampled["M_math"].to_numpy(dtype=float)

    fig_qq, axes_qq = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    il_y, ih_y, mu_y, sigma_y, _, _ = plot_standardized_qq_with_tail_insets(
        axes_qq[0],
        y_values,
        "Reading: standardized Q-Q with GPD tail zooms",
        left_cut_score=400.0,
        fit_left_tail=True,
        fit_right_tail=True,
    )
    il_m, ih_m, mu_m, sigma_m, _, _ = plot_standardized_qq_with_tail_insets(
        axes_qq[1],
        m_values,
        "Mathematics: standardized Q-Q with GPD tail zoom",
        left_cut_score=400.0,
        fit_left_tail=True,
        fit_right_tail=False,
    )
    save_figure(fig_qq, "global_Y_M_heavy_tail_qq")

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(13.2, 4.4), constrained_layout=True)
    plot_density_panel(
        axes_hist[0],
        y_values,
        "Reading: empirical density and tail models",
        il_y,
        ih_y,
        mu_y,
        sigma_y,
        fit_left_tail=True,
        fit_right_tail=True,
    )
    plot_density_panel(
        axes_hist[1],
        m_values,
        "Mathematics: empirical density and left-tail model",
        il_m,
        ih_m,
        mu_m,
        sigma_m,
        fit_left_tail=True,
        fit_right_tail=False,
    )
    save_figure(fig_hist, "global_Y_M_heavy_tail_hist")


def plot_enriched_q_y(sampled: pd.DataFrame) -> None:
    grouped = sampled.groupby("gktchid")["Y_read"]
    mu = grouped.mean().to_numpy(dtype=float)
    var = grouped.var(ddof=1).fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    scatter = ax.scatter(mu, var, c=mu, cmap="viridis", s=38, alpha=0.86, edgecolors="white", linewidths=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Class mean reading score")
    ax.set_title(r"Enriched $Q^y$: class-level Gaussian parameters")
    ax.set_xlabel(r"$\hat{\mu}_i$ for reading score")
    ax.set_ylabel(r"$\hat{\sigma}^2_i$ for reading score")
    save_figure(fig, "enriched_q_Q_y")


def class_gaussian_parameters(sampled: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = sampled.groupby("gktchid")[column]
    mu = grouped.mean().to_numpy(dtype=float)
    var = grouped.var(ddof=1).fillna(0.0).to_numpy(dtype=float)
    return mu, var


def class_binary_probability(sampled: pd.DataFrame, column: str) -> np.ndarray:
    return sampled.groupby("gktchid")[column].mean().to_numpy(dtype=float)


def class_conditional_lunch_given_ethnicity(sampled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_lunch_e0: list[float] = []
    p_lunch_e1: list[float] = []
    for _, group in sampled.groupby("gktchid"):
        lunch = group["L_free"].to_numpy(dtype=float)
        ethnicity = group["E_white_asian"].to_numpy(dtype=float)
        for value, target in ((0.0, p_lunch_e0), (1.0, p_lunch_e1)):
            mask = ethnicity == value
            if np.any(mask):
                target.append(float(np.mean(lunch[mask])))
            else:
                target.append(float(np.mean(lunch)))
    return np.asarray(p_lunch_e0, dtype=float), np.asarray(p_lunch_e1, dtype=float)


def style_small_panel(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)


def plot_binary_q_histogram(ax: plt.Axes, values: np.ndarray, title: str, color: str) -> None:
    ax.hist(values, bins=np.linspace(-0.025, 1.025, 22), color=color, alpha=0.78, edgecolor="white", linewidth=0.45)
    ax.axvline(float(np.mean(values)), color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Class-level probability")
    ax.set_ylabel("Classes")
    style_small_panel(ax)


def plot_enriched_q_probabilities(sampled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), constrained_layout=True)
    fig.suptitle("Estimated class-level HCM probabilities in STAR", fontsize=15)

    plot_binary_q_histogram(axes[0, 0], class_binary_probability(sampled, "A_small"), r"$Q^a$: small-class probability", "#4C78A8")
    plot_binary_q_histogram(axes[0, 1], class_binary_probability(sampled, "G_female"), r"$Q^g$: female probability", "#F58518")
    plot_binary_q_histogram(axes[0, 2], class_binary_probability(sampled, "E_white_asian"), r"$Q^e$: white/asian probability", "#54A24B")
    plot_binary_q_histogram(axes[1, 0], class_binary_probability(sampled, "L_free"), r"$Q^l$: free-lunch probability", "#B279A2")

    ax = axes[1, 1]
    p_lunch_e0, p_lunch_e1 = class_conditional_lunch_given_ethnicity(sampled)
    ax.scatter(
        p_lunch_e0,
        p_lunch_e1,
        c=class_binary_probability(sampled, "L_free"),
        cmap="cividis",
        s=14,
        alpha=0.78,
        edgecolors="none",
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"$Q^{l|e}$: lunch by ethnicity")
    ax.set_xlabel(r"$\hat{P}(L=1\mid E=0)$")
    ax.set_ylabel(r"$\hat{P}(L=1\mid E=1)$")
    style_small_panel(ax)

    ax = axes[1, 2]
    grouped_s = sampled.groupby("gktchid")["S_urbanicity"].first()
    counts = grouped_s.value_counts().sort_index()
    ax.bar([str(int(x)) for x in counts.index], counts.to_numpy(dtype=float), color="#72B7B2", edgecolor="white", linewidth=0.6)
    ax.set_title(r"$S$: class-level urbanicity")
    ax.set_xlabel("Urbanicity category")
    ax.set_ylabel("Classes")
    style_small_panel(ax)

    save_figure(fig, "enriched_q_probabilities_star")


def plot_enriched_q_parameters_and_means(sampled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    fig.suptitle("Estimated class-level HCM parameters and means in STAR", fontsize=15)

    y_mu, y_var = class_gaussian_parameters(sampled, "Y_read")
    ax = axes[0]
    sc = ax.scatter(y_mu, y_var, c=y_mu, cmap="viridis", s=16, alpha=0.78, edgecolors="none")
    ax.set_title(r"$Q^y$: reading Gaussian parameters")
    ax.set_xlabel(r"$\hat{\mu}_i(Y)$")
    ax.set_ylabel(r"$\hat{\sigma}^2_i(Y)$")
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.01).set_label("Mean", fontsize=8)
    style_small_panel(ax)

    m_mu, m_var = class_gaussian_parameters(sampled, "M_math")
    ax = axes[1]
    sc = ax.scatter(m_mu, m_var, c=m_mu, cmap="magma", s=16, alpha=0.78, edgecolors="none")
    ax.set_title(r"$Q^m$: mathematics Gaussian parameters")
    ax.set_xlabel(r"$\hat{\mu}_i(M)$")
    ax.set_ylabel(r"$\hat{\sigma}^2_i(M)$")
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.01).set_label("Mean", fontsize=8)
    style_small_panel(ax)

    ax = axes[2]
    ax.scatter(y_mu, m_mu, c=class_binary_probability(sampled, "A_small"), cmap="plasma", s=16, alpha=0.78, edgecolors="none")
    ax.set_title(r"Joint class means: $Q^y$ vs $Q^m$")
    ax.set_xlabel(r"$\hat{\mu}_i(Y)$")
    ax.set_ylabel(r"$\hat{\mu}_i(M)$")
    style_small_panel(ax)

    save_figure(fig, "enriched_q_parameters_means_star")


def plot_all_enriched_q_quantities(sampled: pd.DataFrame) -> None:
    plot_enriched_q_probabilities(sampled)
    plot_enriched_q_parameters_and_means(sampled)


def normalize_hcm_node(name: str) -> str:
    return name[1:] if name.startswith("_") else name


def node_positions() -> dict[str, tuple[float, float]]:
    return {
        "U": (-0.2, 1.0),
        "S": (3.4, 1.0),
        "A": (0.0, 0.0),
        "E": (1.6, 0.0),
        "G": (3.2, 0.0),
        "L": (4.8, 0.0),
        "M": (6.4, 0.0),
        "Y": (8.0, 0.0),
    }


def interpreted_node_labels() -> dict[str, str]:
    return {
        "U": "Unobserved\nclass heterogeneity",
        "S": "School\nurbanicity",
        "A": "Small-class\nassignment",
        "E": "Ethnicity",
        "G": "Gender",
        "L": "Free lunch",
        "M": "Mathematics\nscore",
        "Y": "Reading\nscore",
    }


def draw_interpreted_node_labels(
    ax: plt.Axes,
    pos: dict[str, tuple[float, float]],
    nodes: list[str],
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    labels = interpreted_node_labels()
    for node in nodes:
        x, y = pos[node]
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            linespacing=0.9,
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.65",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 1.35,
            },
            zorder=4,
        )


def exactbic_edge_curvature() -> dict[tuple[str, str], float]:
    return {
        ("S", "A"): 0.05,
        ("S", "L"): -0.03,
        ("S", "Y"): -0.18,
        ("S", "E"): 0.12,
        ("U", "A"): 0.04,
        ("U", "M"): -0.08,
        ("A", "Y"): 0.16,
        ("E", "L"): 0.08,
        ("E", "M"): 0.16,
        ("G", "Y"): 0.13,
        ("M", "L"): 0.25,
        ("Y", "M"): 0.25,
    }


def draw_curved_edges(
    ax: plt.Axes,
    pos: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    removed_edges: set[tuple[str, str]],
    *,
    graph_name: str,
) -> None:
    curvature = exactbic_edge_curvature() if graph_name == "ExactBIC" else {}
    for source, target in edges:
        is_removed = (source, target) in removed_edges
        color = "#D62728" if is_removed else "#333333"
        linestyle = "--" if is_removed else "-"
        linewidth = 1.85 if is_removed else 1.55
        rad = curvature.get((source, target), 0.08 if not is_removed else 0.18)
        arrow = FancyArrowPatch(
            pos[source],
            pos[target],
            arrowstyle="-|>",
            mutation_scale=12.5,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=19,
            shrinkB=22,
            zorder=2,
        )
        ax.add_patch(arrow)


def plot_effective_graph(graph_name: str, graph_rows: list[dict], factor_summary: dict) -> None:
    display_name = {"DirectLiNGAM": "Graph A", "ExactBIC": "ExactBIC"}[graph_name]
    output_name = {"DirectLiNGAM": "graph_a_with_factor1_effective_graph_curved", "ExactBIC": "ExactBIC_with_factor1_effective_graph_curved"}[graph_name]
    row = next(item for item in graph_rows if item["graph_name"] == graph_name)
    removed_edges = {
        (normalize_hcm_node(a), normalize_hcm_node(b))
        for a, b in factor_summary[graph_name]["removed_edges_present_in_graph"]
    }
    all_edges = [(normalize_hcm_node(a), normalize_hcm_node(b)) for a, b in row["edges_hcm"]]
    if graph_name == "ExactBIC":
        all_edges = [("S", "E") if edge == ("E", "S") else edge for edge in all_edges]
    active_edges = [edge for edge in all_edges if edge not in removed_edges]
    graph = nx.DiGraph()
    graph.add_nodes_from(["U", "S", "A", "E", "G", "L", "M", "Y"])
    graph.add_edges_from(all_edges)
    pos = node_positions()
    fig, ax = plt.subplots(figsize=(10.2, 3.9), constrained_layout=True)
    ax.axhspan(0.65, 1.35, color="#EAF1FF", alpha=0.9)
    ax.axhspan(-0.35, 0.35, color="#FFF3E0", alpha=0.9)
    ax.text(-1.45, 1.12, "Unit level", color="#254B9B", weight="bold", fontsize=11)
    ax.text(-1.45, 0.12, "Student level", color="#9A5A00", weight="bold", fontsize=11)
    unit_nodes = ["U", "S"]
    sub_nodes = ["A", "E", "G", "L", "M", "Y"]
    draw_curved_edges(ax, pos, active_edges + list(removed_edges), removed_edges, graph_name=graph_name)
    draw_interpreted_node_labels(ax, pos, unit_nodes, facecolor="#BFD4FF", edgecolor="#254B9B")
    draw_interpreted_node_labels(ax, pos, sub_nodes, facecolor="#FFD89C", edgecolor="#9A5A00")
    ax.plot([], [], color="#333333", linewidth=1.45, label="Active factor")
    ax.plot([], [], color="#D62728", linewidth=1.6, linestyle="--", label="Normalized factor")
    ax.set_title(f"{display_name}: nominal and effective graph", fontsize=12)
    ax.set_xlim(-1.65, 8.65)
    ax.set_ylim(-0.45, 1.45)
    ax.set_axis_off()
    ax.legend(loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.16), fontsize=8)
    save_figure(fig, output_name)


def plot_effective_graphs() -> None:
    results = json.loads((RESULTS_DIR / "star_hcm_v2_teacher_student.json").read_text(encoding="utf-8"))
    factor_summary = json.loads(FACTOR_SUMMARY.read_text(encoding="utf-8"))
    for graph_name in ("DirectLiNGAM", "ExactBIC"):
        plot_effective_graph(graph_name, results["results"], factor_summary)


def transformed_q_node_positions() -> dict[str, tuple[float, float]]:
    return {
        "U": (-1.0, 1.2),
        "S": (1.2, 1.2),
        "Qa": (-0.9, 0.0),
        "Qe": (1.1, 0.0),
        "Qg": (3.1, 0.0),
        "Qyag": (1.1, -1.2),
        "Qmey": (3.1, -1.2),
        "Qm": (5.3, -0.6),
    }


def transformed_q_node_labels() -> dict[str, str]:
    return {
        "U": "Unobserved\nclass heterogeneity",
        "S": "School\nurbanicity",
        "Qa": r"$Q^a$" + "\nsmall-class\nmechanism",
        "Qe": r"$Q^e$" + "\nethnicity\nmechanism",
        "Qg": r"$Q^g$" + "\ngender\nmechanism",
        "Qyag": r"$Q^{y\mid a,g}$" + "\nreading\nmechanism",
        "Qmey": r"$Q^{m\mid e,y}$" + "\nmathematics\nmechanism",
        "Qm": r"$Q^m$" + "\ntarget class\nmath distribution",
    }


def draw_q_transformed_nodes(ax: plt.Axes, pos: dict[str, tuple[float, float]]) -> None:
    labels = transformed_q_node_labels()
    unit_nodes = {"U", "S"}
    target_nodes = {"Qm"}
    for node, (x, y) in pos.items():
        facecolor = "#BFD4FF" if node in unit_nodes else "#C9E8C9" if node in target_nodes else "#FFE1AD"
        edgecolor = "#254B9B" if node in unit_nodes else "#2E7D32" if node in target_nodes else "#9A5A00"
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            linespacing=0.9,
            bbox={
                "boxstyle": "round,pad=0.30,rounding_size=0.45",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 1.25,
            },
            zorder=4,
        )


def draw_q_transformed_edge(
    ax: plt.Axes,
    pos: dict[str, tuple[float, float]],
    source: str,
    target: str,
    *,
    rad: float = 0.0,
    color: str = "#333333",
    linestyle: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        pos[source],
        pos[target],
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.45,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=25,
        shrinkB=25,
        zorder=2,
    )
    ax.add_patch(arrow)


def plot_exactbic_transformed_q_graph() -> None:
    pos = transformed_q_node_positions()
    active_edges = [
        ("S", "Qa", -0.08),
        ("S", "Qe", 0.00),
        ("U", "Qa", 0.05),
        ("U", "Qm", -0.18),
        ("Qa", "Qm", 0.12),
        ("Qe", "Qm", -0.08),
        ("Qg", "Qm", 0.03),
        ("Qyag", "Qm", 0.00),
        ("Qmey", "Qm", -0.04),
    ]
    normalized_edges = [("S", "Qyag", -0.20)]
    fig, ax = plt.subplots(figsize=(10.4, 4.4), constrained_layout=True)
    ax.axhspan(0.75, 1.65, color="#EAF1FF", alpha=0.9)
    ax.axhspan(-1.65, 0.45, color="#FFF3E0", alpha=0.9)
    ax.text(-1.95, 1.38, "Unit-level context", color="#254B9B", weight="bold", fontsize=11)
    ax.text(-1.95, 0.2, "Collapsed/augmented Q graph", color="#9A5A00", weight="bold", fontsize=11)
    for source, target, rad in active_edges:
        draw_q_transformed_edge(ax, pos, source, target, rad=rad)
    for source, target, rad in normalized_edges:
        draw_q_transformed_edge(ax, pos, source, target, rad=rad, color="#D62728", linestyle="--")
    draw_q_transformed_nodes(ax, pos)
    ax.plot([], [], color="#333333", linewidth=1.45, label="Retained factor/path")
    ax.plot([], [], color="#D62728", linewidth=1.45, linestyle="--", label="Normalized for stability")
    ax.set_title(r"ExactBIC after HCM collapse/augmentation: Q-variable graph", fontsize=12)
    ax.set_xlim(-2.15, 6.15)
    ax.set_ylim(-1.75, 1.75)
    ax.set_axis_off()
    ax.legend(loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    save_figure(fig, "ExactBIC_transformed_q_graph")


def main() -> None:
    apply_publication_style()
    sampled = load_balanced_sample()
    plot_gender_heterogeneity(sampled)
    plot_class_means(sampled)
    plot_global_score_diagnostics(sampled)
    plot_enriched_q_y(sampled)
    plot_all_enriched_q_quantities(sampled)
    plot_effective_graphs()
    plot_exactbic_transformed_q_graph()
    print(f"Wrote paper-ready figures to {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
