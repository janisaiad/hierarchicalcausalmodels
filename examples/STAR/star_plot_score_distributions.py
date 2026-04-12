"""
Diagnostics empiriques pour Y (lecture), M (maths) et S (urbanicité) sur le même
échantillon équilibré que star_hcm_v2_teacher_student.py.

- ``global_Y_M_gaussian_hist_qq.png`` : Y et M — **uniquement** Gaussienne + QQ normal (référence HCM).
- ``global_Y_M_normal_vs_student_t_hist_qq.png`` : Y et M — Normal vs **Student-t MLE** (souvent ν grand ⇒ quasi identique à la Normal) **et** une **t à ν fixe** (échelle calée sur σ) en pointillés pour voir des queues plus lourdes ; QQ normal + QQ-t MLE.
- ``global_Y_M_gaussian_body_heavy_tail_qq_hist.png`` : analyse **complète** par variable : Q-Q standardisé (identité + **courbes** t MLE et t à ν fixe en z) + **insets** Q-Q t sur les queues ; histogramme (**log-normal sur le corps**, Normal corps, LN/N globaux) ; panneau **AIC** (Normal, log-normal, Student-t sur tout l’échantillon).
- ``global_Y_log_M_gauss_fits.png`` : Y — log-normal + QQ log-normal (Gauss. en pointillés) ; M — Gaussienne.
- Par classe : **Gaussienne (pointillés) + log-normal (trait plein)** sur les mêmes scores.
- Moyennes par classe : **Gaussienne + log-normal** superposées.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import lognorm as scipy_lognorm
from scipy.stats import norm as scipy_norm
from scipy.stats import t as scipy_student_t

from star_hcm_v2_teacher_student import N_SUB_PER_CLASS, RANDOM_STATE, RAW_TAB


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
    df["Y_read"] = df["gktreadss"].astype(float)
    df["M_math"] = df["gktmathss"].astype(float)
    df["S_urbanicity"] = df["gksurban"].astype(float)
    class_sizes = df.groupby("gktchid").size()
    valid_classes = class_sizes[class_sizes >= N_SUB_PER_CLASS].index
    df = df[df["gktchid"].isin(valid_classes)].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_parts: list[pd.DataFrame] = []
    for _, g in df.groupby("gktchid"):
        sampled_parts.append(g.sample(n=N_SUB_PER_CLASS, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(sampled_parts, axis=0).sort_values(["gktchid", "stdntid"]).reset_index(drop=True)


def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def fit_lognorm_s_scale(x: np.ndarray) -> tuple[float, float]:
    """Retourne ``(s, scale)`` pour ``scipy.stats.lognorm`` avec ``loc=0`` (MLE sur ``log(x)``)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        return 1e-3, float(np.median(x)) if len(x) else 1.0
    logx = np.log(x)
    mu_l = float(np.mean(logx))
    sig_l = max(float(np.std(logx, ddof=1)), 1e-6)
    return sig_l, float(np.exp(mu_l))


def plot_global_gaussian_hist_qq(
    ax_hist: plt.Axes,
    ax_qq: plt.Axes,
    values: np.ndarray,
    title: str,
    qq_title: str = "QQ normal",
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
    ax_hist.hist(values, bins=40, density=True, color="coral", alpha=0.5, edgecolor="white", label="empirique")
    xs = np.linspace(values.min(), values.max(), 200)
    ax_hist.plot(xs, _norm_pdf(xs, mu, sigma), color="darkred", lw=2, label=f"Gaussienne N({mu:.1f}, {sigma:.1f}²)")
    ax_hist.set_title(title)
    ax_hist.set_ylabel("densité")
    ax_hist.legend(loc="upper right", fontsize=8)
    n = len(values)
    ordered = np.sort(values)
    p = (np.arange(1, n + 1) - 0.5) / n
    z = scipy_norm.ppf(p)
    ax_qq.scatter(z, ordered, s=4, alpha=0.35, c="coral")
    lo, hi = float(z.min()), float(z.max())
    line_x = np.linspace(lo, hi, 50)
    ax_qq.plot(line_x, mu + sigma * line_x, color="darkred", lw=1.5)
    ax_qq.set_xlabel("quantiles N(0,1)")
    ax_qq.set_ylabel("quantiles empiriques")
    ax_qq.set_title(qq_title)


def plot_qq_normal_only(ax_qq: plt.Axes, values: np.ndarray, qq_title: str) -> None:
    """QQ plot seul (référence N(0,1) sur l’axe x, données triées sur y, droite μ+σz)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
    sigma = max(sigma, 1e-9)
    n = len(values)
    ordered = np.sort(values)
    p = (np.arange(1, n + 1) - 0.5) / n
    z = scipy_norm.ppf(p)
    ax_qq.scatter(z, ordered, s=4, alpha=0.35, c="coral")
    lo, hi = float(z.min()), float(z.max())
    line_x = np.linspace(lo, hi, 50)
    ax_qq.plot(line_x, mu + sigma * line_x, color="darkred", lw=1.5)
    ax_qq.set_xlabel("quantiles N(0,1)")
    ax_qq.set_ylabel("quantiles empiriques (score)")
    ax_qq.set_title(qq_title)


def plot_global_lognormal_hist_qq(
    ax_hist: plt.Axes,
    ax_qq: plt.Axes,
    values: np.ndarray,
    title: str,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    s_ln, scale_ln = fit_lognorm_s_scale(values)
    ax_hist.hist(values, bins=40, density=True, color="steelblue", alpha=0.5, edgecolor="white", label="empirique")
    xs = np.linspace(max(values.min(), 1e-6), values.max(), 250)
    pdf = scipy_lognorm.pdf(xs, s=s_ln, scale=scale_ln)
    ax_hist.plot(
        xs,
        pdf,
        color="darkgreen",
        lw=2,
        label=f"log-normal (σ_log={s_ln:.3f}, scale={scale_ln:.1f})",
    )
    mu_g, sig_g = float(np.mean(values)), float(np.std(values, ddof=1))
    ax_hist.plot(
        xs,
        _norm_pdf(xs, mu_g, sig_g),
        color="gray",
        ls="--",
        lw=1.2,
        alpha=0.85,
        label="Gaussienne (réf.)",
    )
    ax_hist.set_title(title)
    ax_hist.set_ylabel("densité")
    ax_hist.legend(loc="upper right", fontsize=7)
    n = len(values)
    ordered = np.sort(values)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = scipy_lognorm.ppf(p, s=s_ln, scale=scale_ln)
    ax_qq.scatter(theo, ordered, s=4, alpha=0.35, c="steelblue")
    lo, hi = float(theo.min()), float(theo.max())
    line_t = np.linspace(lo, hi, 50)
    ax_qq.plot(line_t, line_t, color="darkgreen", lw=1.5)
    ax_qq.set_xlabel("quantiles log-normal théoriques")
    ax_qq.set_ylabel("quantiles empiriques")
    ax_qq.set_title("QQ log-normal (Y)")


def _fit_student_t_safe(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return 30.0, float(np.mean(values)), max(float(np.std(values, ddof=1)), 1e-6)
    try:
        df, loc, scale = scipy_student_t.fit(values)
        df = float(max(df, 1.1))
        scale = float(max(scale, 1e-6))
        return df, float(loc), scale
    except Exception:
        return 30.0, float(np.mean(values)), max(float(np.std(values, ddof=1)), 1e-6)


def plot_hist_normal_and_student_t(
    ax_hist: plt.Axes,
    values: np.ndarray,
    title: str,
    *,
    fixed_df: float = 5.0,
) -> tuple[float, float, float]:
    """Histogramme + PDF normale, Student-t MLE, et t à ``fixed_df`` (échelle = σ√((ν-1)/ν)) pour contraste visuel."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mu = float(np.mean(values))
    sigma = max(float(np.std(values, ddof=1)), 1e-9)
    df_t, loc_t, scale_t = _fit_student_t_safe(values)
    ax_hist.hist(values, bins=40, density=True, color="coral", alpha=0.45, edgecolor="white", label="empirique")
    xs = np.linspace(float(values.min()), float(values.max()), 260)
    ax_hist.plot(xs, scipy_norm.pdf(xs, mu, sigma), color="darkred", lw=2, label=f"Normal μ={mu:.1f}, σ={sigma:.1f}")
    ax_hist.plot(
        xs,
        scipy_student_t.pdf(xs, df_t, loc=loc_t, scale=scale_t),
        color="darkblue",
        lw=2,
        label=f"Student-t MLE ν={df_t:.1f}",
    )
    nu0 = float(max(fixed_df, 2.1))
    scale_fixed = sigma * float(np.sqrt(max(nu0 - 2.0, 0.1) / nu0))
    ax_hist.plot(
        xs,
        scipy_student_t.pdf(xs, nu0, loc=mu, scale=scale_fixed),
        color="teal",
        lw=2,
        ls="--",
        label=f"t fixe ν={nu0:.0f} (Var≈σ²)",
    )
    note = (
        f"Si ν_MLE ≫ 1, la t-MLE est quasi superposée à la Normal.\n"
        f"La courbe pointillée impose ν={nu0:.0f} pour illustrer des queues plus lourdes."
    )
    ax_hist.text(
        0.02,
        0.98,
        note,
        transform=ax_hist.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
    )
    ax_hist.set_title(title)
    ax_hist.set_ylabel("densité")
    ax_hist.legend(loc="upper right", fontsize=7)
    return df_t, loc_t, scale_t


def _standardized_normal_qq_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``x`` = Φ⁻¹(p), ``y`` = z_(i) triés avec z=(v-μ̂)/σ̂ (même convention que le Q-Q identité)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    mu = float(np.mean(v))
    sd = max(float(np.std(v, ddof=1)), 1e-9)
    ys = np.sort((v - mu) / sd)
    n = len(ys)
    p = (np.arange(1, n + 1) - 0.5) / n
    xq = scipy_norm.ppf(p)
    return xq, ys, mu, sd


def _identity_line_crossings_x(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Abscisses où la polyligne (x,y) coupe la droite y=x (plan z vs Φ⁻¹)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out: list[float] = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        denom = dx - dy
        if abs(denom) < 1e-14:
            continue
        t = (y0 - x0) / denom
        if not (0.0 <= t <= 1.0):
            continue
        xc = x0 + t * dx
        yc = y0 + t * dy
        if abs(xc - yc) <= 1e-4 * max(1.0, abs(xc)):
            out.append(float(xc))
    return out


def _body_x_window_from_crossings(xq: np.ndarray, crossings: list[float]) -> tuple[float, float, bool]:
    """Fenêtre [x_lo, x_hi] sur l’axe Φ⁻¹ pour le « corps » ; ``used_fallback`` si quantiles empiriques."""
    inner = sorted(c for c in crossings if -2.7 < c < 2.7)
    if len(inner) >= 2:
        return inner[0], inner[-1], False
    if len(inner) == 1:
        c = inner[0]
        return c - 0.45, c + 0.45, False
    return float(np.quantile(xq, 0.15)), float(np.quantile(xq, 0.85)), True


def _tail_t_qq_inset(
    ax_parent: plt.Axes,
    tail_vals: np.ndarray,
    loc: str,
    subtitle: str,
    *,
    side: str,
) -> None:
    """Petit Q-Q : queue vs Student-t. Queue **droite** : quantiles supérieurs ``1-p`` + scores triés **décroissant**."""
    tv = np.asarray(tail_vals, dtype=float)
    tv = tv[np.isfinite(tv)]
    if len(tv) < 8:
        return
    ax_in = inset_axes(ax_parent, width="34%", height="30%", loc=loc, borderpad=0.8)
    try:
        df, loc_t, sc = scipy_student_t.fit(tv)
        df = float(max(df, 1.2))
        sc = float(max(sc, 1e-6))
    except Exception:
        df, loc_t, sc = 8.0, float(np.mean(tv)), max(float(np.std(tv, ddof=1)), 1e-6)
    m = len(tv)
    p = (np.arange(1, m + 1) - 0.5) / m
    if side == "right":
        ot = np.sort(tv)[::-1]
        theo = scipy_student_t.ppf(1.0 - p, df, loc=loc_t, scale=sc)
        xlab = "t théo. (queue sup.)"
    else:
        ot = np.sort(tv)
        theo = scipy_student_t.ppf(p, df, loc=loc_t, scale=sc)
        xlab = "t théo. (queue inf.)"
    col = "darkred" if side == "left" else "darkblue"
    ax_in.scatter(theo, ot, s=2, alpha=0.45, c=col)
    lo = float(min(theo.min(), ot.min()))
    hi = float(max(theo.max(), ot.max()))
    ax_in.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ax_in.set_title(subtitle + f"\nν̂={df:.1f}", fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.set_xlabel(xlab, fontsize=6)
    ax_in.set_ylabel("score (tri)", fontsize=6)


def _aic_model_scores(v: np.ndarray) -> tuple[list[tuple[str, float, int]], str]:
    """Liste (nom, AIC, k) et nom du meilleur modèle (AIC minimal)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 5:
        return [], "n insuffisant"
    rows: list[tuple[str, float, int]] = []
    mu = float(np.mean(v))
    sig = max(float(np.std(v, ddof=1)), 1e-9)
    ll_n = float(np.sum(scipy_norm.logpdf(v, loc=mu, scale=sig)))
    aic_n = -2.0 * ll_n + 2.0 * 2.0
    rows.append(("Normal", aic_n, 2))
    best_name, best_aic = "Normal", aic_n
    if float(np.min(v)) > 0:
        sh, loc_ln, sc_ln = scipy_lognorm.fit(v, floc=0)
        ll_ln = float(np.sum(scipy_lognorm.logpdf(v, sh, loc_ln, sc_ln)))
        aic_ln = -2.0 * ll_ln + 2.0 * 2.0
        rows.append(("Log-normal (floc=0)", aic_ln, 2))
        if aic_ln < best_aic:
            best_name, best_aic = "Log-normal", aic_ln
    df_t, lc, sc_t = _fit_student_t_safe(v)
    ll_t = float(np.sum(scipy_student_t.logpdf(v, df_t, loc=lc, scale=sc_t)))
    aic_t = -2.0 * ll_t + 2.0 * 3.0
    rows.append((f"Student-t (ν̂={df_t:.2f})", aic_t, 3))
    if aic_t < best_aic:
        best_name, best_aic = "Student-t", aic_t
    return rows, best_name


def _format_aic_panel(v: np.ndarray, label: str) -> str:
    rows, best = _aic_model_scores(v)
    if not rows:
        return f"{label}\n(insuffisant)"
    lines = [f"{label} — critère AIC (plus bas = mieux)", ""]
    for name, aic, _k in rows:
        lines.append(f"  {name:22s}  {aic:,.0f}")
    lines.append("")
    lines.append(f"  → meilleur ajustement global : {best}")
    lines.append("")
    lines.append("(AIC sur tout l’échantillon ; le « corps » LN sur l’hist. est un zoom local.)")
    return "\n".join(lines)


def plot_qq_gaussian_body_heavy_tail(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
) -> tuple[float, float, int, int, bool, float, float]:
    """Q-Q standardisé + identité + courbes t en z + insets queues ; renvoie x_lo,x_hi,il,ih,fallback,μ,σ."""
    v_raw = np.asarray(values, dtype=float)
    v_raw = v_raw[np.isfinite(v_raw)]
    xq, ys, mu, sd = _standardized_normal_qq_arrays(values)
    n = len(xq)
    ax.scatter(xq, ys, s=5, alpha=0.35, c="C0", label="empirique", zorder=2)
    lim = float(max(3.2, np.max(np.abs(xq)) * 1.05))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.3, label="identité (N(0,1))", zorder=1)
    cr = _identity_line_crossings_x(xq, ys)
    x_lo, x_hi, fb = _body_x_window_from_crossings(xq, cr)
    ax.axvline(x_lo, color="purple", lw=1.2, ls="-", zorder=1)
    ax.axvline(x_hi, color="purple", lw=1.2, ls="-", zorder=1)
    ax.axvspan(-lim, x_lo, facecolor="salmon", alpha=0.16, zorder=0)
    ax.axvspan(x_lo, x_hi, facecolor="lightgreen", alpha=0.2, zorder=0)
    ax.axvspan(x_hi, lim, facecolor="salmon", alpha=0.16, zorder=0)
    p_line = np.linspace(0.003, 0.997, 500)
    x_line = scipy_norm.ppf(p_line)
    df_mle, loc_m, sc_m = _fit_student_t_safe(v_raw)
    y_t_mle = (scipy_student_t.ppf(p_line, df_mle, loc=loc_m, scale=sc_m) - mu) / sd
    ax.plot(
        x_line,
        y_t_mle,
        color="darkorange",
        lw=2.0,
        label=f"réf. t MLE → z (ν={df_mle:.1f})",
        zorder=3,
    )
    nu0 = 5.0
    sc0 = sd * float(np.sqrt(max(nu0 - 2.0, 0.1) / nu0))
    y_t0 = (scipy_student_t.ppf(p_line, nu0, loc=mu, scale=sc0) - mu) / sd
    ax.plot(
        x_line,
        y_t0,
        color="teal",
        lw=1.7,
        ls="--",
        label=f"réf. t ν={nu0:.0f} (Var=σ²) → z",
        zorder=3,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Φ⁻¹(p)")
    ax.set_ylabel("z trié = (score−μ̂)/σ̂")
    sub = f"n_croisements={len(cr)}" + (" (fenêtre 15–85)" if fb else "")
    ax.set_title(f"{title}\n{sub}")
    ax.legend(loc="lower right", fontsize=6)
    il = int(np.searchsorted(xq, x_lo, side="left"))
    ih = int(np.searchsorted(xq, x_hi, side="right")) - 1
    il = max(0, min(il, n - 1))
    ih = max(0, min(ih, n - 1))
    if ih < il:
        il, ih = 0, n - 1
    ordered_v = np.sort(v_raw)
    if il >= 8:
        _tail_t_qq_inset(ax, ordered_v[:il], "lower left", "Queue gauche", side="left")
    if n - 1 - ih >= 8:
        _tail_t_qq_inset(ax, ordered_v[ih + 1 :], "lower right", "Queue droite", side="right")
    return x_lo, x_hi, il, ih, fb, mu, sd


def plot_hist_full_and_body_models(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    il: int,
    ih: int,
    mu: float,
    sigma: float,
) -> None:
    """Histogramme : empirique, N/LN globaux, LN (et N) ajustés sur le corps seulement."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=40, density=True, color="lightgray", alpha=0.55, edgecolor="white", label="empirique")
    xs = np.linspace(float(v.min()), float(v.max()), 320)
    ax.plot(xs, scipy_norm.pdf(xs, mu, sigma), color="gray", lw=1.4, ls=":", label="N global")
    if np.min(v) > 0:
        sh_g, loc_g, sc_g = scipy_lognorm.fit(v, floc=0)
        ax.plot(xs, scipy_lognorm.pdf(xs, sh_g, loc_g, sc_g), color="gray", lw=1.4, ls="--", label="LN global")
    ordered = np.sort(v)
    body = ordered[il : ih + 1]
    if len(body) >= 4 and float(np.min(body)) > 0:
        sh_b, loc_b, sc_b = scipy_lognorm.fit(body, floc=0)
        ax.plot(
            xs,
            scipy_lognorm.pdf(xs, sh_b, loc_b, sc_b),
            color="darkgreen",
            lw=2.6,
            label=f"LN « corps » (n={len(body)})",
        )
    if len(body) >= 3:
        mu_b = float(np.mean(body))
        sig_b = max(float(np.std(body, ddof=1)), 1e-9)
        ax.plot(
            xs,
            scipy_norm.pdf(xs, mu_b, sig_b),
            color="darkviolet",
            lw=1.7,
            ls="--",
            label="N « corps »",
        )
    ax.set_title(title)
    ax.set_ylabel("densité")
    ax.legend(loc="upper right", fontsize=6)


def plot_qq_student_t(ax_qq: plt.Axes, values: np.ndarray, df: float, loc: float, scale: float, title: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        ax_qq.text(0.5, 0.5, "n<2", ha="center", va="center", transform=ax_qq.transAxes)
        ax_qq.set_title(title)
        return
    ordered = np.sort(values)
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = scipy_student_t.ppf(p, df, loc=loc, scale=scale)
    ax_qq.scatter(theo, ordered, s=4, alpha=0.35, c="steelblue")
    lo, hi = float(np.min(theo)), float(np.max(theo))
    ax_qq.plot([lo, hi], [lo, hi], color="darkblue", lw=1.5, ls="--")
    ax_qq.set_xlabel("quantiles Student-t ajustés")
    ax_qq.set_ylabel("quantiles empiriques")
    ax_qq.set_title(title)


def plot_s_distribution(ax: plt.Axes, s_unit: np.ndarray, title: str) -> None:
    s_unit = np.asarray(s_unit, dtype=float)
    s_unit = s_unit[np.isfinite(s_unit)]
    uniq, counts = np.unique(s_unit, return_counts=True)
    ax.bar(uniq.astype(int), counts, color="seagreen", alpha=0.75, edgecolor="white")
    ax.set_xticks(uniq.astype(int))
    ax.set_xlabel("gksurban (code)")
    ax.set_ylabel("nombre de classes")
    ax.set_title(title)


def plot_per_class_gaussian_and_lognormal_rows(
    fig: plt.Figure,
    sampled: pd.DataFrame,
    class_ids: list[int],
    var_y: str,
    var_m: str,
) -> None:
    for row, tid in enumerate(class_ids):
        sub = sampled.loc[sampled["gktchid"] == tid]
        raw_y = np.asarray(sub[var_y].to_numpy(), dtype=float)
        raw_m = np.asarray(sub[var_m].to_numpy(), dtype=float)

        ax_y = fig.add_subplot(len(class_ids), 2, 2 * row + 1)
        ax_m = fig.add_subplot(len(class_ids), 2, 2 * row + 2)

        for ax, raw, name, color in (
            (ax_y, raw_y, "Y (lecture)", "C0"),
            (ax_m, raw_m, "M (maths)", "C1"),
        ):
            vals = raw[np.isfinite(raw)]
            if len(vals) < 2:
                ax.text(0.5, 0.5, "données insuffisantes", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"classe {int(tid)} — {name}")
                continue
            mu, sig = float(np.mean(vals)), max(float(np.std(vals, ddof=1)), 1e-9)
            ax.hist(vals, bins=min(8, max(3, len(vals))), density=True, alpha=0.55, color=color, edgecolor="white")
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.02 * max(hi - lo, 1.0)
            xs = np.linspace(lo - pad, hi + pad, 100)
            ax.plot(xs, _norm_pdf(xs, mu, sig), color="darkred", ls="--", lw=1.6, label="Gaussienne")
            pos = vals[vals > 0]
            if len(pos) >= 2:
                s_ln, sc_ln = fit_lognorm_s_scale(pos)
                xs_p = np.linspace(max(pos.min(), 1e-6), pos.max(), 100)
                ax.plot(
                    xs_p,
                    scipy_lognorm.pdf(xs_p, s=s_ln, scale=sc_ln),
                    color="black",
                    lw=1.8,
                    label="log-normal",
                )
            ax.set_title(f"classe {int(tid)} — {name} (n={len(vals)})", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("score")
            ax.set_ylabel("densité")


def plot_class_means_gaussian_and_lognorm(ax: plt.Axes, sampled: pd.DataFrame, col: str, label: str) -> None:
    g = sampled.groupby("gktchid")[col].mean()
    vals = g.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=35, color="slateblue", alpha=0.65, edgecolor="white", density=True)
    mu, sig = float(np.mean(vals)), max(float(np.std(vals, ddof=1)), 1e-9)
    xs = np.linspace(vals.min(), vals.max(), 120)
    ax.plot(xs, _norm_pdf(xs, mu, sig), color="darkred", ls="--", lw=2, label="Gaussienne")
    if np.all(vals > 0):
        s_ln, sc_ln = fit_lognorm_s_scale(vals)
        xs_p = np.linspace(max(vals.min(), 1e-6), vals.max(), 120)
        ax.plot(xs_p, scipy_lognorm.pdf(xs_p, s=s_ln, scale=sc_ln), color="orange", lw=2, label="log-normal")
    ax.set_title(f"Moyennes par classe — {label}")
    ax.set_xlabel("moyenne élèves dans la classe")
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "score_distribution_diagnostics",
        help="Dossier de sortie des PNG.",
    )
    parser.add_argument("--n-classes", type=int, default=6, help="Nombre de classes illustrées au hasard.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sampled = load_balanced_sample()
    y_all = sampled["Y_read"].to_numpy()
    m_all = sampled["M_math"].to_numpy()
    s_unit = sampled.groupby("gktchid")["S_urbanicity"].first().to_numpy()

    fig0, axes0 = plt.subplots(2, 2, figsize=(10, 8))
    plot_global_gaussian_hist_qq(
        axes0[0, 0],
        axes0[0, 1],
        y_all,
        "Y = gktreadss — global : Gaussienne (référence modèle HCM)",
        qq_title="QQ normal (Y)",
    )
    plot_global_gaussian_hist_qq(
        axes0[1, 0],
        axes0[1, 1],
        m_all,
        "M = gktmathss — global : Gaussienne (référence modèle HCM)",
        qq_title="QQ normal (M)",
    )
    fig0.suptitle(
        "Référence tout gaussien (comme distribution_families Y,M dans le HCM)\n"
        f"(échantillon HCM v2 : {N_SUB_PER_CLASS} élèves / classe)",
        fontsize=11,
    )
    fig0.tight_layout()
    p0 = outdir / "global_Y_M_gaussian_hist_qq.png"
    fig0.savefig(p0, dpi=160)
    plt.close(fig0)

    fig_t, axes_t = plt.subplots(2, 3, figsize=(13.5, 8))
    df_y, loc_y, sc_y = plot_hist_normal_and_student_t(
        axes_t[0, 0],
        y_all,
        "Y = gktreadss — Normal vs Student-t (MLE)",
    )
    plot_qq_normal_only(axes_t[0, 1], y_all, "QQ normal (Y)")
    plot_qq_student_t(axes_t[0, 2], y_all, df_y, loc_y, sc_y, "QQ Student-t (Y)")
    df_m, loc_m, sc_m = plot_hist_normal_and_student_t(
        axes_t[1, 0],
        m_all,
        "M = gktmathss — Normal vs Student-t (MLE)",
    )
    plot_qq_normal_only(axes_t[1, 1], m_all, "QQ normal (M)")
    plot_qq_student_t(axes_t[1, 2], m_all, df_m, loc_m, sc_m, "QQ Student-t (M)")
    fig_t.suptitle(
        "Student-t : la t-MLE a souvent un ν grand sur des scores proches d’une Normal → courbes quasi identiques.\n"
        f"La t en pointillés (ν fixe, variance = σ²) montre des queues plus lourdes. {N_SUB_PER_CLASS} élèves / classe.",
        fontsize=10,
    )
    fig_t.tight_layout()
    p_t = outdir / "global_Y_M_normal_vs_student_t_hist_qq.png"
    fig_t.savefig(p_t, dpi=160)
    plt.close(fig_t)

    fig_b, axes_b = plt.subplots(2, 3, figsize=(15.0, 9.2), layout="constrained")
    _, _, il_y, ih_y, fb_y, mu_y, sig_y = plot_qq_gaussian_body_heavy_tail(
        axes_b[0, 0],
        y_all,
        "Y (lecture) — Q-Q standardisé + références à queues lourdes",
    )
    plot_hist_full_and_body_models(
        axes_b[0, 1],
        y_all,
        "Y — densités (LN sur le corps, comparé à N corps et modèles globaux)",
        il_y,
        ih_y,
        mu_y,
        sig_y,
    )
    axes_b[0, 2].axis("off")
    axes_b[0, 2].text(
        0.04,
        0.97,
        _format_aic_panel(y_all, "Y (lecture)"),
        transform=axes_b[0, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
    )
    _, _, il_m, ih_m, fb_m, mu_m, sig_m = plot_qq_gaussian_body_heavy_tail(
        axes_b[1, 0],
        m_all,
        "M (maths) — Q-Q standardisé + références à queues lourdes",
    )
    plot_hist_full_and_body_models(
        axes_b[1, 1],
        m_all,
        "M — densités (LN sur le corps, comparé à N corps et modèles globaux)",
        il_m,
        ih_m,
        mu_m,
        sig_m,
    )
    axes_b[1, 2].axis("off")
    axes_b[1, 2].text(
        0.04,
        0.97,
        _format_aic_panel(m_all, "M (maths)"),
        transform=axes_b[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
    )
    fb_note = ""
    if fb_y or fb_m:
        fb_note = " Fenêtre « corps» par quantiles 15–85 si peu de croisements y=x."
    fig_b.suptitle(
        "Sélection de modèles : Q-Q (identité + t), LN sur le corps, AIC global (Normal / LN / t)."
        + fb_note,
        fontsize=10,
    )
    p_b = outdir / "global_Y_M_gaussian_body_heavy_tail_qq_hist.png"
    fig_b.savefig(p_b, dpi=160)
    plt.close(fig_b)

    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_global_lognormal_hist_qq(
        axes[0, 0],
        axes[0, 1],
        y_all,
        "Y = gktreadss — global : log-normal (+ Gauss. en pointillés)",
    )
    plot_global_gaussian_hist_qq(
        axes[1, 0],
        axes[1, 1],
        m_all,
        "M = gktmathss — global : gaussienne",
        qq_title="QQ normal (M)",
    )
    fig1.suptitle(
        "Fits : Y log-normal, M gaussienne\n"
        f"(échantillon HCM v2 : {N_SUB_PER_CLASS} élèves / classe)",
        fontsize=11,
    )
    fig1.tight_layout()
    p1 = outdir / "global_Y_log_M_gauss_fits.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_s_distribution(ax2, s_unit, "S = gksurban au niveau classe (une valeur par classe)")
    fig2.suptitle(
        "Urbanicité : variable catégorielle / ordinale",
        fontsize=10,
    )
    fig2.tight_layout()
    p2 = outdir / "S_urbanicity_bar.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    rng = np.random.default_rng(args.seed)
    uids = sampled["gktchid"].unique().tolist()
    pick = rng.choice(np.array(uids, dtype=np.int64), size=min(args.n_classes, len(uids)), replace=False).tolist()
    nrows = len(pick)
    fig3 = plt.figure(figsize=(10, 2.4 * nrows))
    plot_per_class_gaussian_and_lognormal_rows(fig3, sampled, pick, "Y_read", "M_math")
    fig3.suptitle(
        f"Par classe : Gaussienne (--) + log-normal (—) (seed={args.seed}, n={N_SUB_PER_CLASS}/classe)",
        fontsize=10,
    )
    fig3.tight_layout()
    p3 = outdir / "per_class_Y_M_gaussian_lognormal_fits.png"
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)

    fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))
    plot_class_means_gaussian_and_lognorm(axes4[0], sampled, "Y_read", "Y lecture")
    plot_class_means_gaussian_and_lognorm(axes4[1], sampled, "M_math", "M maths")
    fig4.suptitle("Moyennes par classe : Gaussienne (--) + log-normal (—)", fontsize=11)
    fig4.tight_layout()
    p4 = outdir / "class_level_means_Y_M_gaussian_lognorm.png"
    fig4.savefig(p4, dpi=160)
    plt.close(fig4)

    print(f"Wrote: {p0}")
    print(f"Wrote: {p_t}")
    print(f"Wrote: {p_b}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")
    print(f"Wrote: {p4}")


if __name__ == "__main__":
    main()
