"""
Graphiques post-estimation pour les morceaux **gaussiens** du pipeline HCM STAR
(``ConditionalDensityEstimator`` famille ``gaussian``, et résumés $Q$ à 2 paramètres).

Utilise ``estimate_causal_effect(..., return_artifacts=True)`` sur un graphe / outcome
choisis (défaut : DirectLiNGAM, lecture, ``do(Q^a)=1``).

Sorties : ``examples/STAR/figures/gaussian_postfit_diagnostics/``

- ``enriched_q_*.png`` : pour chaque $Q$ à 2 paramètres (ex. ``Q^y``, ``Q^m`` sous
  famille gaussienne au niveau classe), nuage $(\\hat\\mu, \\hat\\sigma^2)$ sur les 322 classes.
- ``q_density_*.png`` : échantillons utilisés par les ``QDensityEstimator`` du
  graphe (KDE dans l’espace des paramètres).
- ``cond_gaussian_*.png`` : si un terme $P(Y|X)$ est réellement ajusté en gaussien
  (OLS + résidus) — souvent absent sur LiNGAM/BIC où dominent ``QDensity`` /
  ``categorical`` / ``nonparametric``.

Usage::

    PYTHONPATH=examples/STAR uv run python examples/STAR/star_plot_gaussian_postfit_diagnostics.py
    PYTHONPATH=examples/STAR uv run python examples/STAR/star_plot_gaussian_postfit_diagnostics.py --graph ExactBIC --outcome M --do 0
    PYTHONPATH=examples/STAR uv run python examples/STAR/star_plot_gaussian_postfit_diagnostics.py --run-all --do 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None  # type: ignore[assignment]

from hierarchicalcausalmodels.do_calculus import identify_effect
from hierarchicalcausalmodels.estimation import (
    ConditionalDensityEstimator,
    QDensityEstimator,
    estimate_causal_effect,
)

from star_hcm_v2_teacher_student import (
    N_MC_SAMPLES,
    RANDOM_STATE,
    REPO_ROOT,
    build_hscm,
    graph_to_cgm_for_effect,
    load_graph_specs,
    load_teacher_student_data,
)

GAUSSIAN_YM_FAMILIES: dict[str, str] = {
    "A": "bernoulli",
    "Y": "gaussian",
    "M": "gaussian",
    "G": "bernoulli",
    "E": "bernoulli",
    "L": "bernoulli",
    "S": "categorical",
}


def _key_slug(key: tuple[object, ...]) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", repr(key))[:120]
    h = hashlib.md5(repr(key).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"{s}_{h}"


def _plot_conditional_gaussian(
    est: ConditionalDensityEstimator,
    title: str,
    out_path: Path,
) -> None:
    Y = np.asarray(est._Y, dtype=float).ravel()
    X = est._X
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panneau A : diagnostic régression (toujours pertinent)
    if X is not None and X.size > 0 and getattr(est, "_lr_gauss", None) is not None:
        lr = est._lr_gauss
        y_hat = np.asarray(lr.predict(X), dtype=float).ravel()
        ax = axes[0]
        ax.scatter(y_hat, Y, alpha=0.25, s=12, edgecolors="none")
        lo = float(min(Y.min(), y_hat.min()))
        hi = float(max(Y.max(), y_hat.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = ŷ")
        ax.set_xlabel("ŷ = E[Y|X] (OLS gaussien)")
        ax.set_ylabel("Y observé (niveau classe)")
        ax.set_title("Observé vs prédit")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        res = Y - y_hat
        sig = float(getattr(est, "_sigma", np.std(res) + 1e-9))
        ax = axes[1]
        ax.hist(res, bins=35, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        if sp_stats is not None:
            xs = np.linspace(res.min(), res.max(), 200)
            ax.plot(xs, sp_stats.norm.pdf(xs, loc=0.0, scale=sig), "r-", lw=2, label=f"N(0, σ²), σ={sig:.3g}")
        ax.set_title("Résidus (Y − ŷ)")
        ax.set_xlabel("résidu")
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif X is not None and X.size > 0 and getattr(est, "_torch_gaussian_state", None) is not None:
        axes[0].text(0.5, 0.5, "Backend torch — utiliser numpy pour ces plots", ha="center", va="center")
        axes[1].axis("off")
    else:
        # Marginal gaussien (sans régression multivariée sklearn sur ce terme)
        mu = float(getattr(est, "_mu_marginal", np.mean(Y)))
        sig = float(getattr(est, "_sigma", np.std(Y) + 1e-9))
        ax = axes[0]
        ax.hist(Y, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        if sp_stats is not None:
            xs = np.linspace(Y.min(), Y.max(), 200)
            ax.plot(xs, sp_stats.norm.pdf(xs, loc=mu, scale=sig), "r-", lw=2, label=f"N({mu:.2f}, {sig:.2f}²)")
        ax.set_xlabel("Y")
        ax.set_title("Histogramme marginal + gaussienne ajustée")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        if sp_stats is not None and len(Y) > 5:
            sp_stats.probplot(Y, dist="norm", plot=ax)
            ax.set_title("QQ-plot vs normal")
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_univariate_conditional_scatter(est: ConditionalDensityEstimator, out_path: Path) -> None:
    """Si une seule colonne X et OLS : nuage + droite."""
    Y = np.asarray(est._Y, dtype=float).ravel()
    X = est._X
    if X is None or X.shape[1] != 1 or getattr(est, "_lr_gauss", None) is None:
        return
    lr = est._lr_gauss
    x0 = X[:, 0]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x0, Y, alpha=0.28, s=14, edgecolors="none")
    grid = np.linspace(float(x0.min()), float(x0.max()), 120)
    ax.plot(grid, lr.predict(grid.reshape(-1, 1)), "r-", lw=2, label="OLS μ(x)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Ajustement gaussien conditionnel (μ linéaire)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_q_bivariate_params(arr: np.ndarray, name: str, out_path: Path) -> None:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(a[:, 0], bins=35, alpha=0.7, edgecolor="white")
        ax.set_title(f"{name} (1 param / scalaire par classe)")
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter(a[:, 0], a[:, 1], alpha=0.35, s=16, edgecolors="none")
        ax.set_xlabel("param 0 (ex. μ̂)")
        ax.set_ylabel("param 1 (ex. σ²̂)")
        ax.set_title(f"{name} — Q par classe (famille gaussienne → 2 params)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _run_one_case(
    *,
    graph: str,
    outcome: str,
    do_val: int,
    mc: int,
    data: dict,
    meta: dict,
    specs: List[Any],
) -> None:
    spec = next((s for s in specs if s.name == graph), None)
    if spec is None:
        raise SystemExit(f"Graphe inconnu : {graph!r}")

    n_units, n_sub = data["A"].shape
    hscm = build_hscm(spec, n_units, n_sub)
    cgm, y_node, x_node = graph_to_cgm_for_effect(hscm, outcome_subunit=str(outcome))
    id_result = identify_effect(cgm, Y=y_node, X=x_node, unobserved={"U"})
    if not id_result.identifiable:
        print(f"[skip] {graph} outcome={outcome}: non identifiable")
        return

    iv = {"Q^a": float(do_val)}
    seed = int(RANDOM_STATE) if do_val == 1 else int(RANDOM_STATE + 1)

    val, art = estimate_causal_effect(
        id_result,
        data=data,
        intervention=iv,
        distribution_families=dict(GAUSSIAN_YM_FAMILIES),
        n_mc_samples=mc,
        random_seed=seed,
        estimator_backend="numpy",
        return_artifacts=True,
    )

    out_dir = (
        REPO_ROOT
        / "examples"
        / "STAR"
        / "figures"
        / "gaussian_postfit_diagnostics"
        / f"{graph}_outcome_{outcome}_do{do_val}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted = art["fitted"]
    enriched = art["enriched"]

    for key, est in fitted.items():
        if isinstance(est, ConditionalDensityEstimator) and est.family == "gaussian":
            slug = _key_slug(key)
            title = f"{graph} outcome={outcome} do={do_val}\n{repr(key)}"
            _plot_conditional_gaussian(est, title, out_dir / f"cond_gaussian_{slug}.png")
            _plot_univariate_conditional_scatter(est, out_dir / f"cond_gaussian_scatter1d_{slug}.png")
        elif isinstance(est, QDensityEstimator):
            slug = _key_slug(key)
            qs = getattr(est, "_q_samples", None)
            if qs is not None:
                _plot_q_bivariate_params(
                    np.asarray(qs),
                    f"QDensity samples {repr(key)}",
                    out_dir / f"q_density_{slug}.png",
                )

    for k, v in enriched.items():
        if "Q^" not in k and "Q^{" not in str(k):
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", k)[:80]
        _plot_q_bivariate_params(arr, k, out_dir / f"enriched_q_{safe}.png")

    idx = {
        "graph": graph,
        "outcome": outcome,
        "do_Qa": do_val,
        "estimated_scalar": float(val),
        "n_mc_samples": mc,
        "random_seed": seed,
        "figures_dir": str(out_dir.relative_to(REPO_ROOT)),
        "data_meta": {kk: meta[kk] for kk in ("n_units_classes", "balanced_students_per_class", "n_rows_after_dropna") if kk in meta},
    }
    (out_dir / "plot_run_index.json").write_text(
        json.dumps(idx, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{graph} outcome={outcome} do={do_val}: E[·|do]={val:.6g} → {out_dir.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="DirectLiNGAM")
    parser.add_argument("--outcome", choices=("Y", "M"), default="Y")
    parser.add_argument("--do", type=int, default=1, choices=(0, 1), help="Intervention Q^a")
    parser.add_argument("--n-mc-samples", type=int, default=None)
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="DirectLiNGAM et ExactBIC × outcomes Y et M (do tel que --do).",
    )
    args = parser.parse_args()

    mc = int(args.n_mc_samples) if args.n_mc_samples is not None else int(N_MC_SAMPLES)
    specs = load_graph_specs()
    data, meta = load_teacher_student_data()

    if args.run_all:
        for g in ("DirectLiNGAM", "ExactBIC"):
            for oc in ("Y", "M"):
                _run_one_case(
                    graph=g,
                    outcome=oc,
                    do_val=int(args.do),
                    mc=mc,
                    data=data,
                    meta=meta,
                    specs=specs,
                )
        return

    _run_one_case(
        graph=str(args.graph),
        outcome=str(args.outcome),
        do_val=int(args.do),
        mc=mc,
        data=data,
        meta=meta,
        specs=specs,
    )


if __name__ == "__main__":
    main()
