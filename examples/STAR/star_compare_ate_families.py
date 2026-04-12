"""
Compare les ATE HCM v2 sous plusieurs ``distribution_families`` pour ``Y`` et ``M`` :

- Gaussien (référence classique),
- Mélange gaussien **K=2** par école (``gaussian_mixture``, défaut du pipeline principal),
- Log-normal (``Y``, ``M`` > 0 requis),
- Beta min–max par école (``beta_unit_minmax``).

Le module ``star_hcm_v2_teacher_student`` utilise par défaut le GMM (2) pour ``Y`` et ``M`` ;
ce script garde une colonne **gaussien** explicite pour la comparaison.

Optionnel : sensibilité de ``S`` (``--s-tail-compare``).

Usage::

    PYTHONPATH=examples/STAR uv run python examples/STAR/star_compare_ate_families.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from star_hcm_v2_teacher_student import (
    DEFAULT_DISTRIBUTION_FAMILIES,
    DEFAULT_ESTIMATOR_KWARGS,
    N_MC_SAMPLES,
    REPO_ROOT,
    RESULTS_DIR,
    load_graph_specs,
    load_teacher_student_data,
    run_one_graph,
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

GMM2_YM_FAMILIES: dict[str, str] = {
    "A": "bernoulli",
    "Y": "gaussian_mixture",
    "M": "gaussian_mixture",
    "G": "bernoulli",
    "E": "bernoulli",
    "L": "bernoulli",
    "S": "categorical",
}

LOGNORMAL_YM_FAMILIES: dict[str, str] = {
    "A": "bernoulli",
    "Y": "lognormal",
    "M": "lognormal",
    "G": "bernoulli",
    "E": "bernoulli",
    "L": "bernoulli",
    "S": "categorical",
}

BETA_UNIT_MINMAX_YM_FAMILIES: dict[str, str] = {
    "A": "bernoulli",
    "Y": "beta_unit_minmax",
    "M": "beta_unit_minmax",
    "G": "bernoulli",
    "E": "bernoulli",
    "L": "bernoulli",
    "S": "categorical",
}


def _with_s(base: dict[str, str], s_family: str) -> dict[str, str]:
    d = dict(base)
    d["S"] = s_family
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("Y", "M"), default="M")
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Graphes séparés par des virgules (défaut : tous).",
    )
    parser.add_argument("--n-mc-samples", type=int, default=None)
    parser.add_argument(
        "--s-tail-compare",
        action="store_true",
        help="En plus : GMM2 YM avec S=gaussian puis S=student_t.",
    )
    args = parser.parse_args()

    specs = load_graph_specs()
    if args.graphs:
        want = {g.strip() for g in args.graphs.split(",") if g.strip()}
        specs = [s for s in specs if s.name in want]
        if not specs:
            raise SystemExit(f"Aucun graphe pour --graphs={args.graphs!r}")

    data, meta = load_teacher_student_data()
    y_flat = data["Y"].ravel()
    m_flat = data["M"].ravel()
    if args.outcome == "M" and (m_flat <= 0).any():
        raise SystemExit("Les scores M doivent être > 0 pour la famille lognormal.")
    if args.outcome == "Y" and (y_flat <= 0).any():
        raise SystemExit("Les scores Y doivent être > 0 pour la famille lognormal.")

    mc = args.n_mc_samples if args.n_mc_samples is not None else N_MC_SAMPLES
    rows: list[dict[str, object]] = []

    for spec in specs:
        g = spec.name
        r_g = run_one_graph(
            spec,
            data,
            outcome_subunit=str(args.outcome),
            distribution_families=dict(GAUSSIAN_YM_FAMILIES),
            n_mc_samples=mc,
        )
        r_gmm = run_one_graph(
            spec,
            data,
            outcome_subunit=str(args.outcome),
            distribution_families=dict(GMM2_YM_FAMILIES),
            estimator_kwargs=dict(DEFAULT_ESTIMATOR_KWARGS),
            n_mc_samples=mc,
        )
        r_ln = run_one_graph(
            spec,
            data,
            outcome_subunit=str(args.outcome),
            distribution_families=dict(LOGNORMAL_YM_FAMILIES),
            n_mc_samples=mc,
        )
        r_beta = run_one_graph(
            spec,
            data,
            outcome_subunit=str(args.outcome),
            distribution_families=dict(BETA_UNIT_MINMAX_YM_FAMILIES),
            n_mc_samples=mc,
        )
        ate_g = r_g.get("ATE")
        ate_gmm = r_gmm.get("ATE")
        ate_ln = r_ln.get("ATE")
        ate_b = r_beta.get("ATE")

        def _d(a: object, b: object) -> float | None:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(b) - float(a)
            return None

        row: dict[str, object] = {
            "graph": g,
            "gaussian_status": r_g.get("status"),
            "gmm2_status": r_gmm.get("status"),
            "lognormal_status": r_ln.get("status"),
            "beta_unit_minmax_status": r_beta.get("status"),
            "ATE_gaussian_YM": ate_g,
            "ATE_gmm2_YM": ate_gmm,
            "ATE_lognormal_YM": ate_ln,
            "ATE_beta_unit_minmax_YM": ate_b,
            "delta_gmm2_minus_gauss": _d(ate_g, ate_gmm),
            "delta_lognorm_minus_gauss": _d(ate_g, ate_ln),
            "delta_beta_unit_minmax_minus_gauss": _d(ate_g, ate_b),
            "E_do_1_gauss": r_g.get("E_do_1"),
            "E_do_0_gauss": r_g.get("E_do_0"),
            "E_do_1_gmm2": r_gmm.get("E_do_1"),
            "E_do_0_gmm2": r_gmm.get("E_do_0"),
            "E_do_1_logn": r_ln.get("E_do_1"),
            "E_do_0_logn": r_ln.get("E_do_0"),
            "E_do_1_beta_um": r_beta.get("E_do_1"),
            "E_do_0_beta_um": r_beta.get("E_do_0"),
            "error_gmm2": r_gmm.get("error") if r_gmm.get("status") != "ok" else None,
            "error_lognormal": r_ln.get("error") if r_ln.get("status") != "ok" else None,
            "error_beta_unit_minmax": r_beta.get("error") if r_beta.get("status") != "ok" else None,
        }

        if args.s_tail_compare:
            base_gmm = dict(GMM2_YM_FAMILIES)
            r_s_gauss = run_one_graph(
                spec,
                data,
                outcome_subunit=str(args.outcome),
                distribution_families=_with_s(base_gmm, "gaussian"),
                estimator_kwargs=dict(DEFAULT_ESTIMATOR_KWARGS),
                n_mc_samples=mc,
            )
            r_s_t = run_one_graph(
                spec,
                data,
                outcome_subunit=str(args.outcome),
                distribution_families=_with_s(base_gmm, "student_t"),
                estimator_kwargs=dict(DEFAULT_ESTIMATOR_KWARGS),
                n_mc_samples=mc,
            )
            row["ATE_gmm2_YM_S_gaussian"] = r_s_gauss.get("ATE")
            row["ATE_gmm2_YM_S_student_t"] = r_s_t.get("ATE")
            row["status_gmm2_S_gaussian"] = r_s_gauss.get("status")
            row["status_gmm2_S_student_t"] = r_s_t.get("status")

        rows.append(row)

        sg = "NA" if ate_g is None else f"{float(ate_g):.4f}"
        sgmm = "NA" if ate_gmm is None else f"{float(ate_gmm):.4f}"
        sl = "NA" if ate_ln is None else f"{float(ate_ln):.4f}"
        sb = "NA" if ate_b is None else f"{float(ate_b):.4f}"
        print(f"{g:<14} gauss={sg:<10} gmm2={sgmm:<10} logn={sl:<10} beta_um={sb:<10}")

    out = RESULTS_DIR / f"star_hcm_v2_ate_family_compare_outcome_{args.outcome}.json"
    payload: dict[str, object] = {
        "outcome": str(args.outcome),
        "n_mc_samples": mc,
        "families_gaussian_ym": GAUSSIAN_YM_FAMILIES,
        "families_gmm2_ym": GMM2_YM_FAMILIES,
        "estimator_kwargs_gmm2": DEFAULT_ESTIMATOR_KWARGS,
        "families_lognormal_ym": LOGNORMAL_YM_FAMILIES,
        "families_beta_unit_minmax_ym": BETA_UNIT_MINMAX_YM_FAMILIES,
        "default_pipeline_families_note": "star_hcm_v2_teacher_student.DEFAULT_DISTRIBUTION_FAMILIES vaut GMM2 pour Y,M.",
        "s_tail_compare": bool(args.s_tail_compare),
        "meta": meta,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
