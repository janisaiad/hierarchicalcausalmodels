"""
Exporte les densités / estimateurs appris pour DirectLiNGAM et ExactBIC
(Y et M en gaussien pour Y,M + S catégoriel), pour ``do(Q^a)=1`` et ``do(Q^a)=0``.

Les $Q$ conditionnels sont précalculés **par niveau d'intervention** : deux jeux
de fichiers par (graphe, outcome) — un pour ``do1``, un pour ``do0``.

Sorties (par sous-dossier) :

- ``meta.json`` : ATE implicite, E_do, hyperparamètres, chemins.
- ``enriched.npz`` : toutes les entrées du dict ``enriched`` (arrays numpy).
- ``bundle.pkl`` : ``pickle`` de ``{fitted, formula, unique_terms, ...}`` (objets
  Python ; recharger uniquement dans un env compatible).
- ``fitted_summary.json`` : aperçu lisible des paramètres des
  ``ConditionalDensityEstimator`` / ``QDensityEstimator``.

Usage::

    PYTHONPATH=examples/STAR uv run python examples/STAR/star_export_lingam_bic_gaussian_artifacts.py
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

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
    RESULTS_DIR,
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

TARGET_GRAPHS = ("DirectLiNGAM", "ExactBIC")


def _sanitize_npz_key(k: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]+", "_", k).strip("_")
    return out or "key"


def _summarize_estimator(est: Any) -> dict[str, Any]:
    if isinstance(est, ConditionalDensityEstimator):
        row: dict[str, Any] = {
            "class": "ConditionalDensityEstimator",
            "family": est.family,
            "backend": est.backend,
            "n": getattr(est, "_n", None),
        }
        if getattr(est, "_lr_gauss", None) is not None:
            lr = est._lr_gauss
            row["gaussian_conditional"] = {
                "coef": np.asarray(lr.coef_).ravel().tolist(),
                "intercept": float(lr.intercept_),
                "sigma_residual": float(getattr(est, "_sigma", float("nan"))),
            }
        elif hasattr(est, "_mu_marginal"):
            row["gaussian_marginal_or_fallback"] = {
                "mu": float(getattr(est, "_mu_marginal", float("nan"))),
                "sigma": float(getattr(est, "_sigma", float("nan"))),
            }
        if getattr(est, "_lr_model", None) is not None:
            m = est._lr_model
            if hasattr(m, "coef_"):
                row["lr_coef"] = np.asarray(m.coef_).ravel().tolist()
                row["lr_intercept"] = float(getattr(m, "intercept_", [0.0])[0])
        if getattr(est, "_p_marginal", None) is not None:
            row["p_marginal"] = float(est._p_marginal)
        cond = getattr(est, "_hcm_cond_parent_keys", None)
        if cond is not None:
            row["cond_parent_keys"] = [list(t) for t in cond]
        return row
    if isinstance(est, QDensityEstimator):
        q = getattr(est, "_q_samples", None)
        reg = getattr(est, "_regressor", None)
        out: dict[str, Any] = {
            "class": "QDensityEstimator",
            "bandwidth": est.bandwidth,
            "q_samples_shape": None if q is None else list(np.asarray(q).shape),
            "has_kde": getattr(est, "_kde", None) is not None,
            "fallback_mean": None
            if getattr(est, "_fallback_mean", None) is None
            else np.asarray(est._fallback_mean).tolist(),
            "fallback_std": None
            if getattr(est, "_fallback_std", None) is None
            else np.asarray(est._fallback_std).tolist(),
        }
        if reg is not None and hasattr(reg, "coef_"):
            out["q_on_x_coef"] = np.asarray(reg.coef_).tolist()
            out["q_on_x_intercept"] = np.asarray(reg.intercept_).tolist()
        return out
    return {"class": type(est).__name__, "repr": repr(est)[:500]}


def _fitted_summary(fitted: dict[tuple[Any, ...], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, est in fitted.items():
        sk = repr(key)
        out[sk] = _summarize_estimator(est)
    return out


def main() -> None:
    out_root = RESULTS_DIR / "star_lingam_bic_gaussian_artifacts"
    out_root.mkdir(parents=True, exist_ok=True)

    specs = [s for s in load_graph_specs() if s.name in TARGET_GRAPHS]
    if len(specs) != len(TARGET_GRAPHS):
        found = {s.name for s in specs}
        missing = set(TARGET_GRAPHS) - found
        raise SystemExit(f"Graphes manquants dans causal_discovery : {missing}")

    data, meta = load_teacher_student_data()
    n_units, n_sub = data["A"].shape

    index: list[dict[str, Any]] = []

    for spec in specs:
        for outcome in ("Y", "M"):
            hscm = build_hscm(spec, n_units, n_sub)
            cgm, y_node, x_node = graph_to_cgm_for_effect(hscm, outcome_subunit=outcome)
            id_result = identify_effect(cgm, Y=y_node, X=x_node, unobserved={"U"})
            if not id_result.identifiable:
                print(f"[skip] {spec.name} outcome={outcome} not identifiable")
                continue

            runs: dict[str, float] = {}
            artifacts_by_do: dict[str, dict[str, Any]] = {}

            for tag, iv, seed_off in (
                ("do1", {"Q^a": 1.0}, 0),
                ("do0", {"Q^a": 0.0}, 1),
            ):
                val, art = estimate_causal_effect(
                    id_result,
                    data=data,
                    intervention=iv,
                    distribution_families=dict(GAUSSIAN_YM_FAMILIES),
                    n_mc_samples=int(N_MC_SAMPLES),
                    random_seed=int(RANDOM_STATE + seed_off),
                    estimator_backend="numpy",
                    return_artifacts=True,
                )
                runs[tag] = float(val)
                artifacts_by_do[tag] = art

            ate = runs["do1"] - runs["do0"]
            sub = out_root / f"{spec.name}_outcome_{outcome}"
            sub.mkdir(parents=True, exist_ok=True)

            for tag in ("do1", "do0"):
                art = artifacts_by_do[tag]
                tdir = sub / tag
                tdir.mkdir(parents=True, exist_ok=True)

                enriched = art["enriched"]
                npz_kw = {_sanitize_npz_key(k): np.asarray(v) for k, v in enriched.items()}
                np.savez_compressed(tdir / "enriched.npz", **npz_kw)
                (tdir / "enriched_keys.json").write_text(
                    json.dumps(list(enriched.keys()), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                pkl_payload = {
                    "fitted": art["fitted"],
                    "formula": art["formula"],
                    "unique_terms": art["unique_terms"],
                    "distribution_families": art["distribution_families"],
                    "intervention_map": art["intervention_map"],
                    "intervention_iv_scalar": art["intervention_iv_scalar"],
                    "n_mc_samples": art["n_mc_samples"],
                    "random_seed": art["random_seed"],
                }
                with open(tdir / "bundle.pkl", "wb") as f:
                    pickle.dump(pkl_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                summ = _fitted_summary(art["fitted"])
                (tdir / "fitted_summary.json").write_text(
                    json.dumps(summ, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            meta_run = {
                "graph": spec.name,
                "outcome_subunit": outcome,
                "distribution_families": GAUSSIAN_YM_FAMILIES,
                "n_mc_samples": N_MC_SAMPLES,
                "random_seed_base": RANDOM_STATE,
                "E_do_1": runs["do1"],
                "E_do_0": runs["do0"],
                "ATE": ate,
                "formula_latex": id_result.formula_latex,
                "paths": {
                    "do1": str((sub / "do1").relative_to(REPO_ROOT)),
                    "do0": str((sub / "do0").relative_to(REPO_ROOT)),
                },
            }
            (sub / "meta.json").write_text(
                json.dumps(meta_run, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            index.append(meta_run)
            print(f"Wrote {sub.relative_to(REPO_ROOT)}  ATE={ate:.6g}")

    (out_root / "index.json").write_text(
        json.dumps(
            {"meta_data": meta, "runs": index},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Index: {out_root.relative_to(REPO_ROOT)}/index.json")


if __name__ == "__main__":
    main()
