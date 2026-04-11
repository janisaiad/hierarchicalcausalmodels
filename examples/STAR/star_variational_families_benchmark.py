"""STAR — benchmark HCM avec familles non gaussiennes (beta, gamma, mélange) vs baseline.

Compare ``estimate_causal_effect`` en backend ``numpy`` et ``numpyro`` sur les mêmes
graphes discovery que ``star_hcm_v2_teacher_student`` : les scores ``gktreadss`` ne sont
pas gaussiens ; on documente une approximation beta sur lecture renormalisée dans
$(0,1)$, une gamma sur scores décalés strictement positifs, et un mélange gaussien
comme alternative multi-modale.

Les ATE beta / gamma sont exprimés dans l’échelle transformée (voir colonnes ``scale``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_DIR = REPO_ROOT / "examples" / "STAR"
RESULTS_DIR = STAR_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(STAR_DIR) not in sys.path:
    sys.path.insert(0, str(STAR_DIR))

from hierarchicalcausalmodels.do_calculus import PYAGNUM_AVAILABLE
from star_hcm_v2_teacher_student import GraphSpec, load_graph_specs, load_teacher_student_data

OUT_JSON = RESULTS_DIR / "star_variational_families_benchmark.json"
OUT_MD = STAR_DIR / "star_variational_families_benchmark.md"


@dataclass(frozen=True)
class OutcomeScenario:
    key: str
    scale_note: str
    y_family: str


def _subsample_units(data: dict[str, np.ndarray], max_units: int) -> dict[str, np.ndarray]:
    n = int(data["A"].shape[0])
    m = min(max_units, n)
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        a = np.asarray(arr)
        if a.ndim == 2:
            out[key] = a[:m].copy()
        else:
            out[key] = a[:m].copy()
    return out


def _transform_data_for_y_family(data: dict[str, np.ndarray], scenario: OutcomeScenario) -> dict[str, np.ndarray]:
    base = {k: np.asarray(v, dtype=float).copy() for k, v in data.items()}
    y = base["Y"].astype(float)
    if scenario.key == "gaussian":
        return base
    flat = y.ravel()
    y_min = float(np.min(flat))
    y_max = float(np.max(flat))
    span = max(y_max - y_min, 1e-9)
    if scenario.key == "beta_unit":
        y01 = (y - y_min) / span
        y01 = np.clip(y01, 1e-4, 1.0 - 1e-4)
        base["Y"] = y01
        return base
    if scenario.key == "gamma_positive_shift":
        base["Y"] = y - y_min + 1e-3
        return base
    if scenario.key == "gaussian_mixture_raw":
        return base
    raise KeyError(scenario.key)


def _base_covariate_families() -> dict[str, str]:
    return {
        "A": "bernoulli",
        "M": "gaussian",
        "G": "bernoulli",
        "E": "bernoulli",
        "L": "bernoulli",
        "S": "gaussian",
    }


def _families_for_y(y_family: str) -> dict[str, str]:
    fam = _base_covariate_families()
    fam["Y"] = y_family
    fam["Q^{y|a}"] = y_family
    return fam


def run_one_graph(
    spec: GraphSpec,
    data: dict[str, np.ndarray],
    families: dict[str, str],
    *,
    estimator_backend: str,
    estimator_kwargs: dict[str, Any] | None,
    n_mc_samples: int,
    n_jobs: int,
    parallel_backend: str,
) -> dict[str, Any]:
    from star_hcm_v2_teacher_student import build_hscm, graph_to_cgm_for_effect
    from hierarchicalcausalmodels.do_calculus import identify_effect
    from hierarchicalcausalmodels.estimation import estimate_causal_effect
    import networkx as nx

    n_units, n_sub = data["A"].shape
    hscm = build_hscm(spec, n_units, n_sub)
    result: dict[str, Any] = {
        "graph_name": spec.name,
        "note": spec.note,
        "dag_ok": bool(nx.is_directed_acyclic_graph(hscm.cgm.dag)),
    }
    if not result["dag_ok"]:
        result["status"] = "invalid_dag"
        return result
    try:
        cgm, y_node, _x_node = graph_to_cgm_for_effect(hscm)
        id_result = identify_effect(cgm, Y=y_node, X={"Q^a"}, unobserved={"U"})
        result["identifiable"] = bool(id_result.identifiable)
        result["formula_latex"] = id_result.formula_latex
        if not id_result.identifiable:
            result["status"] = "not_identifiable"
            return result
        ey1 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 1.0},
            distribution_families=families,
            n_mc_samples=n_mc_samples,
            random_seed=42,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,  # type: ignore[arg-type]
            estimator_backend=estimator_backend,
            estimator_kwargs=estimator_kwargs,
        )
        ey0 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 0.0},
            distribution_families=families,
            n_mc_samples=n_mc_samples,
            random_seed=43,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,  # type: ignore[arg-type]
            estimator_backend=estimator_backend,
            estimator_kwargs=estimator_kwargs,
        )
        result["status"] = "ok"
        result["E_do_1"] = float(ey1)
        result["E_do_0"] = float(ey0)
        result["ATE"] = float(ey1 - ey0)
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result


def _numpyro_estimator_kwargs(quick: bool) -> dict[str, Any]:
    if quick:
        return {
            "__default__": {
                "device": "cpu",
                "num_steps": 220,
                "learning_rate": 0.025,
                "num_posterior_samples": 64,
            },
            "gaussian_mixture": {"n_components": 2, "num_steps": 280, "num_posterior_samples": 64},
        }
    return {
        "__default__": {
            "device": "cpu",
            "num_steps": 350,
            "learning_rate": 0.02,
            "num_posterior_samples": 96,
        },
        "gaussian_mixture": {"n_components": 2},
    }


def build_scenarios() -> list[OutcomeScenario]:
    return [
        OutcomeScenario("gaussian", "gktreadss brut (échelle score enseignant)", "gaussian"),
        OutcomeScenario(
            "beta_unit",
            "Y -> (Y-min)/(max-min) dans (0,1) ; ATE sur cette échelle",
            "beta",
        ),
        OutcomeScenario(
            "gamma_positive_shift",
            "Y -> Y - min(Y) + 1e-3 ; ATE sur scores décalés",
            "gamma",
        ),
        OutcomeScenario(
            "gaussian_mixture_raw",
            "gktreadss brut ; Y modélisé comme mélange gaussien (K=2)",
            "gaussian_mixture",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="STAR variational / non-Gaussian HCM benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="moins d’unités, moins d’échantillons MC, SVI plus court",
    )
    parser.add_argument("--max-units", type=int, default=0, help="0 = toutes les classes valides")
    parser.add_argument(
        "--graphs",
        type=str,
        default="ConsensusMean,PC,DirectLiNGAM",
        help="noms de graphes séparés par des virgules (ordre conservé ; voir load_graph_specs)",
    )
    args = parser.parse_args()

    if not PYAGNUM_AVAILABLE:
        raise RuntimeError("pyAgrum est requis pour ce benchmark.")

    from hierarchicalcausalmodels.estimation import NUMPYRO_AVAILABLE

    data_full, meta = load_teacher_student_data()
    max_units = args.max_units if args.max_units > 0 else int(data_full["A"].shape[0])
    if args.quick:
        max_units = min(max_units, 28)
    data = _subsample_units(data_full, max_units)

    wanted = [name.strip() for name in args.graphs.split(",") if name.strip()]
    specs_all = load_graph_specs()
    by_name = {s.name: s for s in specs_all}
    if wanted:
        specs = []
        for w in wanted:
            if w not in by_name:
                raise ValueError(f"Graphe inconnu {w!r}. Disponibles : {sorted(by_name)}")
            specs.append(by_name[w])
    else:
        specs = list(specs_all)

    n_mc = 36 if args.quick else 56
    n_jobs_np = min(6, max(1, data["A"].shape[0] // 4)) if not args.quick else 2
    vi_kw = _numpyro_estimator_kwargs(bool(args.quick))

    scenarios = build_scenarios()
    rows: list[dict[str, Any]] = []

    for scen in scenarios:
        data_scen = _transform_data_for_y_family(data, scen)
        fam = _families_for_y(scen.y_family)
        for spec in specs:
            t0 = time.perf_counter()
            row_np = run_one_graph(
                spec,
                data_scen,
                fam,
                estimator_backend="numpy",
                estimator_kwargs=None,
                n_mc_samples=n_mc,
                n_jobs=n_jobs_np,
                parallel_backend="threads",
            )
            elapsed_np = time.perf_counter() - t0
            row_np["backend"] = "numpy"
            row_np["outcome_scenario"] = scen.key
            row_np["y_family"] = scen.y_family
            row_np["scale_note"] = scen.scale_note
            row_np["elapsed_seconds"] = elapsed_np
            rows.append(row_np)

            if NUMPYRO_AVAILABLE and scen.y_family in {"beta", "gamma", "gaussian_mixture"}:
                t1 = time.perf_counter()
                row_vi = run_one_graph(
                    spec,
                    data_scen,
                    fam,
                    estimator_backend="numpyro",
                    estimator_kwargs=vi_kw,
                    n_mc_samples=n_mc,
                    n_jobs=1,
                    parallel_backend="processes",
                )
                elapsed_vi = time.perf_counter() - t1
                row_vi["backend"] = "numpyro"
                row_vi["outcome_scenario"] = scen.key
                row_vi["y_family"] = scen.y_family
                row_vi["scale_note"] = scen.scale_note
                row_vi["elapsed_seconds"] = elapsed_vi
                rows.append(row_vi)
                if row_np.get("status") == "ok" and row_vi.get("status") == "ok":
                    row_vi["abs_diff_ate_vs_numpy"] = float(abs(row_vi["ATE"] - row_np["ATE"]))
                    row_vi["abs_diff_edo1_vs_numpy"] = float(abs(row_vi["E_do_1"] - row_np["E_do_1"]))
                    row_vi["abs_diff_edo0_vs_numpy"] = float(abs(row_vi["E_do_0"] - row_np["E_do_0"]))
                else:
                    row_vi["abs_diff_ate_vs_numpy"] = None
                    row_vi["abs_diff_edo1_vs_numpy"] = None
                    row_vi["abs_diff_edo0_vs_numpy"] = None

    payload: dict[str, Any] = {
        "meta": {
            **meta,
            "max_units": int(data["A"].shape[0]),
            "quick": bool(args.quick),
            "graphs_requested": args.graphs,
            "numpyro_available": bool(NUMPYRO_AVAILABLE),
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(_build_markdown(payload))
    print(f"Écrit : {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Écrit : {OUT_MD.relative_to(REPO_ROOT)}")


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# STAR — benchmark familles non gaussiennes (numpy vs numpyro)")
    lines.append("")
    lines.append("## Données et échelles")
    lines.append("")
    m = payload["meta"]
    lines.append(f"- unités (classes) : `{m.get('max_units')}`")
    lines.append(f"- quick : `{m.get('quick')}`")
    lines.append(f"- NumPyro disponible : `{m.get('numpyro_available')}`")
    lines.append("")
    lines.append("## Tableau comparatif")
    lines.append("")
    lines.append(
        "| Graphe | Scénario | Famille Y | Backend | Statut | E[do(1)] | ATE | Temps (s) | max(|ΔE_do|) vs numpy | erreur (extrait) |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        k = (row["graph_name"], row["outcome_scenario"], row["backend"])
        by_key[k] = row
    for row in payload["rows"]:
        if row["backend"] != "numpy":
            continue
        e1 = "" if row.get("E_do_1") is None else f"{row['E_do_1']:.5f}"
        ate = "" if row.get("ATE") is None else f"{row['ATE']:.5f}"
        diff = ""
        vi = by_key.get((row["graph_name"], row["outcome_scenario"], "numpyro"))
        if vi is not None and vi.get("abs_diff_edo1_vs_numpy") is not None:
            mde = max(float(vi["abs_diff_edo1_vs_numpy"]), float(vi["abs_diff_edo0_vs_numpy"]))
            diff = f"{mde:.5f}"
        err_np = (row.get("error") or "")[:70].replace("|", "/")
        lines.append(
            f"| `{row['graph_name']}` | `{row['outcome_scenario']}` | `{row['y_family']}` | numpy | "
            f"`{row.get('status')}` | {e1} | {ate} | {row['elapsed_seconds']:.2f} | | {err_np} |"
        )
        if vi is not None:
            e1v = "" if vi.get("E_do_1") is None else f"{vi['E_do_1']:.5f}"
            ate_v = "" if vi.get("ATE") is None else f"{vi['ATE']:.5f}"
            err_vi = (vi.get("error") or "")[:70].replace("|", "/")
            lines.append(
                f"| `{row['graph_name']}` | `{row['outcome_scenario']}` | `{row['y_family']}` | numpyro | "
                f"`{vi.get('status')}` | {e1v} | {ate_v} | {vi['elapsed_seconds']:.2f} | {diff} | {err_vi} |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- La colonne ATE pour `beta_unit` est sur le score de lecture linéairement renormalisé dans $(0,1)$, "
        "pas sur les points bruts du test."
    )
    lines.append(
        "- Pour `gamma_positive_shift`, l’ATE est sur $Y - \\min(Y) + \\epsilon$ (ordre de grandeur comparable au brut)."
    )
    lines.append(
        "- `gaussian_mixture` reste le plus sensible à l’optimisation variationnelle ; comparer surtout l’ordre de grandeur avec numpy."
    )
    lines.append(
        "- Sur les graphes issus de la découverte causale packagée ici, la formule identifiée est souvent réduite à "
        "$P(Q^y)$ sans dépendance effective à $do(Q^a)$ : l’ATE est alors nul alors que $E[\\mathrm{do}(1)]$ reste "
        "l’estimande principal à comparer entre backends."
    )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    main()
