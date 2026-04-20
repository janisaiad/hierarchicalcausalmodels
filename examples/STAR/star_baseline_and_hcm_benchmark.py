from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS

from hierarchicalcausalmodels.do_calculus import PYAGNUM_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_DIR = REPO_ROOT / "examples" / "STAR"
RESULTS_DIR = STAR_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(STAR_DIR) not in sys.path:
    sys.path.insert(0, str(STAR_DIR))

from star_hcm_v2_teacher_student import GraphSpec, load_graph_specs, load_teacher_student_data

RAW_TAB = STAR_DIR / "STA-207" / "STAR_Students.tab"


@dataclass
class HcmFamilySpec:
    name: str
    families: dict[str, str]
    note: str


def load_kindergarten_student_level() -> pd.DataFrame:
    raw = pd.read_csv(RAW_TAB, sep="\t", low_memory=False)
    cols = [
        "stdntid",
        "gender",
        "race",
        "gkclasstype",
        "gkschid",
        "gksurban",
        "gktchid",
        "gkfreelunch",
        "gktreadss",
        "gktmathss",
        "gkclasssize",
        "gktrace",
        "gktyears",
        "gkthighdegree",
    ]
    df = raw[cols].dropna().copy().reset_index(drop=True)
    df["small"] = (df["gkclasstype"] == 1.0).astype(float)
    df["reg_aide"] = (df["gkclasstype"] == 3.0).astype(float)
    df["female"] = (df["gender"] == 2.0).astype(float)
    df["white_asian"] = df["race"].isin([1.0, 3.0]).astype(float)
    df["free_lunch"] = (df["gkfreelunch"] == 1.0).astype(float)
    df["white_teacher"] = (df["gktrace"] == 1.0).astype(float)
    df["masters_plus"] = df["gkthighdegree"].isin([4.0, 5.0]).astype(float)
    return df


def fit_ols_and_rf_like_startenesse(df: pd.DataFrame, outcome_col: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_schools": int(df["gkschid"].nunique()),
        "n_classes": int(df["gktchid"].nunique()),
    }

    formulas = {
        "ols_minimal": f"{outcome_col} ~ small + reg_aide",
        "ols_school_fe": f"{outcome_col} ~ small + reg_aide + C(gkschid)",
        "ols_with_student_controls": f"{outcome_col} ~ small + reg_aide + white_asian + female + free_lunch + C(gkschid)",
        "ols_full_like_startenesse": f"{outcome_col} ~ small + reg_aide + white_asian + female + free_lunch + white_teacher + gktyears + masters_plus + C(gkschid)",
    }

    fits: dict[str, Any] = {}
    for name, formula in formulas.items():
        t0 = time.perf_counter()
        fit = smf.ols(formula, data=df).fit(cov_type="HC1")
        elapsed = time.perf_counter() - t0
        fits[name] = {
            "formula": formula,
            "elapsed_seconds": elapsed,
            "nobs": int(fit.nobs),
            "r2": float(fit.rsquared),
            "coef_small": float(fit.params.get("small", np.nan)),
            "se_small": float(fit.bse.get("small", np.nan)),
            "p_small": float(fit.pvalues.get("small", np.nan)),
            "coef_reg_aide": float(fit.params.get("reg_aide", np.nan)),
            "se_reg_aide": float(fit.bse.get("reg_aide", np.nan)),
            "p_reg_aide": float(fit.pvalues.get("reg_aide", np.nan)),
        }
    results["ols"] = fits
    return results


def fit_iv_like_startenesse(df: pd.DataFrame, outcome_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
    }

    exog = pd.DataFrame(
        {
            "const": 1.0,
            "white_asian": df["white_asian"].to_numpy(dtype=float),
            "female": df["female"].to_numpy(dtype=float),
            "free_lunch": df["free_lunch"].to_numpy(dtype=float),
            "white_teacher": df["white_teacher"].to_numpy(dtype=float),
            "gktyears": df["gktyears"].to_numpy(dtype=float),
            "masters_plus": df["masters_plus"].to_numpy(dtype=float),
        }
    )
    school_dummies = pd.get_dummies(df["gkschid"].astype(int).astype(str), prefix="school", drop_first=True, dtype=float)
    exog_with_endog = pd.concat(
        [
            exog,
            school_dummies,
            pd.DataFrame({"gkclasssize": df["gkclasssize"].to_numpy(dtype=float)}),
        ],
        axis=1,
    )
    instruments = pd.concat(
        [
            exog,
            school_dummies,
            pd.DataFrame(
                {
                    "small": df["small"].to_numpy(dtype=float),
                    "reg_aide": df["reg_aide"].to_numpy(dtype=float),
                }
            ),
        ],
        axis=1,
    )
    y = df[outcome_col].to_numpy(dtype=float)

    t0 = time.perf_counter()
    fit = IV2SLS(y, exog_with_endog.to_numpy(dtype=float), instruments.to_numpy(dtype=float)).fit()
    elapsed = time.perf_counter() - t0

    param_names = list(exog_with_endog.columns)
    idx_classsize = param_names.index("gkclasssize")
    out["iv_2sls"] = {
        "elapsed_seconds": elapsed,
        "nobs": int(len(df)),
        "coef_classsize": float(fit.params[idx_classsize]),
        "se_classsize": float(fit.bse[idx_classsize]),
        "t_classsize": float(fit.tvalues[idx_classsize]),
        "p_classsize": float(fit.pvalues[idx_classsize]),
        "instrument_names": ["small", "reg_aide"],
    }
    return out


def run_hcm_family_benchmarks(data: dict[str, np.ndarray], specs: list[GraphSpec]) -> dict[str, Any]:
    if not PYAGNUM_AVAILABLE:
        raise RuntimeError("pyAgrum is required for HCM benchmarking.")

    family_specs = [
        HcmFamilySpec(
            name="all_gaussian_except_A",
            families={
                "A": "bernoulli",
                "Y": "gaussian",
                "M": "gaussian",
                "G": "gaussian",
                "E": "gaussian",
                "L": "gaussian",
                "S": "categorical",
            },
            note="Near the original first-pass convention.",
        ),
        HcmFamilySpec(
            name="mixed_binary_gmm2_ym",
            families={
                "A": "bernoulli",
                "Y": "gaussian_mixture",
                "M": "gaussian_mixture",
                "G": "bernoulli",
                "E": "bernoulli",
                "L": "bernoulli",
                "S": "categorical",
            },
            note="Aligné sur star_hcm v2 : Y,M = mélange gaussien K=2 par école.",
        ),
        HcmFamilySpec(
            name="mixed_with_poisson_count_proxy",
            families={
                "A": "bernoulli",
                "Y": "gaussian_mixture",
                "M": "gaussian_mixture",
                "G": "bernoulli",
                "E": "bernoulli",
                "L": "bernoulli",
                "S": "poisson",
            },
            note="Stress-test Poisson sur S ; Y,M en GMM K=2.",
        ),
    ]

    benchmark: dict[str, Any] = {}
    for fam_spec in family_specs:
        fam_results: list[dict[str, Any]] = []
        t_family_0 = time.perf_counter()
        for spec in specs:
            t0 = time.perf_counter()
            row = run_one_graph_with_families(spec, data, fam_spec.families)
            row["elapsed_seconds"] = time.perf_counter() - t0
            fam_results.append(row)
        benchmark[fam_spec.name] = {
            "note": fam_spec.note,
            "families": fam_spec.families,
            "elapsed_seconds_total": time.perf_counter() - t_family_0,
            "results": fam_results,
        }
    return benchmark


def run_one_graph_with_families(spec: GraphSpec, data: dict[str, np.ndarray], families: dict[str, str]) -> dict[str, Any]:
    from star_hcm_v2_teacher_student import DEFAULT_ESTIMATOR_KWARGS, build_hscm, graph_to_cgm_for_effect
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
        cgm, y_node, x_node = graph_to_cgm_for_effect(hscm)
        id_result = identify_effect(cgm, Y=y_node, X=x_node, unobserved={"U"})
        result["identifiable"] = bool(id_result.identifiable)
        result["formula_latex"] = id_result.formula_latex
        if not id_result.identifiable:
            result["status"] = "not_identifiable"
            return result
        ek = dict(DEFAULT_ESTIMATOR_KWARGS)
        ey1 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 1.0},
            distribution_families=families,
            n_mc_samples=60,
            random_seed=42,
            estimator_kwargs=ek,
        )
        ey0 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 0.0},
            distribution_families=families,
            n_mc_samples=60,
            random_seed=43,
            estimator_kwargs=ek,
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


def choose_best_hcm_variant(benchmark: dict[str, Any], target_small_effect: float = 5.34) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for name, payload in benchmark.items():
        rows = payload["results"]
        ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("ATE") is not None]
        n_ok = len(ok_rows)
        n_err = sum(1 for row in rows if row.get("status") == "error")
        mean_abs_ate = float(np.mean([abs(row["ATE"]) for row in ok_rows])) if ok_rows else float("inf")
        median_abs_dev = (
            float(np.median([abs(row["ATE"] - target_small_effect) for row in ok_rows]))
            if ok_rows
            else float("inf")
        )
        scored.append(
            {
                "name": name,
                "n_ok": n_ok,
                "n_err": n_err,
                "mean_abs_ate": mean_abs_ate,
                "median_abs_dev_from_startenesse": median_abs_dev,
                "elapsed_seconds_total": payload["elapsed_seconds_total"],
            }
        )
    scored.sort(key=lambda row: (-row["n_ok"], row["n_err"], row["median_abs_dev_from_startenesse"], row["mean_abs_ate"]))
    return {
        "ranking": scored,
        "best_name": scored[0]["name"] if scored else None,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# STAR baseline econometric benchmark and HCM family benchmark")
    lines.append("")
    lines.append(f"- outcome baseline économétrique: `{payload['data_summary']['baseline_outcome_col']}`")
    lines.append("")
    lines.append("## Econometric baseline")
    lines.append("")
    ols = payload["econometric"]["ols"]
    lines.append("| Model | Coef small | SE small | p small | Coef regular+aide | SE regular+aide | R2 | Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in ols.items():
        lines.append(
            f"| `{name}` | {row['coef_small']:.4f} | {row['se_small']:.4f} | {row['p_small']:.4g} | "
            f"{row['coef_reg_aide']:.4f} | {row['se_reg_aide']:.4f} | {row['r2']:.4f} | {row['elapsed_seconds']:.3f} |"
        )
    lines.append("")
    iv = payload["econometric"]["iv"]["iv_2sls"]
    lines.append("| IV model | Coef class size | SE | p-value | Time (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `2SLS small/reg_aide -> class size` | {iv['coef_classsize']:.4f} | {iv['se_classsize']:.4f} | {iv['p_classsize']:.4g} | {iv['elapsed_seconds']:.3f} |"
    )
    lines.append("")
    lines.append("## HCM family benchmark")
    lines.append("")
    for fam_name, fam_payload in payload["hcm_benchmark"].items():
        lines.append(f"### {fam_name}")
        lines.append("")
        lines.append(f"- note: {fam_payload['note']}")
        lines.append(f"- total time: `{fam_payload['elapsed_seconds_total']:.3f}` s")
        lines.append("")
        lines.append("| Graph | Status | ATE | Time (s) |")
        lines.append("|---|---|---:|---:|")
        for row in fam_payload["results"]:
            ate_text = "" if row.get("ATE") is None else f"{row['ATE']:.4f}"
            lines.append(f"| `{row['graph_name']}` | `{row['status']}` | {ate_text} | {row['elapsed_seconds']:.3f} |")
        lines.append("")
    lines.append("## Best variant")
    lines.append("")
    best = payload["best_hcm_variant"]
    lines.append(f"- selected variant: `{best['best_name']}`")
    lines.append("")
    lines.append("| Variant | OK graphs | Error graphs | Median abs deviation from 5.34 | Mean abs ATE | Total time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in best["ranking"]:
        lines.append(
            f"| `{row['name']}` | {row['n_ok']} | {row['n_err']} | {row['median_abs_dev_from_startenesse']:.4f} | {row['mean_abs_ate']:.4f} | {row['elapsed_seconds_total']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outcome",
        type=str,
        default="read",
        choices=("read", "math"),
        help="Outcome baseline econometrique: read=gktreadss, math=gktmathss.",
    )
    args = parser.parse_args()

    outcome_col = "gktmathss" if str(args.outcome) == "math" else "gktreadss"
    if str(args.outcome) == "math":
        out_json = RESULTS_DIR / "star_baseline_math_benchmark.json"
        out_md = STAR_DIR / "star_baseline_math_benchmark.md"
    else:
        out_json = RESULTS_DIR / "star_baseline_and_hcm_benchmark.json"
        out_md = STAR_DIR / "star_baseline_and_hcm_benchmark.md"

    df = load_kindergarten_student_level()
    econometric = {
        "ols": fit_ols_and_rf_like_startenesse(df, outcome_col=outcome_col)["ols"],
        "iv": fit_iv_like_startenesse(df, outcome_col=outcome_col),
    }
    data, meta = load_teacher_student_data()
    specs = load_graph_specs()
    hcm_benchmark = run_hcm_family_benchmarks(data, specs)
    best_hcm_variant = choose_best_hcm_variant(hcm_benchmark)

    payload = {
        "data_summary": {
            "n_rows_complete": int(len(df)),
            "n_schools": int(df["gkschid"].nunique()),
            "n_classes": int(df["gktchid"].nunique()),
            "baseline_outcome_col": outcome_col,
            "meta_hcm": meta,
        },
        "econometric": econometric,
        "hcm_benchmark": hcm_benchmark,
        "best_hcm_variant": best_hcm_variant,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    out_md.write_text(build_markdown(payload))
    print(f"Wrote JSON: {out_json.relative_to(REPO_ROOT)}")
    print(f"Wrote Markdown: {out_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
