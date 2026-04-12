from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from hierarchicalcausalmodels.do_calculus import (
    PYAGNUM_AVAILABLE,
    augment_collapsed_model,
    collapse,
    identify_effect,
    suggest_augment_for_outcome,
)
from hierarchicalcausalmodels.estimation import estimate_causal_effect
from hierarchicalcausalmodels.models import HSCMParametric

from star_hcm_v2_teacher_student import DEFAULT_ESTIMATOR_KWARGS


RANDOM_STATE = 42
N_MC_SAMPLES = 120

REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_DIR = REPO_ROOT / "examples" / "STAR"
RESULTS_DIR = STAR_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERY_JSON = RESULTS_DIR / "causal_discovery_star.json"
DATA_CSV = REPO_ROOT / "data" / "star" / "STAR.csv"
OUT_JSON = RESULTS_DIR / "star_hcm_first_pass.json"
OUT_MD = STAR_DIR / "star_hcm_first_pass.md"


def _noop(_: Any) -> None:
    return None


@dataclass
class GraphSpec:
    name: str
    edges: list[tuple[str, str]]
    note: str


DISCOVERY_TO_HCM = {
    "stark": "A",
    "readk": "Y",
    "schoolk": "S",
    "mathk": "M",
    "gender": "G",
    "ethnicity": "E",
    "lunchk": "L",
}

UNIT_NODES = {"U", "S", "M", "G", "E", "L"}
SUBUNIT_NODES = {"A", "Y"}

ORDER_FOR_ORIENTATION = ["ethnicity", "gender", "schoolk", "lunchk", "mathk", "stark", "readk"]
ORDER_INDEX = {name: idx for idx, name in enumerate(ORDER_FOR_ORIENTATION)}


def parse_directed_edges(labels: list[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for label in labels:
        if "→" not in label:
            continue
        left, right = [part.strip() for part in label.split("→", 1)]
        edges.append((left, right))
    return edges


def orient_by_order(pairs: list[list[int]], names: list[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for i, j in pairs:
        a = names[i]
        b = names[j]
        if ORDER_INDEX[a] <= ORDER_INDEX[b]:
            edges.append((a, b))
        else:
            edges.append((b, a))
    return edges


def translate_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    translated: list[tuple[str, str]] = []
    for src, dst in edges:
        if src not in DISCOVERY_TO_HCM or dst not in DISCOVERY_TO_HCM:
            continue
        translated.append((DISCOVERY_TO_HCM[src], DISCOVERY_TO_HCM[dst]))
    return translated


def unique_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            out.append(edge)
    return out


def load_graph_specs() -> list[GraphSpec]:
    payload = json.loads(DISCOVERY_JSON.read_text())
    names = payload["names"]

    pc_edges = parse_directed_edges(payload["PC"]["directed_edges"])
    pc_edges += orient_by_order(payload["PC"]["undirected_edges"], names)

    fci_edges = orient_by_order(payload["FCI"]["skeleton_pairs"], names)

    lingam_edges = parse_directed_edges(payload["DirectLiNGAM"]["directed_edges"])
    exact_edges = parse_directed_edges(payload["ExactBIC"]["directed_edges"])

    consensus_edges = [
        ("ethnicity", "lunchk"),
        ("gender", "readk"),
        ("schoolk", "lunchk"),
        ("ethnicity", "mathk"),
        ("ethnicity", "schoolk"),
        ("lunchk", "readk"),
        ("mathk", "lunchk"),
        ("mathk", "readk"),
        ("schoolk", "readk"),
    ]

    specs = [
        GraphSpec(
            name="PC",
            edges=translate_edges(unique_edges(pc_edges)),
            note="PC avec orientation heuristique des arêtes non orientées selon un ordre causal fixe.",
        ),
        GraphSpec(
            name="FCI",
            edges=translate_edges(unique_edges(fci_edges)),
            note="FCI ne fournit ici qu’un squelette; orientation heuristique imposée pour obtenir un DAG HCM exécutable.",
        ),
        GraphSpec(
            name="DirectLiNGAM",
            edges=translate_edges(unique_edges(lingam_edges)),
            note="Arcs dirigés repris directement de DirectLiNGAM.",
        ),
        GraphSpec(
            name="ExactBIC",
            edges=translate_edges(unique_edges(exact_edges)),
            note="DAG score-based repris directement de ExactBIC.",
        ),
        GraphSpec(
            name="ConsensusMean",
            edges=translate_edges(unique_edges(consensus_edges)),
            note="Graphe moyen majoritaire d’après `preliminar_results.md`.",
        ),
    ]
    return specs


def build_balanced_star_hcm_data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = pd.read_csv(DATA_CSV)
    cols = ["schoolidk", "schoolk", "stark", "readk", "mathk", "gender", "ethnicity", "lunchk"]
    df = raw[cols].dropna().copy()

    school_map = {"inner-city": 0.0, "rural": 1.0, "suburban": 2.0, "urban": 3.0}
    df["S_code"] = df["schoolk"].map(school_map).astype(float)
    df["A_small"] = (df["stark"] == "small").astype(float)
    df["G_female"] = (df["gender"] == "female").astype(float)
    df["E_white_asian"] = df["ethnicity"].isin(["cauc", "asian"]).astype(float)
    df["L_free"] = (df["lunchk"] == "free").astype(float)

    rng = np.random.default_rng(RANDOM_STATE)
    grouped = df.groupby("schoolidk", group_keys=False)
    n_sub = int(grouped.size().min())

    sampled_parts: list[pd.DataFrame] = []
    for _, g in grouped:
        sampled_parts.append(g.sample(n=n_sub, random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(sampled_parts, axis=0).sort_values(["schoolidk"]).reset_index(drop=True)

    school_ids = sorted(sampled["schoolidk"].unique().tolist())
    school_to_idx = {sid: idx for idx, sid in enumerate(school_ids)}
    sampled["school_idx"] = sampled["schoolidk"].map(school_to_idx)

    n_units = len(school_ids)

    def to_matrix(frame: pd.DataFrame, value_col: str) -> np.ndarray:
        rows: list[np.ndarray] = []
        for sid in school_ids:
            vals = frame.loc[frame["schoolidk"] == sid, value_col].to_numpy(dtype=float)
            rows.append(vals)
        return np.vstack(rows)

    A = to_matrix(sampled, "A_small")
    Y = to_matrix(sampled, "readk")

    per_school = sampled.groupby("schoolidk").agg(
        S=("S_code", "first"),
        M=("mathk", "mean"),
        G=("G_female", "mean"),
        E=("E_white_asian", "mean"),
        L=("L_free", "mean"),
    )
    per_school = per_school.loc[school_ids]

    data = {
        "A": A,
        "Y": Y,
        "S": per_school["S"].to_numpy(dtype=float),
        "M": per_school["M"].to_numpy(dtype=float),
        "G": per_school["G"].to_numpy(dtype=float),
        "E": per_school["E"].to_numpy(dtype=float),
        "L": per_school["L"].to_numpy(dtype=float),
    }

    meta = {
        "n_rows_after_dropna": int(len(df)),
        "n_units_schools": int(n_units),
        "balanced_subunits_per_school": int(n_sub),
        "school_ids_sampled": school_ids,
        "regressor_choices": {
            "A": "binary treatment: 1 if stark == 'small', else 0",
            "Y": "student-level readk",
            "S": "schoolk encoded as 0..3 at unit level",
            "M": "per-school mean mathk",
            "G": "per-school proportion female",
            "E": "per-school proportion white_or_asian",
            "L": "per-school proportion free lunch",
        },
    }
    return data, meta


def build_hscm(spec: GraphSpec, n_units: int, n_sub: int) -> HSCMParametric:
    edges = set(spec.edges)
    edges.update({("U", "A"), ("U", "Y")})

    hscm = HSCMParametric(
        nodes={"U", "S", "M", "G", "E", "L", "A", "Y"},
        edges=edges,
        unit_nodes=UNIT_NODES,
        subunit_nodes=SUBUNIT_NODES,
        sizes=[n_sub] * n_units,
        node_functions={node: _noop for node in {"U", "S", "M", "G", "E", "L", "A", "Y"}},
        data={},
    )
    hscm.cgm.unobserved_variables = {"U"}
    hscm.cgm.observed_variables = {"S", "M", "G", "E", "L", "A", "Y"}
    return hscm


def graph_to_collapsed_for_effect(hscm: HSCMParametric) -> tuple[Any, str, set[str]]:
    collapsed = collapse(hscm)
    collapsed.unobserved_variables = {"U"}
    q_hat, parents = suggest_augment_for_outcome(hscm, "Y")
    if parents == {q_hat} and q_hat in collapsed.dag.nodes:
        return collapsed, q_hat, {"Q^a"}
    augmented = augment_collapsed_model(collapsed, q_hat, parents)
    augmented.unobserved_variables = {"U"}
    return augmented, q_hat, {"Q^a"}


def run_single_graph(spec: GraphSpec, data: dict[str, np.ndarray]) -> dict[str, Any]:
    n_units, n_sub = data["A"].shape
    print(f"[HCM] Running graph: {spec.name}")
    hscm = build_hscm(spec, n_units=n_units, n_sub=n_sub)

    dag_ok = nx.is_directed_acyclic_graph(hscm.cgm.dag)
    result: dict[str, Any] = {
        "graph_name": spec.name,
        "note": spec.note,
        "edges_hcm": sorted([list(edge) for edge in hscm.edges]),
        "dag_ok": bool(dag_ok),
    }
    if not dag_ok:
        result["status"] = "invalid_dag"
        return result

    try:
        cgm_for_id, y_node, x_node = graph_to_collapsed_for_effect(hscm)
        id_result = identify_effect(cgm_for_id, Y=y_node, X=x_node, unobserved={"U"})
        result["identifiable"] = bool(id_result.identifiable)
        result["formula_latex"] = id_result.formula_latex
        result["explanation"] = id_result.explanation
        if not id_result.identifiable:
            result["status"] = "not_identifiable"
            return result

        fam = {
            "A": "bernoulli",
            "Y": "gaussian_mixture",
            "S": "categorical",
            "M": "gaussian_mixture",
            "G": "gaussian",
            "E": "gaussian",
            "L": "gaussian",
        }
        ek = dict(DEFAULT_ESTIMATOR_KWARGS)

        ey1 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 1.0},
            distribution_families=fam,
            n_mc_samples=N_MC_SAMPLES,
            random_seed=RANDOM_STATE,
            estimator_kwargs=ek,
        )
        ey0 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 0.0},
            distribution_families=fam,
            n_mc_samples=N_MC_SAMPLES,
            random_seed=RANDOM_STATE + 1,
            estimator_kwargs=ek,
        )

        result["status"] = "ok"
        result["estimand"] = f"E[{y_node} | do(Q^a)]"
        result["E_do_1"] = float(ey1)
        result["E_do_0"] = float(ey0)
        result["ATE"] = float(ey1 - ey0)
        print(f"[HCM] {spec.name}: ok, identifiable=True, ATE={result['ATE']:.4f}")
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print(f"[HCM] {spec.name}: error, {result['error_type']}: {result['error']}")
        return result


def build_markdown_report(meta: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# STAR — first hierarchical HCM pass")
    lines.append("")
    lines.append("## Chosen schema")
    lines.append("")
    lines.append("- unit `i`: `schoolidk`")
    lines.append("- subunit `j`: student within school `i`")
    lines.append("- latent unit confounder: `U`")
    lines.append("- treatment `A_ij`: `1{stark == small}`")
    lines.append("- outcome `Y_ij`: `readk`")
    lines.append("- observed unit covariates: `S` = `schoolk`, `M` = school mean `mathk`, `G` = school proportion female, `E` = school proportion white/asian, `L` = school proportion free lunch")
    lines.append("")
    lines.append("This first pass keeps the discovery graphs but moves the non-treatment regressors to the unit level as school summaries so that the HCM pipeline remains estimable with a subunit treatment and a subunit outcome.")
    lines.append("")
    lines.append("## Data choices")
    lines.append("")
    lines.append(f"- rows after dropping missing values: `{meta['n_rows_after_dropna']}`")
    lines.append(f"- schools used as units: `{meta['n_units_schools']}`")
    lines.append(f"- balanced students per school: `{meta['balanced_subunits_per_school']}`")
    lines.append("")
    lines.append("## Graph-by-graph HCM results")
    lines.append("")
    lines.append("| Graph | Status | Identifiable | ATE | E[do(1)] | E[do(0)] |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in results:
        identifiable = row.get("identifiable")
        ident_str = "yes" if identifiable else "no"
        ate = row.get("ATE")
        ey1 = row.get("E_do_1")
        ey0 = row.get("E_do_0")
        lines.append(
            f"| `{row['graph_name']}` | `{row['status']}` | {ident_str} | "
            f"{'' if ate is None else f'{ate:.4f}'} | "
            f"{'' if ey1 is None else f'{ey1:.4f}'} | "
            f"{'' if ey0 is None else f'{ey0:.4f}'} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for row in results:
        lines.append(f"### {row['graph_name']}")
        lines.append("")
        lines.append(f"- note: {row['note']}")
        lines.append(f"- status: `{row['status']}`")
        if row.get("formula_latex"):
            lines.append(f"- formula: `${row['formula_latex']}$")
        if row.get("error"):
            lines.append(f"- error: `{row['error_type']}: {row['error']}`")
        elif row.get("explanation"):
            lines.append(f"- explanation: {row['explanation']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if not PYAGNUM_AVAILABLE:
        raise RuntimeError("pyAgrum is required for this first HCM pass.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=None, help="Run only one graph by name.")
    args = parser.parse_args()

    specs = load_graph_specs()
    if args.graph is not None:
        specs = [spec for spec in specs if spec.name == args.graph]
        if not specs:
            raise ValueError(f"Unknown graph name: {args.graph}")
    data, meta = build_balanced_star_hcm_data()
    results = [run_single_graph(spec, data) for spec in specs]

    payload = {
        "schema": {
            "unit_nodes": sorted(list(UNIT_NODES)),
            "subunit_nodes": sorted(list(SUBUNIT_NODES)),
            "latent_unit_node": "U",
            "meta": meta,
        },
        "results": results,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(build_markdown_report(meta, results))

    print(f"Wrote JSON: {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote Markdown: {OUT_MD.relative_to(REPO_ROOT)}")
    for row in results:
        ate = row.get("ATE")
        ate_str = "NA" if ate is None else f"{ate:.4f}"
        print(f"{row['graph_name']:<14} status={row['status']:<18} identifiable={str(row.get('identifiable')):<5} ATE={ate_str}")


if __name__ == "__main__":
    main()
