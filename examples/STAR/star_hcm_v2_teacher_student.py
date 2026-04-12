from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


RANDOM_STATE = 42
N_SUB_PER_CLASS = 10
N_MC_SAMPLES = 60

DEFAULT_DISTRIBUTION_FAMILIES: dict[str, str] = {
    "A": "bernoulli",
    "Y": "gaussian_mixture",
    "M": "gaussian_mixture",
    "G": "bernoulli",
    "E": "bernoulli",
    "L": "bernoulli",
    "S": "categorical",
}

DEFAULT_ESTIMATOR_KWARGS: dict[str, Any] = {
    "Y": {"n_components": 2, "random_state_gmm": RANDOM_STATE, "max_iter_gmm": 400},
    "M": {"n_components": 2, "random_state_gmm": RANDOM_STATE, "max_iter_gmm": 400},
}

REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_DIR = REPO_ROOT / "examples" / "STAR"
RESULTS_DIR = STAR_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERY_JSON = RESULTS_DIR / "causal_discovery_star.json"
RAW_TAB = STAR_DIR / "STA-207" / "STAR_Students.tab"
OUT_JSON = RESULTS_DIR / "star_hcm_v2_teacher_student.json"
OUT_MD = STAR_DIR / "star_hcm_v2_teacher_student_summary.md"


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
    "mathk": "M",
    "gender": "G",
    "ethnicity": "E",
    "lunchk": "L",
    "schoolk": "S",
}

UNIT_NODES = {"U", "S"}
SUBUNIT_NODES = {"A", "Y", "M", "G", "E", "L"}

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
    out: list[tuple[str, str]] = []
    for src, dst in edges:
        if src in DISCOVERY_TO_HCM and dst in DISCOVERY_TO_HCM:
            out.append((DISCOVERY_TO_HCM[src], DISCOVERY_TO_HCM[dst]))
    return out


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

    return [
        GraphSpec("PC", translate_edges(unique_edges(pc_edges)), "PC + orientation heuristique des arêtes non orientées."),
        GraphSpec("FCI", translate_edges(unique_edges(fci_edges)), "FCI squelette seulement; orientation heuristique imposée."),
        GraphSpec("DirectLiNGAM", translate_edges(unique_edges(lingam_edges)), "Arcs dirigés repris de DirectLiNGAM."),
        GraphSpec("ExactBIC", translate_edges(unique_edges(exact_edges)), "DAG score-based repris de ExactBIC."),
        GraphSpec("ConsensusMean", translate_edges(unique_edges(consensus_edges)), "Graphe moyen majoritaire de `preliminar_results.md`."),
    ]


def load_teacher_student_data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    sampled_parts: list[pd.DataFrame] = []
    for _, g in df.groupby("gktchid"):
        sampled_parts.append(g.sample(n=N_SUB_PER_CLASS, random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(sampled_parts, axis=0).sort_values(["gktchid", "stdntid"]).reset_index(drop=True)

    teacher_ids = sorted(sampled["gktchid"].unique().tolist())

    def to_matrix(value_col: str) -> np.ndarray:
        rows: list[np.ndarray] = []
        for tid in teacher_ids:
            vals = sampled.loc[sampled["gktchid"] == tid, value_col].to_numpy(dtype=float)
            rows.append(vals)
        return np.vstack(rows)

    A = to_matrix("A_small")
    Y = to_matrix("Y_read")
    M = to_matrix("M_math")
    G = to_matrix("G_female")
    E = to_matrix("E_white_asian")
    L = to_matrix("L_free")
    S = (
        sampled.groupby("gktchid")["S_urbanicity"]
        .first()
        .loc[teacher_ids]
        .to_numpy(dtype=float)
    )

    data = {
        "A": A,
        "Y": Y,
        "M": M,
        "G": G,
        "E": E,
        "L": L,
        "S": S,
    }

    meta = {
        "source_table": "examples/STAR/STA-207/STAR_Students.tab",
        "grade": "kindergarten",
        "n_rows_after_dropna": int(len(df)),
        "n_units_classes": int(len(teacher_ids)),
        "balanced_students_per_class": int(N_SUB_PER_CLASS),
        "regressor_choice_rationale": [
            "Keep the same discovery notebook variables for comparability.",
            "Move only school urbanicity to the unit level.",
            "Keep math, gender, ethnicity, and lunch at the student level.",
            "Use binary codings for categorical student regressors to stabilize first-pass HCM estimation.",
        ],
        "variable_mapping": {
            "A": "1{gkclasstype == 1} = small class",
            "Y": "gktreadss",
            "M": "gktmathss",
            "G": "1{gender == female}",
            "E": "1{race in {white, asian}}",
            "L": "1{free lunch}",
            "S": "gksurban",
            "unit_id": "gktchid",
        },
    }
    return data, meta


def build_hscm(spec: GraphSpec, n_units: int, n_sub: int) -> HSCMParametric:
    edges = set(spec.edges)
    edges.update({("U", "A"), ("U", "Y")})
    hscm = HSCMParametric(
        nodes={"U", "S", "A", "Y", "M", "G", "E", "L"},
        edges=edges,
        unit_nodes=UNIT_NODES,
        subunit_nodes=SUBUNIT_NODES,
        sizes=[n_sub] * n_units,
        node_functions={node: _noop for node in {"U", "S", "A", "Y", "M", "G", "E", "L"}},
        data={},
    )
    hscm.cgm.unobserved_variables = {"U"}
    hscm.cgm.observed_variables = {"S", "A", "Y", "M", "G", "E", "L"}
    return hscm


def graph_to_cgm_for_effect(
    hscm: HSCMParametric,
    outcome_subunit: str = "Y",
) -> tuple[Any, str, set[str]]:
    """Collapse + augment for a subunit outcome (``Y`` = lecture, ``M`` = maths / ``gktmathss``)."""
    collapsed = collapse(hscm)
    collapsed.unobserved_variables = {"U"}
    q_hat, parents = suggest_augment_for_outcome(hscm, outcome_subunit)
    if parents == {q_hat} and q_hat in collapsed.dag.nodes:
        return collapsed, q_hat, {"Q^a"}
    augmented = augment_collapsed_model(collapsed, q_hat, parents)
    augmented.unobserved_variables = {"U"}
    return augmented, q_hat, {"Q^a"}


def run_one_graph(
    spec: GraphSpec,
    data: dict[str, np.ndarray],
    *,
    outcome_subunit: str = "Y",
    estimator_backend: str = "numpy",
    torch_kwargs: Optional[dict[str, Any]] = None,
    n_mc_samples: int | None = None,
    distribution_families: Optional[dict[str, str]] = None,
    estimator_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    n_units, n_sub = data["A"].shape
    print(f"[HCM-v2] Running graph: {spec.name} (outcome={outcome_subunit})")
    hscm = build_hscm(spec, n_units, n_sub)
    result: dict[str, Any] = {
        "graph_name": spec.name,
        "note": spec.note,
        "edges_hcm": sorted([list(edge) for edge in hscm.edges]),
        "dag_ok": bool(nx.is_directed_acyclic_graph(hscm.cgm.dag)),
    }
    if not result["dag_ok"]:
        result["status"] = "invalid_dag"
        print(f"[HCM-v2] {spec.name}: invalid DAG")
        return result

    try:
        cgm, y_node, x_node = graph_to_cgm_for_effect(hscm, outcome_subunit=outcome_subunit)
        id_result = identify_effect(cgm, Y=y_node, X=x_node, unobserved={"U"})
        result["identifiable"] = bool(id_result.identifiable)
        result["formula_latex"] = id_result.formula_latex
        result["explanation"] = id_result.explanation
        if not id_result.identifiable:
            result["status"] = "not_identifiable"
            print(f"[HCM-v2] {spec.name}: not identifiable")
            return result

        fam = dict(distribution_families) if distribution_families is not None else dict(DEFAULT_DISTRIBUTION_FAMILIES)
        ek: dict[str, Any] = dict(DEFAULT_ESTIMATOR_KWARGS) if estimator_kwargs is None else {**DEFAULT_ESTIMATOR_KWARGS, **estimator_kwargs}

        mc = int(n_mc_samples) if n_mc_samples is not None else int(N_MC_SAMPLES)
        ey1 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 1.0},
            distribution_families=fam,
            n_mc_samples=mc,
            random_seed=RANDOM_STATE,
            estimator_backend=estimator_backend,
            torch_kwargs=torch_kwargs,
            estimator_kwargs=ek,
        )
        ey0 = estimate_causal_effect(
            id_result,
            data=data,
            intervention={"Q^a": 0.0},
            distribution_families=fam,
            n_mc_samples=mc,
            random_seed=RANDOM_STATE + 1,
            estimator_backend=estimator_backend,
            torch_kwargs=torch_kwargs,
            estimator_kwargs=ek,
        )
        result["status"] = "ok"
        result["outcome_subunit"] = outcome_subunit
        result["estimand"] = f"E[{y_node} | do(Q^a)]"
        result["E_do_1"] = float(ey1)
        result["E_do_0"] = float(ey0)
        result["ATE"] = float(ey1 - ey0)
        result["estimator_backend"] = estimator_backend
        result["n_mc_samples"] = mc
        result["distribution_families"] = fam
        result["estimator_kwargs"] = ek
        print(f"[HCM-v2] {spec.name}: ok, ATE={result['ATE']:.4f}")
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print(f"[HCM-v2] {spec.name}: error {result['error_type']}: {result['error']}")
        return result


def build_summary(meta: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    outcome = str(results[0].get("outcome_subunit", "M")) if results else "M"
    outcome_label = "lecture (`gktreadss`)" if outcome.upper() == "Y" else "maths (`gktmathss`)"
    qsym = "Q^y" if outcome.upper() == "Y" else "Q^m"
    lines.append("# STAR — HCM v2 teacher/class -> student")
    lines.append("")
    lines.append(f"- **Outcome analysé** : `{outcome}` = {outcome_label} (intervention toujours sur `Q^a`).")
    lines.append("")
    lines.append("## Schéma")
    lines.append("")
    lines.append("- unité `i` : classe / enseignant (`gktchid`)")
    lines.append("- sous-unité `j` : élève dans la classe `i`")
    lines.append("- variable latente de niveau unité : `U`")
    lines.append("- variable observée de niveau unité : `S = gksurban`")
    lines.append("- traitement de niveau élève : `A_ij = 1{small class}`")
    lines.append("- outcome de niveau élève ciblé par ce run : variable `" + outcome + f"` ({outcome_label})")
    lines.append("- régresseurs de niveau élève :")
    lines.append("  - `M_ij = gktmathss`")
    lines.append("  - `G_ij = 1{female}`")
    lines.append("  - `E_ij = 1{white/asian}`")
    lines.append("  - `L_ij = 1{free lunch}`")
    lines.append("")
    lines.append("Cette v2 garde les variables du notebook de causal discovery pour rester comparable, mais les replace à un niveau plus fidèle à `STAR` : `schoolk` au niveau classe et les covariables individuelles au niveau élève.")
    lines.append("")
    lines.append("## Choix des régresseurs")
    lines.append("")
    for item in meta["regressor_choice_rationale"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Données")
    lines.append("")
    lines.append(f"- source : `{meta['source_table']}`")
    lines.append(f"- grade : `{meta['grade']}`")
    lines.append(f"- classes utilisées comme unités : `{meta['n_units_classes']}`")
    lines.append(f"- élèves retenus par classe : `{meta['balanced_students_per_class']}`")
    lines.append(f"- lignes après suppression des valeurs manquantes et filtrage des classes valides : `{meta['n_rows_after_dropna']}`")
    lines.append("")
    lines.append("## Résultats")
    lines.append("")
    lines.append("| Graph | Status | Identifiable | ATE | E[do(1)] | E[do(0)] |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in results:
        ate_text = "" if row.get("ATE") is None else f"{row['ATE']:.4f}"
        e1_text = "" if row.get("E_do_1") is None else f"{row['E_do_1']:.4f}"
        e0_text = "" if row.get("E_do_0") is None else f"{row['E_do_0']:.4f}"
        lines.append(
            f"| `{row['graph_name']}` | `{row['status']}` | "
            f"{'yes' if row.get('identifiable') else 'no'} | "
            f"{ate_text} | "
            f"{e1_text} | "
            f"{e0_text} |"
        )
    lines.append("")
    lines.append("## Lecture")
    lines.append("")
    lines.append("Cette v2 est meilleure que la v1 au niveau du schéma hiérarchique, car elle colle mieux au mécanisme réel du traitement. En revanche, les graphes de départ restent issus d'une causal discovery plate, puis traduits dans un HCM : cette étape reste donc un choix de modélisation, pas une vérité causale directement lue dans `STAR`.")
    lines.append("")
    lines.append(
        f"Pour `PC`, `FCI` et `ConsensusMean`, la formule identifiée se réduit souvent à `$P({qsym})$` "
        f"(marginalisation sans effet net de `do(Q^a)`). Un `ATE = 0` dans ce cas reflète surtout la "
        "structure graphique, pas une preuve d'absence d'effet de la petite classe."
    )
    lines.append("")
    lines.append(
        "L'estimateur numérique aligne désormais strictement les régresseurs de conditionnement "
        "entre l'ajustement et l'évaluation (même ensemble de parents `Q` que dans la formule, "
        "même règle de colonne pour les profils 2D). Les graphes denses (`DirectLiNGAM`, `ExactBIC`) "
        "peuvent toutefois rester bruyants ou sensibles à la paramétrisation."
    )
    lines.append("")
    for row in results:
        lines.append(f"### {row['graph_name']}")
        lines.append("")
        lines.append(f"- note : {row['note']}")
        lines.append(f"- statut : `{row['status']}`")
        if row.get("formula_latex"):
            lines.append(f"- formule : ${row['formula_latex']}$")
        if row.get("error"):
            lines.append(f"- erreur : `{row['error_type']}: {row['error']}`")
        elif row.get("explanation"):
            lines.append(f"- explication : {row['explanation']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if not PYAGNUM_AVAILABLE:
        raise RuntimeError("pyAgrum is required for the HCM v2 analysis.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=None, help="Run only one graph by name.")
    parser.add_argument(
        "--outcome",
        type=str,
        default="M",
        choices=("Y", "M"),
        help="Sous-unité cible pour l'estimande (Y=gktreadss lecture, M=gktmathss maths).",
    )
    parser.add_argument(
        "--estimator-backend",
        type=str,
        default="numpy",
        choices=("numpy", "torch"),
        help="Backend numérique pour estimate_causal_effect (torch pour GPU si disponible).",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default=None,
        help="Device PyTorch explicite (ex. cuda:0, cpu). Défaut : résolution automatique.",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=None,
        help="Nombre d'échantillons MC (défaut : constante N_MC_SAMPLES du module).",
    )
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Liste de graphes séparés par des virgules (ex. DirectLiNGAM,ExactBIC). Défaut : tous.",
    )
    args = parser.parse_args()

    specs = load_graph_specs()
    if args.graph is not None:
        specs = [spec for spec in specs if spec.name == args.graph]
        if not specs:
            raise ValueError(f"Unknown graph name: {args.graph}")
    elif args.graphs is not None:
        want = {g.strip() for g in str(args.graphs).split(",") if g.strip()}
        specs = [spec for spec in specs if spec.name in want]
        if not specs:
            raise ValueError(f"No graph matched --graphs={args.graphs!r}")
        missing = want - {s.name for s in specs}
        if missing:
            raise ValueError(f"Unknown graph name(s): {sorted(missing)}")

    torch_kwargs: dict[str, Any] | None = None
    if args.estimator_backend == "torch":
        torch_kwargs = {}
        if args.torch_device is not None:
            torch_kwargs["device"] = str(args.torch_device)

    data, meta = load_teacher_student_data()
    results = [
        run_one_graph(
            spec,
            data,
            outcome_subunit=str(args.outcome),
            estimator_backend=str(args.estimator_backend),
            torch_kwargs=torch_kwargs,
            n_mc_samples=args.n_mc_samples,
            distribution_families=None,
        )
        for spec in specs
    ]

    partial = args.graph is not None or args.graphs is not None
    if partial and OUT_JSON.is_file():
        try:
            prev = json.loads(OUT_JSON.read_text())
        except json.JSONDecodeError:
            prev = None
        if prev is not None and str(prev.get("schema", {}).get("outcome_subunit_default")) == str(args.outcome):
            by_name = {r["graph_name"]: r for r in prev.get("results", [])}
            for r in results:
                by_name[r["graph_name"]] = r
            order = [s.name for s in load_graph_specs()]
            results = [by_name[n] for n in order if n in by_name]
            if len(results) < len(order):
                print(
                    "[HCM-v2] Attention : fusion partielle — graphes manquants dans le JSON précédent : "
                    f"{sorted(set(order) - set(by_name))}",
                )

    payload = {
        "schema": {
            "unit_nodes": sorted(list(UNIT_NODES)),
            "subunit_nodes": sorted(list(SUBUNIT_NODES)),
            "latent_unit_node": "U",
            "outcome_subunit_default": str(args.outcome),
            "estimator_backend": str(args.estimator_backend),
            "torch_kwargs": torch_kwargs,
            "merged_partial_run": bool(partial and OUT_JSON.is_file()),
            "meta": meta,
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(build_summary(meta, results))

    print(f"Wrote JSON: {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote Markdown: {OUT_MD.relative_to(REPO_ROOT)}")
    for row in results:
        ate = row.get("ATE")
        ate_str = "NA" if ate is None else f"{ate:.4f}"
        print(f"{row['graph_name']:<14} status={row['status']:<18} identifiable={str(row.get('identifiable')):<5} ATE={ate_str}")


if __name__ == "__main__":
    main()
