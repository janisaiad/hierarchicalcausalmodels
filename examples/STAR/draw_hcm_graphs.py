from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_DIR = REPO_ROOT / "examples" / "STAR"
RESULTS_DIR = STAR_DIR / "results"
FIG_DIR = STAR_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

V1_JSON = RESULTS_DIR / "star_hcm_first_pass.json"
V2_JSON = RESULTS_DIR / "star_hcm_v2_teacher_student.json"


def normalize_node(name: str) -> tuple[str, str]:
    if name.startswith("_"):
        return name[1:], "subunit"
    return name, "unit"


def load_graphs(path: Path) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str]]]]:
    payload = json.loads(path.read_text())
    unit_nodes = payload["schema"]["unit_nodes"]
    subunit_nodes = payload["schema"]["subunit_nodes"]
    graphs: dict[str, list[tuple[str, str]]] = {}
    for row in payload["results"]:
        edges: list[tuple[str, str]] = []
        for src, dst in row["edges_hcm"]:
            src_name, _ = normalize_node(src)
            dst_name, _ = normalize_node(dst)
            edges.append((src_name, dst_name))
        graphs[row["graph_name"]] = edges
    return {"unit": unit_nodes, "subunit": subunit_nodes}, graphs


def node_positions(schema: dict[str, list[str]], variant: str) -> dict[str, tuple[float, float]]:
    if variant == "v1":
        unit_order = ["U", "S", "M", "G", "E", "L"]
        sub_order = ["A", "Y"]
        y_unit = 1.0
        y_sub = 0.0
        pos = {}
        for idx, name in enumerate(unit_order):
            pos[name] = (idx * 1.3, y_unit)
        for idx, name in enumerate(sub_order):
            pos[name] = (2.2 + idx * 1.6, y_sub)
        return pos

    unit_order = ["U", "S"]
    sub_order = ["A", "M", "G", "E", "L", "Y"]
    y_unit = 1.0
    y_sub = 0.0
    pos = {}
    for idx, name in enumerate(unit_order):
        pos[name] = (1.5 + idx * 3.2, y_unit)
    for idx, name in enumerate(sub_order):
        pos[name] = (idx * 1.3, y_sub)
    return pos


def draw_graph(
    schema: dict[str, list[str]],
    edges: list[tuple[str, str]],
    title: str,
    output_path: Path,
    variant: str,
) -> None:
    graph = nx.DiGraph()
    for node in schema["unit"]:
        graph.add_node(node)
    for node in schema["subunit"]:
        graph.add_node(node)
    graph.add_edges_from(edges)

    pos = node_positions(schema, variant)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.axhspan(0.55, 1.25, color="#eef3ff", alpha=0.9)
    ax.axhspan(-0.25, 0.45, color="#fff5e8", alpha=0.95)
    ax.text(-0.2, 1.12, "niveau unité", fontsize=11, weight="bold", color="#355caa")
    ax.text(-0.2, 0.12, "niveau sous-unité", fontsize=11, weight="bold", color="#b36b00")

    unit_nodes = [node for node in schema["unit"] if node in graph.nodes]
    sub_nodes = [node for node in schema["subunit"] if node in graph.nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=unit_nodes,
        node_color="#a9c5ff",
        edgecolors="#355caa",
        linewidths=1.5,
        node_size=1800,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=sub_nodes,
        node_color="#ffd79a",
        edgecolors="#b36b00",
        linewidths=1.5,
        node_size=1800,
        ax=ax,
    )

    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=18,
        width=1.8,
        edge_color="#333333",
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_axis_off()
    ax.set_xlim(-0.6, max(x for x, _ in pos.values()) + 0.8)
    ax.set_ylim(-0.35, 1.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_base_schema(schema: dict[str, list[str]], title: str, output_path: Path, variant: str) -> None:
    draw_graph(schema, [], title, output_path, variant)


def main() -> None:
    schema_v1, graphs_v1 = load_graphs(V1_JSON)
    schema_v2, graphs_v2 = load_graphs(V2_JSON)

    draw_base_schema(schema_v1, "HCM v1 - schema de base (school -> student)", FIG_DIR / "hcm_v1_base_schema.png", "v1")
    for name, edges in graphs_v1.items():
        draw_graph(schema_v1, edges, f"HCM v1 - {name}", FIG_DIR / f"hcm_v1_{name.lower()}.png", "v1")

    draw_base_schema(schema_v2, "HCM v2 - schema de base (teacher/class -> student)", FIG_DIR / "hcm_v2_base_schema.png", "v2")
    for name, edges in graphs_v2.items():
        draw_graph(schema_v2, edges, f"HCM v2 - {name}", FIG_DIR / f"hcm_v2_{name.lower()}.png", "v2")

    print("Wrote HCM graph PNGs to", FIG_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
