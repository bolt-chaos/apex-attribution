"""Phase (c): the causal DAG as a networkx.DiGraph.

Node names == dataframe columns in data/f1_results.parquet, so phase (d) can call
gcm.fit(scm, df) with no renaming.

Design note (Option A for DNFs — see SCHEMA_NOTES.md / project memory):
`finish_pos` is modelled on CLASSIFIED (finished) rows only. Conditioning on "finished"
means `reliability_dnf` has no variation within that subset, so it is NOT a structural
parent of `finish_pos`. Instead it is a parallel leaf consequence of the constructor:

    constructor → reliability_dnf      ("does the car break?")
    constructor → finish_pos           ("how well does the car race, given it finished?")

Both share the constructor cause. In phase (f) we COMBINE them into a single
"expected finish including breakdown risk" number — without ever feeding an imputed DNF
position into the finish_pos mechanism. The gcm SCM we actually fit for attribution and
counterfactuals is the finish_pos subgraph (reliability_dnf dropped); the reliability leaf
is fit/estimated separately on all started rows.

Run directly to validate the graph and render figures/dag.png.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "figures" / "dag.png"

# node -> metadata. `fit_subset` says which rows the node's mechanism is learned from.
NODES: dict[str, dict] = {
    "circuit_type": dict(kind="categorical", role="root", fit_subset="started",
                         desc="permanent / street / road course (context)"),
    "constructor_id": dict(kind="categorical", role="root", fit_subset="started",
                           desc="team/car — observed proxy for latent car pace"),
    "driver_id": dict(kind="categorical", role="root", fit_subset="started",
                      desc="driver — observed proxy for latent skill"),
    "grid": dict(kind="ordinal", role="mediator", fit_subset="classified",
                 desc="starting position (qualifying outcome); PL start imputed 21"),
    "reliability_dnf": dict(kind="binary", role="censoring", fit_subset="started",
                            desc="mechanical DNF? property of the car; combined in phase (f)"),
    "finish_pos": dict(kind="ordinal", role="outcome", fit_subset="classified",
                       desc="finishing position, finished races only (Option A)"),
}

# directed cause -> effect edges
EDGES: list[tuple[str, str]] = [
    ("circuit_type", "grid"),
    ("constructor_id", "grid"),
    ("driver_id", "grid"),
    ("circuit_type", "finish_pos"),
    ("constructor_id", "finish_pos"),
    ("driver_id", "finish_pos"),
    ("grid", "finish_pos"),
    ("constructor_id", "reliability_dnf"),
]


def build_dag() -> nx.DiGraph:
    """The full conceptual DAG (all 6 nodes, incl. the reliability leaf)."""
    g = nx.DiGraph()
    for name, meta in NODES.items():
        g.add_node(name, **meta)
    g.add_edges_from(EDGES)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("graph is not acyclic")
    missing = set(g.nodes) - set(NODES)
    if missing:
        raise ValueError(f"edge references unknown node(s): {missing}")
    return g


def finish_scm_graph(g: nx.DiGraph) -> nx.DiGraph:
    """The subgraph we hand to gcm for the finish_pos SCM: drop the reliability leaf,
    which is estimated separately on all started rows (Option A)."""
    return g.subgraph([n for n in g.nodes if n != "reliability_dnf"]).copy()


def describe(g: nx.DiGraph) -> None:
    print("=" * 64)
    print("CAUSAL DAG (phase c)")
    print("=" * 64)
    order = list(nx.topological_sort(g))
    print(f"nodes ({g.number_of_nodes()}), topological order:")
    for n in order:
        m = g.nodes[n]
        parents = list(g.predecessors(n)) or ["—(root)"]
        print(f"  {n:16s} [{m['kind']:11s} {m['role']:9s} fit:{m['fit_subset']:10s}]")
        print(f"  {'':16s}  parents: {', '.join(parents)}")
        print(f"  {'':16s}  {m['desc']}")
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    print(f"\nroots:  {roots}")
    print(f"leaves: {leaves}")
    print(f"edges:  {g.number_of_edges()}")
    sub = finish_scm_graph(g)
    print(f"\nfinish_pos SCM subgraph handed to gcm (phase d): "
          f"{sorted(sub.nodes)}  ({sub.number_of_edges()} edges)")


def plot_dag(g: nx.DiGraph, path: Path = FIG_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # hand-placed layered layout: roots left, mediators middle, outcome right
    pos = {
        "circuit_type": (0, 2.4), "constructor_id": (0, 1.2), "driver_id": (0, 0.0),
        "grid": (1.4, 1.8), "reliability_dnf": (1.4, 0.2),
        "finish_pos": (2.8, 1.2),
    }
    role_color = {"root": "#cfe8ff", "mediator": "#ffe9b3",
                  "censoring": "#f4c7c3", "outcome": "#c9efc2"}
    colors = [role_color[g.nodes[n]["role"]] for n in g.nodes]

    fig, ax = plt.subplots(figsize=(9, 5))
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=18,
                           edge_color="#555", node_size=4200,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors,
                           node_size=4200, edgecolors="#333")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=12, label=r) for r, c in role_color.items()]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_title("F1 driver-vs-car causal DAG (v1, Option A)\n"
                 "reliability_dnf is a parallel leaf — combined with finish_pos at phase (f)",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    dag = build_dag()
    describe(dag)
    plot_dag(dag)
