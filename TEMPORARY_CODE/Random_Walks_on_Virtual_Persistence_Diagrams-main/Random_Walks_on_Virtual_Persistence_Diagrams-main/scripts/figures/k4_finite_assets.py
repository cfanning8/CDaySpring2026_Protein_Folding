"""Generate K4 finite panel assets."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.pyvista_helpers import visualize_graph_3d_pyvista, PYVISTA_AVAILABLE, FIGURE_DPI


def make_output_dir() -> Path:
    out_dir = Path("results") / "figures" / "finite" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_k4_graph():
    """Build K4 graph."""
    G = nx.Graph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j)
    return G


def k4_positions_3d():
    """3D positions arranged as diamond."""
    pos_3d = {
        0: (0.0, -1.0, 0.0),
        1: (-1.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
    }
    return pos_3d


def generate_k4_graph_image(output_dir: Path) -> None:
    G = build_k4_graph()
    pos_3d = k4_positions_3d()

    highlight_nodes = {0, 3}
    highlight_edges = {(0, 3)}

    output_path = output_dir / "k4_graph.png"

    if not PYVISTA_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)
        pos_2d = {n: (pos_3d[n][0], pos_3d[n][1]) for n in G.nodes()}
        nx.draw_networkx_edges(G, pos_2d, ax=ax, edge_color="#333333", width=2.0, alpha=0.85)
        nx.draw_networkx_edges(
            G,
            pos_2d,
            edgelist=[(0, 3)],
            ax=ax,
            edge_color="#aa0000",
            width=3.0,
            alpha=0.95,
        )
        nx.draw_networkx_nodes(
            G,
            pos_2d,
            nodelist=[n for n in G.nodes() if n not in highlight_nodes],
            node_color="#333333",
            node_size=300,
            alpha=0.85,
        )
        nx.draw_networkx_nodes(
            G,
            pos_2d,
            nodelist=list(highlight_nodes),
            node_color="#aa0000",
            node_size=350,
            alpha=0.95,
        )
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.0, transparent=True)
        plt.close(fig)
        return

    visualize_graph_3d_pyvista(
        G,
        pos_3d,
        edge_labels={},
        output_path=output_path,
        label_type="scalar",
        t_samples=None,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
    )


def _latex_png(text: str, path: Path) -> None:
    """Generate LaTeX text as PNG."""
    fig = plt.figure(figsize=(2, 1), dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=20, color="red")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def generate_labels(output_dir: Path) -> None:
    basepoint_text = r"$[A] \in X/A \mapsto 0 \in K(X,A)$"
    top_text = r"$(123) \in S_3$"
    edge_text = r"$\rho(0,(123)) = 2$"
    mass_top_text = r"$m((123)) = -2$"
    mass_basepoint_text = r"$m(0) = \infty$"

    _latex_png(basepoint_text, output_dir / "basepoint_label.png")
    _latex_png(top_text, output_dir / "top_label.png")
    _latex_png(edge_text, output_dir / "edge_label.png")
    _latex_png(mass_top_text, output_dir / "mass_top_label.png")
    _latex_png(mass_basepoint_text, output_dir / "mass_basepoint_label.png")


def main() -> None:
    output_dir = make_output_dir()
    generate_k4_graph_image(output_dir)
    generate_labels(output_dir)


if __name__ == "__main__":
    main()
