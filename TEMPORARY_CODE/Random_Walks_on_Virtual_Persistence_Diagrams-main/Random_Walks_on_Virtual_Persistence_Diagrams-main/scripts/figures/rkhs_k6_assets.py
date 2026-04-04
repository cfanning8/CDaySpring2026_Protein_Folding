"""Generate RKHS K6 panel assets."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.pyvista_helpers import visualize_graph_3d_pyvista, PYVISTA_AVAILABLE, FIGURE_DPI

matplotlib.use("Agg")


def make_output_dir() -> Path:
    out_dir = Path("results") / "figures" / "RKHS" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def s3_word_metric(g1: str, g2: str) -> int:
    """Word metric distance in S_3 with generating set {(12), (23)}."""
    if g1 == g2:
        return 0
    
    dist_from_e = {
        'e': 0,
        '12': 1,
        '23': 1,
        '123': 2,
        '132': 2,
        '13': 3,
    }
    
    mult_table = {
        ('e', 'e'): 'e',
        ('e', '12'): '12',
        ('e', '23'): '23',
        ('e', '123'): '123',
        ('e', '132'): '132',
        ('e', '13'): '13',
        ('12', 'e'): '12',
        ('12', '12'): 'e',
        ('12', '23'): '123',
        ('12', '123'): '23',
        ('12', '132'): '13',
        ('12', '13'): '132',
        ('23', 'e'): '23',
        ('23', '12'): '132',
        ('23', '23'): 'e',
        ('23', '123'): '12',
        ('23', '132'): '13',
        ('23', '13'): '123',
        ('123', 'e'): '123',
        ('123', '12'): '23',
        ('123', '23'): '13',
        ('123', '123'): '132',
        ('123', '132'): 'e',
        ('123', '13'): '12',
        ('132', 'e'): '132',
        ('132', '12'): '13',
        ('132', '23'): '12',
        ('132', '123'): 'e',
        ('132', '132'): '123',
        ('132', '13'): '23',
        ('13', 'e'): '13',
        ('13', '12'): '132',
        ('13', '23'): '123',
        ('13', '123'): '12',
        ('13', '132'): '23',
        ('13', '13'): 'e',
    }
    
    inv_table = {
        'e': 'e',
        '12': '12',
        '23': '23',
        '123': '132',
        '132': '123',
        '13': '13',
    }
    
    g1_inv = inv_table[g1]
    g1_inv_g2 = mult_table[(g1_inv, g2)]
    return dist_from_e[g1_inv_g2]


def build_k6_graph() -> Tuple[nx.Graph, Dict[int, str]]:
    """Build K6 graph representing S_3."""
    G = nx.Graph()
    elements = ['e', '12', '23', '123', '132', '13']
    node_to_element = {i: elem for i, elem in enumerate(elements)}
    
    G.add_nodes_from(range(6))
    for i in range(6):
        for j in range(i + 1, 6):
            G.add_edge(i, j)
    
    return G, node_to_element


def k6_positions_3d() -> Dict[int, Tuple[float, float, float]]:
    """3D positions in regular hexagon on plane perpendicular to camera view."""
    camera_pos = np.array([5.0, 5.0, 5.0])
    camera_focal = np.array([0.0, 0.0, 0.0])
    camera_up = np.array([0.0, 0.0, 1.0])
    
    view_dir = camera_focal - camera_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    
    plane_normal = view_dir
    
    up_proj = camera_up - np.dot(camera_up, plane_normal) * plane_normal
    if np.linalg.norm(up_proj) < 1e-6:
        up_proj = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), plane_normal) * plane_normal
    up_proj = up_proj / (np.linalg.norm(up_proj) + 1e-10)
    
    right_proj = np.cross(plane_normal, up_proj)
    right_proj = right_proj / (np.linalg.norm(right_proj) + 1e-10)
    
    radius = 1.5
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    
    pos_3d = {}
    for i, angle in enumerate(hex_angles):
        x_local = radius * np.cos(angle)
        y_local = radius * np.sin(angle)
        point_3d = x_local * right_proj + y_local * up_proj
        pos_3d[i] = tuple(point_3d)
    
    return pos_3d


def compute_edge_weights(G: nx.Graph, node_to_element: Dict[int, str], alpha: float = 1.0) -> Dict[Tuple[int, int], float]:
    """Compute edge weights using psi(r) = exp(-alpha * r^2)."""
    weights = {}
    for u, v in G.edges():
        elem_u = node_to_element[u]
        elem_v = node_to_element[v]
        dist = s3_word_metric(elem_u, elem_v)
        weight = np.exp(-alpha * dist * dist)
        weights[(u, v)] = weight
    return weights


def weight_to_color(weight: float, min_weight: float, max_weight: float) -> str:
    """Map edge weight to color from sky blue to crimson."""
    if max_weight == min_weight:
        return "#87CEEB"
    
    t = (weight - min_weight) / (max_weight - min_weight)
    
    sky_blue = np.array([135, 206, 235])
    crimson = np.array([220, 20, 60])
    
    color = t * sky_blue + (1 - t) * crimson
    color = np.clip(color, 0, 255).astype(int)
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def generate_k6_graph_image(output_dir: Path) -> None:
    """Generate K6 graph with edge coloring."""
    G, node_to_element = build_k6_graph()
    pos_3d = k6_positions_3d()
    
    alpha = 0.5
    edge_weights = compute_edge_weights(G, node_to_element, alpha=alpha)
    
    min_weight = min(edge_weights.values())
    max_weight = max(edge_weights.values())
    
    edge_colors = {}
    for (u, v), weight in edge_weights.items():
        edge_colors[(u, v)] = weight_to_color(weight, min_weight, max_weight)
    
    output_path = output_dir / "k6_graph.png"
    
    if not PYVISTA_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=FIGURE_DPI)
        pos_2d = {n: (pos_3d[n][0], pos_3d[n][1]) for n in G.nodes()}
        
        for (u, v), color in edge_colors.items():
            nx.draw_networkx_edges(
                G, pos_2d, edgelist=[(u, v)], ax=ax,
                edge_color=color, width=2.5, alpha=0.85
            )
        
        nx.draw_networkx_nodes(
            G, pos_2d, node_color="#333333", node_size=400, alpha=0.85, ax=ax
        )
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.0, transparent=True)
        plt.close(fig)
        return
    
    import pyvista as pv
    try:
        pv.start_xvfb()
    except (OSError, AttributeError):
        pass
    
    plotter = pv.Plotter(off_screen=True, window_size=[2000, 2000])
    plotter.renderer.SetAutomaticLightCreation(False)
    plotter.renderer.LightFollowCameraOff()
    
    light1 = pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white', intensity=1.5)
    light1.positional = False
    plotter.add_light(light1)
    
    light2 = pv.Light(position=(-3, 3, 2), focal_point=(0, 0, 0), color='white', intensity=0.8)
    light2.positional = False
    plotter.add_light(light2)
    
    plotter.renderer.SetUseShadows(False)
    plotter.renderer.SetTwoSidedLighting(True)
    plotter.set_background([0, 0, 0, 0])
    
    n = G.number_of_nodes()
    node_positions = np.array([list(pos_3d[i]) for i in range(n)])
    
    plotter.camera.position = (5, 5, 5)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.SetViewUp(0, 0, 1)
    
    edge_radius = 0.02
    
    for u, v in G.edges():
        start = np.array(pos_3d[u])
        end = np.array(pos_3d[v])
        line = pv.Line(start, end)
        tube = line.tube(radius=edge_radius, n_sides=12)
        color_hex = edge_colors.get((u, v), edge_colors.get((v, u), "#333333"))
        color_rgb = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))
        actor = plotter.add_mesh(
            tube, color=color_rgb, opacity=0.85, smooth_shading=True,
            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
            show_edges=False, pbr=False, metallic=0.0, roughness=0.6
        )
        actor.GetProperty().SetLighting(True)
    
    node_size = 0.12
    for i in range(n):
        sphere = pv.Sphere(radius=node_size, center=node_positions[i], theta_resolution=30, phi_resolution=30)
        actor = plotter.add_mesh(
            sphere, color='#333333', show_edges=False,
            opacity=0.85, smooth_shading=True,
            ambient=0.3, diffuse=0.6, specular=0.3, specular_power=30,
            pbr=False, metallic=0.0, roughness=0.5
        )
        actor.GetProperty().SetLighting(True)
    
    plotter.renderer.SetUseFXAA(False)
    plotter.renderer.SetBackgroundAlpha(0.0)
    plotter.renderer.SetAutomaticLightCreation(False)
    plotter.renderer.SetTwoSidedLighting(True)
    
    plotter.render()
    
    plotter.screenshot(str(output_path), transparent_background=True)
    plotter.close()


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
    """Generate LaTeX labels for all 6 S_3 elements."""
    labels = {
        'e': r"$e$",
        '12': r"$(12)$",
        '23': r"$(23)$",
        '123': r"$(123)$",
        '132': r"$(132)$",
        '13': r"$(13)$",
    }
    
    for elem, text in labels.items():
        filename = f"label_{elem}.png"
        _latex_png(text, output_dir / filename)


def generate_node_position_map(output_dir: Path) -> None:
    """Generate node position mapping document."""
    G, node_to_element = build_k6_graph()
    pos_3d = k6_positions_3d()
    
    camera_pos = np.array([5.0, 5.0, 5.0])
    camera_focal = np.array([0.0, 0.0, 0.0])
    camera_up = np.array([0.0, 0.0, 1.0])
    
    view_dir = camera_focal - camera_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    
    plane_normal = view_dir
    up_proj = camera_up - np.dot(camera_up, plane_normal) * plane_normal
    if np.linalg.norm(up_proj) < 1e-6:
        up_proj = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), plane_normal) * plane_normal
    up_proj = up_proj / (np.linalg.norm(up_proj) + 1e-10)
    right_proj = np.cross(plane_normal, up_proj)
    right_proj = right_proj / (np.linalg.norm(right_proj) + 1e-10)
    
    node_2d = {}
    for node_idx in range(6):
        pos = np.array(pos_3d[node_idx])
        x_2d = np.dot(pos, right_proj)
        y_2d = np.dot(pos, up_proj)
        node_2d[node_idx] = (x_2d, y_2d)
    
    node_angles = []
    for node_idx in range(6):
        x, y = node_2d[node_idx]
        angle = np.arctan2(y, x)
        node_angles.append((node_idx, angle, x, y))
    
    node_angles.sort(key=lambda x: (x[1] + np.pi/2) % (2*np.pi))
    
    position_labels = [
        "top",
        "top-right",
        "bottom-right",
        "bottom",
        "bottom-left",
        "top-left"
    ]
    
    map_path = output_dir / "node_position_map.txt"
    with open(map_path, 'w') as f:
        f.write("K6 Graph Node Position Mapping\n")
        f.write("=" * 40 + "\n\n")
        f.write("Hexagon layout (regular hexagon on plane perpendicular to camera view)\n")
        f.write("Camera: position (5, 5, 5), focal point (0, 0, 0), up (0, 0, 1)\n\n")
        f.write("Node positions (clockwise from top):\n")
        f.write("-" * 40 + "\n")
        
        for i, (node_idx, angle, x, y) in enumerate(node_angles):
            elem = node_to_element[node_idx]
            pos_label = position_labels[i]
            f.write(f"{pos_label:15s} -> Node {node_idx} -> Label: {elem}\n")
        
        f.write("\n" + "=" * 40 + "\n")
        f.write("Label files:\n")
        for node_idx in range(6):
            elem = node_to_element[node_idx]
            f.write(f"  label_{elem}.png\n")


def main() -> None:
    output_dir = make_output_dir()
    generate_k6_graph_image(output_dir)
    generate_labels(output_dir)
    generate_node_position_map(output_dir)


if __name__ == "__main__":
    main()
