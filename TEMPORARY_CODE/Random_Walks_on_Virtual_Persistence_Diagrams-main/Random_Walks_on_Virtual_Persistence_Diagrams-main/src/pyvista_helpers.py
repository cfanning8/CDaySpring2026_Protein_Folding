"""PyVista 3D graph visualization."""

import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Any, Iterable, Optional

try:
    import pyvista as pv
    try:
        pv.start_xvfb()
    except (OSError, AttributeError):
        pass
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

FIGURE_DPI = 300


def visualize_graph_3d_pyvista(
    G: nx.Graph,
    pos_3d: Dict[int, Tuple[float, float, float]],
    edge_labels: Dict[Tuple[int, int], Any],
    output_path: Path,
    label_type: str = "scalar",
    t_samples: np.ndarray = None,
    highlight_nodes: Optional[Iterable[int]] = None,
    highlight_edges: Optional[Iterable[Tuple[int, int]]] = None,
) -> None:
    """Visualize graph in 3D using PyVista."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    plotter.render()
    
    camera_pos = np.array(plotter.camera.position)
    camera_up = np.array(plotter.camera.up)
    camera_up = camera_up / (np.linalg.norm(camera_up) + 1e-10)
    
    graph_center = np.mean(node_positions, axis=0)
    consistent_view_dir = camera_pos - graph_center
    consistent_view_dir = consistent_view_dir / (np.linalg.norm(consistent_view_dir) + 1e-10)
    
    camera_forward = -consistent_view_dir
    
    camera_right = np.cross(camera_up, camera_forward)
    camera_right = camera_right / (np.linalg.norm(camera_right) + 1e-10)
    camera_up_corrected = np.cross(camera_forward, camera_right)
    camera_up_corrected = camera_up_corrected / (np.linalg.norm(camera_up_corrected) + 1e-10)
    
    consistent_rot_matrix = np.eye(3)
    consistent_rot_matrix[:, 0] = camera_right
    consistent_rot_matrix[:, 1] = camera_up_corrected
    consistent_rot_matrix[:, 2] = camera_forward
    
    edges = list(G.edges())
    edge_midpoints = {}
    for u, v in edges:
        mid_point = (np.array(pos_3d[u]) + np.array(pos_3d[v])) / 2
        edge_midpoints[(u, v)] = mid_point
    
    squiggliest_edge = (1, 2)
    closest_edge = None
    for e in edges:
        if (e[0] == squiggliest_edge[0] and e[1] == squiggliest_edge[1]) or \
           (e[0] == squiggliest_edge[1] and e[1] == squiggliest_edge[0]):
            closest_edge = e
            break
    
    if closest_edge is None:
        edge_distances = []
        for u, v in edges:
            mid_point = edge_midpoints[(u, v)]
            dist = np.linalg.norm(mid_point - camera_pos)
            edge_distances.append(((u, v), dist))
        edge_distances.sort(key=lambda x: x[1])
        if len(edge_distances) >= 1:
            closest_edge = edge_distances[0][0]
        else:
            closest_edge = edges[0] if edges else None
    
    edge_radius = 0.015

    highlight_nodes_set = set(highlight_nodes) if highlight_nodes is not None else set()
    highlight_edges_set = set()
    if highlight_edges is not None:
        for u, v in highlight_edges:
            if (u, v) in edges or (v, u) in edges:
                highlight_edges_set.add((min(u, v), max(u, v)))
    
    def create_curved_edge_from_homeomorphism(u, v, phi_e, t_samples, start, end):
        """Create curved edge from homeomorphism phi_e."""
        if len(phi_e) != len(t_samples):
            t_old = np.linspace(0, 1, len(phi_e))
            phi_e = np.interp(t_samples, t_old, phi_e)
        
        n_points = len(t_samples)
        curve_points = []
        
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-10:
            direction_unit = direction / direction_norm
        else:
            direction_unit = np.array([1, 0, 0])
        
        if abs(direction_unit[2]) < 0.9:
            perp_vec = np.cross(direction_unit, np.array([0, 0, 1]))
        else:
            perp_vec = np.cross(direction_unit, np.array([1, 0, 0]))
        perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-10)
        
        max_offset = direction_norm * 0.15 * 2.0
        
        for i, t in enumerate(t_samples):
            pos_along_line = start + t * direction
            identity_value = t
            deviation = phi_e[i] - identity_value
            offset = perp_vec * deviation * max_offset
            curve_points.append(pos_along_line + offset)
        
        curve_points = np.array(curve_points)
        spline = pv.Spline(curve_points, n_points=n_points)
        return spline.tube(radius=edge_radius, n_sides=12)
    
    if label_type == "homeomorphism" and t_samples is not None:
        for u, v in G.edges():
            edge = tuple(sorted((u, v)))
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            
            phi_e = edge_labels.get(edge, t_samples / t_samples[-1])
            
            is_highlighted = (closest_edge is not None and 
                            ((u, v) == closest_edge or (v, u) == closest_edge))
            
            tube = create_curved_edge_from_homeomorphism(u, v, phi_e, t_samples, start, end)
            edge_color = '#333333'
            actor = plotter.add_mesh(tube, color=edge_color, opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
    else:
        for u, v in G.edges():
            if closest_edge is not None and ((u, v) == closest_edge or (v, u) == closest_edge):
                continue
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            line = pv.Line(start, end)
            tube = line.tube(radius=edge_radius, n_sides=12)
            edge_key = (min(u, v), max(u, v))
            color = '#aa0000' if edge_key in highlight_edges_set else '#333333'
            actor = plotter.add_mesh(tube, color=color, opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
        
        if closest_edge is not None:
            u, v = closest_edge
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            line = pv.Line(start, end)
            tube = line.tube(radius=edge_radius, n_sides=12)
            edge_key = (min(u, v), max(u, v))
            color = '#aa0000' if edge_key in highlight_edges_set else '#333333'
            actor = plotter.add_mesh(tube, color=color, opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
    
    node_size = 0.1
    for i in range(n):
        sphere = pv.Sphere(radius=node_size, center=node_positions[i], theta_resolution=30, phi_resolution=30)
        color = '#aa0000' if i in highlight_nodes_set else '#333333'
        actor = plotter.add_mesh(sphere, color=color, show_edges=False,
                        opacity=0.85, smooth_shading=True,
                        ambient=0.3, diffuse=0.6, specular=0.3, specular_power=30,
                        pbr=False, metallic=0.0, roughness=0.5)
        actor.GetProperty().SetLighting(True)
    
    plotter.renderer.SetUseFXAA(False)
    plotter.renderer.SetBackgroundAlpha(0.0)
    plotter.renderer.SetAutomaticLightCreation(False)
    plotter.renderer.SetTwoSidedLighting(True)
    
    plotter.render()
    
    import tempfile
    temp_img_path = output_path.parent / f"temp_{output_path.name}"
    plotter.screenshot(str(temp_img_path), transparent_background=True)
    plotter.close()
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    img_3d = Image.open(temp_img_path)
    img_width, img_height = img_3d.size
    
    fig, ax = plt.subplots(figsize=(img_width/FIGURE_DPI, img_height/FIGURE_DPI), dpi=FIGURE_DPI)
    ax.imshow(img_3d, origin='upper')
    ax.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=FIGURE_DPI, 
               facecolor='none', transparent=True)
    plt.close(fig)
    
    temp_img_path.unlink()
