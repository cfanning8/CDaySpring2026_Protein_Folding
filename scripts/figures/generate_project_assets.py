from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import pyvista as pv
except ImportError:
    pv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from persistence_diagram_3d_engine import (  # noqa: E402
    render_persistence_diagram_3d,
    render_virtual_persistence_diagram_3d,
)
from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402

COLOR_LOW = "#87CEEB"
COLOR_HIGH = "#DC143C"
COLOR_SECONDARY = "#FF8C00"
COLOR_ACCENT = "#4169E1"
FIGURE_DPI = 300
PRIMARY_SCHOOL_DATASET_KEY = r"primary_school\primaryschool.csv\primaryschool.csv"


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.table_csv)
    features = np.load(args.features_npz)
    output_root = Path(args.figure_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if pv is None:
        raise RuntimeError("PyVista is required for project asset generation.")

    generate_drift_scatter_assets(
        table,
        output_root / "drift_vs_risk_scatter_3d" / "assets",
        dataset_label=_dataset_label_from_key(args.dataset_key),
    )
    generate_rkhs_trajectory_assets(table, features, output_root / "rkhs_trajectory_3d" / "assets")
    generate_persistence_regime_assets(
        table,
        features,
        output_root / "persistence_regimes_3d" / "assets",
        dataset_key=args.dataset_key,
        min_events_per_window=args.min_events_per_window,
    )
    generate_model_comparison_assets(args.model_output_dir, output_root / "model_comparison_3d" / "assets")
    generate_pipeline_spec_assets(args.table_csv, args.features_npz, output_root / "pipeline_overview" / "assets")
    generate_collective_assets(
        output_root=output_root,
        collective_metrics_csv=args.collective_metrics_csv,
        collective_timeseries_csv=args.collective_timeseries_csv,
    )
    print(f"Generated project assets under {output_root}")


def generate_drift_timeseries_assets(table: pd.DataFrame, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    t_values = table["t"].to_numpy()
    y_values = table["y_large_outbreak_prob"].to_numpy()
    g_values = table["g_l2_norm"].to_numpy()
    g_scaled = (g_values - g_values.min()) / (g_values.max() - g_values.min() + 1e-12)
    y_ci = 1.96 * np.sqrt(np.maximum(y_values * (1.0 - y_values), 0.0) / 200.0)
    g_ci = np.full_like(g_scaled, max(float(np.std(g_scaled) * 0.20), 0.03))
    ci_diagnostics = {
        "median_y_ci_width": float(np.median(2.0 * y_ci)),
        "median_g_ci_width": float(np.median(2.0 * g_ci)),
    }
    show_y_ci = ci_diagnostics["median_y_ci_width"] >= 0.01
    show_g_ci = ci_diagnostics["median_g_ci_width"] >= 0.01

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    _add_floor_grid(plotter, float(t_values.min()), float(t_values.max()), -0.2, 1.2, z=0.0, steps=10)

    risk_points = np.column_stack([t_values, np.zeros_like(t_values), y_values])
    drift_points = np.column_stack([t_values, np.ones_like(t_values), g_scaled])

    risk_line = pv.lines_from_points(risk_points, close=False).tube(radius=0.05)
    drift_line = pv.lines_from_points(drift_points, close=False).tube(radius=0.05)
    plotter.add_mesh(risk_line, color=COLOR_HIGH, opacity=0.9)
    plotter.add_mesh(drift_line, color=COLOR_LOW, opacity=0.9)

    if show_y_ci:
        risk_ci = _create_ci_ribbon_surface(
            t_values,
            0.0,
            np.clip(y_values - y_ci, 0.0, 1.0),
            np.clip(y_values + y_ci, 0.0, 1.0),
        )
        plotter.add_mesh(risk_ci, color=COLOR_HIGH, opacity=0.75)
    if show_g_ci:
        drift_ci = _create_ci_ribbon_surface(
            t_values,
            1.0,
            np.clip(g_scaled - g_ci, 0.0, 1.0),
            np.clip(g_scaled + g_ci, 0.0, 1.0),
        )
        plotter.add_mesh(drift_ci, color=COLOR_LOW, opacity=0.75)

    risk_surface = _create_under_curve_surface(t_values, np.zeros_like(t_values), y_values)
    drift_surface = _create_under_curve_surface(t_values, np.ones_like(t_values), g_scaled)
    plotter.add_mesh(risk_surface, color=COLOR_HIGH, opacity=0.33)
    plotter.add_mesh(drift_surface, color=COLOR_LOW, opacity=0.33)

    _set_camera_head_on(plotter, pad=3.4, tilt=0.10)
    _save_plotter_image(plotter, asset_dir / "mesh_timeseries_3d.png")
    plotter.close()

    _create_latex_sticker(r"$Y_t$", asset_dir / "text_sticker_Yt.png", COLOR_HIGH, fontsize=30)
    _create_latex_sticker(r"$\Vert g_t \Vert_2$", asset_dir / "text_sticker_gt.png", COLOR_LOW, fontsize=28)
    if show_y_ci or show_g_ci:
        _create_latex_sticker(r"$95\%\ \mathrm{CI}$", asset_dir / "text_sticker_ci.png", COLOR_SECONDARY, fontsize=24)
    _create_latex_sticker(r"$\Delta t$", asset_dir / "text_sticker_t.png", COLOR_ACCENT, fontsize=24)

    pd.DataFrame(
        {
            "t": t_values,
            "y_t": y_values,
            "g_norm": g_values,
            "delta_y": table["delta_y_large_outbreak_prob"].to_numpy(),
            "y_ci": y_ci,
            "g_ci_scaled": g_ci,
            "show_y_ci": np.repeat(show_y_ci, len(t_values)),
            "show_g_ci": np.repeat(show_g_ci, len(t_values)),
        }
    ).to_csv(asset_dir / "text_timeseries.csv", index=False)
    pd.DataFrame([ci_diagnostics]).to_csv(asset_dir / "text_ci_diagnostics.csv", index=False)


def generate_drift_scatter_assets(table: pd.DataFrame, asset_dir: Path, dataset_label: str) -> None:
    _clear_asset_dir(asset_dir)
    x = table["g_l2_norm"].to_numpy()
    y = table["y_large_outbreak_prob"].to_numpy()
    z = table["delta_y_large_outbreak_prob"].to_numpy()

    _render_scatter_persistence_style(
        x=x,
        y=y,
        z=z,
        output_path=asset_dir / "mesh_scatter_3d.png",
    )

    _create_latex_sticker(r"$\Vert g_t \Vert_2$", asset_dir / "text_sticker_x.png", COLOR_LOW, fontsize=24)
    _create_latex_sticker(r"$Y_t$", asset_dir / "text_sticker_y.png", COLOR_HIGH, fontsize=24)
    _create_latex_sticker(r"$\Delta Y_t$", asset_dir / "text_sticker_z.png", COLOR_SECONDARY, fontsize=24)
    _create_latex_sticker(
        rf"$\mathrm{{{dataset_label}}}$",
        asset_dir / "text_sticker_dataset.png",
        COLOR_ACCENT,
        fontsize=18,
        fig_width=3.8,
    )
    pd.DataFrame({"g_norm": x, "y_t": y, "delta_y": z}).to_csv(asset_dir / "text_scatter.csv", index=False)
    pd.DataFrame(
        [
            {
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "z_min": float(np.min(z)),
                "z_max": float(np.max(z)),
            }
        ]
    ).to_csv(asset_dir / "text_scatter_debug_ranges.csv", index=False)


def generate_rkhs_trajectory_assets(table: pd.DataFrame, features: np.lib.npyio.NpzFile, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    rkhs = features["rkhs_g_t"]
    y_vals = table["y_large_outbreak_prob"].to_numpy()
    pca = PCA(n_components=3, random_state=14, whiten=True)
    coords = pca.fit_transform(rkhs)
    coords = coords * 3.0

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    points = pv.PolyData(coords)
    points["risk"] = y_vals
    mesh = points.glyph(scale=False, orient=False, geom=pv.Sphere(radius=0.08))
    plotter.add_mesh(mesh, scalars="risk", cmap="coolwarm", opacity=0.9, show_scalar_bar=False)
    for idx in range(coords.shape[0] - 1):
        p0 = coords[idx]
        p1 = coords[idx + 1]
        s = float(0.5 * (y_vals[idx] + y_vals[idx + 1]))
        color = _blend_color(COLOR_LOW, COLOR_HIGH, s)
        seg = pv.Line(tuple(p0), tuple(p1)).tube(radius=0.02)
        plotter.add_mesh(seg, color=color, opacity=0.88)
    _set_camera_head_on(plotter, pad=4.8, tilt=0.18)
    _save_plotter_image(plotter, asset_dir / "mesh_rkhs_trajectory.png")
    plotter.close()

    points_only = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    points_only.set_background([0, 0, 0, 0])
    points_only.add_mesh(mesh, scalars="risk", cmap="coolwarm", opacity=0.90, show_scalar_bar=False)
    _set_camera_head_on(points_only, pad=4.8, tilt=0.18)
    _save_plotter_image(points_only, asset_dir / "mesh_rkhs_points_only.png")
    points_only.close()

    _create_latex_sticker(r"$\Phi(g_t)$", asset_dir / "text_sticker_rkhs.png", COLOR_SECONDARY, fontsize=28)

    pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "risk": y_vals}
    ).to_csv(asset_dir / "text_rkhs_trajectory.csv", index=False)
    pd.DataFrame(
        [
            {
                "x_span": float(np.max(coords[:, 0]) - np.min(coords[:, 0])),
                "y_span": float(np.max(coords[:, 1]) - np.min(coords[:, 1])),
                "z_span": float(np.max(coords[:, 2]) - np.min(coords[:, 2])),
            }
        ]
    ).to_csv(asset_dir / "text_rkhs_debug_ranges.csv", index=False)


def generate_persistence_regime_assets(
    table: pd.DataFrame,
    features: np.lib.npyio.NpzFile,
    asset_dir: Path,
    dataset_key: str,
    min_events_per_window: int,
) -> None:
    _clear_asset_dir(asset_dir)
    d_values = features["d_t"]
    grid_size = int(np.sqrt(d_values.shape[1] // 2))
    idx_low = int(np.argmin(table["y_large_outbreak_prob"].to_numpy()))
    idx_high = int(np.argmax(table["y_large_outbreak_prob"].to_numpy()))
    idx_transition = int(np.argmax(np.abs(table["delta_y_large_outbreak_prob"].to_numpy())))
    windows = _reconstruct_windows_from_table(table, dataset_key=dataset_key, min_events=min_events_per_window)
    regimes = [("low", idx_low), ("transition", idx_transition), ("high", idx_high)]
    regime_rows = []

    for label, idx in regimes:
        safe_idx = int(np.clip(idx, 0, d_values.shape[0] - 1))
        next_idx = int(np.clip(safe_idx + 1, 0, d_values.shape[0] - 1))
        if next_idx == safe_idx and safe_idx > 0:
            next_idx = safe_idx - 1
        diag_t = _vpd_vector_to_diagram(d_values[safe_idx], grid_size)
        diag_next = _vpd_vector_to_diagram(d_values[next_idx], grid_size)

        render_persistence_diagram_3d(
            diag_t,
            asset_dir / f"mesh_{label}_persistence_3d.png",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            elev=25,
            azim=-60,
            bar_scale=1.0,
        )
        render_virtual_persistence_diagram_3d(
            diag_next,
            diag_t,
            asset_dir / f"mesh_{label}_virtual_persistence_3d.png",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            elev=25,
            azim=-60,
            bar_scale=1.0,
        )
        if safe_idx < len(windows):
            edges = _aggregate_window_edges(windows[safe_idx])
            _render_network_asset(edges, asset_dir / f"mesh_{label}_network_3d.png")
            regime_rows.append({"label": label, "t": safe_idx, "num_nodes": len(set(edges["source"]).union(set(edges["target"]))), "num_edges": len(edges)})
        if label == "low":
            _create_latex_sticker(r"$\mathrm{low}$", asset_dir / f"text_sticker_{label}.png", COLOR_LOW, fontsize=24)
        elif label == "high":
            _create_latex_sticker(r"$\mathrm{high}$", asset_dir / f"text_sticker_{label}.png", COLOR_HIGH, fontsize=24)
        else:
            _create_latex_sticker(r"$\mathrm{transition}$", asset_dir / f"text_sticker_{label}.png", COLOR_SECONDARY, fontsize=24)

    pd.DataFrame(regime_rows).to_csv(asset_dir / "text_regime_summary.csv", index=False)


def generate_model_comparison_assets(model_output_dir: Path, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    files = [
        ("tgn_baseline_predictions.csv", "TGN"),
        ("tgn_perslay_constraint_predictions.csv", "PersLay"),
        ("tgn_rkhs_constraint_predictions.csv", "RKHS"),
    ]
    rows = []
    for filename, model in files:
        path = model_output_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rmse = float(np.sqrt(np.mean((df["y_pred"] - df["y_true"]) ** 2)))
        rows.append({"model": model, "rmse": rmse})
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(asset_dir / "text_model_scores.csv", index=False)
    model_order = ["TGN", "PersLay", "RKHS"]
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")
    colors = {"TGN": COLOR_LOW, "PersLay": COLOR_SECONDARY, "RKHS": COLOR_HIGH}

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=FIGURE_DPI, facecolor="none")
    x = np.arange(len(summary))
    y = summary["rmse"].to_numpy(dtype=float)
    c = [colors[str(m)] for m in summary["model"]]
    bars = ax.bar(x, y, color=c, alpha=0.88, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].astype(str).tolist())
    ax.set_ylabel("RMSE")
    ax.set_ylim(0.0, max(float(np.max(y)) * 1.15, 1e-6))
    ax.grid(axis="y", linestyle=":", alpha=0.30)
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            float(value) + 0.01 * max(float(np.max(y)), 1.0),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )
    fig.savefig(asset_dir / "curve_model_rmse_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    _create_latex_sticker(r"$\mathrm{RMSE}$", asset_dir / "text_sticker_rmse.png", COLOR_ACCENT, fontsize=22)
    for model in summary["model"].astype(str).tolist():
        _create_latex_sticker(
            rf"$\mathrm{{{model}}}$",
            asset_dir / f"text_sticker_{model.lower()}.png",
            colors[model],
            fontsize=18,
            fig_width=2.6,
        )


def generate_pipeline_spec_assets(table_csv: Path, features_npz: Path, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    steps = [
        {"name": "events_to_windows", "input": str(table_csv), "output": "windowed_events"},
        {"name": "windows_to_D_t", "input": "windowed_events", "output": str(features_npz)},
        {"name": "D_t_to_g_t", "input": str(features_npz), "output": "g_t"},
        {"name": "g_t_to_prediction", "input": "g_t + TGN", "output": "y_t"},
    ]
    pd.DataFrame(steps).to_csv(asset_dir / "text_pipeline_steps.csv", index=False)
    _render_single_box_asset(asset_dir / "mesh_box_blue.png", COLOR_LOW)
    _render_single_box_asset(asset_dir / "mesh_box_red.png", COLOR_HIGH)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_blue_to_red.png", COLOR_LOW, COLOR_HIGH)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_red_to_blue.png", COLOR_HIGH, COLOR_LOW)

    _create_latex_sticker(r"$G_t$", asset_dir / "text_sticker_Gt.png", COLOR_LOW, fontsize=28)
    _create_latex_sticker(r"$D_t$", asset_dir / "text_sticker_Dt.png", COLOR_HIGH, fontsize=28)
    _create_latex_sticker(r"$g_t=D_{t+\Delta}-D_t$", asset_dir / "text_sticker_gt_eq.png", COLOR_SECONDARY, fontsize=18, fig_width=4.8)
    _create_latex_sticker(r"$\hat{Y}_t=\mathbb{P}(A_t\geq\tau)$", asset_dir / "text_sticker_target.png", COLOR_ACCENT, fontsize=20)


def generate_collective_assets(output_root: Path, collective_metrics_csv: Path, collective_timeseries_csv: Path) -> None:
    if collective_metrics_csv.exists():
        metrics = pd.read_csv(collective_metrics_csv)
        _generate_collective_model_comparison(output_root / "model_comparison_3d" / "assets" / "total" / "assets", metrics)
        _generate_collective_results_table(output_root / "results_table_figure" / "assets" / "total" / "assets", metrics)
    if collective_timeseries_csv.exists():
        ts = pd.read_csv(collective_timeseries_csv)
        _generate_collective_timeseries(output_root / "drift_risk_timeseries_3d" / "assets" / "total" / "assets", ts)


def _generate_collective_timeseries(asset_dir: Path, collective_ts: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    datasets = list(dict.fromkeys(collective_ts["dataset"].tolist()))
    cmap = plt.get_cmap("tab10")
    normalized = collective_ts.copy()
    for dataset in datasets:
        ds = normalized[normalized["dataset"] == dataset].sort_values("t")
        if len(ds) <= 1:
            t_rel = np.zeros(len(ds), dtype=float)
        else:
            t_rel = np.linspace(0.0, 1.0, len(ds))
        normalized.loc[ds.index, "t_rel"] = t_rel

    fig, ax = plt.subplots(figsize=(12, 7), dpi=FIGURE_DPI, facecolor="none")
    for idx, dataset in enumerate(datasets):
        ds = normalized[normalized["dataset"] == dataset].sort_values("t_rel")
        x = ds["t_rel"].to_numpy(dtype=float)
        y = ds["y_t"].to_numpy(dtype=float)
        color = cmap(idx % 10)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.0,
            alpha=0.90,
            zorder=3,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$Y_t$")
    ax.grid(alpha=0.25, linestyle=":")
    fig.savefig(asset_dir / "curve_collective_timeseries_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    _create_latex_sticker(r"$\mathrm{Time}$", asset_dir / "text_sticker_time.png", COLOR_ACCENT, fontsize=22, fig_width=2.4)
    _create_latex_sticker(r"$Y_t$", asset_dir / "text_sticker_Yt.png", COLOR_HIGH, fontsize=22, fig_width=1.8)
    for idx, dataset in enumerate(datasets):
        dataset_color = mcolors.to_hex(cmap(idx % 10))
        _create_latex_sticker(
            rf"$\mathrm{{{dataset}}}$",
            asset_dir / f"text_sticker_dataset_{dataset.lower()}.png",
            dataset_color,
            fontsize=16,
            fig_width=3.2,
        )


def _generate_collective_model_comparison(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    model_order = ["TGN", "PersLay", "RKHS"]
    datasets = list(dict.fromkeys(metrics["dataset"].tolist()))
    colors = {"TGN": COLOR_LOW, "PersLay": COLOR_SECONDARY, "RKHS": COLOR_HIGH}

    pivot = metrics.pivot(index="dataset", columns="model", values="rmse").reindex(index=datasets, columns=model_order)
    if pivot.isna().any().any():
        missing = pivot.isna()
        missing_pairs = [(datasets[i], model_order[j]) for i, j in np.argwhere(missing.to_numpy())]
        raise ValueError(f"missing collective model metrics rows: {missing_pairs}")

    x = np.arange(len(datasets), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI, facecolor="none")
    for idx, model in enumerate(model_order):
        y = pivot[model].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, y, width=width, label=model, color=colors[model], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(asset_dir / "curve_collective_model_rmse_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    metrics.to_csv(asset_dir / "text_collective_model_metrics.csv", index=False)


def _generate_collective_results_table(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    rows = []
    model_order = ["TGN", "PersLay", "RKHS"]
    for dataset in dict.fromkeys(metrics["dataset"].tolist()):
        subset = metrics[metrics["dataset"] == dataset]
        for model in model_order:
            row = subset[subset["model"] == model]
            if row.empty:
                continue
            if "brier" not in row.columns:
                raise ValueError("collective metrics is missing required column 'brier'")
            if "ece" not in row.columns:
                raise ValueError("collective metrics is missing required column 'ece'")
            if "eval_count" not in row.columns:
                raise ValueError("collective metrics is missing required column 'eval_count'")
            brier_value = float(row.iloc[0]["brier"])
            ece_value = float(row.iloc[0]["ece"])
            eval_count = float(row.iloc[0]["eval_count"])
            y = {
                "Dataset": dataset,
                "Model": model,
                "RMSE": float(row.iloc[0]["rmse"]),
                "Brier": brier_value,
                "ECE": ece_value,
                "EvalCount": eval_count,
            }
            rows.append(y)
    table_df = pd.DataFrame(rows)
    table_df.to_csv(asset_dir / "text_collective_results_table.csv", index=False)
    _render_collective_results_table_strict(table_df, asset_dir / "table_collective_results_main.png")


def _render_collective_results_table_strict(table_df: pd.DataFrame, output_path: Path) -> None:
    required_cols = {"Dataset", "Model", "RMSE", "Brier", "ECE"}
    missing = required_cols.difference(set(table_df.columns))
    if missing:
        raise ValueError(f"missing collective table columns: {sorted(missing)}")

    grouped = []
    for dataset in dict.fromkeys(table_df["Dataset"].tolist()):
        group = table_df[table_df["Dataset"] == dataset]
        if len(group) != 3:
            raise ValueError(f"expected exactly 3 model rows for dataset '{dataset}', found {len(group)}")
        grouped.append((dataset, group))

    col_widths = [2.5, 1.5, 1.3, 1.3, 1.3]
    row_h = 0.60
    header_h = 0.70
    n_rows = len(table_df)
    fig_w = sum(col_widths)
    fig_h = header_h + n_rows * row_h + 0.30
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIGURE_DPI, facecolor="none")
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    x0 = 0.0
    y_top = fig_h - 0.15
    headers = ["Dataset", "Model", "RMSE", "Brier", "ECE"]
    for col_idx, (w, label) in enumerate(zip(col_widths, headers)):
        rect = plt.Rectangle((x0, y_top - header_h), w, header_h, facecolor=(0.85, 0.85, 0.85, 0.98), edgecolor="#222222", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x0 + w * 0.5, y_top - header_h * 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold")
        x0 += w

    current_y = y_top - header_h
    for dataset, group in grouped:
        dataset_h = row_h * 3.0
        drect = plt.Rectangle((0.0, current_y - dataset_h), col_widths[0], dataset_h, facecolor=(0.92, 0.92, 0.92, 0.95), edgecolor="#222222", linewidth=1.0)
        ax.add_patch(drect)
        ax.text(col_widths[0] * 0.5, current_y - dataset_h * 0.5, str(dataset), ha="center", va="center", fontsize=10)

        row_y = current_y
        for _, row in group.iterrows():
            entries = [
                str(row["Model"]),
                f"{float(row['RMSE']):.3f}",
                f"{float(row['Brier']):.3f}",
                f"{float(row['ECE']):.3f}",
            ]
            x = col_widths[0]
            for idx, txt in enumerate(entries):
                w = col_widths[idx + 1]
                rect = plt.Rectangle((x, row_y - row_h), w, row_h, facecolor=(1.0, 1.0, 1.0, 0.90), edgecolor="#222222", linewidth=1.0)
                ax.add_patch(rect)
                ax.text(x + w * 0.5, row_y - row_h * 0.5, txt, ha="center", va="center", fontsize=10)
                x += w
            row_y -= row_h
        current_y -= dataset_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project figure assets from selected results.")
    parser.add_argument("--table-csv", type=Path, required=True, help="Input table CSV path to use for assets.")
    parser.add_argument("--features-npz", type=Path, required=True, help="Input features NPZ path to use for assets.")
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "output",
        help="Directory containing model prediction CSV outputs.",
    )
    parser.add_argument(
        "--figure-output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
        help="Root directory for generated figure asset folders.",
    )
    parser.add_argument(
        "--collective-metrics-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "collective_metrics.csv",
    )
    parser.add_argument(
        "--collective-timeseries-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "collective_timeseries.csv",
    )
    parser.add_argument("--dataset-key", type=str, default=PRIMARY_SCHOOL_DATASET_KEY)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    return parser.parse_args()


def _create_under_curve_surface(x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray) -> pv.PolyData:
    points = []
    faces = []
    for idx in range(len(x_values) - 1):
        p0 = [x_values[idx], y_values[idx], 0.0]
        p1 = [x_values[idx + 1], y_values[idx + 1], 0.0]
        p2 = [x_values[idx + 1], y_values[idx + 1], z_values[idx + 1]]
        p3 = [x_values[idx], y_values[idx], z_values[idx]]
        base = len(points)
        points.extend([p0, p1, p2, p3])
        faces.extend([4, base, base + 1, base + 2, base + 3])
    poly = pv.PolyData(np.array(points), np.array(faces))
    return poly.triangulate()


def _create_ci_ribbon_surface(x_values: np.ndarray, y_center: float, z_low: np.ndarray, z_high: np.ndarray) -> pv.PolyData:
    points = []
    faces = []
    for idx in range(len(x_values) - 1):
        p0 = [x_values[idx], y_center - 0.06, z_low[idx]]
        p1 = [x_values[idx + 1], y_center - 0.06, z_low[idx + 1]]
        p2 = [x_values[idx + 1], y_center + 0.06, z_high[idx + 1]]
        p3 = [x_values[idx], y_center + 0.06, z_high[idx]]
        base = len(points)
        points.extend([p0, p1, p2, p3])
        faces.extend([4, base, base + 1, base + 2, base + 3])
    poly = pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))
    return poly.triangulate()


def _add_floor_grid(plotter: pv.Plotter, x_min: float, x_max: float, y_min: float, y_max: float, z: float, steps: int) -> None:
    for x in np.linspace(x_min, x_max, steps):
        line = pv.Line((float(x), y_min, z), (float(x), y_max, z))
        plotter.add_mesh(line.tube(radius=0.004), color="#999999", opacity=0.55)
    for y in np.linspace(y_min, y_max, steps):
        line = pv.Line((x_min, float(y), z), (x_max, float(y), z))
        plotter.add_mesh(line.tube(radius=0.004), color="#999999", opacity=0.55)
    plane = pv.Plane(center=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z), i_size=max(x_max - x_min, 1e-6), j_size=max(y_max - y_min, 1e-6))
    plotter.add_mesh(plane, color="white", opacity=0.20)


def _render_scatter_persistence_style(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_path: Path) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    x_n = (x - x_min) / (x_max - x_min + 1e-12)
    y_n = (y - y_min) / (y_max - y_min + 1e-12)
    z_n = (z - z_min) / (z_max - z_min + 1e-12)
    heights = 0.08 + 0.92 * z_n

    fig = plt.figure(figsize=(10, 10), dpi=FIGURE_DPI, facecolor="none")
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    z_grid, z_plane, z_pts = 1, 2, 4
    for val in np.linspace(0.0, 1.0, 6):
        ax.plot([val, val], [0.0, 1.0], [0.0, 0.0], color="#999999", linewidth=2, alpha=0.6, zorder=z_grid)
        ax.plot([0.0, 1.0], [val, val], [0.0, 0.0], color="#999999", linewidth=2, alpha=0.6, zorder=z_grid)
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2))
    zz = np.zeros_like(xx)
    surf = ax.plot_surface(xx, yy, zz, color="white", alpha=0.5, shade=False)
    surf.set_zorder(z_plane)

    for i in range(len(x_n)):
        col = _blend_color(COLOR_LOW, COLOR_HIGH, float(y_n[i]))
        ax.plot([x_n[i], x_n[i]], [y_n[i], y_n[i]], [0.0, heights[i]], color=col, linewidth=3.2, alpha=0.88, zorder=z_pts)
        ax.scatter([x_n[i]], [y_n[i]], [heights[i]], c=[col], s=90, alpha=0.92, edgecolors="black", linewidths=0.7, zorder=z_pts)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.view_init(elev=25, azim=-60)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.2, facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)


def _create_text_sticker(text: str, output_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 0.9), dpi=FIGURE_DPI, facecolor="none")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        fontsize=12,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.9),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.close(fig)


def _dataset_label_from_key(dataset_key: str) -> str:
    token = dataset_key.split("\\")[0].strip().lower()
    mapping = {
        "high_school": "Thiers13",
        "primary_school": "LyonSchool",
        "workplace": "InVS15",
        "hypertext": "HT2009",
        "infectious": "Infectious",
    }
    return mapping.get(token, token)


def _create_latex_sticker(
    latex_text: str,
    output_path: Path,
    color: str,
    fontsize: int,
    fig_width: float = 2.8,
    fig_height: float = 0.9,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIGURE_DPI, facecolor="none")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        latex_text,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(1.0, 1.0, 1.0, 0.55), edgecolor=color, linewidth=0.8),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)


def _clear_asset_dir(asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for path in asset_dir.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)


def _reconstruct_windows_from_table(table: pd.DataFrame, dataset_key: str, min_events: int) -> list[pd.DataFrame]:
    datasets = load_all_datasets(PROJECT_ROOT / "data")
    if dataset_key not in datasets:
        return []
    temporal_result = extract_temporal_events_for_dataset(dataset_key, datasets[dataset_key])
    if temporal_result is None:
        return []
    events = temporal_result.events.copy()
    window_seconds = float(np.median(table["window_end"].to_numpy() - table["window_start"].to_numpy()))
    starts = np.sort(table["window_start"].to_numpy())
    if len(starts) > 1:
        stride_seconds = float(np.median(np.diff(starts)))
    else:
        stride_seconds = max(window_seconds * 0.5, 1.0)
    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    windows = []
    cursor = t_min
    while cursor + window_seconds <= t_max + 1e-9:
        mask = (events["t_start"] >= cursor) & (events["t_start"] < cursor + window_seconds)
        window_events = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window_events) >= min_events:
            windows.append(window_events)
        cursor += stride_seconds
    return windows


def _aggregate_window_edges(window_events: pd.DataFrame) -> pd.DataFrame:
    ordered_source = np.where(
        window_events["source"].astype(str).to_numpy() <= window_events["target"].astype(str).to_numpy(),
        window_events["source"].astype(str).to_numpy(),
        window_events["target"].astype(str).to_numpy(),
    )
    ordered_target = np.where(
        window_events["source"].astype(str).to_numpy() <= window_events["target"].astype(str).to_numpy(),
        window_events["target"].astype(str).to_numpy(),
        window_events["source"].astype(str).to_numpy(),
    )
    edges = pd.DataFrame(
        {
            "source": ordered_source,
            "target": ordered_target,
            "duration_seconds": pd.to_numeric(window_events["duration_seconds"], errors="coerce").fillna(0.0),
        }
    )
    edges = edges[(edges["source"] != edges["target"]) & (edges["duration_seconds"] > 0)]
    if edges.empty:
        return edges
    return (
        edges.groupby(["source", "target"], as_index=False)
        .agg(duration_seconds=("duration_seconds", "sum"))
        .sort_values(["source", "target"])
    )


def _render_network_asset(edges: pd.DataFrame, output_path: Path) -> None:
    if edges.empty:
        return
    nodes = sorted(set(edges["source"]).union(set(edges["target"])))
    angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
    coords = {node: np.array([np.cos(ang), np.sin(ang), 0.0]) for node, ang in zip(nodes, angles)}
    max_w = float(max(edges["duration_seconds"].max(), 1.0))

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    _add_floor_plane(plotter, -1.4, 1.4, -1.4, 1.4, z=-0.03)
    for _, row in edges.iterrows():
        p0 = coords[str(row["source"])]
        p1 = coords[str(row["target"])]
        weight_scale = float(row["duration_seconds"]) / max_w
        line = pv.Line(tuple(p0), tuple(p1)).tube(radius=0.0015 + 0.028 * (weight_scale**1.5))
        edge_color = _blend_color(COLOR_LOW, COLOR_HIGH, weight_scale)
        plotter.add_mesh(line, color=edge_color, opacity=0.12 + 0.80 * weight_scale)
    for node in nodes:
        sphere = pv.Sphere(radius=0.04, center=tuple(coords[node]))
        plotter.add_mesh(sphere, color="#B0B0B0", opacity=0.84)
    _set_camera_overhead(plotter, pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _add_floor_plane(plotter: pv.Plotter, x_min: float, x_max: float, y_min: float, y_max: float, z: float) -> None:
    plane = pv.Plane(
        center=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z),
        i_size=max(x_max - x_min, 1e-6),
        j_size=max(y_max - y_min, 1e-6),
    )
    plotter.add_mesh(plane, color="white", opacity=0.20)


def _vpd_vector_to_diagram(vector: np.ndarray, grid_size: int) -> np.ndarray:
    half = grid_size * grid_size
    h0 = np.asarray(vector[:half], dtype=float).reshape(grid_size, grid_size)
    h1 = np.asarray(vector[half:], dtype=float).reshape(grid_size, grid_size)
    rows: list[list[float]] = []
    for dim, matrix in ((0, h0), (1, h1)):
        for i in range(grid_size):
            for j in range(grid_size):
                multiplicity = int(round(matrix[i, j]))
                if multiplicity <= 0:
                    continue
                birth = (i + 0.5) / grid_size
                death = (j + 0.5) / grid_size
                if death <= birth:
                    continue
                rows.append([float(dim), float(birth), float(death), float(multiplicity)])
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _blend_color(hex_a: str, hex_b: str, t: float) -> tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    a = tuple(int(hex_a.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    b = tuple(int(hex_b.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _set_camera_head_on(plotter: pv.Plotter, pad: float, tilt: float) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = plotter.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    dz = max(z_max - z_min, 1e-6)
    dist = pad * max(dx, dy, dz)
    plotter.camera.position = (cx, cy - dist, cz + tilt * dist)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0, 0, 1)


def _set_camera_overhead(plotter: pv.Plotter, pad: float) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = plotter.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    d = pad * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
    plotter.camera.position = (cx, cy, cz + d)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0, 1, 0)


def _render_single_box_asset(output_path: Path, color: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    box = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=0.6)
    plotter.add_mesh(box, color=color, opacity=0.84)
    _set_camera_head_on(plotter, pad=2.6, tilt=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_arc_arrow(output_path: Path, color_start: str, color_end: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    n = 80
    t_values = np.linspace(0.0, 1.0, n)
    points = []
    for t in t_values:
        x = -0.8 + 1.6 * t
        y = 0.35 * np.sin(np.pi * t)
        z = 0.18 * (1.0 - np.cos(np.pi * t))
        points.append((x, y, z))
    for idx in range(n - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        seg = pv.Line(p0, p1).tube(radius=0.018)
        plotter.add_mesh(seg, color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.18, radius=0.06, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.0, tilt=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _save_plotter_image(plotter: pv.Plotter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = plotter.screenshot(transparent_background=True, return_img=True)
    try:
        from PIL import Image

        Image.fromarray(image).save(output_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    except Exception:
        plt.imsave(output_path, image)


if __name__ == "__main__":
    main()
