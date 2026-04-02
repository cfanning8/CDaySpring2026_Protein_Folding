from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import (  # noqa: E402
    HeatRandomFeatures,
    TopologicalRKHSLoss,
    persistence_diagram_to_vpd_vector,
    topological_loss_batch_numpy,
    virtual_difference_vector,
)


def main() -> None:
    diagram_a = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.9]], dtype=float)
    diagram_b = np.array([[0.1, 0.6], [0.4, 1.0]], dtype=float)

    vpd_a = persistence_diagram_to_vpd_vector(diagram_a, grid_size=16)
    vpd_b = persistence_diagram_to_vpd_vector(diagram_b, grid_size=16)
    vpd_diff = virtual_difference_vector(vpd_b, vpd_a)

    if vpd_a.shape != (256,) or vpd_b.shape != (256,):
        raise SystemExit("VPD vector shape mismatch")
    if not np.any(vpd_diff):
        raise SystemExit("VPD difference unexpectedly all zeros")

    features = HeatRandomFeatures(input_dim=256, n_components=64, random_state=14).transform(vpd_diff)
    if features.shape != (1, 128):
        raise SystemExit("RFF output shape mismatch")

    loss_fn = TopologicalRKHSLoss(grid_size=16, n_components=64, random_state=14)
    loss_value = loss_fn(vpd_diff.astype(np.float32))
    batch_loss = topological_loss_batch_numpy(
        np.vstack([vpd_diff.astype(np.float32), (-vpd_diff).astype(np.float32)]),
        loss_fn,
    )
    if not np.isfinite(loss_value) or loss_value < 0:
        raise SystemExit("Single topological loss invalid")
    if not np.isfinite(batch_loss) or batch_loss < 0:
        raise SystemExit("Batch topological loss invalid")

    print("Paper1 topology smoke test passed.")
    print(f"single_loss={loss_value:.6f}")
    print(f"batch_loss={batch_loss:.6f}")


if __name__ == "__main__":
    main()
