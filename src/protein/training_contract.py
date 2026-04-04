r"""
Contract for **ColabFold / AF2-class fine-tuning** with **competing** topological losses.

This module does **not** wrap ColabFold internals (JAX). It documents and demonstrates the
**loss-side** objects that must attach to predicted coordinates (and native references):

- **Recipe A (Baseline):** \(L = L_{\mathrm{fold}}\) only.
- **Recipe B (Wasserstein):** \(L = L_{\mathrm{fold}} + \lambda_{\mathrm{W}} L_{\mathrm{W}}\)
  (diagram / Wasserstein at chosen **structural depth** — PH backend wired elsewhere).
- **Recipe C (RKHS):** \(L = L_{\mathrm{fold}} + \lambda_{\mathcal{H}} L_{\mathcal{H}}\)
  on **sum-pooled** internal virtual objects (README **Full differential pipeline** §7). This
  module’s demo uses `TopologicalRKHSLossTorch` on a **grid encoding** of diagram mass — gradients
  through **Enc**\((\cdot)\) only; the README gives the **intended** \(\widehat{X}\to\cdots\to\omega\) chain
  (piecewise-smooth strata).

See README **Design freeze** for non-cumulative stage semantics and ColabFold-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.topology.rkhs_torch import TopologicalRKHSLossTorch


Recipe = Literal["baseline", "wasserstein", "rkhs"]


@dataclass(frozen=True)
class TopologyLossContract:
    """Hyperparameters for the RKHS arm (matches NumPy `TopologicalRKHSLoss` init)."""

    grid_size: int = 50
    n_components: int = 256
    temperature: float = 0.2
    random_state: int | None = 42


def demo_rkhs_gradient_on_encoding(
    *,
    grid_size: int = 50,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Prove `TopologicalRKHSLossTorch` receives gradients w.r.t. a **diagram-difference encoding**
    (finite vector on the `grid_size²` grid used throughout the project).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = grid_size * grid_size
    x = torch.randn(g, device=device, dtype=torch.float32, requires_grad=True)
    mod = TopologicalRKHSLossTorch(grid_size=grid_size, random_state=0)
    loss = mod(x.unsqueeze(0))
    loss.backward()
    if x.grad is None:
        raise RuntimeError("expected non-None grad on diagram encoding")
    return x.grad.norm()
