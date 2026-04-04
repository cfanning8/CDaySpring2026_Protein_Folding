#!/usr/bin/env python3
r"""Demonstrate differentiable surrogate RKHS loss on diagram **encodings** (Recipe C plumbing; DEF-S05).

Formal \(L_{\mathcal{H}}\) and \(\partial L_{\mathcal{H}}/\partial\omega\) use characters + spectral
symbol on the **virtual sum** (README **Full differential pipeline**); this script only checks
autograd through ``Enc(Δdiagram)`` → ``TopologicalRKHSLossTorch``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.training_contract import TopologyLossContract, demo_rkhs_gradient_on_encoding


def main() -> None:
    c = TopologyLossContract()
    gn = demo_rkhs_gradient_on_encoding(grid_size=c.grid_size)
    print("topology_contract_ok", "rkhs_encoding_grad_norm", float(gn))


if __name__ == "__main__":
    main()
