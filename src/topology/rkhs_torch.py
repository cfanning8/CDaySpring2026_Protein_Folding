r"""
Differentiable RKHS semimetric loss on finite diagram encodings (RFF on torus).

Uses the same `HeatRandomFeatures` random draw as `TopologicalRKHSLoss` (NumPy) so
init hyperparameters match; forward pass is pure torch for autograd w.r.t. `vpd_diff`.

This closes **DEF-L02** for gradients through the **encoded diagram difference** only.
Gradients through persistent homology itself require a separate surrogate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.topology.loss import TopologicalRKHSLoss


class TopologicalRKHSLossTorch(nn.Module):
    r"""2(k(0) - k(gamma,0)) with RFF embedding, clamped at 0; batch mean loss."""

    def __init__(
        self,
        grid_size: int = 50,
        n_components: int = 256,
        temperature: float = 0.2,
        lambda_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        np_fn = TopologicalRKHSLoss(
            grid_size=grid_size,
            n_components=n_components,
            temperature=temperature,
            lambda_weights=lambda_weights,
            random_state=random_state,
        )
        rff = np_fn.rff
        self._input_dim = int(np_fn.input_dim)
        self._n_components = int(rff.n_components)
        self.register_buffer("omega", torch.from_numpy(np.asarray(rff.omega_, dtype=np.float32)))
        self.register_buffer("bias", torch.from_numpy(np.asarray(rff.bias_, dtype=np.float32)))
        self.register_buffer("scale", torch.from_numpy(np.asarray(rff.scale_, dtype=np.float32)))

        z = torch.zeros(self._input_dim, dtype=torch.float32)
        z_embed = self._transform(z.unsqueeze(0))[0].detach()
        self.register_buffer("_zero_embed", z_embed)
        self.register_buffer("_k_zero", torch.dot(z_embed, z_embed))

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * torch.pi
        om = self.omega.to(device=x.device, dtype=x.dtype)
        ba = self.bias.to(device=x.device, dtype=x.dtype)
        proj = (x @ om.T + ba) % two_pi
        c = torch.cos(proj)
        s = torch.sin(proj)
        nf = torch.empty(x.shape[0], 2 * self._n_components, device=x.device, dtype=x.dtype)
        sc = self.scale.to(device=x.device, dtype=x.dtype)
        nf[:, 0::2] = sc * c
        nf[:, 1::2] = sc * s
        return nf

    def forward(self, vpd_diff: torch.Tensor) -> torch.Tensor:
        if vpd_diff.ndim == 1:
            vpd_diff = vpd_diff.unsqueeze(0)
        x = vpd_diff.float()
        if x.shape[1] != self._input_dim:
            raise ValueError(f"expected dim {self._input_dim}, got {x.shape[1]}")
        emb = self._transform(x)
        ze = self._zero_embed.to(device=x.device, dtype=x.dtype)
        k_gz = (emb * ze.unsqueeze(0)).sum(dim=1)
        k0 = self._k_zero.to(device=x.device, dtype=x.dtype)
        loss = 2.0 * (k0 - k_gz)
        loss = torch.clamp(loss, min=0.0)
        finite = torch.isfinite(loss)
        loss = torch.where(finite, loss, torch.zeros_like(loss))
        return loss.mean()
