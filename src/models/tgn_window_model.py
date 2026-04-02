from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory


class WindowTGNRegressor(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int = 64, time_dim: int = 16) -> None:
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=1,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim=1, memory_dim=memory_dim, time_dim=time_dim),
            aggregator_module=LastAggregator(),
        )
        self.embedding = WindowTemporalEmbedding(
            memory_dim=memory_dim,
            out_dim=memory_dim,
            raw_msg_dim=1,
            time_encoder=self.memory.time_enc,
        )
        self.readout = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid(),
        )

    def reset_state(self) -> None:
        self.memory.reset_state()

    def detach_memory(self) -> None:
        self.memory.detach()

    def encode_window(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t_start: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        raw_msg = duration.view(-1, 1).to(torch.float32)
        self.memory.update_state(src, dst, t_start, raw_msg)

        node_ids = torch.unique(torch.cat([src, dst], dim=0))
        assoc = torch.full((self.memory.num_nodes,), -1, dtype=torch.long, device=src.device)
        assoc[node_ids] = torch.arange(node_ids.size(0), device=src.device)
        edge_index = torch.stack([assoc[src], assoc[dst]], dim=0)
        valid = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        edge_index = edge_index[:, valid]
        if edge_index.numel() == 0:
            memory_values, _ = self.memory(node_ids)
            return memory_values.mean(dim=0)

        event_t = t_start.to(torch.float32)[valid]
        event_msg = raw_msg[valid]
        memory_values, last_update = self.memory(node_ids)
        z = self.embedding(memory_values, last_update, edge_index, event_t, event_msg)
        return z.mean(dim=0)

    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.readout(embedding).view(())


class WindowTemporalEmbedding(nn.Module):
    def __init__(self, memory_dim: int, out_dim: int, raw_msg_dim: int, time_encoder: nn.Module) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        edge_dim = raw_msg_dim + time_encoder.out_channels
        self.conv = TransformerConv(
            in_channels=memory_dim,
            out_channels=out_dim // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        node_memory: torch.Tensor,
        last_update: torch.Tensor,
        edge_index: torch.Tensor,
        event_t: torch.Tensor,
        event_msg: torch.Tensor,
    ) -> torch.Tensor:
        src_idx = edge_index[0]
        # Elapsed time semantics for TGN edge encoding.
        rel_t = event_t - last_update[src_idx].to(torch.float32)
        rel_t = torch.clamp(rel_t, min=0.0)
        rel_t_enc = self.time_encoder(rel_t)
        edge_attr = torch.cat([rel_t_enc, event_msg.to(node_memory.dtype)], dim=-1)
        return self.conv(node_memory, edge_index, edge_attr)
