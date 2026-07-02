"""
The transformer body + action-chunk output head, built on top of
EntityTokenizer (tokenizer.py).

Design, kept deliberately simple for a first working version:
  - A single learned "context" token (same idea as BERT's [CLS]) is
    prepended to the entity token set. Self-attention lets it gather
    information from all entities (proprio, object, goal, obstacles);
    padded obstacle slots are excluded via the attention mask.
  - After N transformer layers, the context token's output vector is fed
    through a small MLP head that predicts the ENTIRE action chunk at once
    (chunk_size * action_dim numbers, reshaped) -- this is the ACT-style
    "chunking" decision from earlier: one forward pass, one chunk of future
    actions, not one action at a time.

This is a v1: a natural upgrade later (not done here) is replacing the
single context token + MLP head with `chunk_size` separate learned query
tokens that cross-attend to the entity tokens (DETR/ACT-style), letting
each future timestep attend differently. Flagged, not built, to keep this
first version easy to follow end-to-end.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .tokenizer import EntityTokenizer


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    chunk_size: int = 12
    action_dim: int = 10


class ActionChunkTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = EntityTokenizer(d_model=cfg.d_model)
        self.context_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.ReLU(),
            nn.Linear(cfg.dim_feedforward, cfg.chunk_size * cfg.action_dim),
        )

    def forward(self, proprio, obj, goal, obstacles, obstacle_mask):
        """
        proprio: (B, F_proprio), obj: (B, F_object), goal: (B, F_goal)
        obstacles: (B, max_obstacles, F_obstacle), obstacle_mask: (B, max_obstacles)
        returns: predicted action chunk, (B, chunk_size, action_dim)
        """
        b = proprio.shape[0]
        tokens, mask = self.tokenizer(proprio, obj, goal, obstacles, obstacle_mask)  # (B, N, d), (B, N)

        ctx = self.context_token.expand(b, 1, self.cfg.d_model)
        tokens = torch.cat([ctx, tokens], dim=1)  # (B, 1+N, d) -- context token is always unmasked
        mask = torch.cat([torch.ones(b, 1, device=mask.device), mask], dim=1)  # (B, 1+N)

        # nn.TransformerEncoder wants a mask where True = IGNORE (opposite
        # convention from our "1 = attend" mask), applied per key.
        key_padding_mask = mask == 0  # (B, 1+N), True where padding

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)  # (B, 1+N, d)
        ctx_out = encoded[:, 0]  # (B, d) -- the context token's updated representation

        flat = self.action_head(ctx_out)  # (B, chunk_size * action_dim)
        return flat.view(b, self.cfg.chunk_size, self.cfg.action_dim)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, chunk_size, action_dim)
    mask: (B, chunk_size) -- 1 = real (padded-out chunk tail from the
          Dataset is 0 here and must not contribute to the loss)
    """
    err2 = (pred - target) ** 2  # (B, chunk_size, action_dim)
    err2 = err2.mean(dim=-1)      # (B, chunk_size)
    err2 = err2 * mask
    return err2.sum() / mask.sum().clamp(min=1.0)
