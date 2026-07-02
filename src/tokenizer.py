"""
EntityTokenizer: turns the grouped raw feature vectors (one per entity --
proprio, object, goal, obstacles) into a set of tagged tokens a transformer
can attend over.

    token = Linear_kind(raw_features) + type_embedding[TYPE_ID[kind]]

Two things are worth being explicit about, since they came up in discussion:
  - WHICH type each token gets is fixed, hardcoded bookkeeping (entities.py
    TYPE_ID) -- not learned, not inferred, decided once by us.
  - WHAT each type's tag vector actually contains (the embedding_table
    rows) IS learned, same as any other weight.

Obstacles are handled as a padded, masked, variable-count group so that
going from 0 obstacles (today) to N obstacles (once that data exists)
requires no architecture change -- only obstacle_features() in entities.py
needs to start returning real numbers instead of an empty array.
"""

import torch
import torch.nn as nn

from . import entities


class EntityTokenizer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # One projection per entity kind -- different kinds have different
        # raw feature dimensions, so they can't share a single Linear.
        self.proj = nn.ModuleDict({
            kind: nn.Linear(dim, d_model)
            for kind, dim in entities.FEATURE_DIM.items()
        })

        # The learned "tag" table: one row per entity kind. Started random,
        # adjusted during training like everything else.
        self.type_embedding = nn.Embedding(entities.NUM_TYPES, d_model)

    def _tag(self, kind: str) -> torch.Tensor:
        type_id = torch.tensor(entities.TYPE_ID[kind])
        return self.type_embedding(type_id)  # (d_model,)

    def forward(
        self,
        proprio: torch.Tensor,   # (B, FEATURE_DIM['proprio'])
        obj: torch.Tensor,       # (B, FEATURE_DIM['object'])
        goal: torch.Tensor,      # (B, FEATURE_DIM['goal'])
        obstacles: torch.Tensor,       # (B, max_obstacles, FEATURE_DIM['obstacle'])
        obstacle_mask: torch.Tensor,   # (B, max_obstacles) -- 1 = real, 0 = padding
    ):
        """
        Returns:
            tokens: (B, 3 + max_obstacles, d_model) -- proprio, object, goal,
                    then one slot per obstacle (padded slots included, zeroed
                    content but flagged in the mask)
            mask:   (B, 3 + max_obstacles) -- 1 = attend to this token,
                    0 = ignore (padding). proprio/object/goal are always
                    real (always 1); only the obstacle tail can be padded.
        """
        b = proprio.shape[0]

        proprio_tok = self.proj["proprio"](proprio) + self._tag("proprio")  # (B, d_model)
        object_tok = self.proj["object"](obj) + self._tag("object")          # (B, d_model)
        goal_tok = self.proj["goal"](goal) + self._tag("goal")               # (B, d_model)

        max_obstacles = obstacles.shape[1]
        if max_obstacles > 0:
            obstacle_tok = self.proj["obstacle"](obstacles) + self._tag("obstacle")  # (B, max_obstacles, d_model)
        else:
            obstacle_tok = obstacles.new_zeros(b, 0, self.d_model)

        fixed_tokens = torch.stack([proprio_tok, object_tok, goal_tok], dim=1)  # (B, 3, d_model)
        tokens = torch.cat([fixed_tokens, obstacle_tok], dim=1)  # (B, 3+max_obstacles, d_model)

        fixed_mask = torch.ones(b, 3, dtype=torch.float32, device=tokens.device)
        mask = torch.cat([fixed_mask, obstacle_mask], dim=1)  # (B, 3+max_obstacles)

        return tokens, mask
