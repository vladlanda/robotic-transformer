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


# --------------------------------------------------------------------------
# Unit tests. Run directly with:  python -m src.model
# --------------------------------------------------------------------------

def _dummy_inputs(batch_size=4, max_obstacles=3, cfg=None, seed=0):
    from . import entities
    cfg = cfg or ModelConfig()
    g = torch.Generator().manual_seed(seed)
    proprio = torch.randn(batch_size, entities.FEATURE_DIM["proprio"], generator=g)
    obj = torch.randn(batch_size, entities.FEATURE_DIM["object"], generator=g)
    goal = torch.randn(batch_size, entities.FEATURE_DIM["goal"], generator=g)
    obstacles = torch.randn(batch_size, max_obstacles, entities.FEATURE_DIM["obstacle"], generator=g)
    obstacle_mask = torch.zeros(batch_size, max_obstacles)
    if max_obstacles > 0:
        obstacle_mask[:, 0] = 1.0
    return proprio, obj, goal, obstacles, obstacle_mask


def test_forward_output_shape():
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=5)
    model = ActionChunkTransformer(cfg)
    b = 4
    pred = model(*_dummy_inputs(b, max_obstacles=3, cfg=cfg))
    assert pred.shape == (b, cfg.chunk_size, cfg.action_dim), f"got {pred.shape}"


def test_forward_works_with_zero_obstacles():
    from . import entities
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=5)
    model = ActionChunkTransformer(cfg)
    b = 4
    proprio, obj, goal, _, _ = _dummy_inputs(b, max_obstacles=0, cfg=cfg)
    obstacles = torch.zeros(b, 0, entities.FEATURE_DIM["obstacle"])
    obstacle_mask = torch.zeros(b, 0)
    pred = model(proprio, obj, goal, obstacles, obstacle_mask)
    assert pred.shape == (b, cfg.chunk_size, cfg.action_dim)


def test_gradients_flow_to_all_parameters():
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=5)
    model = ActionChunkTransformer(cfg)
    pred = model(*_dummy_inputs(4, max_obstacles=3, cfg=cfg))
    pred.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"


def test_padded_obstacle_content_does_not_affect_output():
    # End-to-end correctness check of the masking mechanism -- not just that
    # the tokenizer accepts padded input, but that the model's actual
    # OUTPUT is provably unaffected by whatever garbage sits in masked
    # (mask=0) obstacle slots. If this ever fails, the attention mask is
    # leaking information it shouldn't.
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=5)
    model = ActionChunkTransformer(cfg)
    model.eval()  # disable dropout for a deterministic comparison

    b, max_obs = 4, 3
    proprio, obj, goal, obstacles_a, obstacle_mask = _dummy_inputs(b, max_obs, cfg=cfg, seed=1)
    obstacle_mask[:, 0] = 1.0
    obstacle_mask[:, 1:] = 0.0  # slots 1,2 are padding

    obstacles_b = obstacles_a.clone()
    obstacles_b[:, 1:, :] = torch.randn_like(obstacles_b[:, 1:, :])  # scramble ONLY the padded slots

    with torch.no_grad():
        pred_a = model(proprio, obj, goal, obstacles_a, obstacle_mask)
        pred_b = model(proprio, obj, goal, obstacles_b, obstacle_mask)

    assert torch.allclose(pred_a, pred_b, atol=1e-5), (
        "model output changed even though only MASKED (padding) obstacle content changed -- "
        "the attention mask is not properly excluding padded slots"
    )


def test_masked_mse_loss_zero_when_prediction_matches_target():
    pred = torch.randn(4, 6, 5)
    mask = torch.tensor([[1., 1., 1., 0., 0., 0.]] * 4)
    loss = masked_mse_loss(pred, pred.clone(), mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7), f"got {loss.item()}"


def test_masked_mse_loss_ignores_masked_entries():
    pred = torch.zeros(2, 4, 3)
    target = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]])
    loss_before = masked_mse_loss(pred, target, mask)
    # scramble target ONLY where mask == 0 -- must not change the loss at all
    target[:, 2:, :] = torch.randn(2, 2, 3) * 100
    loss_after = masked_mse_loss(pred, target, mask)
    assert torch.isclose(loss_before, loss_after), f"{loss_before.item()} != {loss_after.item()}"


def test_masked_mse_loss_matches_hand_computed_value():
    pred = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])   # (B=1, chunk=2, action_dim=2)
    target = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])  # second step is masked out, should be ignored
    mask = torch.tensor([[1.0, 0.0]])
    # only step 0 counts: mean squared error over its 2 action dims =
    # mean((1-0)^2, (0-0)^2) = mean(1, 0) = 0.5
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(0.5)), f"got {loss.item()}"


def test_masked_mse_loss_all_zero_mask_does_not_divide_by_zero():
    pred = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4)
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isfinite(loss), f"got {loss.item()} -- should be a finite fallback (0), not NaN/Inf"
    assert torch.isclose(loss, torch.tensor(0.0))


def _run_all_tests():
    import sys
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} tests passed" + (f", FAILED: {failed}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests()
