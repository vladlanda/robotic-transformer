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


# --------------------------------------------------------------------------
# Unit tests. Run directly with:  python -m src.tokenizer
# --------------------------------------------------------------------------

def _dummy_batch(batch_size=4, max_obstacles=3, d_model=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    proprio = torch.randn(batch_size, entities.FEATURE_DIM["proprio"], generator=g)
    obj = torch.randn(batch_size, entities.FEATURE_DIM["object"], generator=g)
    goal = torch.randn(batch_size, entities.FEATURE_DIM["goal"], generator=g)
    obstacles = torch.randn(batch_size, max_obstacles, entities.FEATURE_DIM["obstacle"], generator=g)
    obstacle_mask = torch.zeros(batch_size, max_obstacles)
    if max_obstacles > 0:
        obstacle_mask[:, 0] = 1.0  # pretend the first obstacle slot is real, rest padding
    return proprio, obj, goal, obstacles, obstacle_mask


def test_output_shapes():
    b, max_obs, d = 4, 3, 16
    tok = EntityTokenizer(d_model=d)
    tokens, mask = tok(*_dummy_batch(b, max_obs, d))
    assert tokens.shape == (b, 3 + max_obs, d), f"got {tokens.shape}"
    assert mask.shape == (b, 3 + max_obs), f"got {mask.shape}"


def test_zero_obstacles_still_works():
    b, d = 4, 16
    tok = EntityTokenizer(d_model=d)
    proprio, obj, goal, _, _ = _dummy_batch(b, max_obstacles=0, d_model=d)
    obstacles = torch.zeros(b, 0, entities.FEATURE_DIM["obstacle"])
    obstacle_mask = torch.zeros(b, 0)
    tokens, mask = tok(proprio, obj, goal, obstacles, obstacle_mask)
    assert tokens.shape == (b, 3, d)
    assert mask.shape == (b, 3)


def test_fixed_entity_mask_is_always_one():
    b, max_obs, d = 4, 5, 16
    tok = EntityTokenizer(d_model=d)
    _, mask = tok(*_dummy_batch(b, max_obs, d))
    assert torch.all(mask[:, :3] == 1.0), "proprio/object/goal slots must never be masked out"


def test_obstacle_mask_is_passed_through_unchanged():
    b, max_obs, d = 4, 5, 16
    tok = EntityTokenizer(d_model=d)
    proprio, obj, goal, obstacles, obstacle_mask = _dummy_batch(b, max_obs, d)
    obstacle_mask = torch.tensor([[1., 1., 0., 0., 0.]] * b)
    _, mask = tok(proprio, obj, goal, obstacles, obstacle_mask)
    assert torch.equal(mask[:, 3:], obstacle_mask), "obstacle mask should pass through unmodified"


def test_same_raw_features_get_different_tokens_by_type_tag():
    # goal and obstacle both happen to have FEATURE_DIM==3 (see entities.py),
    # which lets us feed the SAME raw numbers in as two different kinds and
    # directly confirm the type tag (not just different weights) makes a
    # difference -- this is the core mechanism from the "how does it know
    # what tag to add" conversation, tested directly rather than assumed.
    assert entities.FEATURE_DIM["goal"] == entities.FEATURE_DIM["obstacle"], (
        "this test relies on goal/obstacle sharing a feature dim -- update the test if that changes"
    )
    d = 16
    tok = EntityTokenizer(d_model=d)
    same_features = torch.randn(1, 3)

    goal_token = tok.proj["goal"](same_features) + tok._tag("goal")
    obstacle_token = tok.proj["obstacle"](same_features) + tok._tag("obstacle")
    assert not torch.allclose(goal_token, obstacle_token), (
        "identical raw features tagged as different entity kinds must produce different tokens"
    )


def test_type_embedding_rows_differ_at_init():
    # Not mathematically guaranteed (random init could theoretically produce
    # duplicates), but overwhelmingly true in practice and worth catching if
    # something pathological (e.g. an embedding table initialized to zeros)
    # ever gets introduced.
    tok = EntityTokenizer(d_model=16)
    rows = tok.type_embedding.weight.detach()
    for i in range(rows.shape[0]):
        for j in range(i + 1, rows.shape[0]):
            assert not torch.allclose(rows[i], rows[j]), f"type_embedding rows {i} and {j} are identical"


def test_gradients_flow_to_all_parameters():
    b, max_obs, d = 4, 3, 16
    tok = EntityTokenizer(d_model=d)
    tokens, mask = tok(*_dummy_batch(b, max_obs, d))
    loss = (tokens * mask.unsqueeze(-1)).sum()
    loss.backward()
    for name, p in tok.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"


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
