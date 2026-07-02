"""
End-to-end smoke test: Dataset -> DataLoader batch -> EntityTokenizer.

Confirms:
  - grouped entity tensors come out of the Dataset with the right shapes
  - the tokenizer turns them into a consistent set of tagged tokens
  - the obstacle mask path works even with zero real obstacles (all-zero,
    fully-masked block) -- this is the part that needs to keep working
    once real obstacle data exists.

Usage:
    python scripts/smoke_test_tokenizer.py [data_dir]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from src import EpisodeChunkDataset, EntityTokenizer  # noqa: E402


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data")

    ds = EpisodeChunkDataset(data_dir, chunk_size=12, history=2, max_obstacles=8, action_mode="next_state")
    print(f"dataset size: {len(ds)}")

    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  batch[{k!r}]: {tuple(v.shape)} {v.dtype}")

    # collapse the history dim into batch for the tokenizer smoke test
    # (a real model would handle history explicitly; here we just check
    # the tokenizer works on a single timestep slice)
    b, h = batch["proprio"].shape[:2]
    tok = EntityTokenizer(d_model=128)
    tokens, mask = tok(
        proprio=batch["proprio"][:, -1],
        obj=batch["object"][:, -1],
        goal=batch["goal"][:, -1],
        obstacles=batch["obstacles"][:, -1],
        obstacle_mask=batch["obstacle_mask"][:, -1],
    )
    print(f"\ntokens: {tuple(tokens.shape)}  (expect: batch={b}, 3+max_obstacles={3 + ds.max_obstacles}, d_model=128)")
    print(f"mask:   {tuple(mask.shape)}")
    print(f"mask values (first sample): {mask[0].tolist()}  (expect [1,1,1] then all 0s -- no obstacle data yet)")

    assert tokens.shape == (b, 3 + ds.max_obstacles, 128)
    assert mask.shape == (b, 3 + ds.max_obstacles)
    assert torch.all(mask[:, :3] == 1.0), "proprio/object/goal tokens should always be unmasked"
    assert torch.all(mask[:, 3:] == 0.0), "no obstacle data exists yet -- all obstacle slots should be masked out"
    assert torch.isfinite(tokens).all(), "found non-finite values in tokens"

    print("\nsmoke test passed.")


if __name__ == "__main__":
    main()
