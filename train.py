"""
Full training script for ActionChunkTransformer.

    python train.py
    python train.py --epochs 100 --batch-size 128 --lr 3e-4
    python train.py --data-dir data --val-fraction 0.15 --seed 1

What it does:
  1. Splits episodes (not individual timesteps!) into train/val, so
     validation never sees a chunk from an episode the model trained on.
  2. action_dim and per-entity feature dims are read directly off a real
     sample from the dataset rather than hardcoded -- this avoids the
     action_mode-dependent dimension mismatch flagged in conversation
     (next_state produces a 10-dim action, finite_diff_vel currently
     produces 9-dim; this script adapts automatically either way instead of
     silently assuming 10).
  3. Standard supervised loop: masked MSE loss, Adam, one epoch = one pass
     over all training chunks.
  4. Saves THREE things to --checkpoint-dir:
       best_model.pt         -- lowest validation loss seen so far
       last_model.pt         -- overwritten every epoch (for resuming)
       training_history.json -- per-epoch train/val loss + full run config
  5. Optional --patience for early stopping (0 = disabled, the default).
  6. Optional --resume PATH to continue training from a saved checkpoint.

No simulator involved anywhere here -- purely supervised learning over the
recorded episode csvs. See notebooks/01_project_walkthrough.ipynb for the
same process explained step by step; this script is the "just run it"
version.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import EpisodeChunkDataset, ActionChunkTransformer, ModelConfig, masked_mse_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="directory of env_*_episode_*.csv files")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="where to save checkpoints + history")
    p.add_argument("--val-fraction", type=float, default=0.1, help="fraction of EPISODES held out for validation")
    p.add_argument("--seed", type=int, default=0)

    # dataset / representation
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--max-obstacles", type=int, default=8)
    p.add_argument("--action-mode", default="next_state", choices=["next_state", "finite_diff_vel"])

    # model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0, help="stop if val loss doesn't improve for this many epochs (0 = disabled)")

    p.add_argument("--resume", default=None, help="path to a checkpoint (e.g. checkpoints/last_model.pt) to resume from")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto-detect)")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def report_device_availability(chosen: torch.device, requested: str = None):
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("GPU check:")
    print(f"  CUDA available: {cuda_ok}" + (f"  ({torch.cuda.get_device_name(0)})" if cuda_ok else ""))
    print(f"  MPS  available: {mps_ok}" + ("  (Apple Silicon GPU)" if mps_ok else ""))
    print(f"  -> using device: {chosen}" + ("  (forced via --device)" if requested else "  (auto-detected)"))
    if chosen.type == "cpu" and (cuda_ok or mps_ok):
        print("  note: a GPU is available but --device cpu was requested/detected as unused.")
    if chosen.type == "cpu":
        print("  training on CPU will be considerably slower than GPU/MPS for this model.")


def device_smoke_test(model, cfg, device: torch.device):
    """
    Runs one tiny forward+backward pass with synthetic data on `device`
    BEFORE the real training loop starts. Purpose: device-placement bugs
    (e.g. a tensor built without a device= argument inside a submodule,
    which silently stays on CPU no matter where the model was moved) are
    invisible on CPU-only setups but crash on MPS/CUDA -- usually deep
    inside the first real training batch, with a cryptic low-level error
    ("Placeholder storage has not been allocated on MPS device!" is a real
    example that came up in practice). Failing fast here, with a clear
    message, beats discovering that minutes into a real run.
    """
    from src import entities
    try:
        b = 2
        proprio = torch.randn(b, entities.FEATURE_DIM["proprio"], device=device)
        obj = torch.randn(b, entities.FEATURE_DIM["object"], device=device)
        goal = torch.randn(b, entities.FEATURE_DIM["goal"], device=device)
        obstacles = torch.randn(b, 2, entities.FEATURE_DIM["obstacle"], device=device)
        obstacle_mask = torch.zeros(b, obstacles.shape[1], device=device)
        target = torch.randn(b, cfg.chunk_size, cfg.action_dim, device=device)
        mask = torch.ones(b, cfg.chunk_size, device=device)

        # training-mode pass (forward + backward)
        model.train()
        pred = model(proprio, obj, goal, obstacles, obstacle_mask)
        loss = masked_mse_loss(pred, target, mask)
        loss.backward()
        model.zero_grad(set_to_none=True)

        # EVAL-mode pass, no_grad, WITH a real padding mask -- this is the
        # exact combination (eval() + src_key_padding_mask) that triggered
        # a real bug in practice: nn.TransformerEncoder's internal nested-
        # tensor fastpath only activates under these conditions and wasn't
        # implemented on MPS, so a training-only smoke test missed it
        # entirely. The validation pass in run_epoch() hits exactly this.
        model.eval()
        with torch.no_grad():
            model(proprio, obj, goal, obstacles, obstacle_mask)
        model.train()
    except RuntimeError as e:  # NotImplementedError is a RuntimeError subclass, caught here too
        raise RuntimeError(
            f"Device smoke test failed on {device} before any real training happened. "
            f"If this is a missing/unimplemented op (common on MPS), the fix is usually "
            f"either avoiding that op or setting PYTORCH_ENABLE_MPS_FALLBACK=1 as a slower "
            f"fallback. If it's a device-mismatch error, it's almost always a tensor "
            f"created somewhere without an explicit device= argument. "
            f"Original error: {e}"
        ) from e
    print(f"  device smoke test passed on {device} (train and eval mode).")


def split_episode_files(data_dir: str, val_fraction: float, seed: int):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no episode csvs found in {data_dir}")
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val_files = sorted(shuffled[:n_val])
    train_files = sorted(shuffled[n_val:])
    return train_files, val_files


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_epoch(model, dl, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    for batch in dl:
        batch = move_batch(batch, device)
        pred = model(
            batch["proprio"][:, -1], batch["object"][:, -1], batch["goal"][:, -1],
            batch["obstacles"][:, -1], batch["obstacle_mask"][:, -1],
        )
        loss = masked_mse_loss(pred, batch["action_chunk"], batch["action_mask"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = batch["proprio"].shape[0]
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    report_device_availability(device, args.device)
    train_files, val_files = split_episode_files(args.data_dir, args.val_fraction, args.seed)
    print(f"episodes: {len(train_files)} train / {len(val_files)} val "
          f"(val_fraction={args.val_fraction}, seed={args.seed})")

    train_ds = EpisodeChunkDataset(files=train_files, chunk_size=args.chunk_size, history=args.history,
                                    max_obstacles=args.max_obstacles, action_mode=args.action_mode)
    val_ds = EpisodeChunkDataset(files=val_files, chunk_size=args.chunk_size, history=args.history,
                                  max_obstacles=args.max_obstacles, action_mode=args.action_mode)
    print(f"samples: {len(train_ds)} train / {len(val_ds)} val")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Read the real action_dim off one actual sample instead of hardcoding
    # it -- action_mode changes this (10 for next_state, 9 for
    # finite_diff_vel as of this writing; see conversation/README).
    sample = train_ds[0]
    action_dim = sample["action_chunk"].shape[-1]
    print(f"action_dim (from data, action_mode={args.action_mode!r}): {action_dim}")

    cfg = ModelConfig(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        chunk_size=args.chunk_size, action_dim=action_dim,
    )
    model = ActionChunkTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    device_smoke_test(model, cfg, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    if args.resume:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", history)

    def save_checkpoint(path: str, epoch: int, val_loss: float):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "action_mode": args.action_mode,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "args": vars(args),
        }, path)

    epochs_since_improvement = 0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, train_dl, device, optimizer=optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_dl, device, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_checkpoint(os.path.join(args.checkpoint_dir, "best_model.pt"), epoch + 1, val_loss)
        else:
            epochs_since_improvement += 1
        save_checkpoint(os.path.join(args.checkpoint_dir, "last_model.pt"), epoch + 1, val_loss)

        elapsed = time.time() - t0
        flag = "  <- best" if improved else ""
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train={train_loss:.5f}  val={val_loss:.5f}"
              f"  best={best_val_loss:.5f}  ({elapsed:.0f}s elapsed){flag}")

        with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
            json.dump({
                "history": history,
                "best_val_loss": best_val_loss,
                "epochs_run": epoch + 1,
                "config": cfg.__dict__,
                "args": vars(args),
                "train_files": train_files,
                "val_files": val_files,
            }, f, indent=2)

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(f"no val improvement for {args.patience} epochs -- stopping early")
            break

    print(f"\ndone. best val loss: {best_val_loss:.5f}")
    print(f"best checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
    print(f"last checkpoint: {os.path.join(args.checkpoint_dir, 'last_model.pt')}")


if __name__ == "__main__":
    main()
