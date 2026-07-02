"""
Compute normalization statistics across the whole dataset and save them to
stats/normalization.json.

We save BOTH mean/std and robust (median / IQR) stats, because several
velocity channels (cube_angvel, robot_box_angvel_z) have heavy-tailed
contact-transient spikes (seen up to ~40 rad/s against a typical range of
~1 rad/s) that will badly distort a plain z-score. Robust stats are the
recommended default for those; mean/std is provided for channels that look
well-behaved and for comparison.

Usage:
    python scripts/compute_stats.py [data_dir] [action_mode]
"""
import sys
import os
import glob
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import load_episode  # noqa: E402


FIELDS = [
    "cube_pos_ego", "cube_linvel_ego",
    "target_pos_ego",
    "endpoint_pos_ego", "endpoint_linvel_ego",
    "gripper_gear_pos", "gripper_gear_vel",
    "gripper_rot_sincos", "gripper_rot_vel",
    "arm_joint_pos", "arm_joint_vel",
    "action_base_vxy", "action_base_yaw_rate",
    "action_arm_joint", "action_gripper_gear",
]


def stats_for(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    p5, p50, p95 = np.percentile(arr, [5, 50, 95], axis=0)
    return {
        "shape_per_step": list(arr.shape[1:]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "p5": p5.tolist(),
        "p50_median": p50.tolist(),
        "p95": p95.tolist(),
        "robust_scale_iqr90": (p95 - p5).tolist(),  # use as a robust denominator instead of std
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data")
    action_mode = sys.argv[2] if len(sys.argv) > 2 else "next_state"
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"scanning {len(files)} episodes (action_mode={action_mode})")

    accum = {f: [] for f in FIELDS}
    n_episodes = 0
    for f in files:
        ep = load_episode(f, action_mode=action_mode)
        n_episodes += 1
        for field in FIELDS:
            val = getattr(ep, field)
            if val is None:
                continue
            accum[field].append(np.asarray(val))

    out = {"n_episodes": n_episodes, "action_mode": action_mode, "fields": {}}
    for field, chunks in accum.items():
        if not chunks:
            continue
        full = np.concatenate(chunks, axis=0)
        out["fields"][field] = stats_for(full)
        print(f"  {field:25s} n={full.shape[0]:6d}  mean={np.round(full.mean(axis=0), 4)}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "stats")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"normalization_{action_mode}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
