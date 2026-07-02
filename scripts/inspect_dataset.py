"""
Sanity-check the data pipeline against every episode in data/.

Usage:
    python scripts/inspect_dataset.py [data_dir]

Checks:
  - every file loads without error, under both action_mode settings
  - no NaNs/Infs anywhere in the derived arrays
  - reports episode length distribution and ego-frame sanity (e.g. does the
    ego-frame target position actually stay put in a sensible range, given
    the base is moving under it)
"""
import sys
import glob
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import load_episode  # noqa: E402


def check_episode(path: str):
    problems = []
    for mode in ("next_state", "finite_diff_vel"):
        ep = load_episode(path, action_mode=mode)
        arrays = {
            "cube_pos_ego": ep.cube_pos_ego,
            "cube_linvel_ego": ep.cube_linvel_ego,
            "target_pos_ego": ep.target_pos_ego,
            "endpoint_pos_ego": ep.endpoint_pos_ego,
            "endpoint_linvel_ego": ep.endpoint_linvel_ego,
            "gripper_rot_sincos": ep.gripper_rot_sincos,
            "action_base_vxy": ep.action_base_vxy,
            "action_base_yaw_rate": ep.action_base_yaw_rate,
            "action_arm_joint": ep.action_arm_joint,
            "action_gripper_gear": ep.action_gripper_gear,
            "action_gripper_rot": ep.action_gripper_rot,
        }
        for name, arr in arrays.items():
            if arr is None:
                continue
            if not np.all(np.isfinite(arr)):
                problems.append(f"[{mode}] {name} has non-finite values")
    return ep, problems


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"found {len(files)} episode files in {data_dir}")

    lengths = []
    all_problems = []
    last_ep = None
    for f in files:
        try:
            ep, problems = check_episode(f)
        except Exception as e:
            all_problems.append(f"{f}: FAILED TO LOAD -- {e}")
            continue
        lengths.append(len(ep.t))
        last_ep = ep
        if problems:
            all_problems.append(f"{f}: " + "; ".join(problems))

    lengths = np.array(lengths)
    print(f"loaded {len(lengths)}/{len(files)} episodes successfully")
    print(f"episode length: min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")

    if all_problems:
        print(f"\n{len(all_problems)} PROBLEMS FOUND:")
        for p in all_problems[:20]:
            print("  -", p)
        if len(all_problems) > 20:
            print(f"  ... and {len(all_problems) - 20} more")
    else:
        print("\nno NaN/Inf problems found across the dataset.")

    if last_ep is not None:
        print("\n--- sample episode (last loaded):", last_ep.path, "---")
        print("target_pos_ego over the episode (expected to shrink toward the robot")
        print("as the base approaches -- it is NOT constant, only the world-frame")
        print("target is constant):")
        print("  at t=0 (base starts at world origin, yaw=0, so ego==world):",
              last_ep.target_pos_ego[0])
        print("  world-frame target (constant by construction):        ", last_ep.target_pos_world[0, :2])
        print("  at final step:                                        ", last_ep.target_pos_ego[-1])


if __name__ == "__main__":
    main()
