"""
Entity/token definitions for the tag-embedding scheme.

The core idea (see conversation/README for the full explanation): every
"thing in the scene" becomes its own token, built from
    token = linear(raw_features) + type_embedding[TYPE_ID[kind]]
The TYPE_ID mapping below is fixed, hardcoded bookkeeping decided by us --
it is NOT learned. What IS learned (in tokenizer.py) is the *content* of
each row of the type-embedding table, and the linear projection weights.

Entity kinds:
  - "proprio":  always exactly 1 token. The robot's own body: arm joint
                angles+vel, gripper open/close+vel, wrist roll (sin/cos)+vel,
                base linear/angular velocity. (Base *position* isn't
                included here -- by definition the base is always at the
                origin of its own ego frame, so it carries no information.)
  - "object":   1 token today (the cube). Written as a list so multi-object
                scenes are a natural extension later, not a rewrite.
  - "goal":     1 token, the target position.
  - "obstacle": 0..N tokens. NO OBSTACLE DATA EXISTS IN THE CURRENT DATASET
                -- this always produces an empty (padded+masked) set for
                now. The slot is reserved so that when obstacle data does
                exist, only obstacle_features() and the data loader need to
                change; the tokenizer/model do not.
"""

from dataclasses import dataclass

import numpy as np

from .episode import Episode

TYPE_ID = {
    "proprio": 0,
    "object": 1,
    "goal": 2,
    "obstacle": 3,
}
NUM_TYPES = len(TYPE_ID)

# Raw feature-vector length for one token of each kind, BEFORE the linear
# projection into the model's internal (d_model) width. Keep these in sync
# with the *_features() functions below.
FEATURE_DIM = {
    "proprio": 4 + 4 + 2 + 2 + 1 + 2 + 1 + 2 + 1 + 2,  # arm(pos+vel) + gear(pos,vel) + rot(sincos,vel) + base(linvel_ego,yaw_rate) + endpoint(pos_ego,z,linvel_ego) = 21
    "object": 2 + 1 + 2,   # pos_ego(2) + z(1) + linvel_ego(2)
    "goal": 2 + 1,          # pos_ego(2) + z(1)
    "obstacle": 3,           # pos_ego(2) + radius(1) -- placeholder, no real data yet
}


def proprio_features(ep: Episode, i: int) -> np.ndarray:
    """Robot's own body state at step i, including the gripper tip (endpoint)
    -- forward-kinematics output of the arm, still part of "the robot," not
    a separate world entity like the cube. Length must match FEATURE_DIM['proprio']."""
    return np.concatenate([
        ep.arm_joint_pos[i], ep.arm_joint_vel[i],                    # 4 + 4
        [ep.gripper_gear_pos[i], ep.gripper_gear_vel[i]],            # 1 + 1
        ep.gripper_rot_sincos[i], [ep.gripper_rot_vel[i]],           # 2 + 1
        ep.base_linvel_ego[i],                                       # 2 -- rotated into ego frame, see transforms.rotate_to_ego
        [ep.base_yaw_rate[i]],                                       # 1
        ep.endpoint_pos_ego[i], [ep.endpoint_z[i]],                  # 2 + 1
        ep.endpoint_linvel_ego[i],                                   # 2
    ]).astype(np.float32)


def object_features(ep: Episode, i: int) -> np.ndarray:
    """The manipulated cube's state at step i, in the base's ego frame."""
    return np.concatenate([
        ep.cube_pos_ego[i], [ep.cube_z[i]],
        ep.cube_linvel_ego[i],
    ]).astype(np.float32)


def goal_features(ep: Episode, i: int) -> np.ndarray:
    """The target position at step i (constant across the episode), ego frame."""
    return np.concatenate([
        ep.target_pos_ego[i], [ep.target_z[i]],
    ]).astype(np.float32)


def obstacle_features(ep: Episode, i: int) -> np.ndarray:
    """
    Returns shape (0, FEATURE_DIM['obstacle']) -- no obstacle data exists in
    the current dataset. This function is the ONLY thing that needs to
    change once obstacle logging is added (e.g. read obstacle_N_world_x/y
    columns and ego-transform them the same way object_features does).
    """
    return np.zeros((0, FEATURE_DIM["obstacle"]), dtype=np.float32)


def validate_feature_dims(ep: Episode) -> None:
    """Sanity check: catches the *_features functions drifting out of sync
    with the FEATURE_DIM dict (easy to do by hand -- happened once already
    while writing this file). Call this from tests/scripts, not hot paths."""
    checks = {
        "proprio": proprio_features(ep, 0),
        "object": object_features(ep, 0),
        "goal": goal_features(ep, 0),
    }
    for kind, vec in checks.items():
        expected = FEATURE_DIM[kind]
        if vec.shape != (expected,):
            raise AssertionError(
                f"{kind}_features() returned shape {vec.shape}, but FEATURE_DIM['{kind}']={expected}"
            )
    obs = obstacle_features(ep, 0)
    if obs.shape[1] != FEATURE_DIM["obstacle"]:
        raise AssertionError(f"obstacle_features() last dim {obs.shape[1]} != FEATURE_DIM['obstacle']={FEATURE_DIM['obstacle']}")
