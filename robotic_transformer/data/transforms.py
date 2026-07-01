"""
Geometric utilities for expressing world-frame quantities in the mobile
base's own (ego-centric) frame.

Why this matters: robot_box_x/y/yaw change continuously throughout an
episode (the base drives while the arm manipulates -- this is a whole-body
task, not "drive then reach"). A policy conditioned on raw world coordinates
has to re-derive "where is the goal right now" as its own reference frame
drifts underneath it, on every single timestep, for every episode. Ego-frame
features turn that into a much simpler, translation/rotation-invariant
target: "the goal is at this bearing and distance from me right now."

All functions here are vectorized over a leading time axis, i.e. shaped
(T, ...), so they can be applied to a full episode at once.
"""

import numpy as np


def yaw_from_quat_wz(quat_w: np.ndarray, quat_z: np.ndarray) -> np.ndarray:
    """
    Extract yaw (rotation about the vertical axis) from a quaternion that is
    known to have zero x/y components (planar/SE(2) rotation only -- true
    for robot_box_quat across the whole dataset, see schema.py).

    quat_w, quat_z: shape (T,)
    returns: yaw in radians, shape (T,), range (-pi, pi]
    """
    return 2.0 * np.arctan2(quat_z, quat_w)


def world_to_ego(
    point_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_yaw: np.ndarray,
) -> np.ndarray:
    """
    Rotate+translate a world-frame 2D point into the base's ego frame.

    point_world:    (T, 2) world-frame xy
    base_pos_world: (T, 2) base xy in world frame
    base_yaw:       (T,)   base yaw in world frame (radians)
    returns:        (T, 2) point expressed in the base frame
    """
    rel = point_world - base_pos_world  # (T, 2), world-frame relative vector
    cos_t, sin_t = np.cos(base_yaw), np.sin(base_yaw)
    # R(-yaw) applied to rel -- rotates a world-frame vector into the frame
    # whose x-axis points along the base's heading.
    x_ego = cos_t * rel[..., 0] + sin_t * rel[..., 1]
    y_ego = -sin_t * rel[..., 0] + cos_t * rel[..., 1]
    return np.stack([x_ego, y_ego], axis=-1)


def world_vel_to_ego(
    vel_world: np.ndarray,
    point_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_vel_world: np.ndarray,
    base_yaw: np.ndarray,
    base_yaw_rate: np.ndarray,
) -> np.ndarray:
    """
    Express a point's world-frame linear velocity as seen from the (moving,
    rotating) base frame. This is the *proper* rigid-body transform: it
    accounts for the base's own translation AND its rotation rate, not just
    a static rotation of the velocity vector.

    v_ego = R(-yaw) @ (v_world - v_base_world - yaw_rate * perp(point - base))

    where perp((rx, ry)) = (-ry, rx) is the effect of the rotating frame on
    a point offset from its origin.

    vel_world:      (T, 2)
    point_world:    (T, 2)  world position of the point (needed for the
                             rotation-coupling term)
    base_pos_world: (T, 2)
    base_vel_world: (T, 2)
    base_yaw:       (T,)
    base_yaw_rate:  (T,)
    returns:        (T, 2) velocity as observed in the base frame
    """
    rel = point_world - base_pos_world  # (T, 2)
    perp_rel = np.stack([-rel[..., 1], rel[..., 0]], axis=-1)  # (T, 2)
    v_rel_world = vel_world - base_vel_world - base_yaw_rate[..., None] * perp_rel

    cos_t, sin_t = np.cos(base_yaw), np.sin(base_yaw)
    x_ego = cos_t * v_rel_world[..., 0] + sin_t * v_rel_world[..., 1]
    y_ego = -sin_t * v_rel_world[..., 0] + cos_t * v_rel_world[..., 1]
    return np.stack([x_ego, y_ego], axis=-1)


def sin_cos_encode(angle: np.ndarray) -> np.ndarray:
    """
    Encode an angle (radians, any range -- including unbounded/continuous
    joints that exceed +-pi) as (sin, cos) to avoid wraparound discontinuity.

    angle: (T,)
    returns: (T, 2) stacked [sin, cos]
    """
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)
