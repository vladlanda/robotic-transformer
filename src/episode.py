"""
Load a single episode CSV into structured numpy arrays, with:
  1. raw columns as logged
  2. derived ego-frame (base-relative) features for cube/target/endpoint
  3. sin/cos encoding for the wrist-roll joint
  4. pseudo-action labels for the controllable DOFs

--------------------------------------------------------------------------
OPEN DESIGN QUESTION -- action labeling
--------------------------------------------------------------------------
The logs contain no recorded control/command signal -- every column is a
measured *state* (robot_box_*, motor_angle_*, gripper_actual_*, ...). There
is nothing here that says "what did the RL policy actually command at time
t". Two standard workarounds, both implemented below via `action_mode`:

  - "next_state" (default): treat the controllable state at t+1 as the
    action target for time t. This assumes a well-tracked low-level
    controller (i.e. actual position at t+1 ~= commanded target at t),
    which is a common and usually reasonable assumption for position-
    controlled joints, but it silently inherits any tracking lag in the
    underlying controller.
  - "finite_diff_vel": derive a velocity action as
    (state[t+1] - state[t]) / dt for position-like DOFs, and use the logged
    *_vel columns directly for the joints/gripper (which already have
    measured velocities). This sidesteps tracking-lag assumptions but is
    noisier (single-step finite differences amplify sensor/sim noise).

We have NOT settled on which of these should be the final training target --
flagging this explicitly rather than quietly picking one. Both are
implemented so we can compare. Default is "next_state" for now.
--------------------------------------------------------------------------

Base yaw-rate note: robot_box_angvel_z is logged directly (not finite-
differenced), so it is used as-is for the ego-frame velocity transform
regardless of action_mode.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import schema
from .transforms import yaw_from_quat_wz, world_to_ego, world_vel_to_ego, sin_cos_encode

ActionMode = Literal["next_state", "finite_diff_vel"]


@dataclass
class Episode:
    path: str
    t: np.ndarray  # (T,) sim time

    # Raw, world-frame (kept for reference / debugging / plotting)
    base_pos_world: np.ndarray  # (T, 2) xy only, z confirmed constant
    base_yaw: np.ndarray  # (T,)
    base_linvel_world: np.ndarray  # (T, 2)
    base_yaw_rate: np.ndarray  # (T,)
    cube_pos_world: np.ndarray  # (T, 3)
    endpoint_pos_world: np.ndarray  # (T, 3)
    target_pos_world: np.ndarray  # (T, 3), constant across the episode

    # Ego-frame (base-relative) features -- primary model inputs
    cube_pos_ego: np.ndarray  # (T, 2) xy relative to base, z left in world (height is frame-invariant)
    cube_z: np.ndarray  # (T,)
    cube_linvel_ego: np.ndarray  # (T, 2)
    target_pos_ego: np.ndarray  # (T, 2)
    target_z: np.ndarray  # (T,)
    endpoint_pos_ego: np.ndarray  # (T, 2)
    endpoint_z: np.ndarray  # (T,)
    endpoint_linvel_ego: np.ndarray  # (T, 2)

    # Gripper / arm state
    gripper_gear_pos: np.ndarray  # (T,)
    gripper_gear_vel: np.ndarray  # (T,)
    gripper_rot_sincos: np.ndarray  # (T, 2)
    gripper_rot_vel: np.ndarray  # (T,)
    arm_joint_pos: np.ndarray  # (T, 4)
    arm_joint_vel: np.ndarray  # (T, 4)

    # Pseudo-action labels (see module docstring)
    action_mode: str = ""
    action_base_vxy: np.ndarray = None  # (T-1, 2) ego-frame if finite_diff_vel; (T-1,2) next-state world xy otherwise -- see note in loader
    action_base_yaw_rate: np.ndarray = None  # (T-1,)
    action_arm_joint: np.ndarray = None  # (T-1, 4)
    action_gripper_gear: np.ndarray = None  # (T-1,)
    action_gripper_rot: np.ndarray = None  # (T-1,)


def load_episode(path: str, action_mode: ActionMode = "next_state") -> Episode:
    df = pd.read_csv(path)

    t = df[schema.TIME_COL].to_numpy()
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], atol=1e-4):
        raise ValueError(f"{path}: non-uniform timestep detected ({dt.min()=}, {dt.max()=})")
    dt = float(dt[0])

    base_pos_world = df[["robot_box_x", "robot_box_y"]].to_numpy()
    base_yaw = yaw_from_quat_wz(df["robot_box_quat_w"].to_numpy(), df["robot_box_quat_z"].to_numpy())
    base_linvel_world = df[["robot_box_linvel_x", "robot_box_linvel_y"]].to_numpy()
    base_yaw_rate = df["robot_box_angvel_z"].to_numpy()

    cube_pos_world = df[schema.CUBE_POS].to_numpy()
    cube_linvel_world = df[schema.CUBE_LINVEL].to_numpy()[:, :2]
    endpoint_pos_world = df[schema.ENDPOINT_POS].to_numpy()
    endpoint_linvel_world = df[schema.ENDPOINT_LINVEL].to_numpy()[:, :2]
    target_pos_world = df[schema.TARGET_POS].to_numpy()

    cube_pos_ego = world_to_ego(cube_pos_world[:, :2], base_pos_world, base_yaw)
    cube_linvel_ego = world_vel_to_ego(
        cube_linvel_world, cube_pos_world[:, :2], base_pos_world, base_linvel_world, base_yaw, base_yaw_rate
    )
    target_pos_ego = world_to_ego(target_pos_world[:, :2], base_pos_world, base_yaw)
    endpoint_pos_ego = world_to_ego(endpoint_pos_world[:, :2], base_pos_world, base_yaw)
    endpoint_linvel_ego = world_vel_to_ego(
        endpoint_linvel_world, endpoint_pos_world[:, :2], base_pos_world, base_linvel_world, base_yaw, base_yaw_rate
    )

    gripper_gear_pos = df[schema.GRIPPER_GEAR_POS[0]].to_numpy()
    gripper_gear_vel = df[schema.GRIPPER_GEAR_VEL[0]].to_numpy()
    gripper_rot_pos = df[schema.GRIPPER_ROT_POS[0]].to_numpy()
    gripper_rot_sincos = sin_cos_encode(gripper_rot_pos)
    gripper_rot_vel = df[schema.GRIPPER_ROT_VEL[0]].to_numpy()
    arm_joint_pos = df[schema.JOINT_POS].to_numpy()
    arm_joint_vel = df[schema.JOINT_VEL].to_numpy()

    ep = Episode(
        path=path,
        t=t,
        base_pos_world=base_pos_world,
        base_yaw=base_yaw,
        base_linvel_world=base_linvel_world,
        base_yaw_rate=base_yaw_rate,
        cube_pos_world=cube_pos_world,
        endpoint_pos_world=endpoint_pos_world,
        target_pos_world=target_pos_world,
        cube_pos_ego=cube_pos_ego,
        cube_z=cube_pos_world[:, 2],
        cube_linvel_ego=cube_linvel_ego,
        target_pos_ego=target_pos_ego,
        target_z=target_pos_world[:, 2],
        endpoint_pos_ego=endpoint_pos_ego,
        endpoint_z=endpoint_pos_world[:, 2],
        endpoint_linvel_ego=endpoint_linvel_ego,
        gripper_gear_pos=gripper_gear_pos,
        gripper_gear_vel=gripper_gear_vel,
        gripper_rot_sincos=gripper_rot_sincos,
        gripper_rot_vel=gripper_rot_vel,
        arm_joint_pos=arm_joint_pos,
        arm_joint_vel=arm_joint_vel,
    )
    _attach_actions(ep, dt, action_mode)
    return ep


def _attach_actions(ep: Episode, dt: float, action_mode: ActionMode) -> None:
    ep.action_mode = action_mode

    if action_mode == "next_state":
        # Action at t = controllable state observed at t+1 (assumes the
        # low-level controller tracks its target closely -- see module
        # docstring). Base xy/yaw are expressed as the ego-frame offset from
        # the CURRENT base pose to next step's base pose, since a raw next
        # world xy is not directly meaningful as an "action" once you're
        # already thinking in ego-frame terms.
        next_base_pos_ego = world_to_ego(ep.base_pos_world[1:], ep.base_pos_world[:-1], ep.base_yaw[:-1])
        ep.action_base_vxy = next_base_pos_ego  # (T-1, 2): where the base ends up, in its own current frame
        ep.action_base_yaw_rate = (ep.base_yaw[1:] - ep.base_yaw[:-1] + np.pi) % (2 * np.pi) - np.pi  # wrap-safe delta yaw
        ep.action_arm_joint = ep.arm_joint_pos[1:]
        ep.action_gripper_gear = ep.gripper_gear_pos[1:]
        # sin/cos target, not raw radians -- consistent with the state
        # encoding and avoids a wraparound discontinuity in the loss.
        ep.action_gripper_rot = ep.gripper_rot_sincos[1:]
    elif action_mode == "finite_diff_vel":
        ep.action_base_vxy = (ep.base_pos_world[1:] - ep.base_pos_world[:-1]) / dt  # world-frame velocity estimate
        ep.action_base_yaw_rate = ep.base_yaw_rate[:-1]  # logged directly, already a rate
        ep.action_arm_joint = ep.arm_joint_vel[:-1]  # logged directly
        ep.action_gripper_gear = ep.gripper_gear_vel[:-1]  # logged directly
        ep.action_gripper_rot = ep.gripper_rot_vel[:-1]  # logged directly
    else:
        raise ValueError(f"unknown action_mode: {action_mode}")
