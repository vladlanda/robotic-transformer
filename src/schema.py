"""
Column schema for the mobile-manipulator episode logs.

This reflects the 48-column format found in data/*.csv (298 episodes, all
from env_000, all sharing this exact schema as of this commit — verified with
scripts/inspect_dataset.py).

IMPORTANT — this is an OBSERVATION log, not an action log. Every column here
is a measured/actual quantity (robot_box_*, cube_*, endpoint_*,
gripper_actual_*, motor_angle_*, motor_speed_*). There is no recorded
"commanded" or "target" control signal for the controllable DOFs (only
target_cube_world_* which is the task goal, not a motor command). See
robotic_transformer/data/episode.py for how we currently derive pseudo-action
labels from this, and the README for the open design question this raises.

Robot structure (confirmed against the sim, see conversation history):
  - Base: planar (SE(2)) mobile base. robot_box_z, quat_x, quat_y, angvel_x,
    angvel_y are identically 0 across the dataset -> only (x, y, yaw) and
    (vx, vy, yaw_rate) carry signal.
  - Arm: continuum/soft backbone visually made of many links, but only 4
    independently actuated bend angles (joint1..4), each applied uniformly
    across the links belonging to that segment.
  - Gripper: 2 DOF - a linear "gear" joint (open/close) and a "rotating"
    joint (wrist roll). The rotating joint is NOT bounded to (-pi/2, pi/2) in
    practice (seen ranges: [0, 1.85] in one episode, [-1.42, 0] in another) -
    treat it as an unbounded/continuous joint, encode via sin/cos.
"""

# --- Base (mobile platform, SE(2)) ---
BASE_POS = ["robot_box_x", "robot_box_y", "robot_box_z"]
BASE_QUAT = ["robot_box_quat_w", "robot_box_quat_x", "robot_box_quat_y", "robot_box_quat_z"]
BASE_LINVEL = ["robot_box_linvel_x", "robot_box_linvel_y", "robot_box_linvel_z"]
BASE_ANGVEL = ["robot_box_angvel_x", "robot_box_angvel_y", "robot_box_angvel_z"]

# --- Manipulated object ---
CUBE_POS = ["cube_world_x", "cube_world_y", "cube_world_z"]
CUBE_QUAT = ["cube_quat_w", "cube_quat_x", "cube_quat_y", "cube_quat_z"]
CUBE_LINVEL = ["cube_linvel_x", "cube_linvel_y", "cube_linvel_z"]
CUBE_ANGVEL = ["cube_angvel_x", "cube_angvel_y", "cube_angvel_z"]

# --- Goal (static per episode) ---
TARGET_POS = ["target_cube_world_x", "target_cube_world_y", "target_cube_world_z"]

# --- End effector (position + linear velocity only -- NO orientation logged) ---
ENDPOINT_POS = ["endpoint_world_x", "endpoint_world_y", "endpoint_world_z"]
ENDPOINT_LINVEL = ["endpoint_linvel_x", "endpoint_linvel_y", "endpoint_linvel_z"]

# --- Gripper ---
GRIPPER_GEAR_POS = ["gripper_actual_pos_rad_gp_leftgearjoint"]       # open/close
GRIPPER_GEAR_VEL = ["gripper_actual_vel_rad_s_gp_leftgearjoint"]
GRIPPER_ROT_POS = ["gripper_actual_pos_rad_gp_rotatingjoint"]        # wrist roll, unbounded
GRIPPER_ROT_VEL = ["gripper_actual_vel_rad_s_gp_rotatingjoint"]

# --- Arm segments (4 actuated bend angles) ---
JOINT_POS = [f"motor_angle_rad_joint{i}" for i in range(1, 5)]
JOINT_VEL = [f"motor_speed_rad_s_joint{i}" for i in range(1, 5)]

TIME_COL = "sim_time_s"

ALL_COLUMNS = (
    [TIME_COL]
    + BASE_POS + BASE_QUAT + BASE_LINVEL + BASE_ANGVEL
    + CUBE_POS + CUBE_QUAT + CUBE_LINVEL + CUBE_ANGVEL
    + ENDPOINT_POS + ENDPOINT_LINVEL
    + TARGET_POS
    + GRIPPER_GEAR_POS + GRIPPER_GEAR_VEL + GRIPPER_ROT_POS + GRIPPER_ROT_VEL
    + JOINT_POS + JOINT_VEL
)

# Columns confirmed constant (zero) across the dataset -- degenerate but kept
# for forward-compatibility with a possible future non-planar base.
DEGENERATE_COLUMNS = [
    "robot_box_z", "robot_box_quat_x", "robot_box_quat_y",
    "robot_box_angvel_x", "robot_box_angvel_y",
]

# The controllable DOFs of the robot -- what an "action" ultimately needs to
# specify, one way or another (see episode.py for how this is currently
# turned into training targets).
CONTROLLABLE_STATE = {
    "base_xy": ["robot_box_x", "robot_box_y"],
    "base_yaw_from_quat": ["robot_box_quat_w", "robot_box_quat_z"],
    "arm_joints": JOINT_POS,
    "gripper_gear": GRIPPER_GEAR_POS,
    "gripper_rot": GRIPPER_ROT_POS,
}
