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


# --------------------------------------------------------------------------
# Unit tests. Run directly with:  python -m src.schema
# These aren't testing "functions" (schema.py is just data) -- they're
# testing that the data is internally consistent, which is exactly the
# class of bug (miscounted/duplicated/missing columns) that's easy to
# introduce by hand when a schema like this grows.
# --------------------------------------------------------------------------

def test_all_columns_no_duplicates():
    assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS)), "duplicate column name in ALL_COLUMNS"


def test_all_columns_count_matches_known_dataset_width():
    # The real dataset (verified in scripts/inspect_dataset.py) has 48
    # columns. If this ever drifts, either the schema or the real data
    # changed and it's worth knowing which.
    assert len(ALL_COLUMNS) == 48, f"expected 48 columns, schema declares {len(ALL_COLUMNS)}"


def test_group_lengths_match_expected_dimensionality():
    assert len(BASE_POS) == 3 and len(BASE_QUAT) == 4 and len(BASE_LINVEL) == 3 and len(BASE_ANGVEL) == 3
    assert len(CUBE_POS) == 3 and len(CUBE_QUAT) == 4 and len(CUBE_LINVEL) == 3 and len(CUBE_ANGVEL) == 3
    assert len(TARGET_POS) == 3
    assert len(ENDPOINT_POS) == 3 and len(ENDPOINT_LINVEL) == 3
    assert len(GRIPPER_GEAR_POS) == 1 and len(GRIPPER_GEAR_VEL) == 1
    assert len(GRIPPER_ROT_POS) == 1 and len(GRIPPER_ROT_VEL) == 1
    assert len(JOINT_POS) == 4 and len(JOINT_VEL) == 4


def test_degenerate_columns_are_a_subset_of_base_columns():
    base_columns = set(BASE_POS + BASE_QUAT + BASE_ANGVEL)
    assert set(DEGENERATE_COLUMNS).issubset(base_columns), (
        "DEGENERATE_COLUMNS should only reference base pose/orientation/angvel columns"
    )


def test_controllable_state_references_only_real_columns():
    all_cols = set(ALL_COLUMNS)
    for name, cols in CONTROLLABLE_STATE.items():
        for c in cols:
            assert c in all_cols, f"CONTROLLABLE_STATE[{name!r}] references unknown column {c!r}"


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
