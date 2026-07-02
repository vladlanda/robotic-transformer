"""
sim_inference.py -- single-step inference for a live simulator loop.

Usage:
    from sim_inference import load_model, predict_action

    model, cfg = load_model("checkpoints/best_model.pt")

    row = [...]  # one row of raw values, in schema.ALL_COLUMNS order --
                 # exactly what one row of the training CSVs looks like
    action = predict_action(row, model, cfg)
    # action = {
    #     "base_delta_x": ..., "base_delta_y": ..., "base_delta_yaw": ...,
    #     "arm_joint1": ..., "arm_joint2": ..., "arm_joint3": ..., "arm_joint4": ...,
    #     "gripper_gear": ...,
    #     "gripper_rotation_rad": ...,   # decoded back from (sin, cos), NOT the raw pair
    # }

ONLY supports checkpoints trained with action_mode="next_state" (see
episode.py) -- that's the mode confirmed as the right fit, and its action
values are what the dict keys below assume (base deltas + absolute joint/
gripper targets + a sin/cos-encoded wrist angle). A checkpoint trained with
finite_diff_vel has a different, currently unit-inconsistent action layout
(see README) and is deliberately rejected here rather than silently
mis-decoded.

No obstacles are passed in (empty set, not an all-masked block of 8) --
this is mathematically identical to the training-time all-masked-and-zero
obstacle block, since padded/masked tokens are proven not to affect the
model's output (see src/model.py's test_padded_obstacle_content_does_not_affect_output).
It's just less code to pass zero than to pass eight masked zeros.
"""
import numpy as np
import torch

from src import schema, entities, ActionChunkTransformer, ModelConfig
from src.episode import Episode
from src.transforms import yaw_from_quat_wz, world_to_ego, world_vel_to_ego, rotate_to_ego, sin_cos_encode

# Order matches the action vector's index groups exactly (see conversation /
# README's action-space table). The wrist roll collapses from 2 raw values
# (sin, cos) down to 1 key here, since it gets decoded back to radians.
ACTION_KEYS = [
    "base_delta_x", "base_delta_y",
    "base_delta_yaw",
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
    "gripper_gear",
]  # + "gripper_rotation_rad" appended after decoding, see decode_action_vector()


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Loads a checkpoint saved by train.py. Returns (model, cfg)."""
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    action_mode = ckpt.get("action_mode", "next_state")
    if action_mode != "next_state":
        raise ValueError(
            f"checkpoint was trained with action_mode={action_mode!r}, but sim_inference.py "
            f"only supports 'next_state' (its action layout -- base deltas + absolute joint/"
            f"gripper targets + sin/cos wrist -- is what predict_action()/decode_action_vector() "
            f"assume). See the README's open-question section on why finite_diff_vel isn't "
            f"unit-consistent with this yet."
        )

    cfg = ModelConfig(**ckpt["config"])
    model = ActionChunkTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def row_to_episode(row: dict) -> Episode:
    """
    Build a single-timestep (T=1) Episode from one raw row of values, using
    the SAME transform functions load_episode() uses at training time --
    not a re-implementation, so there's no risk of the two silently
    drifting apart. Only the observation-side fields are computable from a
    single row (no t+1 needed, unlike the action_* fields, which is exactly
    why this works for live single-step inference).
    """
    base_pos_world = np.array([[row["robot_box_x"], row["robot_box_y"]]])
    base_yaw = yaw_from_quat_wz(np.array([row["robot_box_quat_w"]]), np.array([row["robot_box_quat_z"]]))
    base_linvel_world = np.array([[row["robot_box_linvel_x"], row["robot_box_linvel_y"]]])
    base_yaw_rate = np.array([row["robot_box_angvel_z"]])
    base_linvel_ego = rotate_to_ego(base_linvel_world, base_yaw)

    cube_pos_world = np.array([[row["cube_world_x"], row["cube_world_y"], row["cube_world_z"]]])
    cube_linvel_world = np.array([[row["cube_linvel_x"], row["cube_linvel_y"]]])
    endpoint_pos_world = np.array([[row["endpoint_world_x"], row["endpoint_world_y"], row["endpoint_world_z"]]])
    endpoint_linvel_world = np.array([[row["endpoint_linvel_x"], row["endpoint_linvel_y"]]])
    target_pos_world = np.array([[row["target_cube_world_x"], row["target_cube_world_y"], row["target_cube_world_z"]]])

    cube_pos_ego = world_to_ego(cube_pos_world[:, :2], base_pos_world, base_yaw)
    cube_linvel_ego = world_vel_to_ego(
        cube_linvel_world, cube_pos_world[:, :2], base_pos_world, base_linvel_world, base_yaw, base_yaw_rate
    )
    target_pos_ego = world_to_ego(target_pos_world[:, :2], base_pos_world, base_yaw)
    endpoint_pos_ego = world_to_ego(endpoint_pos_world[:, :2], base_pos_world, base_yaw)
    endpoint_linvel_ego = world_vel_to_ego(
        endpoint_linvel_world, endpoint_pos_world[:, :2], base_pos_world, base_linvel_world, base_yaw, base_yaw_rate
    )

    gripper_rot_pos = np.array([row["gripper_actual_pos_rad_gp_rotatingjoint"]])

    return Episode(
        path="<live_sim_row>",
        t=np.array([row.get(schema.TIME_COL, 0.0)]),
        base_pos_world=base_pos_world, base_yaw=base_yaw,
        base_linvel_world=base_linvel_world, base_linvel_ego=base_linvel_ego,
        base_yaw_rate=base_yaw_rate,
        cube_pos_world=cube_pos_world, endpoint_pos_world=endpoint_pos_world, target_pos_world=target_pos_world,
        cube_pos_ego=cube_pos_ego, cube_z=cube_pos_world[:, 2], cube_linvel_ego=cube_linvel_ego,
        target_pos_ego=target_pos_ego, target_z=target_pos_world[:, 2],
        endpoint_pos_ego=endpoint_pos_ego, endpoint_z=endpoint_pos_world[:, 2], endpoint_linvel_ego=endpoint_linvel_ego,
        gripper_gear_pos=np.array([row["gripper_actual_pos_rad_gp_leftgearjoint"]]),
        gripper_gear_vel=np.array([row["gripper_actual_vel_rad_s_gp_leftgearjoint"]]),
        gripper_rot_sincos=sin_cos_encode(gripper_rot_pos),
        gripper_rot_vel=np.array([row["gripper_actual_vel_rad_s_gp_rotatingjoint"]]),
        arm_joint_pos=np.array([[row[c] for c in schema.JOINT_POS]]),
        arm_joint_vel=np.array([[row[c] for c in schema.JOINT_VEL]]),
    )


def preprocess_row(row: list) -> tuple:
    """
    row: a flat list of raw values in schema.ALL_COLUMNS order (i.e. one
    row of the training CSVs, read left to right).
    Returns (proprio, object, goal) as (1, dim) float32 torch tensors,
    ready to feed into the model -- built with entities.py's own feature
    functions, so this is guaranteed consistent with training.
    """
    if len(row) != len(schema.ALL_COLUMNS):
        raise ValueError(f"expected {len(schema.ALL_COLUMNS)} values (schema.ALL_COLUMNS), got {len(row)}")
    row_dict = dict(zip(schema.ALL_COLUMNS, row))
    ep = row_to_episode(row_dict)

    proprio = torch.from_numpy(entities.proprio_features(ep, 0)).unsqueeze(0).float()
    obj = torch.from_numpy(entities.object_features(ep, 0)).unsqueeze(0).float()
    goal = torch.from_numpy(entities.goal_features(ep, 0)).unsqueeze(0).float()
    return proprio, obj, goal


def decode_action_vector(vec) -> dict:
    """
    vec: 10 raw numbers in the model's action layout (see ACTION_KEYS +
    the trailing sin/cos pair). Returns a dict with human-readable keys,
    wrist roll converted back from (sin, cos) to a single radian value.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.shape[-1] != 10:
        raise ValueError(f"expected a 10-dim next_state action vector, got shape {vec.shape}")
    result = dict(zip(ACTION_KEYS, vec[:8]))
    sin_val, cos_val = vec[8], vec[9]
    result["gripper_rotation_rad"] = float(np.arctan2(sin_val, cos_val))  # atan2(sin, cos) -- order matters
    return result


def predict_action(row: list, model, cfg: ModelConfig, device: str = "cpu", chunk_index: int = 0) -> dict:
    """
    Full pipeline: raw row -> preprocessing -> model -> decoded action dict.

    chunk_index: the model predicts a `cfg.chunk_size`-step chunk in one
    forward pass (see model.py); chunk_index picks which predicted step to
    decode into the returned dict. 0 = the very next action, which is what
    a live control loop normally wants (predict a chunk, execute step 0,
    re-query next tick with fresh sensor data -- rather than blindly
    executing the whole stale chunk).
    """
    device = torch.device(device)
    proprio, obj, goal = preprocess_row(row)
    proprio, obj, goal = proprio.to(device), obj.to(device), goal.to(device)
    obstacles = torch.zeros(1, 0, entities.FEATURE_DIM["obstacle"], device=device)
    obstacle_mask = torch.zeros(1, 0, device=device)

    with torch.no_grad():
        pred = model(proprio, obj, goal, obstacles, obstacle_mask)  # (1, chunk_size, action_dim)

    vec = pred[0, chunk_index].cpu().numpy()
    return decode_action_vector(vec)


# --------------------------------------------------------------------------
# Unit tests. Run directly with:  python -m sim_inference
# Uses a synthetic row + an untrained model, so this works without a real
# checkpoint file (checkpoints/ is gitignored, so a fresh clone won't have
# one). If a real checkpoint DOES exist locally, also runs one real demo
# prediction against actual data, purely as a sanity spot-check.
# --------------------------------------------------------------------------

def _synthetic_row() -> list:
    """One row's worth of raw values, in schema.ALL_COLUMNS order, reusing
    the same hand-chosen numbers as episode.py's synthetic test CSV (step 0)."""
    values = {
        "sim_time_s": 0.0,
        "robot_box_x": 0.0, "robot_box_y": 0.0, "robot_box_z": 0.0,
        "robot_box_quat_w": 1.0, "robot_box_quat_x": 0.0, "robot_box_quat_y": 0.0, "robot_box_quat_z": 0.0,
        "robot_box_linvel_x": 10.0, "robot_box_linvel_y": 0.0, "robot_box_linvel_z": 0.0,
        "robot_box_angvel_x": 0.0, "robot_box_angvel_y": 0.0, "robot_box_angvel_z": 0.0,
        "cube_world_x": 5.0, "cube_world_y": 2.0, "cube_world_z": 0.42,
        "cube_quat_w": 1.0, "cube_quat_x": 0.0, "cube_quat_y": 0.0, "cube_quat_z": 0.0,
        "cube_linvel_x": 0.0, "cube_linvel_y": 0.0, "cube_linvel_z": 0.0,
        "cube_angvel_x": 0.0, "cube_angvel_y": 0.0, "cube_angvel_z": 0.0,
        "endpoint_world_x": 1.0, "endpoint_world_y": 1.0, "endpoint_world_z": 0.5,
        "endpoint_linvel_x": 10.0, "endpoint_linvel_y": 0.0, "endpoint_linvel_z": 0.0,
        "target_cube_world_x": 8.0, "target_cube_world_y": 3.0, "target_cube_world_z": 0.42,
        "gripper_actual_pos_rad_gp_leftgearjoint": -1.57,
        "gripper_actual_vel_rad_s_gp_leftgearjoint": 3.7,
        "gripper_actual_pos_rad_gp_rotatingjoint": 0.0,
        "gripper_actual_vel_rad_s_gp_rotatingjoint": 5.0,
        "motor_angle_rad_joint1": 0.0, "motor_angle_rad_joint2": 0.0,
        "motor_angle_rad_joint3": 0.0, "motor_angle_rad_joint4": 0.0,
        "motor_speed_rad_s_joint1": 1.0, "motor_speed_rad_s_joint2": 1.1,
        "motor_speed_rad_s_joint3": 1.2, "motor_speed_rad_s_joint4": 1.3,
    }
    return [values[c] for c in schema.ALL_COLUMNS]


def test_row_to_episode_matches_known_values():
    # Same numbers/expected outputs as episode.py's synthetic-CSV test at
    # step 0 (see test_load_episode_ego_frame_values) -- cross-checks that
    # single-row reconstruction agrees with the multi-row CSV path.
    row = dict(zip(schema.ALL_COLUMNS, _synthetic_row()))
    ep = row_to_episode(row)
    assert np.allclose(ep.cube_pos_ego, [[5.0, 2.0]])
    assert np.allclose(ep.target_pos_ego, [[8.0, 3.0]])
    assert np.allclose(ep.endpoint_pos_ego, [[1.0, 1.0]])
    assert np.allclose(ep.endpoint_linvel_ego, [[0.0, 0.0]], atol=1e-10)
    assert np.allclose(ep.cube_linvel_ego, [[-10.0, 0.0]])


def test_preprocess_row_shapes():
    proprio, obj, goal = preprocess_row(_synthetic_row())
    assert proprio.shape == (1, entities.FEATURE_DIM["proprio"])
    assert obj.shape == (1, entities.FEATURE_DIM["object"])
    assert goal.shape == (1, entities.FEATURE_DIM["goal"])


def test_preprocess_row_rejects_wrong_length():
    try:
        preprocess_row(_synthetic_row()[:-1])
        raise AssertionError("expected ValueError for a short row, none was raised")
    except ValueError:
        pass


def test_decode_action_vector_keys_and_wrist_roundtrip():
    angle = 1.234
    vec = [0.1, -0.2, 0.03, 0.4, 0.5, 0.6, 0.7, -0.8, np.sin(angle), np.cos(angle)]
    result = decode_action_vector(vec)
    assert set(result.keys()) == set(ACTION_KEYS) | {"gripper_rotation_rad"}
    assert np.isclose(result["gripper_rotation_rad"], angle)
    assert np.isclose(result["base_delta_x"], 0.1)
    assert np.isclose(result["arm_joint4"], 0.7)
    assert np.isclose(result["gripper_gear"], -0.8)


def test_decode_action_vector_rejects_wrong_dim():
    try:
        decode_action_vector([0.0] * 9)
        raise AssertionError("expected ValueError for a 9-dim vector, none was raised")
    except ValueError:
        pass


def test_predict_action_end_to_end_untrained_model():
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=10)
    model = ActionChunkTransformer(cfg)
    model.eval()
    action = predict_action(_synthetic_row(), model, cfg)
    assert set(action.keys()) == set(ACTION_KEYS) | {"gripper_rotation_rad"}
    for k, v in action.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    assert -np.pi <= action["gripper_rotation_rad"] <= np.pi


def test_load_model_rejects_finite_diff_vel_checkpoint(tmp_path=None):
    import tempfile, os
    cfg = ModelConfig(d_model=16, nhead=2, num_layers=2, dim_feedforward=32, chunk_size=6, action_dim=9)
    model = ActionChunkTransformer(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad_checkpoint.pt")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__,
                    "action_mode": "finite_diff_vel"}, path)
        try:
            load_model(path)
            raise AssertionError("expected ValueError for a finite_diff_vel checkpoint, none was raised")
        except ValueError:
            pass


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


def _demo_with_real_checkpoint_if_present():
    import os, glob
    import pandas as pd

    ckpt_path = "checkpoints/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"\n(skipping real-checkpoint demo -- {ckpt_path} not found; "
              f"train one with train.py to see this)")
        return

    data_files = sorted(glob.glob("data/*.csv"))
    if not data_files:
        print("\n(skipping real-checkpoint demo -- no files found in data/)")
        return

    print(f"\nreal-checkpoint demo: {ckpt_path} on the first row of {data_files[0]}")
    model, cfg = load_model(ckpt_path)
    row = pd.read_csv(data_files[0]).iloc[0][schema.ALL_COLUMNS].tolist()
    action = predict_action(row, model, cfg)
    for k, v in action.items():
        print(f"  {k:22s} {v:+.5f}")


if __name__ == "__main__":
    _run_all_tests()
    _demo_with_real_checkpoint_if_present()
