"""
sim_inference.py -- load the model, feed it one row of raw simulator
values, get back the predicted next action as a dict.
"""
import numpy as np
import torch

from src import schema, ActionChunkTransformer, ModelConfig
from src.transforms import yaw_from_quat_wz, world_to_ego, world_vel_to_ego, rotate_to_ego, sin_cos_encode


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str = None):
    device = device or get_device()
    print(f"using device: {device}")
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = ActionChunkTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def predict_action(row: list, model, cfg) -> dict:
    """
    row: flat list of 48 raw values, in schema.ALL_COLUMNS order (one row
    of the training csvs). Returns the predicted next action as a dict.
    """
    r = dict(zip(schema.ALL_COLUMNS, row))
    device = next(model.parameters()).device  # match wherever the model actually is

    # ego-frame transform (same math used at training time) -- the model
    # never sees raw world coordinates, only "relative to me right now"
    base_pos = np.array([[r["robot_box_x"], r["robot_box_y"]]])
    base_yaw = yaw_from_quat_wz(np.array([r["robot_box_quat_w"]]), np.array([r["robot_box_quat_z"]]))
    base_linvel = np.array([[r["robot_box_linvel_x"], r["robot_box_linvel_y"]]])
    base_yaw_rate = np.array([r["robot_box_angvel_z"]])
    base_linvel_ego = rotate_to_ego(base_linvel, base_yaw)[0]

    cube_pos = np.array([[r["cube_world_x"], r["cube_world_y"]]])
    cube_linvel = np.array([[r["cube_linvel_x"], r["cube_linvel_y"]]])
    cube_pos_ego = world_to_ego(cube_pos, base_pos, base_yaw)[0]
    cube_linvel_ego = world_vel_to_ego(cube_linvel, cube_pos, base_pos, base_linvel, base_yaw, base_yaw_rate)[0]

    target_pos = np.array([[r["target_cube_world_x"], r["target_cube_world_y"]]])
    target_pos_ego = world_to_ego(target_pos, base_pos, base_yaw)[0]

    endpoint_pos = np.array([[r["endpoint_world_x"], r["endpoint_world_y"]]])
    endpoint_linvel = np.array([[r["endpoint_linvel_x"], r["endpoint_linvel_y"]]])
    endpoint_pos_ego = world_to_ego(endpoint_pos, base_pos, base_yaw)[0]
    endpoint_linvel_ego = world_vel_to_ego(
        endpoint_linvel, endpoint_pos, base_pos, base_linvel, base_yaw, base_yaw_rate
    )[0]

    wrist_sincos = sin_cos_encode(np.array([r["gripper_actual_pos_rad_gp_rotatingjoint"]]))[0]

    proprio = np.concatenate([
        [r["motor_angle_rad_joint1"], r["motor_angle_rad_joint2"], r["motor_angle_rad_joint3"], r["motor_angle_rad_joint4"]],
        [r["motor_speed_rad_s_joint1"], r["motor_speed_rad_s_joint2"], r["motor_speed_rad_s_joint3"], r["motor_speed_rad_s_joint4"]],
        [r["gripper_actual_pos_rad_gp_leftgearjoint"], r["gripper_actual_vel_rad_s_gp_leftgearjoint"]],
        wrist_sincos, [r["gripper_actual_vel_rad_s_gp_rotatingjoint"]],
        base_linvel_ego, [r["robot_box_angvel_z"]],
        endpoint_pos_ego, [r["endpoint_world_z"]], endpoint_linvel_ego,
    ]).astype(np.float32)

    obj = np.concatenate([cube_pos_ego, [r["cube_world_z"]], cube_linvel_ego]).astype(np.float32)
    goal = np.concatenate([target_pos_ego, [r["target_cube_world_z"]]]).astype(np.float32)

    proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(device)
    obj_t = torch.from_numpy(obj).unsqueeze(0).to(device)
    goal_t = torch.from_numpy(goal).unsqueeze(0).to(device)
    obstacles = torch.zeros(1, 0, 3, device=device)      # no obstacle data exists -- empty set
    obstacle_mask = torch.zeros(1, 0, device=device)

    with torch.no_grad():
        pred = model(proprio_t, obj_t, goal_t, obstacles, obstacle_mask)  # (1, chunk_size, action_dim)

    a = pred[0, 0].cpu().numpy()  # first step of the predicted chunk = the next action to take

    return {
        "base_delta_x": float(a[0]),          # ego frame, meters
        "base_delta_y": float(a[1]),          # ego frame, meters
        "base_delta_yaw": float(a[2]),        # radians
        "arm_joint1": float(a[3]),            # radians, absolute target
        "arm_joint2": float(a[4]),
        "arm_joint3": float(a[5]),
        "arm_joint4": float(a[6]),
        "gripper_gear": float(a[7]),          # radians, absolute target
        "gripper_rotation_rad": float(np.arctan2(a[8], a[9])),  # decoded from (sin, cos)
    }


if __name__ == "__main__":
    model, cfg = load_model("checkpoints/best_model.pt")

    # one example row, in schema.ALL_COLUMNS order:
    row = [
        0.0,                  # sim_time_s
        0.0, 0.0, 0.0,        # robot_box_x, y, z
        1.0, 0.0, 0.0, 0.0,   # robot_box_quat_w, x, y, z
        0.0, 0.0, 0.0,        # robot_box_linvel_x, y, z
        0.0, 0.0, 0.0,        # robot_box_angvel_x, y, z
        0.33, -0.73, 0.42,    # cube_world_x, y, z
        1.0, 0.0, 0.0, 0.0,   # cube_quat_w, x, y, z
        0.0, 0.0, 0.0,        # cube_linvel_x, y, z
        0.0, 0.0, 0.0,        # cube_angvel_x, y, z
        0.10, 0.10, 0.50,     # endpoint_world_x, y, z
        0.0, 0.0, 0.0,        # endpoint_linvel_x, y, z
        -0.35, -0.74, 0.42,   # target_cube_world_x, y, z
        -1.57,                # gripper_actual_pos_rad_gp_leftgearjoint
        0.0,                  # gripper_actual_vel_rad_s_gp_leftgearjoint
        0.0,                  # gripper_actual_pos_rad_gp_rotatingjoint
        0.0,                  # gripper_actual_vel_rad_s_gp_rotatingjoint
        0.0, 0.0, 0.0, 0.0,   # motor_angle_rad_joint1..4
        0.0, 0.0, 0.0, 0.0,   # motor_speed_rad_s_joint1..4
    ]

    action = predict_action(row, model, cfg)
    print(action)
