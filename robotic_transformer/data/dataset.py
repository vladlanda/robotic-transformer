"""
A minimal, deliberately simple PyTorch Dataset over the episode files.

Design choices made here (open to revisiting):
  - Each sample is one timestep's observation paired with an ACTION CHUNK of
    `chunk_size` future pseudo-actions (ACT-style chunking, to reduce
    compounding error at inference vs. single-step prediction).
  - `history` past steps of proprioception are included as context (0 = no
    history, just the current step).
  - Padding at episode ends is done by repeating the last valid action and
    masking it out via `action_mask`, rather than dropping incomplete
    windows -- keeps every timestep in an episode usable as a starting
    point, which matters given episodes are short (19-59 steps).

This does NOT yet decide the model architecture or the tokenization scheme
fed into a transformer -- it produces plain flat feature vectors per step.
That mapping (which fields become which tokens) is the next thing to design
once the action_mode question above is settled.
"""
import glob
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .episode import Episode, load_episode, ActionMode


def _obs_vector(ep: Episode, i: int) -> np.ndarray:
    """Flatten one timestep's observation into a single feature vector."""
    return np.concatenate([
        ep.cube_pos_ego[i], [ep.cube_z[i]],
        ep.cube_linvel_ego[i],
        ep.target_pos_ego[i], [ep.target_z[i]],
        ep.endpoint_pos_ego[i], [ep.endpoint_z[i]],
        ep.endpoint_linvel_ego[i],
        [ep.gripper_gear_pos[i], ep.gripper_gear_vel[i]],
        ep.gripper_rot_sincos[i], [ep.gripper_rot_vel[i]],
        ep.arm_joint_pos[i], ep.arm_joint_vel[i],
    ]).astype(np.float32)


def _action_vector(ep: Episode, i: int) -> np.ndarray:
    """Flatten the pseudo-action at step i (see episode.py for action_mode semantics)."""
    parts = [ep.action_base_vxy[i], [ep.action_base_yaw_rate[i]], ep.action_arm_joint[i],
              [ep.action_gripper_gear[i]]]
    if ep.action_gripper_rot is not None:
        parts.append(np.atleast_1d(ep.action_gripper_rot[i]).ravel())
    return np.concatenate(parts).astype(np.float32)


class EpisodeChunkDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 12,
        history: int = 0,
        action_mode: ActionMode = "next_state",
        file_glob: str = "*.csv",
    ):
        self.chunk_size = chunk_size
        self.history = history
        self.action_mode = action_mode

        files = sorted(glob.glob(os.path.join(data_dir, file_glob)))
        if not files:
            raise FileNotFoundError(f"no episode files found in {data_dir} matching {file_glob}")

        self.episodes: list[Episode] = [load_episode(f, action_mode=action_mode) for f in files]

        # Index of (episode_idx, start_step) for every valid starting point.
        # An episode of length T has actions defined for steps [0, T-2]
        # (action at t needs state at t+1), so valid starts are [0, T-2].
        self.index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            n_actions = len(ep.action_base_yaw_rate)
            for start in range(n_actions):
                self.index.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        n_actions = len(ep.action_base_yaw_rate)

        # --- observation (+ optional history) ---
        hist_start = max(0, start - self.history)
        obs_steps = list(range(hist_start, start + 1))
        obs = np.stack([_obs_vector(ep, i) for i in obs_steps], axis=0)
        if len(obs_steps) < self.history + 1:
            pad = np.repeat(obs[:1], self.history + 1 - len(obs_steps), axis=0)
            obs = np.concatenate([pad, obs], axis=0)

        # --- action chunk, padded at episode end + mask ---
        end = min(start + self.chunk_size, n_actions)
        chunk = np.stack([_action_vector(ep, i) for i in range(start, end)], axis=0)
        mask = np.ones(len(chunk), dtype=np.float32)
        if len(chunk) < self.chunk_size:
            pad_n = self.chunk_size - len(chunk)
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad_n, axis=0)], axis=0)
            mask = np.concatenate([mask, np.zeros(pad_n, dtype=np.float32)], axis=0)

        return {
            "obs": torch.from_numpy(obs),          # (history+1, obs_dim)
            "action_chunk": torch.from_numpy(chunk),  # (chunk_size, action_dim)
            "action_mask": torch.from_numpy(mask),    # (chunk_size,)
            "episode_path": ep.path,
            "step": start,
        }
