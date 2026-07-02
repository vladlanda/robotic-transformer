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

As of this version, observations are returned as SEPARATE, TAGGED entity
groups (proprio / object / goal / obstacles+mask) rather than one flat
vector -- see entities.py and tokenizer.py. `max_obstacles` reserves space
for obstacle tokens even though the current dataset has none (they come
back as an all-zero, fully-masked block); this is what lets obstacle
support be added later without changing the Dataset or model interface,
only entities.obstacle_features().
"""
import glob
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .episode import Episode, load_episode, ActionMode
from . import entities


def _action_vector(ep: Episode, i: int) -> np.ndarray:
    """Flatten the pseudo-action at step i (see episode.py for action_mode semantics)."""
    parts = [ep.action_base_vxy[i], [ep.action_base_yaw_rate[i]], ep.action_arm_joint[i],
              [ep.action_gripper_gear[i]]]
    if ep.action_gripper_rot is not None:
        parts.append(np.atleast_1d(ep.action_gripper_rot[i]).ravel())
    return np.concatenate(parts).astype(np.float32)


def _stack_with_history(fn, ep: Episode, steps: list[int]) -> np.ndarray:
    return np.stack([fn(ep, i) for i in steps], axis=0)


class EpisodeChunkDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 12,
        history: int = 0,
        max_obstacles: int = 8,
        action_mode: ActionMode = "next_state",
        file_glob: str = "*.csv",
    ):
        self.chunk_size = chunk_size
        self.history = history
        self.max_obstacles = max_obstacles
        self.action_mode = action_mode

        files = sorted(glob.glob(os.path.join(data_dir, file_glob)))
        if not files:
            raise FileNotFoundError(f"no episode files found in {data_dir} matching {file_glob}")

        self.episodes: list[Episode] = [load_episode(f, action_mode=action_mode) for f in files]
        entities.validate_feature_dims(self.episodes[0])  # fail fast if *_features() drifts from FEATURE_DIM

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

    def _obstacle_block(self, ep: Episode, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features, mask) padded/truncated to exactly max_obstacles rows.
        Today ep's obstacle_features() always returns 0 rows, so this is an
        all-zero, fully-masked block -- exercising the real padding/masking
        code path even though there's nothing to pad yet."""
        raw = entities.obstacle_features(ep, i)  # (n_obs, FEATURE_DIM['obstacle'])
        n_obs = min(raw.shape[0], self.max_obstacles)
        feat = np.zeros((self.max_obstacles, entities.FEATURE_DIM["obstacle"]), dtype=np.float32)
        mask = np.zeros(self.max_obstacles, dtype=np.float32)
        if n_obs > 0:
            feat[:n_obs] = raw[:n_obs]
            mask[:n_obs] = 1.0
        return feat, mask

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        n_actions = len(ep.action_base_yaw_rate)

        # --- observation (+ optional history), per entity group ---
        hist_start = max(0, start - self.history)
        obs_steps = list(range(hist_start, start + 1))
        pad_n = self.history + 1 - len(obs_steps)
        if pad_n > 0:
            obs_steps = [obs_steps[0]] * pad_n + obs_steps  # repeat earliest valid step

        proprio = _stack_with_history(entities.proprio_features, ep, obs_steps)  # (H+1, 21)
        obj = _stack_with_history(entities.object_features, ep, obs_steps)       # (H+1, 5)
        goal = _stack_with_history(entities.goal_features, ep, obs_steps)        # (H+1, 3)
        obstacle_feat, obstacle_mask = zip(*[self._obstacle_block(ep, i) for i in obs_steps])
        obstacle_feat = np.stack(obstacle_feat, axis=0)   # (H+1, max_obstacles, 3)
        obstacle_mask = np.stack(obstacle_mask, axis=0)   # (H+1, max_obstacles)

        # --- action chunk, padded at episode end + mask ---
        end = min(start + self.chunk_size, n_actions)
        chunk = np.stack([_action_vector(ep, i) for i in range(start, end)], axis=0)
        act_mask = np.ones(len(chunk), dtype=np.float32)
        if len(chunk) < self.chunk_size:
            act_pad_n = self.chunk_size - len(chunk)
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], act_pad_n, axis=0)], axis=0)
            act_mask = np.concatenate([act_mask, np.zeros(act_pad_n, dtype=np.float32)], axis=0)

        return {
            "proprio": torch.from_numpy(proprio),              # (H+1, 21)
            "object": torch.from_numpy(obj),                    # (H+1, 5)
            "goal": torch.from_numpy(goal),                     # (H+1, 3)
            "obstacles": torch.from_numpy(obstacle_feat),        # (H+1, max_obstacles, 3)
            "obstacle_mask": torch.from_numpy(obstacle_mask),    # (H+1, max_obstacles)
            "action_chunk": torch.from_numpy(chunk),             # (chunk_size, action_dim)
            "action_mask": torch.from_numpy(act_mask),           # (chunk_size,)
            "episode_path": ep.path,
            "step": start,
        }
