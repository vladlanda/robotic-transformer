# robotic-transformer

Distilling an RL-trained mobile-manipulator policy into a goal-conditioned
transformer, trained via behavior cloning on recorded rollouts, with the
aim of generalizing pick-and-place beyond the exact configurations seen
during RL training.

## The robot

A mobile manipulator: a planar (SE(2): x, y, yaw) base carrying a
continuum/soft arm. The arm is visually made of many small links, but only
4 bend angles are independently actuated (`joint1..4`), each applied
uniformly across the links in that segment. The gripper has 2 DOF: a
linear open/close joint and a wrist-roll ("rotating") joint. See
`robotic_transformer/data/schema.py` for the full column-level breakdown
and the reasoning behind it.

## Data

`data/*.csv` — 298 episodes, single env (`env_000`), all sharing the same
48-column schema (verified in `scripts/inspect_dataset.py`). Each episode
is a short (19–59 step, 60Hz) pick-and-place: the RL policy drives the base
+ arm to a randomly placed cube, grasps it, and carries it to a randomly
placed target zone. Cube starts and targets are drawn from two fixed,
non-overlapping regions (mirroring the gray-zone → green-zone setup in the
reference screenshot).

## Repo layout

```
robotic_transformer/
  data/
    schema.py      column groupings + notes on what's actually in the logs
    transforms.py  SE(2) ego-frame transform, sin/cos angle encoding
    episode.py     loads one CSV -> Episode (raw + ego-frame + pseudo-actions)
    dataset.py     PyTorch Dataset: chunked (obs, action_chunk) windows
scripts/
  inspect_dataset.py   sanity-checks the pipeline against every episode
  compute_stats.py     per-channel normalization stats -> stats/*.json
stats/
  normalization_*.json precomputed stats, one file per action_mode
```

## Key design decisions made so far

- **Ego-frame, not world-frame.** The base moves continuously throughout
  each episode (this is whole-body manipulation, not "drive then reach"),
  so cube/target/endpoint positions and velocities are transformed into the
  base's own frame before being used as features. This is implemented
  properly (including the base's own rotation rate, not just a static
  rotation) in `transforms.world_to_ego` / `world_vel_to_ego`.
- **Wrist roll as sin/cos.** `gripper_actual_pos_rad_gp_rotatingjoint` is
  not bounded to a fixed range across episodes (seen: `[0, 1.85]` in one,
  `[-1.42, 0]` in another) — treated as unbounded/continuous and encoded as
  `(sin, cos)` to avoid a wraparound discontinuity.
- **Action chunking.** The dataset yields a `chunk_size`-step action window
  per sample (ACT-style), not single-step actions, to reduce compounding
  error at rollout time.

## OPEN QUESTION — there is no recorded action/command column

Every column in the logs is a *measured state* (`robot_box_*`,
`motor_angle_*`, `gripper_actual_*`, ...). Nothing records what the RL
policy actually commanded at each step. `episode.py` implements two
workarounds behind an `action_mode` flag, and **we have not yet picked
one as final**:

- `"next_state"` (current default): action at `t` = controllable state
  observed at `t+1`. Assumes the low-level controller tracks its target
  closely; inherits any tracking lag.
- `"finite_diff_vel"`: action at `t` = `(state[t+1] - state[t]) / dt` for
  position-like DOFs, or the logged `*_vel` column directly where available.
  No tracking-lag assumption, but noisier.

Both are implemented and both have precomputed stats in `stats/`, so we can
compare empirically once there's a model to compare them with, rather than
guessing up front.

## Running the checks

```
pip install -r requirements.txt
python scripts/inspect_dataset.py      # loads all 298 episodes, checks for NaN/Inf
python scripts/compute_stats.py        # writes stats/normalization_next_state.json
python scripts/compute_stats.py data finite_diff_vel
```

## Not yet done

- Model architecture (transformer encoder over the flattened per-step
  features -> action-chunk head). `dataset.py` produces plain flat vectors;
  the tokenization scheme for the transformer itself isn't decided yet.
- Endpoint orientation is not logged at all — fine if every grasp uses a
  fixed approach orientation (looks true from the reference episodes), but
  worth confirming before assuming it away entirely.
- Cube/target orientation is not yet expressed in the base's ego frame
  (only positions are); low priority unless grasp orientation ends up
  mattering.
