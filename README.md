# robotic-transformer

Distilling an RL-trained mobile-manipulator policy into a goal-conditioned
transformer, trained via behavior cloning on recorded rollouts, with the
aim of generalizing pick-and-place beyond the exact configurations seen
during RL training.

## Start here

`notebooks/01_project_walkthrough.ipynb` — an educational, tab-by-tab
walkthrough of everything below: the raw data, the ego-frame transform,
entity tokens, the tokenizer, the model, and a full training loop at the
end (already executed, with plots, so you can read it without rerunning).
Does **not** cover inference/rollout evaluation — that's a separate
notebook. The rest of this README is the written reference for the same
material.

## The robot

A mobile manipulator: a planar (SE(2): x, y, yaw) base carrying a
continuum/soft arm. The arm is visually made of many small links, but only
4 bend angles are independently actuated (`joint1..4`), each applied
uniformly across the links in that segment. The gripper has 2 DOF: a
linear open/close joint and a wrist-roll ("rotating") joint. See
`src/schema.py` for the full column-level breakdown
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
src/
  schema.py      column groupings + notes on what's actually in the logs
  transforms.py  SE(2) ego-frame transform (points + velocities), sin/cos encoding
  episode.py     loads one CSV -> Episode (raw + ego-frame + pseudo-actions)
  entities.py    per-entity feature extraction (proprio/object/goal/obstacle)
                 + fixed TYPE_ID tags -- see "Entity tokens" below
  tokenizer.py   EntityTokenizer (nn.Module): raw features -> tagged tokens
  dataset.py     PyTorch Dataset: chunked (entity tokens, action_chunk) windows
scripts/
  inspect_dataset.py       sanity-checks the pipeline against every episode
  compute_stats.py         per-channel normalization stats -> stats/*.json
  smoke_test_tokenizer.py  end-to-end: Dataset -> batch -> EntityTokenizer
stats/
  normalization_*.json precomputed stats, one file per action_mode
```

## Entity tokens (tag-embedding scheme)

Every "thing in the scene" becomes its own token:
`token = Linear_kind(raw_features) + type_embedding[TYPE_ID[kind]]`.
`TYPE_ID` (in `entities.py`) is fixed, hardcoded bookkeeping -- not learned.
What IS learned is the content of each type's tag vector, and the linear
projection weights (in `tokenizer.py`).

Four kinds today: `proprio` (1 token: arm + gripper + base velocity +
gripper-tip pose, always present), `object` (1 token: the cube), `goal` (1
token: the target position), `obstacle` (0..N tokens, padded to
`max_obstacles` and masked).

**No obstacle data exists in the current dataset.** `entities.obstacle_features()`
always returns zero rows today, so obstacle tokens are always fully masked
(the model sees a "no obstacles present" signal, not fake obstacles at the
origin). This is intentional and already tested end-to-end in
`scripts/smoke_test_tokenizer.py` — adding real obstacles later should only
require changing `obstacle_features()` to read real columns; the Dataset,
tokenizer, and any model built on top of it should not need to change.

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

## Testing

Two layers of checks:

**1. Per-module unit tests.** Every module under `src/` is independently
runnable and self-testing -- each file ends with `test_*` functions (plain
assertions, no pytest dependency) and an `if __name__ == "__main__"` block
that runs them and prints PASS/FAIL:

```
python -m src.schema
python -m src.transforms
python -m src.episode
python -m src.entities
python -m src.tokenizer
python -m src.dataset
python -m src.model
```

Or all of them in one pass: `python scripts/run_all_unit_tests.py`
(49 assertions total, as of this writing).

These run against small synthetic/hand-computable data (e.g. `episode.py`
builds a synthetic CSV with a base moving in a straight line at constant
velocity, so the expected ego-frame values can be worked out by hand and
checked exactly) -- not the real dataset. Notably: `transforms.py` includes
a regression test for the exact "`y_ego=0` doesn't mean world-y matches"
scenario a conversation surfaced; `entities.py` tests that
`validate_feature_dims` actually catches a deliberately broken dimension
(this exact bug happened once while writing the file); `model.py` tests
that scrambling only *masked* obstacle content leaves the model's output
unchanged -- proof the attention mask protects the output, not just that
the tokenizer accepts padded input.

**2. Integration checks against the real dataset:**

```
pip install -r requirements.txt
python scripts/inspect_dataset.py      # loads all 298 episodes, checks for NaN/Inf
python scripts/compute_stats.py        # writes stats/normalization_next_state.json
python scripts/compute_stats.py data finite_diff_vel
python scripts/smoke_test_tokenizer.py # Dataset -> batch -> EntityTokenizer, end to end
```

## Not yet done

- The transformer body itself (attention over the entity tokens) and the
  action-chunk output head. `tokenizer.py` produces tagged tokens; nothing
  yet consumes the full token *set* with self-attention -- that's the next
  piece.
- Real obstacle data. The `obstacle` token slot is reserved and tested with
  an empty/masked set, but nothing has been evaluated with actual obstacles
  present.
- Endpoint orientation is not logged at all — fine if every grasp uses a
  fixed approach orientation (looks true from the reference episodes), but
  worth confirming before assuming it away entirely.
- Cube/target orientation is not yet expressed in the base's ego frame
  (only positions are); low priority unless grasp orientation ends up
  mattering.
