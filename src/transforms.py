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


def rotate_to_ego(vec_world: np.ndarray, base_yaw: np.ndarray) -> np.ndarray:
    """
    Rotate a world-frame 2D vector (e.g. a velocity) into the base's ego
    frame -- rotation only, no translation. Use this for vectors that don't
    have a "position" (nothing to subtract the base's position from),
    unlike world_to_ego which is for points.

    vec_world: (T, 2)
    base_yaw:  (T,)
    returns:   (T, 2)
    """
    cos_t, sin_t = np.cos(base_yaw), np.sin(base_yaw)
    x_ego = cos_t * vec_world[..., 0] + sin_t * vec_world[..., 1]
    y_ego = -sin_t * vec_world[..., 0] + cos_t * vec_world[..., 1]
    return np.stack([x_ego, y_ego], axis=-1)


def sin_cos_encode(angle: np.ndarray) -> np.ndarray:
    """
    Encode an angle (radians, any range -- including unbounded/continuous
    joints that exceed +-pi) as (sin, cos) to avoid wraparound discontinuity.

    angle: (T,)
    returns: (T, 2) stacked [sin, cos]
    """
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


# --------------------------------------------------------------------------
# Unit tests. Run directly with:  python -m src.transforms
# (must be run as a module from the repo root so the package-relative
# imports elsewhere in src/ aren't disturbed by how this file is invoked)
# --------------------------------------------------------------------------

def test_yaw_from_quat_wz():
    # identity rotation: w=1, z=0 -> yaw=0
    assert np.isclose(yaw_from_quat_wz(np.array([1.0]), np.array([0.0]))[0], 0.0)
    # 90 degree rotation: quat = (cos(45deg), sin(45deg)) for a half-angle representation
    half = np.pi / 4
    yaw = yaw_from_quat_wz(np.array([np.cos(half)]), np.array([np.sin(half)]))[0]
    assert np.isclose(yaw, np.pi / 2), f"expected pi/2, got {yaw}"
    # -90 degrees
    yaw = yaw_from_quat_wz(np.array([np.cos(half)]), np.array([-np.sin(half)]))[0]
    assert np.isclose(yaw, -np.pi / 2), f"expected -pi/2, got {yaw}"
    # vectorized over T
    w = np.array([1.0, np.cos(half), np.cos(half)])
    z = np.array([0.0, np.sin(half), -np.sin(half)])
    yaws = yaw_from_quat_wz(w, z)
    assert np.allclose(yaws, [0.0, np.pi / 2, -np.pi / 2])


def test_world_to_ego_identity():
    # base at world origin, facing yaw=0 -> ego frame IS world frame
    point = np.array([[3.0, -2.0]])
    base_pos = np.array([[0.0, 0.0]])
    base_yaw = np.array([0.0])
    ego = world_to_ego(point, base_pos, base_yaw)
    assert np.allclose(ego, point)


def test_world_to_ego_translation_only():
    point = np.array([[5.0, 5.0]])
    base_pos = np.array([[2.0, 1.0]])
    base_yaw = np.array([0.0])
    ego = world_to_ego(point, base_pos, base_yaw)
    assert np.allclose(ego, [[3.0, 4.0]])  # just point - base, no rotation


def test_world_to_ego_rotation_90deg():
    # base at origin, facing +90 degrees (i.e. facing world +y).
    # A point directly ahead of the base in the room (world +y) should show
    # up as "straight ahead" in ego coords, i.e. ego x > 0, ego y ~ 0.
    point = np.array([[0.0, 5.0]])  # 5 units along world +y from the base
    base_pos = np.array([[0.0, 0.0]])
    base_yaw = np.array([np.pi / 2])
    ego = world_to_ego(point, base_pos, base_yaw)
    assert np.allclose(ego, [[5.0, 0.0]], atol=1e-10), f"got {ego}"


def test_world_to_ego_matches_manual_example_from_conversation():
    # Regression test for the exact scenario a user question surfaced:
    # y_ego == 0 does NOT imply point_world_y == base_world_y when the base
    # is rotated. Reconstructed from real episode data (env_000_episode_0008,
    # step 25): base yaw ~ -70.4 degrees, cube world y = 0.752, base world y
    # = 0.039 (clearly not equal), yet cube's y_ego ~ 0.
    cube_world = np.array([[-0.250, 0.752]])
    base_pos = np.array([[-0.0000833, 0.0392]])
    base_yaw = np.array([np.deg2rad(-70.4)])
    ego = world_to_ego(cube_world, base_pos, base_yaw)
    assert abs(ego[0, 1]) < 0.01, f"expected y_ego ~ 0, got {ego[0, 1]}"
    assert not np.isclose(cube_world[0, 1], base_pos[0, 1], atol=0.1), (
        "sanity check on the test data itself: world-y values should NOT match"
    )


def test_world_vel_to_ego_comoving_point_has_zero_relative_velocity():
    # If a point moves with EXACTLY the base's velocity and there's no
    # rotation, its velocity in the ego frame must be zero (it's not moving
    # relative to the robot at all).
    vel_world = np.array([[1.5, -0.5]])
    point_world = np.array([[3.0, 3.0]])
    base_pos = np.array([[0.0, 0.0]])
    base_vel = np.array([[1.5, -0.5]])
    base_yaw = np.array([0.3])  # rotation shouldn't matter for a zero vector
    base_yaw_rate = np.array([0.0])
    ego_vel = world_vel_to_ego(vel_world, point_world, base_pos, base_vel, base_yaw, base_yaw_rate)
    assert np.allclose(ego_vel, [[0.0, 0.0]], atol=1e-10), f"got {ego_vel}"


def test_world_vel_to_ego_rotation_coupling_term():
    # A point that is stationary in the WORLD frame, sitting 2 units along
    # the base's local +x at zero yaw, observed from a base that is only
    # spinning (yaw_rate=1, no translation) should appear to move backwards
    # along ego -y at speed = yaw_rate * distance = 2.0 (the classic
    # "-omega x r" rotating-frame term), not zero.
    vel_world = np.array([[0.0, 0.0]])       # point isn't moving in the room
    point_world = np.array([[2.0, 0.0]])     # 2 units ahead of the base
    base_pos = np.array([[0.0, 0.0]])
    base_vel = np.array([[0.0, 0.0]])        # base isn't translating
    base_yaw = np.array([0.0])
    base_yaw_rate = np.array([1.0])          # but IS spinning
    ego_vel = world_vel_to_ego(vel_world, point_world, base_pos, base_vel, base_yaw, base_yaw_rate)
    assert np.allclose(ego_vel, [[0.0, -2.0]], atol=1e-10), f"got {ego_vel}"


def test_rotate_to_ego_matches_world_to_ego_when_base_at_origin():
    # rotate_to_ego (no translation) and world_to_ego (with translation)
    # must agree when the base sits at the world origin, since translation
    # is a no-op there.
    vec = np.array([[4.0, -1.0]])
    yaw = np.array([0.77])
    base_pos = np.array([[0.0, 0.0]])
    assert np.allclose(rotate_to_ego(vec, yaw), world_to_ego(vec, base_pos, yaw))


def test_rotate_to_ego_preserves_vector_length():
    # rotation must not change a vector's magnitude
    vec = np.array([[3.0, 4.0], [1.0, 1.0], [-2.0, 5.0]])
    yaw = np.array([0.1, 1.5, -2.3])
    rotated = rotate_to_ego(vec, yaw)
    assert np.allclose(np.linalg.norm(vec, axis=-1), np.linalg.norm(rotated, axis=-1))


def test_sin_cos_encode():
    angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2, 5.0, -100.0])
    sc = sin_cos_encode(angles)
    assert sc.shape == (len(angles), 2)
    # every (sin, cos) pair must lie on the unit circle, however large the
    # input angle -- this is the whole point of the encoding (no unbounded
    # radian value to worry about downstream)
    assert np.allclose(sc[:, 0] ** 2 + sc[:, 1] ** 2, 1.0)
    assert np.allclose(sc[0], [0.0, 1.0])            # angle 0
    assert np.allclose(sc[1], [1.0, 0.0], atol=1e-10)  # pi/2
    assert np.allclose(sc[3], [-1.0, 0.0], atol=1e-10)  # -pi/2


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
