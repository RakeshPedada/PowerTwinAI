import numpy as np

from geometry.incremental_triangulator import (
    IncrementalTriangulator,
    Observation,
    TriangulatorOptions,
)


class Camera:
    def cam_from_img(self, p):
        return np.asarray(p, dtype=float) / 800.0


camera = Camera()
pose = np.eye(4)

tri = IncrementalTriangulator(
    TriangulatorOptions(
        ignore_two_view_tracks=True,
    )
)

# ---------------------------------------------------------
# 1. Insufficient observations
# ---------------------------------------------------------

obs1 = Observation(
    image_id=1,
    point2D_idx=0,
    xy=np.array([40.0, 20.0]),
    camera=camera,
    cam_from_world=pose,
)

result = tri.TriangulateImage(1, [obs1])

print("INSUFFICIENT OBS:", result)

if result:
    raise RuntimeError("Insufficient-observation guard failed")

# ---------------------------------------------------------
# 2. Two-view track must be ignored
# ---------------------------------------------------------

obs2 = Observation(
    image_id=2,
    point2D_idx=0,
    xy=np.array([-160.0, 20.0]),
    camera=camera,
    cam_from_world=pose.copy(),
)

tri2 = IncrementalTriangulator(
    TriangulatorOptions(
        ignore_two_view_tracks=True,
    )
)

tri2.AddCorrespondence(obs1, obs2)

result2 = tri2.TriangulateImage(
    1,
    [obs1, obs2],
)

print("TWO VIEW CREATED:", len(result2))

if result2:
    raise RuntimeError("ignore_two_view_tracks guard failed")

# ---------------------------------------------------------
# 3. Invalid camera must be rejected
# ---------------------------------------------------------

class InvalidCamera:
    def has_bogus_params(
        self,
        min_focal_length_ratio,
        max_focal_length_ratio,
        max_extra_param,
    ):
        return True

invalid_camera = InvalidCamera()

invalid_obs = Observation(
    image_id=3,
    point2D_idx=0,
    xy=np.array([40.0, 20.0]),
    camera=invalid_camera,
    cam_from_world=pose,
)

valid = tri.IsCameraValid(
    99,
    invalid_camera,
)

print("INVALID CAMERA:", valid)

if valid:
    raise RuntimeError("Invalid-camera guard failed")

# ---------------------------------------------------------
# 4. Invalid pose must not be usable
# ---------------------------------------------------------

bad_pose = np.zeros((3, 3))

bad_obs = Observation(
    image_id=4,
    point2D_idx=0,
    xy=np.array([40.0, 20.0]),
    camera=camera,
    cam_from_world=bad_pose,
)

result4 = tri.TriangulateImage(
    4,
    [bad_obs, obs2],
)

print("INVALID POSE RESULT:", result4)

if result4:
    raise RuntimeError("Invalid-pose guard failed")

print("EDGE CASE REGRESSION: PASS")
