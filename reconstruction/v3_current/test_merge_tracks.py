import numpy as np

from geometry.incremental_triangulator import (
    IncrementalTriangulator,
    Point3D,
    Observation,
    TriangulatorOptions,
)


class Camera:
    def __init__(self):
        self.K = np.array([
            [800.0, 0.0, 0.0],
            [0.0, 800.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def cam_from_img(self, p):
        return np.array([
            p[0] / 800.0,
            p[1] / 800.0,
        ])


camera = Camera()

poses = [
    np.eye(4),
    np.eye(4),
    np.eye(4),
    np.eye(4),
]

poses[1][0, 3] = -1.0
poses[2][1, 3] = -0.5
poses[3][0, 3] = 0.5

X = np.array([0.2, 0.1, 4.0])


def make_observation(image_id, pose, point3D_id):
    Xc = pose[:3, :3] @ X + pose[:3, 3]

    xy = np.array([
        800.0 * Xc[0] / Xc[2],
        800.0 * Xc[1] / Xc[2],
    ])

    return Observation(
        image_id=image_id,
        point2D_idx=0,
        xy=xy,
        camera=camera,
        cam_from_world=pose,
        point3D_id=point3D_id,
    )


tri = IncrementalTriangulator(
    TriangulatorOptions()
)

# ---------------------------------------------------------
# First two compatible tracks
# ---------------------------------------------------------

p1 = Point3D(
    point3D_id=1,
    xyz=X.copy(),
    observations=[(1, 0), (2, 0)],
)

p2 = Point3D(
    point3D_id=2,
    xyz=X.copy(),
    observations=[(3, 0)],
)

tri.points3D[1] = p1
tri.points3D[2] = p2

obs1 = make_observation(1, poses[0], 1)
obs2 = make_observation(2, poses[1], 1)
obs3 = make_observation(3, poses[2], 2)

for observation in (obs1, obs2, obs3):
    tri._observation_registry[
        (observation.image_id, observation.point2D_idx)
    ] = observation

# ---------------------------------------------------------
# MergeTracks()
# ---------------------------------------------------------

merged = tri.MergeTracks([(1, 2)])

print("MERGED:", merged)
print("POINTS AFTER MergeTracks:", len(tri.points3D))

if merged != 1:
    raise RuntimeError("MergeTracks failed")

if len(tri.points3D) != 1:
    raise RuntimeError("MergeTracks did not leave one point")

# ---------------------------------------------------------
# Create another compatible point
# ---------------------------------------------------------

remaining_id = next(iter(tri.points3D))
remaining = tri.points3D[remaining_id]

p3 = Point3D(
    point3D_id=3,
    xyz=remaining.xyz.copy(),
    observations=[(4, 0)],
)

tri.points3D[3] = p3

obs4 = make_observation(4, poses[3], 3)

tri._observation_registry[
    (obs4.image_id, obs4.point2D_idx)
] = obs4

# ---------------------------------------------------------
# MergeAllTracks()
# ---------------------------------------------------------

merged_all = tri.MergeAllTracks(
    [(remaining_id, 3)]
)

print("MERGED ALL:", merged_all)
print("POINTS AFTER MergeAllTracks:", len(tri.points3D))

passed = (
    merged == 1
    and merged_all == 1
    and len(tri.points3D) == 1
)

print(
    "MERGE TRACKS TEST: PASS"
    if passed
    else "MERGE TRACKS TEST: FAIL"
)

if not passed:
    raise RuntimeError("Merge wrapper test failed")
