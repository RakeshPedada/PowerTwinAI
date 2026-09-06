import numpy as np

from geometry.incremental_triangulator import (
    IncrementalTriangulator,
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
]

poses[1][0, 3] = -1.0
poses[2][1, 3] = -0.5

X = np.array([0.2, 0.1, 4.0])

obs = []

for i, pose in enumerate(poses):
    Xc = pose[:3, :3] @ X + pose[:3, 3]

    xy = np.array([
        800.0 * Xc[0] / Xc[2],
        800.0 * Xc[1] / Xc[2],
    ])

    obs.append(
        Observation(
            image_id=i + 1,
            point2D_idx=0,
            xy=xy,
            camera=camera,
            cam_from_world=pose,
        )
    )

tri = IncrementalTriangulator(
    TriangulatorOptions(
        ignore_two_view_tracks=True,
    )
)

tri.AddCorrespondence(obs[0], obs[1])
tri.AddCorrespondence(obs[1], obs[2])

created = tri.TriangulateImage(
    1,
    obs,
)

if not created:
    raise RuntimeError("Initial 3-view track creation failed")

point = created[0]

before = len(point.observations)

added = tri.CompleteTracks(obs)

after = len(point.observations)

print("INITIAL CREATED:", len(created))
print("COMPLETE TRACKS ADDED:", added)
print("TRACK BEFORE:", before)
print("TRACK AFTER:", after)
print("TRACK:", point.observations)

all_added = tri.CompleteAllTracks()

print("COMPLETE ALL TRACKS ADDED:", all_added)
print("FINAL TRACK:", point.observations)

passed = (
    len(created) == 1
    and len(tri.points3D) == 1
    and len(point.observations) == 3
    and all(
        key in point.observations
        for key in [(1, 0), (2, 0), (3, 0)]
    )
)

print(
    "COMPLETE TRACKS TEST: PASS"
    if passed
    else "COMPLETE TRACKS TEST: FAIL"
)
