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

observations = []

for i, pose in enumerate(poses):
    Xc = pose[:3, :3] @ X + pose[:3, 3]

    xy = np.array([
        800.0 * Xc[0] / Xc[2],
        800.0 * Xc[1] / Xc[2],
    ])

    observations.append(
        Observation(
            image_id=i + 1,
            point2D_idx=0,
            xy=xy,
            camera=camera,
            cam_from_world=pose,
        )
    )

triangulator = IncrementalTriangulator(
    TriangulatorOptions(
        complete_max_reproj_error=4.0,
    )
)

for obs in observations:
    triangulator.RegisterObservation(obs)

triangulator.AddCorrespondence(
    observations[0],
    observations[1],
)

triangulator.AddCorrespondence(
    observations[1],
    observations[2],
)

result = triangulator.CompleteImage(
    image_id=1,
    observations=observations,
)

print("COMPLETE IMAGE RESULT:", result)
print("POINTS3D:", len(triangulator.points3D))

if triangulator.points3D:
    point = next(iter(triangulator.points3D.values()))
    print("POINT:", point.xyz)
    print("TRACK:", point.observations)

print(
    "COMPLETE IMAGE TEST: PASS"
    if result > 0 and len(triangulator.points3D) == 1
    else "COMPLETE IMAGE TEST: FAIL"
)
