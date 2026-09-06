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
    TriangulatorOptions()
)

triangulator.AddCorrespondence(
    observations[0],
    observations[1],
)

triangulator.AddCorrespondence(
    observations[1],
    observations[2],
)

created = triangulator.TriangulateImage(
    image_id=1,
    observations=observations,
)

print("CREATED:", len(created))
print("POINTS3D:", len(triangulator.points3D))

if created:
    print("POINT:", created[0].xyz)
    print("TRACK:", created[0].observations)

passed = (
    len(created) == 1
    and len(triangulator.points3D) == 1
    and np.allclose(
        created[0].xyz,
        X,
        atol=1e-5,
    )
)

print(
    "TRIANGULATE IMAGE TEST: PASS"
    if passed
    else "TRIANGULATE IMAGE TEST: FAIL"
)
