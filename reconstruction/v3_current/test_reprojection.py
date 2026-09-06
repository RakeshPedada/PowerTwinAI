import numpy as np
from geometry.estimate_triangulation import EstimateTriangulation
from geometry.triangulation_estimator import TriangulationResidualType

class Camera:
    pass

c = Camera()
c.K = np.array([
    [800., 0., 0.],
    [0., 800., 0.],
    [0., 0., 1.]
])

c.cam_from_img = lambda p: np.array([
    p[0] / 800.,
    p[1] / 800.
])

poses = [
    np.eye(4),
    np.eye(4),
    np.eye(4)
]

poses[1][0, 3] = -1.
poses[2][1, 3] = -0.5

X = np.array([0.2, 0.1, 4.])

points = []
cameras = []

for p in poses:
    X_cam = p[:3, :3] @ X + p[:3, 3]

    points.append(np.array([
        800.0 * X_cam[0] / X_cam[2],
        800.0 * X_cam[1] / X_cam[2]
    ]))

    cameras.append(c)

r = EstimateTriangulation(
    points,
    poses,
    cameras,
    residual_type=TriangulationResidualType.REPROJECTION_ERROR,
    max_error=16.0
)

print("POINTS:", points)
print("SUCCESS:", r.success)
print("POINT:", r.point3D)
print("INLIERS:", int(np.count_nonzero(r.inlier_mask)))
print("MASK:", r.inlier_mask)

print(
    "REPROJECTION TEST: PASS"
    if r.success and int(np.count_nonzero(r.inlier_mask)) >= 2
    else "REPROJECTION TEST: FAIL"
)
