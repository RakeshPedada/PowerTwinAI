import numpy as np
from types import SimpleNamespace
from geometry import IncrementalTriangulator, TriangulatorOptions, Observation, Point3D

K = np.array([
    [800., 0., 320.],
    [0., 800., 240.],
    [0., 0., 1.]
])

cam = SimpleNamespace(
    K=K,
    CamFromImg=lambda p: (
        np.linalg.inv(K) @ np.array([p[0], p[1], 1.])
    )[:2]
)

opt = TriangulatorOptions(
    ignore_two_view_tracks=False,
    merge_max_reproj_error=4.0
)

t = IncrementalTriangulator(opt)

X = np.array([0.2, 0.1, 4.0])

P1 = np.eye(4)
P2 = np.array([
    [1., 0., 0., -0.1],
    [0., 1., 0.,  0.0],
    [0., 0., 1.,  0.0],
    [0., 0., 0.,  1.0]
])
P3 = np.array([
    [1., 0., 0., -0.2],
    [0., 1., 0.,  0.0],
    [0., 0., 1.,  0.0],
    [0., 0., 0.,  1.0]
])

def project(P, X):
    q = K @ P[:3, :] @ np.r_[X, 1.]
    return q[:2] / q[2]

pixels = [
    project(P1, X),
    project(P2, X),
    project(P3, X)
]

obs = [
    Observation(
        image_id=i + 1,
        point2D_idx=0,
        xy=pixels[i],
        camera=cam,
        cam_from_world=P,
    )
    for i, P in enumerate([P1, P2, P3])
]

t.RegisterObservations(obs)

# Two independent 3D points with the same true location.
p1 = Point3D(
    point3D_id=1,
    xyz=X.copy(),
    observations=[(1, 0), (2, 0)]
)

p2 = Point3D(
    point3D_id=2,
    xyz=X.copy(),
    observations=[(3, 0)]
)

t.points3D[1] = p1
t.points3D[2] = p2

for o in obs:
    o.point3D_id = 1 if o.image_id < 3 else 2

print("BEFORE:", len(t.points3D))
print("P1 OBS:", p1.observations)
print("P2 OBS:", p2.observations)

merged = t.Merge(1, 2)

print("MERGED:", merged is not None)
print("AFTER:", len(t.points3D))
print("XYZ:", None if merged is None else merged.xyz)
print("OBS:", None if merged is None else merged.observations)
print("ERROR:", None if merged is None else np.linalg.norm(merged.xyz - X))
