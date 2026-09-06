import numpy as np
from types import SimpleNamespace
from geometry import (
    IncrementalTriangulator,
    TriangulatorOptions,
    Observation,
)

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
    min_angle=0.0,
    create_max_angle_error=2.0,
    continue_max_angle_error=2.0,
    re_max_angle_error=5.0,
    re_min_ratio=0.2,
    re_max_trials=1,
)

t = IncrementalTriangulator(opt)

X = np.array([0.2, 0.1, 4.0])

poses = [
    np.eye(4),
    np.array([
        [1., 0., 0., -0.1],
        [0., 1., 0.,  0.0],
        [0., 0., 1.,  0.0],
        [0., 0., 0.,  1.0]
    ]),
    np.array([
        [1., 0., 0., -0.2],
        [0., 1., 0.,  0.0],
        [0., 0., 1.,  0.0],
        [0., 0., 0.,  1.0]
    ])
]

def project(P, X):
    q = K @ P[:3, :] @ np.r_[X, 1.]
    return q[:2] / q[2]

pixels = [project(P, X) for P in poses]

obs = [
    Observation(
        image_id=i + 1,
        point2D_idx=0,
        xy=pixels[i],
        camera=cam,
        cam_from_world=poses[i],
    )
    for i, P in enumerate(poses)
]

t.RegisterObservations(obs)

# Establish an existing 3D point from views 1 and 2.
point = t.Create(obs[:2])

print("INITIAL_POINT:", point.xyz if point else None)
print("INITIAL_TRACK:", point.observations if point else None)

# Retriangulate with only view 3 missing from the existing track.
changed = t.Retriangulate(
    [(obs[0], obs[2])],
    num_tri_corrs=0,
    num_total_corrs=10,
)

print("CHANGED:", changed)
print("RETRIALS:", t.retriangulation_trials)
print("FINAL_TRACK:", point.observations if point else None)
print("OBS3_POINT_ID:", obs[2].point3D_id)
print("POINTS3D:", len(t.points3D))
