import numpy as np
from types import SimpleNamespace
from geometry import IncrementalTriangulator, TriangulatorOptions, Observation

K = np.array([
    [800., 0., 320.],
    [0., 800., 240.],
    [0., 0., 1.]
])

cam = SimpleNamespace(
    K=K,
    CamFromImg=lambda p: (np.linalg.inv(K) @ np.array([p[0], p[1], 1.]))[:2]
)

opt = TriangulatorOptions(
    ignore_two_view_tracks=False,
    min_angle=0.0,
    create_max_angle_error=2.0,
    complete_max_reproj_error=4.0
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
        cam_from_world=poses[i]
    )
    for i in range(3)
]

t.RegisterObservations(obs)
t.AddCorrespondence(obs[1], obs[2])

t.AddCorrespondence(obs[1], obs[2])
# Create initial 3D point from observations 1 and 2.
point = t.Create(obs[:2])

print("CREATE:", point.xyz if point else None)

# Complete the track through observation 3.
if point:
    completed = t.Complete(point)
    print("COMPLETE:", completed)
    print("TRACK_SIZE:", len(point.observations))
    print("POINTS3D:", len(t.points3D))
    print("HAS_OBS_3:", (3, 0) in point.observations)
