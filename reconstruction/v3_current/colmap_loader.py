import os
import numpy as np


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to rotation matrix."""

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Invalid zero-length quaternion")

    qw, qx, qy, qz = q / norm

    return np.array([
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw)
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw)
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy)
        ]
    ], dtype=np.float64)


def load_colmap_cameras(cameras_file):
    """Read camera calibration from COLMAP cameras.txt."""

    cameras = {}

    with open(cameras_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            params = np.asarray(
                [float(value) for value in parts[4:]],
                dtype=np.float64
            )

            # Build K for the common COLMAP camera models.
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                f_value, cx, cy = params[:3]
                fx = fy = f_value

            elif model in (
                "PINHOLE",
                "OPENCV",
                "OPENCV_FISHEYE",
                "FULL_OPENCV"
            ):
                fx, fy, cx, cy = params[:4]

            else:
                raise ValueError(
                    f"Unsupported COLMAP camera model: {model}"
                )

            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "K": K
            }

    if not cameras:
        raise RuntimeError("No cameras found in cameras.txt")

    return cameras


def load_colmap_images(images_file, cameras):
    """Read registered images and preserve filename -> pose mapping."""

    registered_images = {}

    with open(images_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # images.txt contains two lines per registered image:
    # pose line followed by the POINTS2D line.
    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        # A valid image pose line:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10:

            try:
                image_id = int(parts[0])

                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])

                t = np.array([
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7])
                ], dtype=np.float64)

                camera_id = int(parts[8])

                # Allows filenames containing spaces.
                image_name = " ".join(parts[9:])

                if camera_id not in cameras:
                    raise RuntimeError(
                        f"Camera {camera_id} used by {image_name} "
                        "was not found in cameras.txt"
                    )

                R = quaternion_to_rotation_matrix(
                    qw, qx, qy, qz
                )

                # COLMAP:
                # X_cam = R * X_world + t
                #
                # Therefore camera center in world coordinates:
                camera_center = -R.T @ t

                registered_images[image_name] = {
                    "image_id": image_id,
                    "image_name": image_name,
                    "camera_id": camera_id,
                    "R": R,
                    "t": t,
                    "camera_center": camera_center,
                    "K": cameras[camera_id]["K"].copy(),
                    "width": cameras[camera_id]["width"],
                    "height": cameras[camera_id]["height"],
                    "camera_model": cameras[camera_id]["model"]
                }

                # Skip POINTS2D line.
                i += 2
                continue

            except (ValueError, IndexError):
                pass

        i += 1

    if not registered_images:
        raise RuntimeError(
            "No registered images found in images.txt"
        )

    return registered_images


def load_colmap_points(points_file):
    sparse_points = []
    sparse_colors = []

    with open(points_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            try:
                sparse_points.append([
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ])

                sparse_colors.append([
                    int(parts[4]) / 255.0,
                    int(parts[5]) / 255.0,
                    int(parts[6]) / 255.0
                ])

            except (ValueError, IndexError):
                continue

    return (
        np.asarray(sparse_points, dtype=np.float64),
        np.asarray(sparse_colors, dtype=np.float64)
    )


def load_colmap_model(folder):

    cameras_file = os.path.join(folder, "cameras.txt")
    images_file = os.path.join(folder, "images.txt")
    points_file = os.path.join(folder, "points3D.txt")

    for path in (
        cameras_file,
        images_file,
        points_file
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing COLMAP model file: {path}"
            )

    cameras = load_colmap_cameras(
        cameras_file
    )

    registered_images = load_colmap_images(
        images_file,
        cameras
    )

    sparse_points, sparse_colors = load_colmap_points(
        points_file
    )

    camera_positions = np.asarray(
        [
            data["camera_center"]
            for data in registered_images.values()
        ],
        dtype=np.float64
    )

    print(
        f"[COLMAP LOADER] Cameras: {len(cameras)}"
    )

    print(
        f"[COLMAP LOADER] Registered images: "
        f"{len(registered_images)}"
    )

    print(
        f"[COLMAP LOADER] Sparse points: "
        f"{len(sparse_points)}"
    )

    return (
        cameras,
        registered_images,
        camera_positions,
        sparse_points,
        sparse_colors
    )