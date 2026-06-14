import numpy as np


def quaternion_to_rotation_matrix(qw, qx, qy, qz):

    return np.array([
        [
            1 - 2*qy*qy - 2*qz*qz,
            2*qx*qy - 2*qz*qw,
            2*qx*qz + 2*qy*qw
        ],
        [
            2*qx*qy + 2*qz*qw,
            1 - 2*qx*qx - 2*qz*qz,
            2*qy*qz - 2*qx*qw
        ],
        [
            2*qx*qz - 2*qy*qw,
            2*qy*qz + 2*qx*qw,
            1 - 2*qx*qx - 2*qy*qy
        ]
    ])


def load_colmap_model(folder):

    cameras_file = f"{folder}/cameras.txt"
    images_file = f"{folder}/images.txt"
    points_file = f"{folder}/points3D.txt"

    camera_positions = []
    global_rotations = {}
    global_translations = {}

    sparse_points = []
    sparse_colors = []

    # ============================
    # LOAD IMAGES
    # ============================

    with open(images_file, "r") as f:

        lines = f.readlines()

    image_index = 0

    for line in lines:

        if line.startswith("#"):
            continue

        parts = line.strip().split()

        if len(parts) < 10:
            continue

        try:

            image_id = int(parts[0])

            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])

            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])

            R = quaternion_to_rotation_matrix(
                qw,
                qx,
                qy,
                qz
            )

            t = np.array(
                [tx, ty, tz]
            )

            camera_center = -R.T @ t

            camera_positions.append(
                camera_center
            )

            global_rotations[
                image_index
            ] = R

            global_translations[
                image_index
            ] = t.reshape(3, 1)
            
            image_index += 1

        except:
            pass

    # ============================
    # LOAD POINTS
    # ============================

    with open(points_file, "r") as f:

        lines = f.readlines()

    for line in lines:

        if line.startswith("#"):
            continue

        parts = line.strip().split()

        if len(parts) < 7:
            continue

        try:

            X = float(parts[1])
            Y = float(parts[2])
            Z = float(parts[3])

            R = int(parts[4])
            G = int(parts[5])
            B = int(parts[6])

            sparse_points.append(
                [X, Y, Z]
            )

            sparse_colors.append(
                [
                    R / 255.0,
                    G / 255.0,
                    B / 255.0
                ]
            )

        except:
            pass

    return (
        np.array(camera_positions),
        global_rotations,
        global_translations,
        np.array(sparse_points),
        np.array(sparse_colors)
    )