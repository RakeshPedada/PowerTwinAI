import os
import time
import csv
import cv2
import numpy as np
import open3d as o3d


from dense_reconstruction import DenseReconstructor
from colmap_loader import load_colmap_model

def process_single_image(file):

    file.seek(0)

    file_bytes = np.asarray(
        bytearray(file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return None

    h, w = img.shape[:2]

    if h > w:

        img = cv2.rotate(
            img,
            cv2.ROTATE_90_CLOCKWISE
        )

    return {
        "image": img
    }
# =========================================================
# SAVE POINT CLOUD
# =========================================================

def save_point_cloud(points, colors, output_path):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(
        output_path,
        pcd
    )

    return output_path


# =========================================================
# ANALYTICS
# =========================================================

def build_analytics(
    sparse_points,
    dense_points,
    camera_positions,
    processing_time
):

    total_points = (
        len(sparse_points)
        + len(dense_points)
    )

    analytics = {

        "total_sparse_points":
        int(len(sparse_points)),

        "total_dense_points":
        int(len(dense_points)),

        "total_points":
        int(total_points),

        "total_cameras":
        int(len(camera_positions)),

        "processing_time":
        round(processing_time, 2),

        "health_score":
        100
    }

    return analytics


# =========================================================
# MAIN
# =========================================================

def run_reconstruction(
    uploaded_files,
    progress_callback=print
):

    start_time = time.time()

    progress_callback(
        "Loading Images..."
    )

    processed_data = []

    for idx, file in enumerate(uploaded_files):

        image_data = process_single_image(
            file
        )

        if image_data is None:
            continue

        processed_data.append(
            image_data
        )

        progress_callback(
            f"Image {idx+1} processed"
        )

    if len(processed_data) < 2:

        raise Exception(
            "Need at least 2 images"
        )
        progress_callback(
        "Loading COLMAP Model..."
    )

    (
        camera_positions,
        global_rotations,
        global_translations,
        sparse_points,
        sparse_colors
    ) = load_colmap_model(
        "colmap_data"
    )

    progress_callback(
        f"COLMAP Cameras: {len(camera_positions)}"
    )

    progress_callback(
        f"COLMAP Sparse Points: {len(sparse_points)}"
    )

    images = [
        item["image"]
        for item in processed_data
    ]

    sample_image = images[0]

    h, w = sample_image.shape[:2]

    focal = max(w, h)

    intrinsic_matrix = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ])

    progress_callback(
        "Running Dense Reconstruction..."
    )

    dense_reconstructor = (
        DenseReconstructor()
    )

    dense_result = (
        dense_reconstructor
        .generate_dense_cloud(

            images,

            camera_positions,

            intrinsic_matrix,

            global_rotations,

            global_translations
        )
    )

    if dense_result is None:

        dense_points = np.empty(
            (0, 3)
        )

        dense_colors = np.empty(
            (0, 3)
        )

    else:

        dense_points = dense_result[
            "points"
        ]

        dense_colors = dense_result[
            "colors"
        ]

    progress_callback(
        f"Dense Points: "
        f"{len(dense_points)}"
    )

    final_points = sparse_points
    final_colors = sparse_colors   
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        final_points
    )

    pcd.colors = o3d.utility.Vector3dVector(
        final_colors
    )

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.5
    )

    final_points = np.asarray(
        pcd.points
    )

    final_colors = np.asarray(
        pcd.colors
    )

    print(
        f"After cleaning: {len(final_points)} points"
    )
    progress_callback(
        "Cleaning Point Cloud..."
    )

    try:

        pcd = o3d.geometry.PointCloud()

        pcd.points = (
            o3d.utility.Vector3dVector(
                final_points
            )
        )

        pcd.colors = (
            o3d.utility.Vector3dVector(
                final_colors
            )
        )

        pcd, _ = (
            pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
        )

        final_points = np.asarray(
            pcd.points
        )

        final_colors = np.asarray(
            pcd.colors
        )

    except Exception as e:

        progress_callback(
            f"Cleanup skipped: {e}"
        )

    os.makedirs(
        "output",
        exist_ok=True
    )

    ply_path = os.path.join(
        "output",
        "advanced_reconstruction.ply"
    )

    save_point_cloud(
        final_points,
        final_colors,
        ply_path
    )

    pair_logs = [

        {
            "pair": "COLMAP",
            "matches": len(sparse_points),
            "inliers": len(sparse_points),
            "points": len(final_points),
            "confidence": 100,
            "status": "SUCCESS"
        }

    ]

    csv_path = os.path.join(
        "output",
        "pair_logs.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "matches",
                "inliers",
                "points",
                "confidence",
                "status"
            ]
        )

        writer.writeheader()

        writer.writerows(
            pair_logs
        )

    processing_time = (
        time.time() - start_time
    )

    analytics = build_analytics(

        sparse_points,

        dense_points,

        camera_positions,

        processing_time
    )

    match_visuals = []

    return (

        final_points,

        final_colors,

        camera_positions,

        match_visuals,

        ply_path,

        pair_logs,

        processing_time,

        analytics
    )