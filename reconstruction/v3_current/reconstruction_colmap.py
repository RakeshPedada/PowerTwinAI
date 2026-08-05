import os
import time
import csv

import cv2
import numpy as np
import open3d as o3d

from dense_reconstruction import DenseReconstructor
from colmap_loader import load_colmap_model
from mesh_generator import generate_mesh


print("COLMAP RECONSTRUCTION.PY IS RUNNING")


# =========================================================
# LOAD IMAGE
# =========================================================

def process_single_image(image_info):
    """
    image_info must contain:
        {
            "path": "...",
            "name": "image_001.JPG"
        }

    IMPORTANT:
    Do NOT rotate images here.
    COLMAP poses/intrinsics correspond to the original image orientation.
    """

    path = image_info["path"]
    name = image_info["name"]

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:
        print(
            f"[WARNING] Failed to load image: {path}"
        )
        return None

    return {
        "image": img,
        "name": name,
        "path": path
    }


# =========================================================
# SAVE POINT CLOUD
# =========================================================

def save_point_cloud(points, colors, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )

    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64)
    )

    success = o3d.io.write_point_cloud(
        output_path,
        pcd
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {output_path}"
        )

    return output_path


# =========================================================
# ANALYTICS
# =========================================================

def build_analytics(
    sparse_points,
    dense_points,
    camera_positions,
    processing_time,
    total_input_images,
    registered_images
):

    total_points = (
        len(sparse_points)
        + len(dense_points)
    )

    registration_ratio = (
        len(registered_images)
        / max(total_input_images, 1)
    )

    # Temporary audit metric.
    # We will replace this with proper quality metrics
    # during the analytics stage.
    health_score = round(
        registration_ratio * 100.0,
        2
    )

    return {

        "total_sparse_points":
            int(len(sparse_points)),

        "total_dense_points":
            int(len(dense_points)),

        "total_points":
            int(total_points),

        "total_cameras":
            int(len(camera_positions)),

        "total_input_images":
            int(total_input_images),

        "registered_images":
            int(len(registered_images)),

        "registration_ratio":
            round(registration_ratio * 100.0, 2),

        "processing_time":
            round(processing_time, 2),

        # Temporary placeholders.
        # Stage 5 will replace these.
        "successful_pairs":
            0,

        "failed_pairs":
            0,

        "avg_confidence":
            0.0,

        "avg_inliers":
            0,

        "avg_points":
            int(total_points),

        "health_score":
            health_score
    }


# =========================================================
# MAIN
# =========================================================

def run_reconstruction(
    uploaded_files,
    progress_callback=print
):

    start_time = time.time()

    progress_callback(
        "Loading Images."
    )

    # =====================================================
    # LOAD INPUT IMAGES
    # =====================================================

    processed_data = []

    for idx, image_info in enumerate(uploaded_files):

        image_data = process_single_image(
            image_info
        )

        if image_data is None:
            continue

        processed_data.append(
            image_data
        )

        progress_callback(
            f"Image {idx + 1} processed"
        )

    if len(processed_data) < 2:

        raise RuntimeError(
            "Need at least 2 valid images"
        )

    total_input_images = len(
        processed_data
    )

    # =====================================================
    # LOAD COLMAP MODEL
    # =====================================================

    progress_callback(
        "Loading COLMAP Model."
    )

    (
        cameras,
        registered_images,
        _camera_positions,
        sparse_points,
        sparse_colors
    ) = load_colmap_model(
        "colmap_data"
    )

    progress_callback(
        f"COLMAP Registered Images: "
        f"{len(registered_images)}"
    )

    progress_callback(
        f"COLMAP Sparse Points: "
        f"{len(sparse_points)}"
    )

    # =====================================================
    # ALIGN INPUT IMAGES WITH COLMAP POSES
    # =====================================================

    aligned_images = []

    camera_positions = []

    global_rotations = []

    global_translations = []

    intrinsic_matrices = []

    aligned_names = []

    skipped_names = []

    for item in processed_data:

        image_name = item["name"]

        pose = registered_images.get(
            image_name
        )

        if pose is None:

            skipped_names.append(
                image_name
            )

            continue

        aligned_images.append(
            item["image"]
        )

        aligned_names.append(
            image_name
        )

        camera_positions.append(
            pose["camera_center"]
        )

        global_rotations.append(
            pose["R"]
        )

        global_translations.append(
            pose["t"]
        )

        intrinsic_matrices.append(
            pose["K"]
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(aligned_images) < 2:

        raise RuntimeError(
            "Fewer than 2 input images were registered "
            "by COLMAP."
        )

    camera_positions = np.asarray(
        camera_positions,
        dtype=np.float64
    )

    global_rotations = np.asarray(
        global_rotations,
        dtype=np.float64
    )

    global_translations = np.asarray(
        global_translations,
        dtype=np.float64
    )

    intrinsic_matrices = np.asarray(
        intrinsic_matrices,
        dtype=np.float64
    )

    progress_callback(
        f"Aligned Images: "
        f"{len(aligned_images)}/{total_input_images}"
    )

    if skipped_names:

        progress_callback(
            f"Skipped Unregistered Images: "
            f"{len(skipped_names)}"
        )

    # =====================================================
    # DEBUG CAMERA INFORMATION
    # =====================================================

    print("\n===== COLMAP ALIGNMENT DEBUG =====")

    print(
        "Input images:",
        total_input_images
    )

    print(
        "Registered images:",
        len(registered_images)
    )

    print(
        "Aligned images:",
        len(aligned_images)
    )

    print(
        "Camera positions:",
        camera_positions.shape
    )

    print(
        "Rotations:",
        global_rotations.shape
    )

    print(
        "Translations:",
        global_translations.shape
    )

    print(
        "Intrinsics:",
        intrinsic_matrices.shape
    )

    if aligned_names:

        print(
            "First aligned image:",
            aligned_names[0]
        )

        print(
            "First intrinsic matrix:\n",
            intrinsic_matrices[0]
        )

    print(
        "==================================\n"
    )

    # =====================================================
    # DENSE RECONSTRUCTION
    # =====================================================

    progress_callback(
        "Running Dense Reconstruction."
    )

    dense_reconstructor = (
        DenseReconstructor()
    )

    # -----------------------------------------------------
    # TEMPORARY STAGE-1 COMPATIBILITY
    # -----------------------------------------------------
    #
    # Current DenseReconstructor accepts one intrinsic
    # matrix for the full sequence.
    #
    # For the current owl dataset COLMAP uses the same
    # camera model for all images, so use the first
    # calibrated K temporarily.
    #
    # Stage 2 will change DenseReconstructor so each image
    # can use its own K and proper stereo geometry.
    # -----------------------------------------------------

    intrinsic_matrix = (
        intrinsic_matrices[0]
    )

    dense_result = (
        dense_reconstructor
        .generate_dense_cloud(

            aligned_images,

            camera_positions,

            intrinsic_matrix,

            global_rotations=
                global_rotations,

            global_translations=
                global_translations
        )
    )

    # =====================================================
    # HANDLE DENSE FAILURE
    # =====================================================

    if (
        dense_result is None
        or
        len(dense_result.get("points", [])) == 0
    ):

        progress_callback(
            "[WARNING] Dense reconstruction "
            "generated no points."
        )

        dense_points = np.empty(
            (0, 3),
            dtype=np.float64
        )

        dense_colors = np.empty(
            (0, 3),
            dtype=np.float64
        )

    else:

        dense_points = np.asarray(
            dense_result["points"],
            dtype=np.float64
        )

        dense_colors = np.asarray(
            dense_result["colors"],
            dtype=np.float64
        )

    progress_callback(
        f"Dense Points: "
        f"{len(dense_points):,}"
    )

    # =====================================================
    # COMBINE SPARSE + DENSE
    # =====================================================

    point_sets = []
    color_sets = []

    if len(sparse_points) > 0:

        point_sets.append(
            sparse_points
        )

        color_sets.append(
            sparse_colors
        )

    if len(dense_points) > 0:

        point_sets.append(
            dense_points
        )

        color_sets.append(
            dense_colors
        )

    if not point_sets:

        raise RuntimeError(
            "No reconstruction points available."
        )

    all_points = np.vstack(
        point_sets
    )

    all_colors = np.vstack(
        color_sets
    )

    # =====================================================
    # REMOVE INVALID VALUES
    # =====================================================

    finite_mask = (
        np.all(
            np.isfinite(all_points),
            axis=1
        )
        &
        np.all(
            np.isfinite(all_colors),
            axis=1
        )
    )

    all_points = all_points[
        finite_mask
    ]

    all_colors = all_colors[
        finite_mask
    ]

    # =====================================================
    # GEOMETRIC POINT FILTER (AUDIT ONLY)
    # =====================================================

    progress_callback(
        "Filtering Geometric Outliers."
    )

    distances = np.linalg.norm(
        all_points,
        axis=1
    )

    print(
        f"[FILTER] Distance Range: "
        f"{distances.min():.4f} -> "
        f"{distances.max():.4f}"
    )

    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:

        print(
            f"[FILTER] {p}% = "
            f"{np.percentile(distances, p):.4f}"
        )

    if len(all_points) == 0:

        raise RuntimeError(
            "All reconstruction points were invalid."
        )

    # Keep colors valid for Open3D.
    all_colors = np.clip(
        all_colors,
        0.0,
        1.0
    )

    # =====================================================
    # CLEAN POINT CLOUD
    # =====================================================

    progress_callback(
        "Cleaning Point Cloud."
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            all_points
        )
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(
            all_colors
        )
    )

 

    # Keep this conservative during Stage 1.
    # We will audit filtering properly in Stage 3.
    if len(all_points) >= 50:

        try:

            pcd, _ = (
                pcd.remove_statistical_outlier(
                    nb_neighbors=20,
                    std_ratio=2.0
                )
            )

        except Exception as e:

            print(
                "[WARNING] Point-cloud cleaning "
                f"failed: {e}"
            )

    final_points = np.asarray(
        pcd.points
    )

    final_colors = np.asarray(
        pcd.colors
    )

    if len(final_points) == 0:

        raise RuntimeError(
            "Point-cloud cleaning removed all points."
        )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        "output",
        exist_ok=True
    )

    # =====================================================
    # SAVE POINT CLOUD
    # =====================================================

    ply_path = save_point_cloud(
        final_points,
        final_colors,
        os.path.join(
            "output",
            "reconstruction.ply"
        )
    )

    progress_callback(
        f"Point Cloud Saved: {ply_path}"
    )

    # =====================================================
    # MESH
    # =====================================================

    progress_callback(
        "Generating Mesh."
    )

    try:

        mesh_result = generate_mesh(
            final_points,
            final_colors,
            output_dir="output"
        )

        if mesh_result is not None:

            progress_callback(
                f"Mesh Vertices: "
                f"{mesh_result['vertices']:,}"
            )

            progress_callback(
                f"Mesh Triangles: "
                f"{mesh_result['triangles']:,}"
            )

    except Exception as e:

        # Mesh failure should not destroy a valid
        # point-cloud reconstruction.
        print(
            f"[WARNING] Mesh generation failed: {e}"
        )

        mesh_result = None

    # =====================================================
    # PAIR LOGS
    # =====================================================

    # The old file fabricated pair confidence/inlier
    # statistics. During Stage 1 we intentionally stop
    # claiming those values.
    pair_logs = []

    pair_log_path = os.path.join(
        "output",
        "pair_logs.csv"
    )

    with open(
        pair_log_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "pair",
            "status",
            "confidence",
            "inliers",
            "points"
        ])

    # =====================================================
    # ANALYTICS
    # =====================================================

    processing_time = (
        time.time() - start_time
    )

    analytics = build_analytics(
        sparse_points,
        dense_points,
        camera_positions,
        processing_time,
        total_input_images,
        registered_images
    )

    # Existing runner expects this field.
    # Stage 5 will replace it with real pair metrics.
    match_visuals = []

    progress_callback(
        f"Final Points: "
        f"{len(final_points):,}"
    )

    progress_callback(
        f"Registered Cameras: "
        f"{len(camera_positions)}"
    )

    progress_callback(
        f"Processing Time: "
        f"{processing_time:.2f} sec"
    )

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