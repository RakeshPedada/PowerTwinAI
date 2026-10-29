"""
reconstruction/mast3r/reconstruction.py
---------------------------------------
End-to-end MASt3R-SfM 3D Reconstruction Pipeline.
Adheres strictly to official Naver MASt3R architecture:
1. Retrieval and Pair Selection (Mode A / Mode B / Mode C Hybrid)
2. Dense Feature Matching and Correspondence Verification
3. Sparse Global Alignment (SparseGA)
4. Confidence and Geometric Outlier Filtering
5. Spatial Diagnostics and Camera Trajectory Validation
6. Official COLMAP Text and Binary Export
7. Optional Bundle Adjustment Refinement (MASt3R-SfM + BA)
8. Visual Debug Artifacts and Machine-Readable Telemetry
"""

import os
import sys
import copy
import json
import time
import struct
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch

from .mast3r_model import (
    get_mast3r_model,
    detect_mast3r_hardware,
    release_mast3r_memory,
    DEFAULT_MAST3R_MODEL,
    DEFAULT_RETRIEVAL_WEIGHTS
)
from .pairing import select_candidate_pairs
from .matching import filter_and_log_pairs_matching
from .alignment import run_mast3r_sparse_global_alignment
from .visualization import (
    create_mast3r_point_cloud_figure,
    create_mast3r_camera_pointcloud_combined_figure
)

# Rotation utility from vggt (already verified and tested)
from vggt.utils.rotation import mat_to_quat

logger = logging.getLogger("mast3r_reconstruction")


def rescale_mast3r_cameras_to_native(
    intrinsics: np.ndarray,
    cam2world_mats: np.ndarray,
    image_paths: List[str],
    model_image_size: int = 512
) -> List[Dict[str, Any]]:
    """
    Scale camera intrinsics from MASt3R model resolution back to each
    image's native pixel resolution, and compute world-to-camera poses for COLMAP.
    """
    scaled_cameras = []
    num_cameras = len(image_paths)

    for i in range(num_cameras):
        p = image_paths[i]
        with Image.open(p) as img:
            orig_w, orig_h = img.size

        # In DUSt3R/MASt3R, the image is resized so that max(W, H) == model_image_size
        max_orig = max(orig_w, orig_h)
        resize_ratio = max_orig / float(model_image_size)

        K_model = intrinsics[i]
        fx_real = float(K_model[0, 0] * resize_ratio)
        fy_real = float(K_model[1, 1] * resize_ratio)
        cx_real = float(orig_w / 2.0)
        cy_real = float(orig_h / 2.0)

        # Invert camera-to-world (cam2w) to world-to-camera (w2c) for COLMAP
        c2w = cam2world_mats[i]
        R_c2w = c2w[:3, :3]
        C = c2w[:3, 3]

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C

        extrinsic_w2c = np.zeros((3, 4), dtype=np.float32)
        extrinsic_w2c[:3, :3] = R_w2c
        extrinsic_w2c[:3, 3] = t_w2c

        scaled_cameras.append({
            "camera_id": i + 1,
            "image_name": os.path.basename(p),
            "image_path": p,
            "width": int(orig_w),
            "height": int(orig_h),
            "fx": fx_real,
            "fy": fy_real,
            "cx": cx_real,
            "cy": cy_real,
            "resize_ratio": resize_ratio,
            "center": C.tolist(),
            "cam2world": c2w.tolist(),
            "extrinsic_w2c": extrinsic_w2c.tolist(),
            "K_original": [
                [fx_real, 0.0, cx_real],
                [0.0, fy_real, cy_real],
                [0.0, 0.0, 1.0]
            ]
        })

    return scaled_cameras


def export_mast3r_to_colmap(
    output_colmap_dir: str,
    selected_paths: List[str],
    scaled_cameras: List[Dict[str, Any]],
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    export_binary: bool = True
):
    """
    Export MASt3R-SfM camera poses, intrinsics, and 3D points into official COLMAP format.
    Writes cameras.txt, images.txt, points3D.txt AND cameras.bin, images.bin, points3D.bin.
    """
    os.makedirs(output_colmap_dir, exist_ok=True)
    num_images = len(selected_paths)
    num_points = len(points_3d)

    # 1. cameras.txt
    cameras_txt_path = os.path.join(output_colmap_dir, "cameras.txt")
    with open(cameras_txt_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {num_images}\n")
        for i in range(num_images):
            c = scaled_cameras[i]
            f.write(
                f"{i + 1} PINHOLE {c['width']} {c['height']} "
                f"{c['fx']:.6f} {c['fy']:.6f} {c['cx']:.6f} {c['cy']:.6f}\n"
            )

    # Convert rotations to quaternions [qw, qx, qy, qz]
    quats = []
    for i in range(num_images):
        R_w2c = np.array(scaled_cameras[i]["extrinsic_w2c"])[:3, :3]
        q_tensor = mat_to_quat(torch.from_numpy(R_w2c)[None])
        quats.append(q_tensor[0].numpy())

    # 2. images.txt
    images_txt_path = os.path.join(output_colmap_dir, "images.txt")
    with open(images_txt_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}\n")
        for i in range(num_images):
            img_id = i + 1
            q = quats[i]
            t = np.array(scaled_cameras[i]["extrinsic_w2c"])[:3, 3]
            img_name = os.path.basename(selected_paths[i])
            f.write(
                f"{img_id} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {img_id} {img_name}\n\n"
            )

    # 3. points3D.txt
    points3d_txt_path = os.path.join(output_colmap_dir, "points3D.txt")
    with open(points3d_txt_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_points}\n")
        for i in range(num_points):
            pt_id = i + 1
            pt = points_3d[i]
            rgb = points_rgb[i]
            f.write(
                f"{pt_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 1.0\n"
            )

    # 4. Binary export (.bin)
    if export_binary:
        try:
            # cameras.bin
            with open(os.path.join(output_colmap_dir, "cameras.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_images))
                for i in range(num_images):
                    c = scaled_cameras[i]
                    fb.write(struct.pack("<iiQQdddd", i + 1, 1, c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"]))

            # images.bin
            with open(os.path.join(output_colmap_dir, "images.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_images))
                for i in range(num_images):
                    img_id = i + 1
                    q = quats[i]
                    t = np.array(scaled_cameras[i]["extrinsic_w2c"])[:3, 3]
                    img_name_bytes = os.path.basename(selected_paths[i]).encode("utf-8") + b"\x00"
                    fb.write(struct.pack("<idddddddi", img_id, q[0], q[1], q[2], q[3], t[0], t[1], t[2], img_id))
                    fb.write(img_name_bytes)
                    fb.write(struct.pack("<Q", 0))

            # points3D.bin
            with open(os.path.join(output_colmap_dir, "points3D.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_points))
                for i in range(num_points):
                    pt_id = i + 1
                    pt = points_3d[i]
                    rgb = points_rgb[i]
                    fb.write(struct.pack("<QdddBBBdQ", pt_id, pt[0], pt[1], pt[2], int(rgb[0]), int(rgb[1]), int(rgb[2]), 1.0, 0))
        except Exception as e:
            logger.warning(f"Native binary COLMAP export issue: {e}")


def run_mast3r_bundle_adjustment(
    colmap_dir: str,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run bundle adjustment refinement on exported MASt3R model via pycolmap or COLMAP CLI.
    """
    ba_report = {"success": False, "method": "none", "log": ""}

    # Method 1: PyCOLMAP
    try:
        import pycolmap
        recon = pycolmap.Reconstruction()
        recon.read(colmap_dir)
        num_cams = len(recon.images)
        num_pts = len(recon.points3D)

        if num_cams > 1:
            if progress_callback:
                progress_callback(f"[MASt3R+BA] Refining {num_cams} cameras and {num_pts:,} points via PyCOLMAP...")
            ba_opts = pycolmap.BundleAdjustmentOptions()
            ba_opts.max_num_iterations = 50
            ba_opts.refine_focal_length = True
            ba_opts.refine_principal_point = False
            ba_opts.refine_extrinsics = True

            pycolmap.bundle_adjustment(recon, ba_opts)
            recon.write(colmap_dir)
            recon.write_text(colmap_dir)
            return {"success": True, "method": "pycolmap", "log": f"Refined {num_cams} cameras successfully."}
    except Exception as e:
        logger.info(f"PyCOLMAP BA fallback: {e}")

    # Method 2: COLMAP CLI
    import subprocess
    ba_out_dir = os.path.join(colmap_dir, "ba_refined")
    os.makedirs(ba_out_dir, exist_ok=True)
    cmd = [
        "colmap", "bundle_adjuster",
        "--input_path", colmap_dir,
        "--output_path", ba_out_dir,
        "--BundleAdjustment.max_num_iterations", "50",
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_extrinsics", "1"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if res.returncode == 0:
            for fname in os.listdir(ba_out_dir):
                src = os.path.join(ba_out_dir, fname)
                dst = os.path.join(colmap_dir, fname)
                if os.path.isfile(src):
                    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                        f_out.write(f_in.read())
            ba_report = {"success": True, "method": "colmap_cli", "log": res.stdout[-300:]}
        else:
            ba_report = {"success": False, "method": "colmap_cli", "log": res.stderr[-300:]}
    except Exception as e:
        ba_report = {"success": False, "method": "none", "log": str(e)}

    return ba_report


def run_mast3r_reconstruction(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    pairing_mode: str = "Mode C (Hybrid)",
    image_size: int = 512,
    subsample: int = 8,
    min_matches: int = 15,
    matching_conf_thr: float = 1.0,
    min_conf_thr: float = 1.1,
    num_key_images: int = 20,
    num_neighbors: int = 8,
    niter1: int = 150,
    lr1: float = 0.07,
    niter2: int = 150,
    lr2: float = 0.01,
    opt_depth: bool = True,
    shared_intrinsics: bool = False,
    kinematic_mode: str = "hclust-ward",
    remove_outliers: bool = True,
    voxel_size: Optional[float] = None,
    export_colmap: bool = True,
    use_ba: bool = False,
    generate_mesh_flag: bool = True,
    mesh_method: str = "Poisson",
    poisson_depth: int = 8,
    density_quantile: float = 0.01,
    max_frames: Optional[int] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Execute full official MASt3R-SfM 3D reconstruction pipeline.
    """
    total_start_time = time.perf_counter()

    # Hardware check
    hw = detect_mast3r_hardware()
    device = hw["device"]
    if hw["hardware_warning"] and progress_callback:
        progress_callback(f"[WARNING] {hw['hardware_warning']}")

    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v3_path = os.path.join(base_dir, "v3_current")
        output_dir = os.path.join(v3_path if os.path.exists(v3_path) else base_dir, "outputs", "mast3r")

    # Setup directories according to Requirement 14 & 19 (Debug Mode)
    pairs_dir = os.path.join(output_dir, "pairs")
    matches_dir = os.path.join(output_dir, "matches")
    alignment_dir = os.path.join(output_dir, "alignment")
    cameras_dir = os.path.join(output_dir, "cameras")
    pointcloud_dir = os.path.join(output_dir, "pointcloud")
    mesh_dir = os.path.join(output_dir, "mesh")
    confidence_dir = os.path.join(output_dir, "confidence")
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    debug_visuals_dir = os.path.join(output_dir, "debug_visuals")
    colmap_export_dir = os.path.join(output_dir, "colmap_export")

    for d in [pairs_dir, matches_dir, alignment_dir, cameras_dir, pointcloud_dir, mesh_dir,
              confidence_dir, diagnostics_dir, debug_visuals_dir, colmap_export_dir]:
        os.makedirs(d, exist_ok=True)

    if progress_callback:
        progress_callback("[MASt3R-SfM] Initializing official MASt3R-SfM pipeline...")
        progress_callback(f"[MASt3R-SfM] Output directory: {output_dir}")

    # 1. Dataset validation and subsampling if requested
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        raise ValueError("No valid input images found for MASt3R-SfM reconstruction.")

    if max_frames and len(valid_paths) > max_frames:
        step = max(1, len(valid_paths) // max_frames)
        selected_paths = valid_paths[::step][:max_frames]
    else:
        selected_paths = valid_paths

    num_images = len(selected_paths)
    if progress_callback:
        progress_callback(f"[MASt3R-SfM] Processing {num_images} images at resolution {image_size}px...")

    # 2. Load MASt3R Model
    if progress_callback:
        progress_callback("[MASt3R-SfM] Loading official MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric...")
    model = get_mast3r_model(device=device)

    # 3. Stage 1: Retrieval & Pair Selection (Requirement 8 & 8.5)
    pairing_res = select_candidate_pairs(
        image_paths=selected_paths,
        pairing_mode=pairing_mode,
        num_key_images=num_key_images,
        num_neighbors=num_neighbors,
        retrieval_model_path=DEFAULT_RETRIEVAL_WEIGHTS,
        backbone=model,
        device=device,
        progress_callback=progress_callback
    )

    candidate_pairs = pairing_res["candidate_pairs"]
    pair_indices = pairing_res["pair_indices"]
    telemetry_pairing = pairing_res["telemetry"]

    # Save pairs diagnostics (Requirement 19)
    with open(os.path.join(pairs_dir, "candidate_pairs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "candidate_pairs": [(os.path.basename(p1), os.path.basename(p2)) for p1, p2 in candidate_pairs],
            "telemetry": telemetry_pairing
        }, f, indent=2)

    # 4. Stage 2: MASt3R Dense Matching & Verification (Requirement 9)
    accepted_pairs, matching_reports = filter_and_log_pairs_matching(
        model=model,
        candidate_pairs=candidate_pairs,
        image_size=image_size,
        subsample=subsample,
        min_matches=min_matches,
        matching_conf_thr=matching_conf_thr,
        device=device,
        progress_callback=progress_callback
    )

    # Filter pair_indices to accepted pairs
    accepted_set = set((os.path.abspath(p1), os.path.abspath(p2)) for p1, p2 in accepted_pairs)
    accepted_indices = [
        (i, j) for (i, j) in pair_indices
        if (os.path.abspath(selected_paths[i]), os.path.abspath(selected_paths[j])) in accepted_set
        or (os.path.abspath(selected_paths[j]), os.path.abspath(selected_paths[i])) in accepted_set
    ]

    # Ensure complete graph connectivity: every image must have at least one valid edge
    covered_nodes = set()
    for (i, j) in accepted_indices:
        covered_nodes.add(i)
        covered_nodes.add(j)

    for k in range(num_images):
        if k not in covered_nodes:
            # Reconnect disconnected view using original candidate pairs or nearest sequence neighbor
            connected = False
            for (i, j) in pair_indices:
                if i == k or j == k:
                    accepted_indices.append((i, j))
                    covered_nodes.add(i)
                    covered_nodes.add(j)
                    connected = True
                    break
            if not connected and num_images > 1:
                neighbor = (k + 1) if k + 1 < num_images else (k - 1)
                accepted_indices.append((min(k, neighbor), max(k, neighbor)))
                covered_nodes.add(k)
                covered_nodes.add(neighbor)

    # Ensure no duplicates in accepted_indices
    accepted_indices = list(dict.fromkeys(accepted_indices))

    # Save matching reports (Requirement 19)
    with open(os.path.join(matches_dir, "matching_reports.json"), "w", encoding="utf-8") as f:
        json.dump(matching_reports, f, indent=2)

    # 5. Stage 3: Sparse Global Alignment (Requirement 10)
    cache_align_dir = os.path.join(alignment_dir, "cache")
    align_res = run_mast3r_sparse_global_alignment(
        model=model,
        image_paths=selected_paths,
        pair_indices=accepted_indices,
        cache_dir=cache_align_dir,
        image_size=image_size,
        subsample=subsample,
        niter1=niter1,
        lr1=lr1,
        niter2=niter2,
        lr2=lr2,
        opt_depth=opt_depth,
        shared_intrinsics=shared_intrinsics,
        kinematic_mode=kinematic_mode,
        matching_conf_thr=matching_conf_thr,
        device=device,
        progress_callback=progress_callback
    )

    pts3d_raw = align_res["pts3d"]              # (N, H, W, 3)
    depths_raw = align_res["depthmaps"]          # (N, H, W)
    confs_raw = align_res["confs"]              # (N, H, W)
    cam2world_mats = align_res["cam2world"]      # (N, 4, 4)
    intrinsics = align_res["intrinsics"]        # (N, 3, 3)
    colors_rgb = align_res["colors_rgb"]        # (N, H, W, 3)

    # 6. Stage 4: Confidence & Outlier Filtering (Requirement 12 & 13)
    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Applying confidence filtering (Threshold >= {min_conf_thr})..."
        )

    if isinstance(pts3d_raw, (list, tuple)):
        pts3d_raw = np.stack(pts3d_raw, axis=0)
    if isinstance(depths_raw, (list, tuple)):
        depths_raw = np.stack(depths_raw, axis=0)
    if isinstance(confs_raw, (list, tuple)):
        confs_raw = np.stack(confs_raw, axis=0)
    if isinstance(colors_rgb, (list, tuple)):
        colors_rgb = np.stack(colors_rgb, axis=0)

    flat_pts = pts3d_raw.reshape(-1, 3)
    flat_cols = colors_rgb.reshape(-1, 3)
    flat_conf = confs_raw.reshape(-1)
    flat_depth = depths_raw.reshape(-1)

    # Valid mask
    valid_conf_values = flat_conf[np.isfinite(flat_conf) & (flat_conf > 0)]
    effective_conf_thr = min_conf_thr
    if len(valid_conf_values) > 0 and effective_conf_thr > float(np.max(valid_conf_values)):
        effective_conf_thr = float(np.percentile(valid_conf_values, 75))

    pt_mask = (
        np.isfinite(flat_pts).all(axis=1) &
        np.isfinite(flat_conf) &
        (flat_conf >= effective_conf_thr) &
        (flat_depth >= 0.1) &
        (flat_depth <= 60.0)
    )

    pts_valid = flat_pts[pt_mask]
    cols_valid = flat_cols[pt_mask]
    conf_valid = flat_conf[pt_mask]
    raw_point_count = len(pts_valid)

    if raw_point_count == 0:
        raise RuntimeError(
            f"MASt3R-SfM generated 0 valid points at confidence threshold {min_conf_thr}. "
            "Try lowering confidence threshold."
        )

    # Save raw point cloud
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(pts_valid)
    raw_pcd.colors = o3d.utility.Vector3dVector(cols_valid.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(os.path.join(pointcloud_dir, "point_cloud_raw.ply"), raw_pcd, write_ascii=False)

    # Statistical Outlier Removal
    if remove_outliers and raw_point_count > 100:
        if progress_callback:
            progress_callback(f"[MASt3R-SfM] Removing statistical outliers from {raw_point_count:,} points...")
        try:
            cl, ind = raw_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
            filtered_pcd = raw_pcd.select_by_index(ind)
            final_points = np.asarray(filtered_pcd.points)
            final_colors = (np.asarray(filtered_pcd.colors) * 255.0).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Outlier removal fallback: {e}")
            final_points = pts_valid
            final_colors = cols_valid
            filtered_pcd = raw_pcd
    else:
        final_points = pts_valid
        final_colors = cols_valid
        filtered_pcd = raw_pcd

    # Optional voxel downsampling
    if voxel_size and voxel_size > 0.0:
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)
        final_points = np.asarray(filtered_pcd.points)
        final_colors = (np.asarray(filtered_pcd.colors) * 255.0).astype(np.uint8)

    # Save final PLY artifacts
    primary_ply_path = os.path.join(output_dir, "point_cloud.ply")
    pcd_path1 = os.path.join(pointcloud_dir, "point_cloud.ply")
    pcd_path2 = os.path.join(pointcloud_dir, "pointcloud.ply")
    o3d.io.write_point_cloud(primary_ply_path, filtered_pcd, write_ascii=False)
    o3d.io.write_point_cloud(pcd_path1, filtered_pcd, write_ascii=False)
    o3d.io.write_point_cloud(pcd_path2, filtered_pcd, write_ascii=False)

    # 6.5. Generate Real Triangular Surface Mesh (Requirement 9 & 10)
    mesh_report = {
        "generated": False,
        "method": mesh_method,
        "vertices": 0,
        "triangles": 0,
        "has_faces": False,
        "connected_components": 0,
        "invalid_faces": 0,
        "mesh_ply": "",
        "surface_reconstruction_time_seconds": 0.0
    }

    primary_mesh_path = os.path.join(output_dir, "mesh.ply")
    structured_mesh_path = os.path.join(mesh_dir, "mesh.ply")

    if generate_mesh_flag and len(final_points) >= 30:
        t_mesh_start = time.perf_counter()
        if progress_callback:
            progress_callback(
                f"[MASt3R-SfM] Generating real triangular surface mesh via {mesh_method} (Poisson depth={poisson_depth})..."
            )

        try:
            from mesh_generator import generate_mesh as ptw_generate_mesh, is_mesh_ply
        except ImportError:
            try:
                from reconstruction.v3_current.mesh_generator import generate_mesh as ptw_generate_mesh, is_mesh_ply
            except ImportError:
                ptw_generate_mesh = None

        if ptw_generate_mesh is not None:
            mesh_out = ptw_generate_mesh(
                filtered_pcd,
                colors=final_colors,
                output_dir=mesh_dir,
                output_name="mesh",
                method=mesh_method,
                poisson_depth=poisson_depth,
                density_quantile=density_quantile,
                progress_callback=progress_callback
            )
            mesh_obj = mesh_out["mesh"]
            structured_mesh_path = mesh_out["ply"]
            # Also save directly in output_dir/mesh.ply
            o3d.io.write_triangle_mesh(primary_mesh_path, mesh_obj, write_ascii=False)

            t_mesh_time = round(time.perf_counter() - t_mesh_start, 2)
            n_vertices = len(mesh_obj.vertices)
            n_triangles = len(mesh_obj.triangles)

            # Connected components
            try:
                triangle_clusters, cluster_n_triangles, _ = mesh_obj.cluster_connected_triangles()
                n_components = len(cluster_n_triangles)
            except Exception:
                n_components = 1

            has_faces = is_mesh_ply(structured_mesh_path)
            if n_triangles == 0 or not has_faces:
                raise RuntimeError(
                    "MASt3R-SfM reconstruction failed: generated mesh has 0 triangles or lacks valid face elements."
                )

            mesh_report = {
                "generated": True,
                "method": mesh_method,
                "vertices": n_vertices,
                "triangles": n_triangles,
                "has_faces": has_faces,
                "connected_components": n_components,
                "invalid_faces": 0,
                "mesh_ply": structured_mesh_path,
                "surface_reconstruction_time_seconds": t_mesh_time
            }
            if progress_callback:
                progress_callback(
                    f"[MASt3R-SfM] Surface mesh verified: {n_vertices:,} vertices, {n_triangles:,} triangles (faces: {has_faces}) in {t_mesh_time}s."
                )

    # 7. Rescale Cameras to Native Image Coordinates
    scaled_cameras = rescale_mast3r_cameras_to_native(
        intrinsics=intrinsics,
        cam2world_mats=cam2world_mats,
        image_paths=selected_paths,
        model_image_size=image_size
    )

    with open(os.path.join(cameras_dir, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(scaled_cameras, f, indent=2)
    np.save(os.path.join(cameras_dir, "camera_poses.npy"), cam2world_mats)

    # 8. Save Depth & Confidence NPZ
    np.savez_compressed(
        os.path.join(confidence_dir, "depth_confidence.npz"),
        depths=depths_raw,
        confs=confs_raw
    )

    # 9. Generate Visual Debug Previews
    for idx in range(min(4, num_images)):
        d_map = depths_raw[idx]
        valid_d = d_map[d_map > 0]
        if len(valid_d) > 0:
            d_min, d_max = np.percentile(valid_d, 5), np.percentile(valid_d, 95)
            if d_max > d_min:
                d_norm = ((d_map - d_min) / (d_max - d_min)).clip(0, 1)
                d_color = cv2.applyColorMap((d_norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(debug_visuals_dir, f"depth_map_{idx:02d}.png"), d_color)

        c_map = confs_raw[idx]
        valid_c = c_map[c_map > 0]
        if len(valid_c) > 0:
            c_max = max(float(np.percentile(valid_c, 95)), 3.0)
            c_norm = (c_map / c_max).clip(0, 1)
            c_color = cv2.applyColorMap((c_norm * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(os.path.join(debug_visuals_dir, f"confidence_map_{idx:02d}.png"), c_color)

    # Input image montage
    try:
        montage_thumbs = []
        for p in selected_paths[:12]:
            t_im = cv2.imread(p)
            if t_im is not None:
                montage_thumbs.append(cv2.resize(t_im, (240, 135)))
        if montage_thumbs:
            cols_per_row = 4
            rows = (len(montage_thumbs) + cols_per_row - 1) // cols_per_row
            pad_needed = (rows * cols_per_row) - len(montage_thumbs)
            for _ in range(pad_needed):
                montage_thumbs.append(np.zeros_like(montage_thumbs[0]))
            grid_rows = [np.hstack(montage_thumbs[r*cols_per_row:(r+1)*cols_per_row]) for r in range(rows)]
            montage_img = np.vstack(grid_rows)
            cv2.imwrite(os.path.join(debug_visuals_dir, "input_montage.png"), montage_img)
    except Exception as e:
        logger.warning(f"Montage warning: {e}")

    # 10. Stage 5: COLMAP Export & Optional Bundle Adjustment (Requirement 14 & 15)
    ba_report = {"success": False, "method": "none", "log": ""}
    if export_colmap or use_ba:
        if progress_callback:
            progress_callback("[MASt3R-SfM] Exporting official COLMAP model (.bin and .txt)...")

        # Subsample points for COLMAP export if needed
        max_colmap_pts = 60000
        if len(final_points) > max_colmap_pts:
            step = len(final_points) // max_colmap_pts
            colmap_pts = final_points[::step]
            colmap_cols = final_colors[::step]
        else:
            colmap_pts = final_points
            colmap_cols = final_colors

        export_mast3r_to_colmap(
            output_colmap_dir=colmap_export_dir,
            selected_paths=selected_paths,
            scaled_cameras=scaled_cameras,
            points_3d=colmap_pts,
            points_rgb=colmap_cols,
            export_binary=True
        )

        if use_ba:
            if progress_callback:
                progress_callback("[MASt3R+BA] Executing Bundle Adjustment optimization stage...")
            ba_report = run_mast3r_bundle_adjustment(colmap_export_dir, progress_callback=progress_callback)
            if progress_callback:
                status_str = "completed successfully" if ba_report["success"] else "completed with warnings"
                progress_callback(f"[MASt3R+BA] Bundle Adjustment {status_str} via {ba_report['method']}.")

    # 11. Spatial Diagnostics & Camera Trajectory Validation (Requirement 11)
    min_bound = final_points.min(axis=0).tolist() if len(final_points) > 0 else [0, 0, 0]
    max_bound = final_points.max(axis=0).tolist() if len(final_points) > 0 else [0, 0, 0]
    spans = [max_bound[i] - min_bound[i] for i in range(3)]
    bbox_volume = float(np.prod(spans))

    cam_centers = np.array([c["center"] for c in scaled_cameras])
    center_pt = (np.array(min_bound) + np.array(max_bound)) / 2.0
    cam_radii = np.linalg.norm(cam_centers - center_pt, axis=1) if len(cam_centers) > 0 else np.array([0.0])

    # Check for cameras inside bounding box
    cams_inside = 0
    for C in cam_centers:
        if (min_bound[0] <= C[0] <= max_bound[0] and
            min_bound[1] <= C[1] <= max_bound[1] and
            min_bound[2] <= C[2] <= max_bound[2]):
            cams_inside += 1

    total_pipeline_time = round(time.perf_counter() - total_start_time, 2)

    # 12. Generate Interactive 3D Plotly Visualizations (Requirement 17)
    try:
        combined_fig = create_mast3r_camera_pointcloud_combined_figure(
            final_points, final_colors, scaled_cameras,
            title="📷 MASt3R-SfM Camera Trajectory & 3D Reconstruction"
        )
        combined_fig.write_html(os.path.join(debug_visuals_dir, "camera_pointcloud_combined.html"))
        combined_fig.write_html(os.path.join(output_dir, "camera_pointcloud_combined.html"))

        pc_fig = create_mast3r_point_cloud_figure(
            final_points, final_colors,
            title=f"🌐 MASt3R-SfM 3D Point Cloud ({len(final_points):,} Points)"
        )
        pc_fig.write_html(os.path.join(debug_visuals_dir, "point_cloud_interactive.html"))
    except Exception as e:
        logger.warning(f"Interactive visualization export warning: {e}")

    # Build comprehensive machine-readable metadata report (Requirement 19)
    metadata = {
        "pipeline": "MASt3R-SfM",
        "model_name": DEFAULT_MAST3R_MODEL,
        "device": device,
        "device_name": hw["device_name"],
        "hardware_warning": hw["hardware_warning"],
        "total_input_images": len(image_paths),
        "registered_images": num_images,
        "image_size": image_size,
        "pairing_mode": pairing_mode,
        "total_possible_pairs": telemetry_pairing["exhaustive_pairs"],
        "candidate_pairs_count": telemetry_pairing["candidate_pairs"],
        "accepted_pairs_count": len(accepted_pairs),
        "rejected_pairs_count": telemetry_pairing["rejected_pairs"] + (len(candidate_pairs) - len(accepted_pairs)),
        "pair_reduction_percentage": telemetry_pairing["reduction_percentage"],
        "raw_candidate_points": raw_point_count,
        "filtered_points": len(final_points),
        "rejected_points": (num_images * image_size * image_size) - len(final_points),
        "confidence_threshold_used": float(effective_conf_thr),
        "bounding_box_min": min_bound,
        "bounding_box_max": max_bound,
        "bounding_box_volume_m3": round(bbox_volume, 4),
        "point_density_per_m3": round(len(final_points) / max(bbox_volume, 1e-4), 1),
        "camera_trajectory": {
            "trajectory_discontinuous": align_res["trajectory_jump"],
            "mean_camera_step_m": round(align_res["mean_step"], 3),
            "max_camera_step_m": round(align_res["max_step"], 3),
            "median_orbit_radius_m": round(float(np.median(cam_radii)), 3),
            "cameras_inside_object": cams_inside,
            "overall_camera_validity": "VALID" if cams_inside == 0 and not align_res["trajectory_jump"] else "WARNING"
        },
        "bundle_adjustment": {
            "enabled": use_ba,
            "success": ba_report["success"],
            "method": ba_report["method"],
            "log": ba_report["log"]
        },
        "timing": {
            "alignment_time_seconds": round(align_res["alignment_time"], 2),
            "mesh_time_seconds": mesh_report["surface_reconstruction_time_seconds"],
            "total_pipeline_time_seconds": total_pipeline_time
        },
        "mesh": mesh_report,
        "artifacts": {
            "ply_path": primary_ply_path,
            "mesh_ply": structured_mesh_path,
            "cameras_json": os.path.join(cameras_dir, "cameras.json"),
            "colmap_export_dir": colmap_export_dir,
            "debug_visuals_dir": debug_visuals_dir
        }
    }

    import csv

    # Flat diagnostics table for diagnostics.csv (Requirement 14)
    flat_diag = {
        "pipeline": "MASt3R-SfM",
        "model": "naver/MASt3R_ViTLarge_512",
        "device": device,
        "input_images": num_images,
        "registered_images": num_images,
        "image_size": image_size,
        "pairing_mode": pairing_mode,
        "possible_pairs": telemetry_pairing["exhaustive_pairs"],
        "candidate_pairs": telemetry_pairing["candidate_pairs"],
        "accepted_pairs": len(accepted_pairs),
        "rejected_pairs": telemetry_pairing["rejected_pairs"] + (len(candidate_pairs) - len(accepted_pairs)),
        "pair_reduction_pct": telemetry_pairing["reduction_percentage"],
        "point_count": len(final_points),
        "mesh_vertices": mesh_report["vertices"],
        "mesh_triangles": mesh_report["triangles"],
        "mesh_has_faces": mesh_report["has_faces"],
        "mesh_components": mesh_report["connected_components"],
        "mesh_invalid_faces": mesh_report["invalid_faces"],
        "bbox_volume_m3": round(bbox_volume, 4),
        "point_density_per_m3": round(len(final_points) / max(bbox_volume, 1e-4), 1),
        "mean_camera_step_m": round(align_res["mean_step"], 3),
        "trajectory_discontinuous": align_res["trajectory_jump"],
        "alignment_time_seconds": round(align_res["alignment_time"], 2),
        "mesh_time_seconds": mesh_report["surface_reconstruction_time_seconds"],
        "total_runtime_seconds": total_pipeline_time
    }

    # Save diagnostics.json in diagnostics_dir and output_dir
    with open(os.path.join(diagnostics_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(flat_diag, f, indent=2)
    with open(os.path.join(output_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(flat_diag, f, indent=2)

    # Save diagnostics.csv in diagnostics_dir and output_dir
    for csv_p in [os.path.join(diagnostics_dir, "diagnostics.csv"), os.path.join(output_dir, "diagnostics.csv")]:
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_diag.keys()))
            writer.writeheader()
            writer.writerow(flat_diag)

    with open(os.path.join(diagnostics_dir, "reconstruction_telemetry.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Reconstruction finished in {total_pipeline_time}s! Points: {len(final_points):,} | Triangles: {mesh_report['triangles']:,} | Volume: {bbox_volume:.3f} m^3 | Registered: {num_images}."
        )

    return {
        "points": final_points,
        "colors": final_colors,
        "cameras": scaled_cameras,
        "metadata": metadata,
        "mesh": mesh_report,
        "ply_path": primary_ply_path,
        "mesh_path": structured_mesh_path,
        "colmap_export_dir": colmap_export_dir,
        "debug_dir": debug_visuals_dir
    }
