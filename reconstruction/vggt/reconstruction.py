"""
reconstruction/vggt/reconstruction.py
-------------------------------------
End-to-end VGGT 3D reconstruction pipeline following the official
facebookresearch/VGGT reference architecture:
- Square aspect-ratio-preserving unprojection with padding exclusion
- Dual reconstruction paths (depth-map unprojection vs direct point head)
- Configurable absolute and percentile confidence filtering
- Robust outlier removal and geometric filtering
- Official COLMAP text and binary (.bin) export with scaled camera intrinsics
- Optional Bundle Adjustment refinement (VGGT + BA)
- Rich visual debug artifacts and comprehensive telemetry
"""

import os
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
import torch
from PIL import Image

logger = logging.getLogger(__name__)

from .vggt_model import get_vggt_model, detect_vggt_hardware
from .inference import run_vggt_inference
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from vggt.utils.rotation import quat_to_mat, mat_to_quat


def rescale_camera_parameters_to_original(
    intrinsics: np.ndarray,
    orig_coords: np.ndarray,
    vggt_resolution: int = 518
) -> List[Dict[str, Any]]:
    """
    Scale camera intrinsics from the 518x518 square model coordinate space
    back to each original image's native pixel resolution following official VGGT.
    """
    scaled_cameras = []
    num_cameras = len(intrinsics)

    for i in range(num_cameras):
        orig_w = float(orig_coords[i, 4])
        orig_h = float(orig_coords[i, 5])
        max_dim = max(orig_w, orig_h)
        resize_ratio = max_dim / float(vggt_resolution)

        K = intrinsics[i]
        fx_real = float(K[0, 0] * resize_ratio)
        fy_real = float(K[1, 1] * resize_ratio)
        cx_real = float(orig_w / 2.0)
        cy_real = float(orig_h / 2.0)

        scaled_cameras.append({
            "camera_id": i + 1,
            "width": int(orig_w),
            "height": int(orig_h),
            "fx": fx_real,
            "fy": fy_real,
            "cx": cx_real,
            "cy": cy_real,
            "resize_ratio": resize_ratio,
            "K_original": np.array([
                [fx_real, 0.0, cx_real],
                [0.0, fy_real, cy_real],
                [0.0, 0.0, 1.0]
            ])
        })

    return scaled_cameras


def export_vggt_to_colmap_standard(
    output_colmap_dir: str,
    selected_paths: List[str],
    extrinsics: np.ndarray,
    scaled_cameras: List[Dict[str, Any]],
    orig_coords: np.ndarray,
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    points_xyf: Optional[np.ndarray] = None,
    multi_view_tracks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    export_binary: bool = True
):
    """
    Export VGGT camera poses, intrinsics, and 3D points into official COLMAP format.
    Generates cameras.txt, images.txt, points3D.txt AND cameras.bin, images.bin, points3D.bin.
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
        R_mat = extrinsics[i, :3, :3]
        q_tensor = mat_to_quat(torch.from_numpy(R_mat)[None])
        quats.append(q_tensor[0].numpy())

    # Build per-image 2D-3D observation map
    image_observations = {i + 1: [] for i in range(num_images)}
    point_tracks = {pt_idx + 1: [] for pt_idx in range(num_points)}

    if multi_view_tracks is not None:
        for pt_id, trk in multi_view_tracks.items():
            for (img_id, pt2d_idx, px, py) in trk:
                image_observations[img_id].append((px, py, pt_id))
                point_tracks[pt_id].append((img_id, len(image_observations[img_id]) - 1))
    elif points_xyf is not None:
        # Single frame observation fallback
        for pt_idx in range(num_points):
            fidx = int(points_xyf[pt_idx, 2])
            img_id = fidx + 1
            x_518 = points_xyf[pt_idx, 0]
            y_518 = points_xyf[pt_idx, 1]
            x1 = orig_coords[fidx, 0]
            y1 = orig_coords[fidx, 1]
            ratio = scaled_cameras[fidx]["resize_ratio"]
            px = float((x_518 - x1) * ratio)
            py = float((y_518 - y1) * ratio)
            pt_id = pt_idx + 1
            image_observations[img_id].append((px, py, pt_id))
            point_tracks[pt_id].append((img_id, len(image_observations[img_id]) - 1))

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
            t = extrinsics[i, :3, 3]
            img_name = os.path.basename(selected_paths[i])
            f.write(
                f"{img_id} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {img_id} {img_name}\n"
            )
            obs = image_observations.get(img_id, [])
            if obs:
                obs_str = " ".join([f"{o[0]:.2f} {o[1]:.2f} {o[2]}" for o in obs])
                f.write(obs_str + "\n")
            else:
                f.write("\n")

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
            trk = point_tracks.get(pt_id, [])
            trk_str = " ".join([f"{t[0]} {t[1]}" for t in trk])
            f.write(
                f"{pt_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 1.0 {trk_str}\n"
            )

    # 4. Binary export (.bin)
    if export_binary:
        try:
            # cameras.bin
            with open(os.path.join(output_colmap_dir, "cameras.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_images))
                for i in range(num_images):
                    c = scaled_cameras[i]
                    # model_id=1 is PINHOLE
                    fb.write(struct.pack("<iiQQdddd", i + 1, 1, c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"]))

            # images.bin
            with open(os.path.join(output_colmap_dir, "images.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_images))
                for i in range(num_images):
                    img_id = i + 1
                    q = quats[i]
                    t = extrinsics[i, :3, 3]
                    img_name_bytes = os.path.basename(selected_paths[i]).encode("utf-8") + b"\x00"
                    obs = image_observations.get(img_id, [])
                    num_obs = len(obs)
                    fb.write(struct.pack("<idddddddi", img_id, q[0], q[1], q[2], q[3], t[0], t[1], t[2], img_id))
                    fb.write(img_name_bytes)
                    fb.write(struct.pack("<Q", num_obs))
                    for o in obs:
                        fb.write(struct.pack("<ddq", o[0], o[1], o[2]))

            # points3D.bin
            with open(os.path.join(output_colmap_dir, "points3D.bin"), "wb") as fb:
                fb.write(struct.pack("<Q", num_points))
                for i in range(num_points):
                    pt_id = i + 1
                    pt = points_3d[i]
                    rgb = points_rgb[i]
                    trk = point_tracks.get(pt_id, [])
                    fb.write(struct.pack("<QdddBBBdQ", pt_id, pt[0], pt[1], pt[2], int(rgb[0]), int(rgb[1]), int(rgb[2]), 1.0, len(trk)))
                    for t in trk:
                        fb.write(struct.pack("<ii", t[0], t[1]))

        except Exception as e:
            logger.warning(f"Native binary COLMAP export encountered issue: {e}")


def build_geometric_multiview_tracks(
    points_3d: np.ndarray,
    points_xyf: np.ndarray,
    depth_map: np.ndarray,
    extrinsics: np.ndarray,
    scaled_cameras: List[Dict[str, Any]],
    orig_coords: np.ndarray,
    depth_tol: float = 0.12
) -> Dict[int, List[Tuple[int, int, float, float]]]:
    """
    Project confident 3D points into adjacent cameras and verify geometric
    depth consistency to synthesize multi-view 2D-3D tracks for Bundle Adjustment.
    """
    num_cameras = len(extrinsics)
    num_points = len(points_3d)
    multi_view_tracks = {}

    S, H, W, _ = depth_map.shape

    for pt_idx in range(num_points):
        pt_id = pt_idx + 1
        origin_frame = int(points_xyf[pt_idx, 2])
        P_world = points_3d[pt_idx]

        # Origin frame observation
        x_518 = points_xyf[pt_idx, 0]
        y_518 = points_xyf[pt_idx, 1]
        x1_o = orig_coords[origin_frame, 0]
        y1_o = orig_coords[origin_frame, 1]
        ratio_o = scaled_cameras[origin_frame]["resize_ratio"]
        px_o = float((x_518 - x1_o) * ratio_o)
        py_o = float((y_518 - y1_o) * ratio_o)

        observations = [(origin_frame + 1, 0, px_o, py_o)]

        # Check adjacent / candidate views
        for target_frame in range(num_cameras):
            if target_frame == origin_frame:
                continue

            R_tgt = extrinsics[target_frame, :3, :3]
            t_tgt = extrinsics[target_frame, :3, 3]
            cam_tgt = scaled_cameras[target_frame]

            P_cam = R_tgt @ P_world + t_tgt
            z_cam = P_cam[2]
            if z_cam <= 0.1:
                continue

            K_real = cam_tgt["K_original"]
            u_real = (K_real[0, 0] * P_cam[0] / z_cam) + K_real[0, 2]
            v_real = (K_real[1, 1] * P_cam[1] / z_cam) + K_real[1, 2]

            if (0 <= u_real < cam_tgt["width"]) and (0 <= v_real < cam_tgt["height"]):
                # Map back to 518x518 to check predicted depth
                x1_t = orig_coords[target_frame, 0]
                y1_t = orig_coords[target_frame, 1]
                ratio_t = cam_tgt["resize_ratio"]
                u_518 = int(np.clip(round(u_real / ratio_t + x1_t), 0, W - 1))
                v_518 = int(np.clip(round(v_real / ratio_t + y1_t), 0, H - 1))

                pred_d = depth_map[target_frame, v_518, u_518, 0]
                if pred_d > 0:
                    rel_diff = abs(z_cam - pred_d) / max(z_cam, 1e-4)
                    if rel_diff <= depth_tol:
                        observations.append((target_frame + 1, len(observations), float(u_real), float(v_real)))

        if len(observations) >= 2:
            multi_view_tracks[pt_id] = observations

    return multi_view_tracks


def run_bundle_adjustment_refinement(
    colmap_dir: str,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run bundle adjustment refinement on exported COLMAP model using either pycolmap
    or the system COLMAP CLI executable.
    """
    ba_report = {"success": False, "method": "none", "log": ""}

    # Method 1: Try pycolmap
    try:
        import pycolmap
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(colmap_dir)
        num_pts = len(reconstruction.points3D)
        num_cams = len(reconstruction.images)

        if num_pts > 0 and num_cams > 1:
            if progress_callback:
                progress_callback(f"[VGGT+BA] Optimizing {num_cams} cameras and {num_pts:,} 3D points via pycolmap...")
            ba_opts = pycolmap.BundleAdjustmentOptions()
            ba_opts.max_num_iterations = 50
            ba_opts.refine_focal_length = True
            ba_opts.refine_principal_point = False
            ba_opts.refine_extra_params = False
            ba_opts.refine_extrinsics = True

            pycolmap.bundle_adjustment(reconstruction, ba_opts)
            reconstruction.write(colmap_dir)
            reconstruction.write_text(colmap_dir)
            ba_report = {
                "success": True,
                "method": "pycolmap",
                "log": f"Successfully refined {num_cams} cameras and {num_pts} points."
            }
            return ba_report
    except Exception as e:
        logger.info(f"pycolmap BA fallback to colmap CLI: {e}")

    # Method 2: Try COLMAP CLI
    import subprocess
    ba_out_dir = os.path.join(colmap_dir, "ba_refined")
    os.makedirs(ba_out_dir, exist_ok=True)
    cmd = [
        "colmap", "bundle_adjuster",
        "--input_path", colmap_dir,
        "--output_path", ba_out_dir,
        "--BundleAdjustment.max_num_iterations", "50",
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extrinsics", "1"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if res.returncode == 0:
            # Copy refined files back
            for fname in os.listdir(ba_out_dir):
                src = os.path.join(ba_out_dir, fname)
                dst = os.path.join(colmap_dir, fname)
                if os.path.isfile(src):
                    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                        f_out.write(f_in.read())
            ba_report = {"success": True, "method": "colmap_cli", "log": res.stdout[-400:]}
        else:
            ba_report = {"success": False, "method": "colmap_cli", "log": res.stderr[-400:]}
    except Exception as e:
        ba_report = {"success": False, "method": "none", "log": str(e)}

    return ba_report


def run_vggt_reconstruction(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    reconstruction_mode: str = "depth",  # "depth" or "point_head"
    conf_threshold_val: float = 1.8,
    max_frames: Optional[int] = 24,
    stride: int = 1,
    coherent_order: bool = True,
    remove_outliers: bool = True,
    voxel_size: Optional[float] = None,
    export_colmap: bool = True,
    use_ba: bool = False,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Execute complete end-to-end VGGT 3D reconstruction pipeline adhering to
    the official facebookresearch/VGGT specification.

    Args:
        image_paths: List of local image file paths
        output_dir: Output directory
        reconstruction_mode: "depth" (unprojection, recommended) or "point_head" (direct)
        conf_threshold_val: Minimum confidence score to retain a point (default: 3.5)
        max_frames: Maximum sequence frames to process
        stride: Frame sampling stride
        coherent_order: Sort images by visual overlap into continuous orbit
        remove_outliers: Apply statistical outlier removal
        voxel_size: Optional voxel grid downsampling in meters
        export_colmap: Export official COLMAP-compatible format (.bin and .txt)
        use_ba: Optional Bundle Adjustment refinement stage (VGGT+BA)
        progress_callback: Progress logging callable
    """
    total_start_time = time.perf_counter()

    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v3_path = os.path.join(base_dir, "v3_current")
        output_dir = os.path.join(v3_path if os.path.exists(v3_path) else base_dir, "outputs", "vggt")

    os.makedirs(output_dir, exist_ok=True)
    cameras_dir = os.path.join(output_dir, "cameras")
    depth_dir = os.path.join(output_dir, "depth")
    point_maps_dir = os.path.join(output_dir, "point_maps")
    debug_dir = os.path.join(output_dir, "debug_visuals")
    colmap_export_dir = os.path.join(output_dir, "colmap_export")

    for d in [cameras_dir, depth_dir, point_maps_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)

    if progress_callback:
        progress_callback("[VGGT] Initializing official VGGT 3D Reconstruction Pipeline...")
        progress_callback(f"[VGGT] Output Directory: {output_dir}")

    # 1. Run feed-forward multi-view inference
    cache_folder = os.path.join(output_dir, "_cache")
    infer_res = run_vggt_inference(
        image_paths,
        max_frames=max_frames,
        stride=stride,
        coherent_order=coherent_order,
        cache_dir=cache_folder,
        use_cache=True,
        progress_callback=progress_callback
    )

    extrinsics = infer_res["extrinsic"]          # (S, 3, 4)
    intrinsics = infer_res["intrinsic"]          # (S, 3, 3) 518x518
    depth_map = infer_res["depth_map"]            # (S, H, W, 1)
    depth_conf = infer_res["depth_conf"]          # (S, H, W)
    world_points_head = infer_res["world_points"]# (S, H, W, 3)
    world_points_conf = infer_res["world_points_conf"]
    orig_coords = infer_res["orig_coords"]        # (S, 6)
    images_np = infer_res["images_np"]            # (S, 3, H, W) in [0, 1]
    selected_paths = infer_res["selected_paths"]
    num_views = len(selected_paths)

    if depth_map.ndim == 3:
        depth_map = depth_map[..., None]

    # Convert images to (S, H, W, 3) uint8 RGB
    colors_rgb = (images_np.transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)

    # 2. Rescale intrinsics to original image dimensions
    scaled_cameras = rescale_camera_parameters_to_original(intrinsics, orig_coords, vggt_resolution=518)

    # 3. Unproject depth map to 3D world points (Path A)
    if progress_callback:
        progress_callback("[VGGT] Unprojecting predicted depth maps to 3D world points...")
    pts_from_depth = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)
    S, H, W, _ = pts_from_depth.shape

    # 4. Compare Path A (Depth Unprojection) vs Path B (Point Head)
    comparison_stats = {}
    if world_points_head is not None:
        eval_thresh = min(conf_threshold_val, 1.5)
        valid_both = (depth_conf >= eval_thresh) & (world_points_conf >= eval_thresh)
        if valid_both.sum() > 0:
            diffs = np.linalg.norm(pts_from_depth[valid_both] - world_points_head[valid_both], axis=-1)
            comparison_stats = {
                "evaluated_points": int(valid_both.sum()),
                "mean_distance_meters": float(np.mean(diffs)),
                "median_distance_meters": float(np.median(diffs)),
                "rms_distance_meters": float(np.sqrt(np.mean(diffs ** 2))),
                "p95_distance_meters": float(np.percentile(diffs, 95)),
                "max_distance_meters": float(np.max(diffs))
            }
            if progress_callback:
                progress_callback(
                    f"[VGGT] Evaluated {valid_both.sum():,} points: Depth Unprojection vs Point Head "
                    f"mean deviation = {np.mean(diffs)*1000:.1f} mm (RMS: {np.sqrt(np.mean(diffs**2))*1000:.1f} mm)."
                )

    # Select primary point set based on requested mode
    if reconstruction_mode == "point_head" and world_points_head is not None:
        active_points = world_points_head
        active_conf = world_points_conf if world_points_conf is not None else depth_conf
        mode_used = "point_head"
    else:
        active_points = pts_from_depth
        active_conf = depth_conf
        mode_used = "depth_unprojection"

    # 5. Build strict valid image pixel mask (EXCLUDE BLACK PADDING)
    valid_pixel_mask = np.zeros((S, H, W), dtype=bool)
    for s in range(S):
        x1, y1, x2, y2 = orig_coords[s, :4]
        valid_pixel_mask[s, int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = True

    flat_pts = active_points.reshape(-1, 3)
    flat_cols = colors_rgb.reshape(-1, 3)
    flat_conf = active_conf.reshape(-1)
    flat_depth = depth_map.reshape(-1)
    flat_valid = valid_pixel_mask.reshape(-1)

    # Adaptive confidence threshold adjustment if user threshold exceeds scene range
    valid_conf_values = flat_conf[flat_valid]
    if len(valid_conf_values) > 0 and conf_threshold_val > float(np.max(valid_conf_values)):
        conf_threshold_val = float(np.percentile(valid_conf_values, 75))
        logger.info(f"Adjusted confidence threshold to {conf_threshold_val:.2f} based on scene distribution.")

    # 6. Apply multi-stage filtering
    if progress_callback:
        progress_callback(
            f"[VGGT] Filtering points (Padding excluded · Confidence >= {conf_threshold_val:.2f} · Depth 0.1-50m)..."
        )

    # Pixel coordinate grid (xyf)
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    xyf_grid = np.zeros((S, H, W, 3), dtype=np.float32)
    for s in range(S):
        xyf_grid[s, :, :, 0] = u_grid
        xyf_grid[s, :, :, 1] = v_grid
        xyf_grid[s, :, :, 2] = s
    flat_xyf = xyf_grid.reshape(-1, 3)

    # Geometric validity mask
    mask = (
        flat_valid &
        np.isfinite(flat_pts).all(axis=1) &
        (flat_depth >= 0.1) &
        (flat_depth <= 50.0) &
        np.isfinite(flat_conf) &
        (flat_conf >= conf_threshold_val)
    )

    pts_valid = flat_pts[mask]
    cols_valid = flat_cols[mask]
    conf_valid = flat_conf[mask]
    xyf_valid = flat_xyf[mask]
    raw_point_count = len(pts_valid)

    if raw_point_count == 0:
        raise RuntimeError(
            f"VGGT generated 0 valid points at confidence threshold {conf_threshold_val}. "
            "Try lowering confidence threshold or inspecting input images."
        )

    # Save raw un-denoised point cloud artifact
    raw_ply_path = os.path.join(debug_dir, "point_cloud_raw.ply")
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(pts_valid)
    raw_pcd.colors = o3d.utility.Vector3dVector(cols_valid.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(raw_ply_path, raw_pcd, write_ascii=False)

    # 7. Statistical Outlier Removal
    if remove_outliers and raw_point_count > 100:
        if progress_callback:
            progress_callback(f"[VGGT] Removing statistical outliers from {raw_point_count:,} candidate points...")
        try:
            cl, ind = raw_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
            filtered_pcd = raw_pcd.select_by_index(ind)
            final_points = np.asarray(filtered_pcd.points)
            final_colors = (np.asarray(filtered_pcd.colors) * 255.0).astype(np.uint8)
            xyf_valid = xyf_valid[ind]
            conf_valid = conf_valid[ind]
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

    # Estimate surface normals for clean mesh generation / shading
    try:
        filtered_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )
        filtered_pcd.orient_normals_consistent_tangent_plane(15)
    except Exception:
        pass

    final_point_count = len(final_points)
    rejected_point_count = (S * H * W) - final_point_count

    # Save primary filtered point cloud PLY artifact
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, filtered_pcd, write_ascii=False)

    # Also save in debug_visuals
    o3d.io.write_point_cloud(os.path.join(debug_dir, "point_cloud_filtered.ply"), filtered_pcd, write_ascii=False)

    # 8. Save Cameras
    cameras_list = []
    cam_to_world_mats = closed_form_inverse_se3(extrinsics)  # shape (S, 4, 4)
    np.save(os.path.join(cameras_dir, "camera_poses.npy"), cam_to_world_mats)

    for i in range(num_views):
        c2w = cam_to_world_mats[i]
        cam_center = c2w[:3, 3].tolist()
        c_meta = scaled_cameras[i]
        cameras_list.append({
            "camera_id": i + 1,
            "image_name": os.path.basename(selected_paths[i]),
            "image_path": selected_paths[i],
            "center": cam_center,
            "extrinsic": extrinsics[i].tolist(),
            "intrinsic_original": c_meta["K_original"].tolist(),
            "intrinsic_518": intrinsics[i].tolist(),
            "focal_x": c_meta["fx"],
            "focal_y": c_meta["fy"],
            "cx": c_meta["cx"],
            "cy": c_meta["cy"],
            "width": c_meta["width"],
            "height": c_meta["height"]
        })

    with open(os.path.join(cameras_dir, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(cameras_list, f, indent=2)

    # 9. Save Depth & Point Map npz
    np.savez_compressed(
        os.path.join(depth_dir, "depth_maps.npz"),
        depth_map=depth_map,
        depth_conf=depth_conf
    )
    np.savez_compressed(
        os.path.join(point_maps_dir, "point_maps.npz"),
        world_points=pts_from_depth,
        world_points_head=world_points_head if world_points_head is not None else np.zeros((0,))
    )

    # 10. Generate Visual Debug Previews (Depth, Confidence, Montage)
    for idx in range(min(4, num_views)):
        # Depth preview
        d_map = depth_map[idx].squeeze()
        v_mask = valid_pixel_mask[idx]
        valid_d = d_map[v_mask & (d_map > 0)]
        if len(valid_d) > 0:
            d_min, d_max = np.percentile(valid_d, 5), np.percentile(valid_d, 95)
            if d_max > d_min:
                d_norm = ((d_map - d_min) / (d_max - d_min)).clip(0, 1)
                d_color = cv2.applyColorMap((d_norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
                d_color[~v_mask] = 0
                cv2.imwrite(os.path.join(depth_dir, f"depth_preview_{idx:02d}.png"), d_color)
                cv2.imwrite(os.path.join(debug_dir, f"depth_map_{idx:02d}.png"), d_color)

        # Confidence preview
        c_map = depth_conf[idx]
        valid_c = c_map[v_mask]
        if len(valid_c) > 0:
            c_max = max(float(np.percentile(valid_c, 95)), 5.0)
            c_norm = (c_map / c_max).clip(0, 1)
            c_color = cv2.applyColorMap((c_norm * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            c_color[~v_mask] = 0
            cv2.imwrite(os.path.join(debug_dir, f"confidence_map_{idx:02d}.png"), c_color)

    # Input image montage
    try:
        montage_thumbs = []
        for p in selected_paths[:12]:
            t_im = cv2.imread(p)
            if t_im is not None:
                t_im = cv2.resize(t_im, (240, 135))
                cv2.putText(t_im, os.path.basename(p)[:12], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                montage_thumbs.append(t_im)
        if montage_thumbs:
            cols_per_row = min(4, len(montage_thumbs))
            rows = (len(montage_thumbs) + cols_per_row - 1) // cols_per_row
            pad_needed = (rows * cols_per_row) - len(montage_thumbs)
            for _ in range(pad_needed):
                montage_thumbs.append(np.zeros_like(montage_thumbs[0]))
            grid_rows = [np.hstack(montage_thumbs[r*cols_per_row:(r+1)*cols_per_row]) for r in range(rows)]
            montage_img = np.vstack(grid_rows)
            cv2.imwrite(os.path.join(debug_dir, "input_montage.png"), montage_img)
    except Exception as e:
        logger.warning(f"Montage generation warning: {e}")

    # 11. COLMAP Export & Optional Bundle Adjustment
    ba_report = {"success": False, "method": "none", "log": ""}
    multi_view_tracks = None

    if export_colmap or use_ba:
        if progress_callback:
            progress_callback(f"[VGGT] Building multi-view observation tracks for COLMAP export...")

        # Subsample points for COLMAP export if needed to keep file size lightweight
        max_colmap_pts = 60000
        if len(final_points) > max_colmap_pts:
            step = len(final_points) // max_colmap_pts
            colmap_pts = final_points[::step]
            colmap_cols = final_colors[::step]
            colmap_xyf = xyf_valid[::step]
        else:
            colmap_pts = final_points
            colmap_cols = final_colors
            colmap_xyf = xyf_valid

        # Synthesize multi-view tracks
        multi_view_tracks = build_geometric_multiview_tracks(
            colmap_pts,
            colmap_xyf,
            depth_map,
            extrinsics,
            scaled_cameras,
            orig_coords,
            depth_tol=0.12
        )

        export_vggt_to_colmap_standard(
            colmap_export_dir,
            selected_paths,
            extrinsics,
            scaled_cameras,
            orig_coords,
            colmap_pts,
            colmap_cols,
            points_xyf=colmap_xyf,
            multi_view_tracks=multi_view_tracks,
            export_binary=True
        )

        if use_ba:
            if progress_callback:
                progress_callback("[VGGT+BA] Executing Bundle Adjustment optimization stage...")
            ba_report = run_bundle_adjustment_refinement(colmap_export_dir, progress_callback=progress_callback)
            if progress_callback:
                status_str = "Completed successfully" if ba_report["success"] else "Finished with warnings"
                progress_callback(f"[VGGT+BA] Bundle Adjustment {status_str} via {ba_report['method']}.")

    # 12. Spatial Metrics & Camera Trajectory Validation
    min_bound = final_points.min(axis=0).tolist() if len(final_points) > 0 else [0, 0, 0]
    max_bound = final_points.max(axis=0).tolist() if len(final_points) > 0 else [0, 0, 0]
    spans = [max_bound[i] - min_bound[i] for i in range(3)]
    bbox_volume = float(np.prod(spans))

    cam_centers = np.array([c["center"] for c in cameras_list])
    cam_dists = np.linalg.norm(cam_centers[1:] - cam_centers[:-1], axis=1) if len(cam_centers) > 1 else np.array([0.0])
    mean_cam_step = float(np.mean(cam_dists))
    max_cam_step = float(np.max(cam_dists))

    # Detect cameras inside or behind reconstructed object
    center_pt = (np.array(min_bound) + np.array(max_bound)) / 2.0
    cam_radii = np.linalg.norm(cam_centers - center_pt, axis=1)
    median_radius = float(np.median(cam_radii))
    interior_cameras = int(np.sum(cam_radii < 0.25 * median_radius))
    outlier_cameras = int(np.sum(cam_radii > 3.0 * median_radius))

    validation_diagnostics = {
        "trajectory_discontinuous": bool(max_cam_step > 3.5 * mean_cam_step and max_cam_step > 2.0),
        "mean_camera_step_meters": round(mean_cam_step, 3),
        "max_camera_step_meters": round(max_cam_step, 3),
        "median_orbit_radius_meters": round(median_radius, 3),
        "cameras_inside_object": interior_cameras,
        "outlier_cameras": outlier_cameras,
        "is_camera_pose_valid": bool(interior_cameras == 0 and outlier_cameras == 0)
    }

    total_pipeline_time = time.perf_counter() - total_start_time

    metadata = {
        "pipeline": "VGGT-1B",
        "model_id": "facebook/VGGT-1B",
        "device": infer_res["device_used"],
        "reconstruction_mode_used": mode_used,
        "total_input_images": len(image_paths),
        "registered_images": num_views,
        "raw_candidate_points": raw_point_count,
        "filtered_points": final_point_count,
        "rejected_points": rejected_point_count,
        "confidence_threshold_value": conf_threshold_val,
        "depth_unprojection_vs_point_head": comparison_stats,
        "bundle_adjustment_used": use_ba,
        "bundle_adjustment_report": ba_report,
        "camera_validation": validation_diagnostics,
        "runtime_seconds": round(total_pipeline_time, 2),
        "inference_time_seconds": round(infer_res["inference_time"], 2),
        "bounding_box_min": min_bound,
        "bounding_box_max": max_bound,
        "bounding_box_volume": round(bbox_volume, 4),
        "point_density_per_m3": round(final_point_count / max(bbox_volume, 1e-4), 1),
        "ply_path": ply_path,
        "cameras_dir": cameras_dir,
        "depth_dir": depth_dir,
        "point_maps_dir": point_maps_dir,
        "debug_dir": debug_dir,
        "colmap_export_dir": colmap_export_dir if export_colmap else None
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback(
            f"[VGGT] Reconstruction completed in {total_pipeline_time:.2f}s! "
            f"Points: {final_point_count:,} · Volume: {bbox_volume:.3f} m³ · Registered Views: {num_views}"
        )

    return {
        "points": final_points,
        "colors": final_colors,
        "cameras": cameras_list,
        "metadata": metadata,
        "ply_path": ply_path,
        "output_dir": output_dir,
        "runtime": total_pipeline_time,
        "validation_diagnostics": validation_diagnostics
    }
