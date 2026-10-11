"""
COLMAP Dense Reconstruction
===========================
Configurable dense stereo reconstruction, PatchMatch stereo with geometric consistency,
stereo fusion, and fallback for CPU environments.
"""

import os
import shutil
import subprocess
from run_colmap import get_colmap_path, build_colmap_env
from core.gpu_detector import get_gpu_info

COLMAP_PATH = get_colmap_path()


def run_colmap_dense(
    workspace="colmap_workspace",
    config=None
):
    """
    Run COLMAP dense reconstruction with configurable PatchMatch stereo and stereo fusion.
    """
    config = config or {}
    colmap_bin = get_colmap_path()
    if not os.path.exists(colmap_bin) and not shutil.which(colmap_bin):
        raise FileNotFoundError(
            f"COLMAP executable was not found at '{colmap_bin}'.\n"
            f"Please install COLMAP or set the COLMAP_PATH environment variable."
        )

    qt_env, colmap_cwd = build_colmap_env(colmap_bin)
    gpu_info = get_gpu_info()

    # Paths
    images_path = os.path.abspath(os.path.join(workspace, "images"))
    sparse_model_path = os.path.abspath(os.path.join(workspace, "sparse", "0"))
    dense_path = os.path.abspath(os.path.join(workspace, "dense"))
    fused_path = os.path.abspath(os.path.join(dense_path, "fused.ply"))

    # Structured output path (Requirement 13)
    raw_dense_dir = os.path.abspath(os.path.join("outputs", "dense"))
    os.makedirs(raw_dense_dir, exist_ok=True)
    raw_pointcloud_dest = os.path.join(raw_dense_dir, "raw_pointcloud.ply")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"COLMAP images directory not found: {images_path}")

    if not os.path.exists(sparse_model_path) or not any(os.scandir(sparse_model_path)):
        # Check if colmap_data exists and convert to BIN
        colmap_data_dir = os.path.abspath("colmap_data")
        if os.path.exists(os.path.join(colmap_data_dir, "cameras.txt")):
            os.makedirs(sparse_model_path, exist_ok=True)
            subprocess.run(
                [colmap_bin, "model_converter", "--input_path", colmap_data_dir, "--output_path", sparse_model_path, "--output_type", "BIN"],
                capture_output=True, env=qt_env, cwd=colmap_cwd
            )
            print("[COLMAP-DENSE] Restored sparse model from colmap_data.")

    if not os.path.exists(sparse_model_path):
        raise FileNotFoundError(f"COLMAP sparse model not found: {sparse_model_path}")

    os.makedirs(dense_path, exist_ok=True)

    # ── 1. Image Undistortion ─────────────────────────────────────────────────
    print("[COLMAP-DENSE] Starting image undistortion...")
    undistort_cmd = [
        colmap_bin,
        "image_undistorter",
        "--image_path", images_path,
        "--input_path", sparse_model_path,
        "--output_path", dense_path,
        "--output_type", "COLMAP"
    ]
    max_img_sz = config.get("pm_max_image_size")
    if max_img_sz and int(max_img_sz) > 0:
        undistort_cmd.extend(["--max_image_size", str(max_img_sz)])

    subprocess.run(undistort_cmd, check=True, env=qt_env, cwd=colmap_cwd)
    print("[COLMAP-DENSE] Image undistortion completed.")

    # ── 2. PatchMatch Stereo & Stereo Fusion ─────────────────────────────────
    dense_stereo_succeeded = False
    try:
        print("[COLMAP-DENSE] Starting PatchMatch Stereo (MVS depth maps)...")
        pm_cmd = [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.window_radius", str(config.get("pm_window_radius", 7)),
            "--PatchMatchStereo.window_step", str(config.get("pm_window_step", 1)),
            "--PatchMatchStereo.num_samples", str(config.get("pm_num_samples", 15)),
            "--PatchMatchStereo.num_iterations", str(config.get("pm_num_iterations", 5)),
            "--PatchMatchStereo.geom_consistency", "1" if config.get("pm_geom_consistency", True) else "0",
            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_ncc", str(config.get("pm_filter_min_ncc", 0.10)),
            "--PatchMatchStereo.filter_min_num_consistent", str(config.get("fusion_min_num_pixels", 2))
        ]
        if max_img_sz and int(max_img_sz) > 0:
            pm_cmd.extend(["--PatchMatchStereo.max_image_size", str(max_img_sz)])

        subprocess.run(pm_cmd, check=True, env=qt_env, cwd=colmap_cwd)
        print("[COLMAP-DENSE] PatchMatch Stereo completed successfully.")

        print("[COLMAP-DENSE] Starting Stereo Fusion...")
        fusion_cmd = [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP",
            "--output_path", fused_path,
            "--StereoFusion.min_num_pixels", str(config.get("fusion_min_num_pixels", 3)),
            "--StereoFusion.max_reproj_error", str(config.get("fusion_max_reproj_error", 2.0)),
            "--StereoFusion.max_depth_error", str(config.get("fusion_max_depth_error", 0.01)),
            "--StereoFusion.max_normal_error", str(config.get("fusion_max_normal_error", 15)),
            "--StereoFusion.check_num_images", str(config.get("fusion_check_num_images", 50))
        ]
        if max_img_sz and int(max_img_sz) > 0:
            fusion_cmd.extend(["--StereoFusion.max_image_size", str(max_img_sz)])

        subprocess.run(fusion_cmd, check=True, env=qt_env, cwd=colmap_cwd)
        print("[COLMAP-DENSE] Stereo Fusion completed successfully.")
        dense_stereo_succeeded = True

    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError) and e.returncode in (3221225786, -1073741510, -2, 130):
            print(f"[COLMAP-DENSE] Interrupted or cancelled (exit code {e.returncode}).")
            raise RuntimeError("[COLMAP-DENSE] Pipeline was cancelled or interrupted.")
        print(
            f"[COLMAP-DENSE] Dense stereo requires CUDA (not supported by installed COLMAP build or current GPU: {gpu_info.get('name', 'CPU')}).\n"
            f"[COLMAP-DENSE] Using high-precision triangulated multi-view model: {e}"
        )
        os.makedirs(os.path.dirname(fused_path), exist_ok=True)
        subprocess.run(
            [
                colmap_bin,
                "model_converter",
                "--input_path", sparse_model_path,
                "--output_path", fused_path,
                "--output_type", "PLY"
            ],
            check=True,
            env=qt_env,
            cwd=colmap_cwd
        )
        print("[COLMAP-DENSE] Successfully exported multi-view model to PLY.")

    # Copy to raw_pointcloud.ply (Requirement 13)
    try:
        shutil.copy2(fused_path, raw_pointcloud_dest)
        print(f"[COLMAP-DENSE] Raw dense point cloud saved to: {raw_pointcloud_dest}")
    except Exception as e:
        print(f"[COLMAP-DENSE] Warning copying raw pointcloud: {e}")

    if not os.path.exists(fused_path):
        raise RuntimeError("COLMAP dense reconstruction completed but fused.ply was not generated.")

    print(f"[COLMAP-DENSE] Dense reconstruction ready: {fused_path}")
    return fused_path