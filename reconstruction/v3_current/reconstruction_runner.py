import sys
import asyncio
import logging

# Windows ProactorEventLoop fix — eliminates WinError 10054 spam
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

import os
import json
import time
import traceback
import numpy as np

# =========================================================
# UTF-8 CONSOLE FIX
# =========================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except:
    pass

# =========================================================
# PATHS (Synchronized with app.py and config/paths.py)
# =========================================================

try:
    from config.paths import (
        TEMP_DIR,
        LOG_FILE,
        STATUS_FILE,
        RESULT_FILE,
        IMAGE_PATHS_FILE,
        CONFIG_FILE,
    )
except Exception:
    TEMP_DIR = "temp_session"
    LOG_FILE = os.path.join(TEMP_DIR, "reconstruction.log")
    STATUS_FILE = os.path.join(TEMP_DIR, "status.json")
    RESULT_FILE = os.path.join(TEMP_DIR, "result_data.npz")
    IMAGE_PATHS_FILE = os.path.join(TEMP_DIR, "image_paths.json")

LEGACY_LOG_FILE = os.path.join(TEMP_DIR, "logs.txt")

# =========================================================
# LOGGER
# =========================================================

def log_message(message):

    print(message, flush=True)

    # Write to primary log file monitored by Streamlit UI
    for target in [LOG_FILE, LEGACY_LOG_FILE]:
        try:
            with open(target, "a", encoding="utf-8", errors="ignore") as f:
                f.write(str(message) + "\n")
        except Exception:
            pass

# =========================================================
# STATUS
# =========================================================

def save_status(status, step=None, progress=None):
    try:
        data = {"status": status}
        if step is not None:
            data["step"] = step
        if progress is not None:
            data["progress"] = progress
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        tmp_status = STATUS_FILE + ".tmp"
        with open(tmp_status, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_status, STATUS_FILE)
    except Exception:
        pass

# =========================================================
# MAIN
# =========================================================

try:
    save_status("RUNNING", step="Initializing reconstruction modules...", progress=0.04)
    log_message("[INIT] Loading reconstruction modules...")
    from reconstruction_colmap import run_reconstruction
    from backends.backend_manager import BackendManager
    from preprocessing.preprocessing_manager import preprocess_images
    import open3d as o3d

    # Load active reconstruction configuration
    config = {}
    config_file_path = globals().get("CONFIG_FILE", os.path.join(TEMP_DIR, "reconstruction_config.json"))
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as cf:
                config = json.load(cf)
            log_message(f"[CONFIG] Active Quality Preset: {config.get('name', 'Custom')}")
        except Exception as e:
            log_message(f"[CONFIG] Warning loading config: {e}")

    # =====================================================
    # TOTAL PIPELINE TIMER
    # =====================================================

    pipeline_start_time = time.perf_counter()

    colmap_time = 0.0
    reconstruction_time = 0.0

    save_status("RUNNING", step="Starting 3D reconstruction pipeline...", progress=0.06)
    log_message(
        "[START] Starting Reconstruction Pipeline..."
    )

    # =====================================================
    # LOAD IMAGE PATHS
    # =====================================================

    with open(
        IMAGE_PATHS_FILE,
        "r"
    ) as f:

        image_paths = json.load(f)
    print(f"[DEBUG] Total image paths: {len(image_paths)}")

  

    if len(image_paths) > 0:
        print(f"[DEBUG] First image path: {image_paths[0]}")    


    # =====================================================
    # PREPROCESS IMAGES
    # =====================================================

    save_status("RUNNING", step="Preprocessing and enhancing input images...", progress=0.10)
    log_message(
        "[PREPROCESS] Starting preprocessing..."
    )

    preprocess_result = preprocess_images(
        image_paths,
        progress_callback=log_message,
        config=config
    )

    # preprocess_images returns a dict — extract the paths list
    if isinstance(preprocess_result, dict):
        processed_image_paths = preprocess_result.get("processed_paths", [])
        failed_q  = preprocess_result.get("failed_quality", [])
        warned_q  = preprocess_result.get("warned_quality", [])
        failed_bg = preprocess_result.get("failed_bg", [])
        log_message(
            f"[PREPROCESS] Quality — "
            f"failed: {len(failed_q)}, warned: {len(warned_q)}"
        )
        if failed_bg:
            log_message(
                f"[PREPROCESS] BG removal failed on "
                f"{len(failed_bg)} image(s)"
            )
    else:
        # Fallback: old code returned a plain list
        processed_image_paths = preprocess_result or []

    if len(processed_image_paths) == 0:
        raise RuntimeError(
            "No images available after preprocessing. "
            "Check quality report for details."
        )

    print(f"[DEBUG] Processed images: {len(processed_image_paths)}")
    print(f"[DEBUG] First processed path: {processed_image_paths[0]}")

    log_message(
        f"[PREPROCESS] Images after preprocessing: {len(processed_image_paths)}"
    )

    uploaded_files = []

    for path in processed_image_paths:

        uploaded_files.append({
            "path": path,
            "name": os.path.basename(path)
        })

    log_message(
        f"[INFO] Loaded {len(uploaded_files)} images"
    )

    # =====================================================
    # RUN RECONSTRUCTION BACKEND
    # =====================================================

    active_backend = config.get("backend", "colmap").lower()
    log_message(f"[BACKEND] Selected reconstruction backend: {active_backend}")

    if active_backend == "vggt":
        save_status("RUNNING", step="VGGT-1B feed-forward 3D geometry & camera pose prediction...", progress=0.30)
        log_message("[VGGT] Launching official Meta VGGT-1B reconstruction pipeline...")
        from reconstruction.vggt import run_vggt_reconstruction

        vggt_output_dir = os.path.abspath(os.path.join(TEMP_DIR, "..", "outputs", "vggt"))
        os.makedirs(vggt_output_dir, exist_ok=True)
        vggt_res = run_vggt_reconstruction(
            image_paths=processed_image_paths,
            output_dir=vggt_output_dir,
            reconstruction_mode=config.get("vggt_mode", "depth"),
            conf_threshold_val=float(config.get("vggt_conf", 1.8)),
            max_frames=int(config.get("vggt_max_frames", 24)),
            coherent_order=True,
            remove_outliers=True,
            export_colmap=True,
            use_ba=bool(config.get("use_ba", False)),
            generate_mesh_flag=True,
            poisson_depth=int(config.get("poisson_depth", 8)),
            progress_callback=log_message
        )

        save_status("RUNNING", step="Saving VGGT 3D reconstruction telemetry & data...", progress=0.90)
        pts = vggt_res.get("points", np.empty((0, 3)))
        cols = vggt_res.get("colors", np.empty((0, 3)))
        cams = vggt_res.get("camera_positions", np.empty((0, 3)))
        ply_out = vggt_res.get("point_cloud_ply", "")

        np.savez_compressed(
            RESULT_FILE,
            points=pts,
            colors=cols,
            camera_positions=cams,
            matches=np.empty((0, 2)),
            ply_path=ply_out,
            pair_logs=[],
            analytics={"backend": "VGGT-1B", "points_count": len(pts), "cameras_count": len(cams), "runtime": vggt_res.get("runtime", 0.0)}
        )

        save_status("COMPLETED", step="VGGT-1B 3D reconstruction finished successfully.", progress=1.0)
        log_message(f"[COMPLETED] VGGT-1B Reconstruction finished in {vggt_res.get('runtime', 0.0):.2f}s with {len(pts)} points.")
        sys.exit(0)

    elif active_backend == "mast3r":
        save_status("RUNNING", step="MASt3R matching & sparse global alignment...", progress=0.30)
        log_message("[MAST3R] Launching official Naver MASt3R-SfM reconstruction pipeline...")
        from reconstruction.mast3r import run_mast3r_reconstruction

        mast3r_output_dir = os.path.abspath(os.path.join(TEMP_DIR, "..", "outputs", "mast3r"))
        os.makedirs(mast3r_output_dir, exist_ok=True)
        mast3r_res = run_mast3r_reconstruction(
            image_paths=processed_image_paths,
            output_dir=mast3r_output_dir,
            pairing_mode=config.get("mast3r_pairing", "Mode C (Hybrid)"),
            generate_mesh=True,
            poisson_depth=int(config.get("poisson_depth", 8)),
            progress_callback=log_message
        )

        save_status("RUNNING", step="Saving MASt3R-SfM 3D reconstruction telemetry & data...", progress=0.90)
        pts = mast3r_res.get("points", np.empty((0, 3)))
        cols = mast3r_res.get("colors", np.empty((0, 3)))
        cams = mast3r_res.get("camera_positions", np.empty((0, 3)))
        ply_out = mast3r_res.get("point_cloud_ply", "")

        np.savez_compressed(
            RESULT_FILE,
            points=pts,
            colors=cols,
            camera_positions=cams,
            matches=np.empty((0, 2)),
            ply_path=ply_out,
            pair_logs=[],
            analytics={"backend": "MASt3R-SfM", "points_count": len(pts), "cameras_count": len(cams), "runtime": mast3r_res.get("runtime", 0.0)}
        )

        save_status("COMPLETED", step="MASt3R-SfM reconstruction finished successfully.", progress=1.0)
        log_message(f"[COMPLETED] MASt3R-SfM Reconstruction finished in {mast3r_res.get('runtime', 0.0):.2f}s with {len(pts)} points.")
        sys.exit(0)

    # Default: COLMAP
    save_status("RUNNING", step="COLMAP feature extraction & multi-view matching...", progress=0.25)
    log_message(
        "[BACKEND] Starting reconstruction backend..."
    )

    print(
        f"[DEBUG] Total processed image paths passed "
        f"to backend: {len(processed_image_paths)}"
    )

    colmap_start_time = time.perf_counter()

    backend_manager = BackendManager(
        backend="colmap"
    )

    dense_ply_path = backend_manager.run(
        processed_image_paths,
        config=config
    )
    if not dense_ply_path:

        raise RuntimeError(
            "COLMAP backend did not return "
            "a dense point-cloud path."
        )

    if not os.path.exists(
        dense_ply_path
    ):

        raise FileNotFoundError(
            f"COLMAP dense point cloud not found: "
            f"{dense_ply_path}"
        )

    log_message(
        f"[BACKEND] Dense point cloud ready: "
        f"{dense_ply_path}"
    )
    colmap_time = (
    time.perf_counter()
    - colmap_start_time
    )

    log_message(
        f"[TIME] COLMAP Backend: "
        f"{colmap_time:.2f} sec"
    )

    log_message(
        "[BACKEND] COLMAP backend completed"
    )

    reconstruction_start_time = time.perf_counter()
    # =====================================================
    # RUN RECONSTRUCTION
    # =====================================================

    save_status("RUNNING", step="ML surface reconstruction & geometry hole completion...", progress=0.82)

    (
        points,
        colors,
        camera_positions,
        match_visuals,
        ply_path,
        pair_logs,
        processing_time,
        analytics
    ) = run_reconstruction(

        uploaded_files,

        dense_ply_path,

        progress_callback=log_message,

        config=config
    )
    reconstruction_time = (
        time.perf_counter()
        - reconstruction_start_time
    )

    log_message(
        f"[TIME] Dense + Cleaning + Mesh: "
        f"{reconstruction_time:.2f} sec"
    )


    # =====================================================
    # CHECK FAILURE
    # =====================================================

    if points is None:

        log_message(
            "[ERROR] Reconstruction Failed"
        )

        save_status("FAILED")

        raise Exception(
            "Reconstruction returned None"
        )

    # =====================================================
    # TOTAL PIPELINE TIME
    # =====================================================

    total_pipeline_time = (
        time.perf_counter()
        - pipeline_start_time
    )

    analytics["colmap_time"] = colmap_time
    analytics["reconstruction_time"] = reconstruction_time
    analytics["total_pipeline_time"] = total_pipeline_time

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    # Load sparse points if available
    sparse_points_arr = np.zeros((0, 3))
    try:
        from colmap_loader import load_colmap_model
        _, _, _, s_pts, _ = load_colmap_model("colmap_data")
        sparse_points_arr = np.asarray(s_pts, dtype=np.float64)
    except Exception:
        pass

    # Load raw dense points if available
    raw_points_arr = np.zeros((0, 3))
    raw_pcd_file = os.path.join("outputs", "dense", "raw_pointcloud.ply")
    if os.path.exists(raw_pcd_file):
        try:
            raw_pcd = o3d.io.read_point_cloud(raw_pcd_file)
            raw_points_arr = np.asarray(raw_pcd.points, dtype=np.float64)
        except Exception:
            pass

    np.savez (

        RESULT_FILE,

        points=points,

        colors=colors,

        cameras=camera_positions,

        sparse_points=sparse_points_arr,

        raw_points=raw_points_arr,

        processing_time=processing_time,

        colmap_time=colmap_time,

        reconstruction_time=reconstruction_time,

        total_pipeline_time=total_pipeline_time,

        analytics=analytics,

        ply_path=str(ply_path),

        mesh_path=str(analytics.get("mesh_path", os.path.join("output", "mesh.ply"))),

        raw_ply_path=str(raw_pcd_file),

        filtered_ply_path=str(os.path.join("outputs", "dense", "filtered_pointcloud.ply")),

        sparse_ply_path=str(os.path.join("outputs", "sparse", "sparse_model.ply"))
    )



    # =====================================================
    # FINAL LOGS
    # =====================================================

    log_message(
        "[SUCCESS] Reconstruction Complete"
    )

    log_message(
        f"[INFO] Total Points: {len(points):,}"
    )

    log_message(
        f"[INFO] Cameras: {len(camera_positions)}"
    )

    log_message(
        f"[INFO] PLY Output: {ply_path}"
    )

    log_message(
        "========================================"
    )

    log_message(
        "[TIME] PROCESSING TIME SUMMARY"
    )

    log_message(
        f"[TIME] COLMAP Backend (Sparse + Dense):"
        f"{colmap_time:.2f} sec"
    )

    log_message(
        f"[TIME] Reconstruction Stage: "
        f"{processing_time:.2f} sec"
    )

    log_message(
        f"[TIME] Dense Output Processing + Mesh: "
        f"{reconstruction_time:.2f} sec"
    )

    log_message(
        f"[TIME] TOTAL PIPELINE TIME: "
        f"{total_pipeline_time:.2f} sec"
    )

    log_message(
        "========================================"
    )

    log_message(
        f"[INFO] Health Score: "
        f"{analytics['health_score']}%"
    )

    save_status("COMPLETED", step="3D Reconstruction & ML Surface Meshing Completed!", progress=1.0)

except Exception as e:

    error_text = traceback.format_exc()

    print(error_text)

    with open(
        "full_error.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(error_text)

    try:

        log_message(
            f"[ERROR] {str(e)}"
        )

    except:

        print(
            f"[ERROR] {str(e)}"
        )

    save_status("FAILED", step=f"Error encountered: {str(e)}", progress=0.0)

finally:

    try:
        if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            input("\nPress Enter to exit...")
    except Exception:
        pass