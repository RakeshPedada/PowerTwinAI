import os
import sys
import json
import time
import traceback
import numpy as np
from reconstruction_colmap import run_reconstruction
from run_colmap import run_colmap

# =========================================================
# UTF-8 CONSOLE FIX
# =========================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except:
    pass

# =========================================================
# PATHS
# =========================================================

TEMP_DIR = "temp_session"

LOG_FILE = os.path.join(
    TEMP_DIR,
    "logs.txt"
)

STATUS_FILE = os.path.join(
    TEMP_DIR,
    "status.json"
)

RESULT_FILE = os.path.join(
    TEMP_DIR,
    "result_data.npz"
)

IMAGE_PATHS_FILE = os.path.join(
    TEMP_DIR,
    "image_paths.json"
)

# =========================================================
# LOGGER
# =========================================================

def log_message(message):

    print(message, flush=True)

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            str(message) + "\n"
        )

# =========================================================
# STATUS
# =========================================================

def save_status(status):

    with open(
        STATUS_FILE,
        "w"
    ) as f:

        json.dump(
            {"status": status},
            f
        )

# =========================================================
# MAIN
# =========================================================

try:
    # =====================================================
    # TOTAL PIPELINE TIMER
    # =====================================================

    pipeline_start_time = time.perf_counter()

    colmap_time = 0.0
    reconstruction_time = 0.0

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

    uploaded_files = []

    for path in image_paths:

        uploaded_files.append({
            "path": path,
            "name": os.path.basename(path)
        })

    log_message(
        f"[INFO] Loaded {len(uploaded_files)} images"
    )
    # =====================================================
    # RUN COLMAP AUTOMATICALLY
    # =====================================================

    log_message(
        "[COLMAP] Starting automatic sparse reconstruction..."
    )

    print(f"[DEBUG] Total image paths passed to COLMAP: {len(image_paths)}")

    colmap_start_time = time.perf_counter()

    run_colmap(image_paths)

    colmap_time = (
        time.perf_counter()
        - colmap_start_time
    )

    log_message(
        f"[TIME] COLMAP Reconstruction: "
        f"{colmap_time:.2f} sec"
    )

    log_message(
        "[COLMAP] Sparse reconstruction completed"
    )
    reconstruction_start_time = time.perf_counter()
    # =====================================================
    # RUN RECONSTRUCTION
    # =====================================================

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

        progress_callback=log_message
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

    np.savez (

        RESULT_FILE,

        points=points,

        colors=colors,

        cameras=camera_positions,

        processing_time=processing_time,

        colmap_time=colmap_time,

        reconstruction_time=reconstruction_time,

        total_pipeline_time=total_pipeline_time,

        analytics=analytics,

        ply_path=str(ply_path)
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
        f"[TIME] COLMAP Sparse Reconstruction: "
        f"{colmap_time:.2f} sec"
    )

    log_message(
        f"[TIME] Reconstruction Stage: "
        f"{processing_time:.2f} sec"
    )

    log_message(
        f"[TIME] Dense + Cleaning + Mesh: "
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

    save_status("COMPLETED")

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

    save_status("FAILED")

finally:

    input(
        "\nPress Enter to exit..."
    )