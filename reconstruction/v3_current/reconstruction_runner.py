import os
import sys
import json
import traceback
import numpy as np
from reconstruction_colmap import run_reconstruction

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

    uploaded_files = []

    # =====================================================
    # LOAD FILES
    # =====================================================

    for path in image_paths:

        file = open(path, "rb")

        uploaded_files.append(file)

    log_message(
        f"[INFO] Loaded {len(uploaded_files)} images"
    )

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
    # SAVE RESULTS
    # =====================================================

    np.savez(

        RESULT_FILE,

        points=points,

        colors=colors,

        cameras=camera_positions,

        processing_time=processing_time,

        analytics=analytics
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
        f"[INFO] Processing Time: "
        f"{processing_time:.2f} sec"
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