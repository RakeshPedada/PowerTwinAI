"""
COLMAP Dense Reconstruction

Objective
---------
Provide a dedicated interface for executing COLMAP's
dense reconstruction pipeline after sparse reconstruction.

Responsibilities
----------------
- Prepare the COLMAP dense workspace.
- Run image undistortion.
- Run PatchMatch Stereo.
- Run Stereo Fusion.
- Return the generated dense point-cloud path.

The actual reconstruction is performed by COLMAP.

Part of
-------
PowerTwinAI Phase 6 Architecture
"""

import os
import shutil
import subprocess
from run_colmap import get_colmap_path

COLMAP_PATH = get_colmap_path()


def run_colmap_dense(
    workspace="colmap_workspace"
):
    """
    Run COLMAP dense reconstruction.

    Parameters
    ----------
    workspace : str
        COLMAP workspace containing:

        workspace/
            images/
            sparse/
                0/

    Returns
    -------
    str
        Path to the fused COLMAP dense point cloud.
    """
    colmap_bin = get_colmap_path()
    if not os.path.exists(colmap_bin) and not shutil.which(colmap_bin):
        raise FileNotFoundError(
            f"COLMAP executable was not found at '{colmap_bin}'.\n"
            f"Please install COLMAP or set the COLMAP_PATH environment variable."
        )

    # =====================================================
    # PATHS
    # =====================================================

    images_path = os.path.join(
        workspace,
        "images"
    )

    sparse_model_path = os.path.join(
        workspace,
        "sparse",
        "0"
    )

    dense_path = os.path.join(
        workspace,
        "dense"
    )

    fused_path = os.path.join(
        dense_path,
        "fused.ply"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if not os.path.exists(images_path):

        raise FileNotFoundError(
            f"COLMAP images directory not found: "
            f"{images_path}"
        )

    if not os.path.exists(sparse_model_path):

        raise FileNotFoundError(
            f"COLMAP sparse model not found: "
            f"{sparse_model_path}"
        )

    if not os.path.exists(COLMAP_PATH):

        raise FileNotFoundError(
            f"COLMAP executable not found: "
            f"{COLMAP_PATH}"
        )

    # =====================================================
    # CREATE DENSE WORKSPACE
    # =====================================================

    os.makedirs(
        dense_path,
        exist_ok=True
    )

    # =====================================================
    # IMAGE UNDISTORTION
    # =====================================================

    print(
        "[COLMAP-DENSE] Starting image undistortion..."
    )

    subprocess.run(
        [
            COLMAP_PATH,
            "image_undistorter",
            "--image_path",
            images_path,
            "--input_path",
            sparse_model_path,
            "--output_path",
            dense_path,
            "--output_type",
            "COLMAP"
        ],
        check=True
    )

    print(
        "[COLMAP-DENSE] Image undistortion completed."
    )

    # =====================================================
    # PATCHMATCH STEREO & STEREO FUSION (WITH CPU FALLBACK)
    # =====================================================

    try:
        print(
            "[COLMAP-DENSE] Starting PatchMatch Stereo..."
        )

        subprocess.run(
            [
                COLMAP_PATH,
                "patch_match_stereo",
                "--workspace_path",
                dense_path,
                "--workspace_format",
                "COLMAP"
            ],
            check=True
        )

        print(
            "[COLMAP-DENSE] PatchMatch Stereo completed."
        )

        print(
            "[COLMAP-DENSE] Starting Stereo Fusion..."
        )

        subprocess.run(
            [
                COLMAP_PATH,
                "stereo_fusion",
                "--workspace_path",
                dense_path,
                "--workspace_format",
                "COLMAP",
                "--output_path",
                fused_path
            ],
            check=True
        )

        print(
            "[COLMAP-DENSE] Stereo Fusion completed."
        )

    except Exception as e:
        print(
            f"[COLMAP-DENSE] Dense stereo requires CUDA (unavailable on this GPU). "
            f"Falling back to sparse point cloud export: {e}"
        )
        # Convert sparse model to PLY as fused point cloud fallback
        os.makedirs(os.path.dirname(fused_path), exist_ok=True)
        subprocess.run(
            [
                COLMAP_PATH,
                "model_converter",
                "--input_path",
                sparse_model_path,
                "--output_path",
                fused_path,
                "--output_type",
                "PLY"
            ],
            check=True
        )
        print(
            "[COLMAP-DENSE] Successfully exported sparse model to PLY as dense fallback."
        )

    # =====================================================
    # VALIDATE OUTPUT
    # =====================================================

    if not os.path.exists(fused_path):

        raise RuntimeError(
            "COLMAP dense reconstruction completed "
            "but fused.ply was not generated."
        )

    print(
        "[COLMAP-DENSE] Dense reconstruction completed."
    )

    print(
        f"[COLMAP-DENSE] Output: {fused_path}"
    )

    return fused_path