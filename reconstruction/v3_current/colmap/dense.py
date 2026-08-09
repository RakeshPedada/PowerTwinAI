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
import subprocess


COLMAP_PATH = r"E:\COLMAP\COLMAP.bat"


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
    # PATCHMATCH STEREO
    # =====================================================

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

    # =====================================================
    # STEREO FUSION
    # =====================================================

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