"""
PowerTwinAI COLMAP Dense Reconstruction

Pipeline
--------
Sparse COLMAP model
        ↓
Image Undistortion
        ↓
PatchMatch Stereo
        ↓
Geometric Consistency
        ↓
Stereo Fusion
        ↓
Dense Point Cloud

The dense reconstruction itself is performed by COLMAP.
"""

import os
import subprocess


COLMAP_PATH = r"E:\COLMAP\COLMAP.bat"


def run_colmap_dense(
    workspace="colmap_workspace",
    sparse_model_path=None
):
    """
    Run the COLMAP dense reconstruction pipeline.

    Parameters
    ----------
    workspace : str
        COLMAP workspace containing the input images.

    sparse_model_path : str
        Selected COLMAP sparse reconstruction model.

    Returns
    -------
    str
        Path to the fused dense point cloud.
    """

    # =====================================================
    # PATHS
    # =====================================================

    images_path = os.path.join(
        workspace,
        "images"
    )

    if sparse_model_path is None:
        raise ValueError(
            "Selected COLMAP sparse model path "
            "was not provided."
        )

    dense_path = os.path.join(
        workspace,
        "dense"
    )

    fused_path = os.path.join(
        dense_path,
        "fused_geometric.ply"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if not os.path.isdir(images_path):
        raise FileNotFoundError(
            "COLMAP images directory not found: "
            f"{images_path}"
        )

    if not os.path.isdir(sparse_model_path):
        raise FileNotFoundError(
            "COLMAP sparse model not found: "
            f"{sparse_model_path}"
        )

    if not os.path.exists(COLMAP_PATH):
        raise FileNotFoundError(
            "COLMAP executable not found: "
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

    print("=" * 70)
    print("[COLMAP-DENSE] IMAGE UNDISTORTION")
    print("=" * 70)

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

    print("=" * 70)
    print("[COLMAP-DENSE] PATCHMATCH STEREO")
    print("=" * 70)

    subprocess.run(
        [
            COLMAP_PATH,
            "patch_match_stereo",
            "--workspace_path",
            dense_path,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true"
        ],
        check=True
    )

    print(
        "[COLMAP-DENSE] PatchMatch Stereo completed."
    )

    # =====================================================
    # STEREO FUSION
    # =====================================================

    print("=" * 70)
    print("[COLMAP-DENSE] STEREO FUSION")
    print("=" * 70)

    subprocess.run(
        [
            COLMAP_PATH,
            "stereo_fusion",
            "--workspace_path",
            dense_path,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_type",
            "PLY",
            "--output_path",
            fused_path,
            "--StereoFusion.min_num_pixels",
            "2"
        ],
        check=True
    )
    print(
        "[COLMAP-DENSE] Stereo Fusion completed."
    )

    # =====================================================
    # VALIDATE OUTPUT
    # =====================================================

    if not os.path.isfile(fused_path):
        raise RuntimeError(
            "COLMAP dense reconstruction completed "
            "but fused_geometric.ply was not generated."
        )

    print("=" * 70)
    print("[COLMAP-DENSE] DENSE RECONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"[COLMAP-DENSE] Output: {fused_path}"
    )

    return fused_path