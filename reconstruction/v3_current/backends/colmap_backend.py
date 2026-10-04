"""
COLMAP Reconstruction Backend

Objective
---------
Provide a unified backend interface for executing the
complete COLMAP reconstruction pipeline.

Responsibilities
----------------
- Accept prepared image paths.
- Execute COLMAP sparse reconstruction.
- Execute COLMAP dense reconstruction.
- Return the generated dense point cloud.
- Keep COLMAP-specific implementation details isolated
  from the rest of PowerTwinAI.

Pipeline
--------
Image Paths
    ↓
COLMAP Feature Extraction
    ↓
Feature Matching
    ↓
Sparse Reconstruction
    ↓
COLMAP Dense Reconstruction
    ↓
Fused Dense Point Cloud

Part of
-------
PowerTwinAI Phase 6 Architecture
"""

import os

from run_colmap import run_colmap
from colmap.dense import run_colmap_dense


def run_colmap_backend(
    image_paths
):
    """
    Execute the complete COLMAP reconstruction backend.

    Parameters
    ----------
    image_paths : list[str]
        Paths to the prepared input images.

    Returns
    -------
    str
        Path to the COLMAP fused dense point cloud.
    """

    if not image_paths:

        raise ValueError(
            "No image paths were provided "
            "to the COLMAP backend."
        )

    # =====================================================
    # COLMAP SPARSE RECONSTRUCTION
    # =====================================================

    print(
        "[BACKEND] Starting COLMAP sparse reconstruction..."
    )

    run_colmap(
        image_paths
    )

    print(
        "[BACKEND] COLMAP sparse reconstruction completed."
    )

    # =====================================================
    # COLMAP DENSE RECONSTRUCTION
    # =====================================================

    print(
        "[BACKEND] Starting AI Dense Reconstruction (CPU Mode)..."
    )

    from backends.ai_dense_fusion import run_ai_dense_fusion
    dense_ply_path = run_ai_dense_fusion()

    if not dense_ply_path:

        raise RuntimeError(
            "COLMAP dense reconstruction did not "
            "return a point-cloud path."
        )

    if not os.path.exists(
        dense_ply_path
    ):

        raise FileNotFoundError(
            "COLMAP dense point cloud was not found: "
            f"{dense_ply_path}"
        )

    print(
        "[BACKEND] COLMAP dense reconstruction completed."
    )

    print(
        f"[BACKEND] Dense point cloud: "
        f"{dense_ply_path}"
    )

    return dense_ply_path