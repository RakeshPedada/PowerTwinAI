"""
backends/colmap_backend.py
--------------------------
Unified interface for executing COLMAP sparse and dense reconstruction pipelines.
"""

import os
from run_colmap import run_colmap
from colmap.dense import run_colmap_dense


def run_colmap_backend(
    image_paths,
    config=None
):
    """
    Execute the complete COLMAP reconstruction backend with active configuration.
    """
    if not image_paths:
        raise ValueError("No image paths were provided to the COLMAP backend.")

    print("[BACKEND] Starting COLMAP sparse reconstruction...")
    run_colmap(
        image_paths,
        config=config
    )
    print("[BACKEND] COLMAP sparse reconstruction completed.")

    print("[BACKEND] Starting COLMAP dense reconstruction...")
    dense_ply_path = run_colmap_dense(
        config=config
    )

    if not dense_ply_path:
        raise RuntimeError("COLMAP dense reconstruction did not return a point-cloud path.")

    if not os.path.exists(dense_ply_path):
        raise FileNotFoundError(f"COLMAP dense point cloud not found: {dense_ply_path}")

    return dense_ply_path