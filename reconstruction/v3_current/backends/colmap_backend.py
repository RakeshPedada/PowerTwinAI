"""
COLMAP Reconstruction Backend

Objective
---------
Provide a clean backend-level interface for executing the
COLMAP reconstruction pipeline.

Responsibilities
----------------
- Accept prepared image paths.
- Execute the existing COLMAP pipeline.
- Keep COLMAP-specific execution details isolated from
  the rest of PowerTwinAI.

The actual COLMAP commands remain implemented in
run_colmap.py. This module acts as the backend adapter.

Current Backend
---------------
COLMAP

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
COLMAP TXT Export

Future
------
COLMAP Dense reconstruction will be integrated into this
backend for PowerTwinAI v1.0.

Part of
-------
PowerTwinAI Phase 6 Architecture
"""

from run_colmap import run_colmap


def run_colmap_backend(image_paths):
    """
    Execute the COLMAP reconstruction backend.

    Parameters
    ----------
    image_paths : list[str]
        Paths to the prepared input images.

    Returns
    -------
    None
        The current run_colmap() implementation writes its
        outputs to the configured COLMAP workspace and
        colmap_data directory.
    """

    if not image_paths:

        raise ValueError(
            "No image paths were provided to the COLMAP backend."
        )

    print(
        "[BACKEND] Starting COLMAP backend..."
    )

    run_colmap(
        image_paths
    )

    print(
        "[BACKEND] COLMAP backend completed."
    )