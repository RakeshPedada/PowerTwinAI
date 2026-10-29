"""
reconstruction/mast3r package
Official Naver MASt3R-SfM 3D Reconstruction Integration for PowerTwinAI.
"""

from .mast3r_model import (
    get_mast3r_model,
    detect_mast3r_hardware,
    release_mast3r_memory,
    is_mast3r_weights_cached,
    DEFAULT_MAST3R_MODEL,
    DEFAULT_RETRIEVAL_WEIGHTS,
    DEFAULT_RETRIEVAL_CODEBOOK
)
from .pairing import select_candidate_pairs
from .matching import match_image_pair_mast3r, filter_and_log_pairs_matching
from .alignment import run_mast3r_sparse_global_alignment
from .reconstruction import run_mast3r_reconstruction
from .visualization import (
    create_mast3r_point_cloud_figure,
    create_mast3r_mesh_figure,
    create_mast3r_camera_pointcloud_combined_figure,
    create_three_way_comparison_figure
)
from .utils import compare_three_backends

__all__ = [
    "run_mast3r_reconstruction",
    "get_mast3r_model",
    "detect_mast3r_hardware",
    "release_mast3r_memory",
    "is_mast3r_weights_cached",
    "select_candidate_pairs",
    "match_image_pair_mast3r",
    "filter_and_log_pairs_matching",
    "run_mast3r_sparse_global_alignment",
    "create_mast3r_point_cloud_figure",
    "create_mast3r_mesh_figure",
    "create_mast3r_camera_pointcloud_combined_figure",
    "create_three_way_comparison_figure",
    "compare_three_backends"
]
