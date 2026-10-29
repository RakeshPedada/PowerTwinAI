"""
reconstruction/vggt/__init__.py
-------------------------------
Public API for the independent VGGT-based 3D reconstruction pipeline.
"""

from .vggt_model import (
    get_vggt_model,
    detect_vggt_hardware,
    release_vggt_memory,
    is_vggt_weights_cached,
)
from .inference import (
    run_vggt_inference,
    select_image_subset,
    order_images_by_overlap,
)
from .reconstruction import (
    run_vggt_reconstruction,
    generate_vggt_surface_mesh,
    validate_vggt_mesh_ply,
    export_vggt_to_colmap_standard,
    rescale_camera_parameters_to_original,
    build_geometric_multiview_tracks,
    run_bundle_adjustment_refinement,
)
from .visualization import (
    create_vggt_point_cloud_figure,
    create_vggt_mesh_figure,
    create_vggt_camera_trajectory_figure,
    create_comparison_point_cloud_figure,
    save_html_debug_visualizations,
)
from .utils import (
    validate_image_files,
    compute_point_cloud_metrics,
    compare_colmap_and_vggt,
    load_vggt_summary,
    load_colmap_summary,
    run_dataset_diagnostics,
)

__all__ = [
    "get_vggt_model",
    "detect_vggt_hardware",
    "release_vggt_memory",
    "is_vggt_weights_cached",
    "run_vggt_inference",
    "select_image_subset",
    "order_images_by_overlap",
    "run_vggt_reconstruction",
    "generate_vggt_surface_mesh",
    "validate_vggt_mesh_ply",
    "export_vggt_to_colmap_standard",
    "rescale_camera_parameters_to_original",
    "build_geometric_multiview_tracks",
    "run_bundle_adjustment_refinement",
    "create_vggt_point_cloud_figure",
    "create_vggt_mesh_figure",
    "create_vggt_camera_trajectory_figure",
    "create_comparison_point_cloud_figure",
    "save_html_debug_visualizations",
    "validate_image_files",
    "compute_point_cloud_metrics",
    "compare_colmap_and_vggt",
    "load_vggt_summary",
    "load_colmap_summary",
    "run_dataset_diagnostics",
]
