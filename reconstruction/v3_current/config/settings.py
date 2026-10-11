"""
Global application settings.
"""

AUTO_REFRESH_INTERVAL = 7000

MAX_PREVIEW_IMAGES = 8

MAX_DISPLAY_POINTS = 100000

DISPLAY_OUTLIER_PERCENTILE = 95

POINT_MARKER_SIZE = 2

POINT_OPACITY = 0.8

PLOT_HEIGHT = 900

PLOT_BACKGROUND = "#0a0614"

SCENE_BACKGROUND = "#05030a"


# =========================================================
# RECONSTRUCTION QUALITY PRESETS
# =========================================================

PRESET_FAST = {
    "name": "Fast",
    "image_max_size": 1600,
    "sift_max_features": 8192,
    "sift_peak_threshold": 0.008,
    "sift_edge_threshold": 10,
    "pm_max_image_size": 1600,
    "pm_window_radius": 5,
    "pm_window_step": 1,
    "pm_num_samples": 7,
    "pm_num_iterations": 3,
    "pm_geom_consistency": False,
    "fusion_min_num_pixels": 4,
    "fusion_max_reproj_error": 2.0,
    "fusion_max_depth_error": 0.01,
    "fusion_check_num_images": 30,
    "outlier_std_ratio": 2.2,
    "outlier_nb_neighbors": 20,
    "poisson_depth": 7,
    "mesh_density_trim": 0.12,
    "meshing_method": "Poisson",
    "remove_floating_components": True,
    "matcher": "ml_ranked",
    "ml_pair_ranking": True,
    "ml_top_k": 6,
    "ml_min_pairs_per_image": 3,
    "ml_similarity_threshold": 0.0,
    "ml_ensure_connected": True,
}

PRESET_STANDARD = {
    "name": "Standard",
    "image_max_size": 2400,
    "sift_max_features": 10240,
    "sift_peak_threshold": 0.005,
    "sift_edge_threshold": 12,
    "pm_max_image_size": 2400,
    "pm_window_radius": 7,
    "pm_window_step": 1,
    "pm_num_samples": 12,
    "pm_num_iterations": 4,
    "pm_geom_consistency": True,
    "fusion_min_num_pixels": 3,
    "fusion_max_reproj_error": 2.0,
    "fusion_max_depth_error": 0.01,
    "fusion_check_num_images": 50,
    "outlier_std_ratio": 1.8,
    "outlier_nb_neighbors": 25,
    "poisson_depth": 8,
    "mesh_density_trim": 0.10,
    "meshing_method": "Poisson",
    "remove_floating_components": True,
    "matcher": "ml_ranked",
    "ml_pair_ranking": True,
    "ml_top_k": 8,
    "ml_min_pairs_per_image": 4,
    "ml_similarity_threshold": 0.0,
    "ml_ensure_connected": True,
}

PRESET_HIGH_DENSITY = {
    "name": "High Density",
    "image_max_size": 2800,  # Covers full 2736x1540 native resolution without exceeding memory limits
    "sift_max_features": 12288,
    "sift_peak_threshold": 0.004,
    "sift_edge_threshold": 12,
    "pm_max_image_size": 2800,
    "pm_window_radius": 7,
    "pm_window_step": 1,
    "pm_num_samples": 12,
    "pm_num_iterations": 4,
    "pm_geom_consistency": True,
    "fusion_min_num_pixels": 2,
    "fusion_max_reproj_error": 2.5,
    "fusion_max_depth_error": 0.015,
    "fusion_check_num_images": 60,
    "outlier_std_ratio": 1.5,
    "outlier_nb_neighbors": 30,
    "poisson_depth": 9,
    "mesh_density_trim": 0.01,
    "meshing_method": "Poisson",
    "remove_floating_components": True,
    "ml_geometry_enhancement": True,
    "ml_complete_blind_spots": True,
    "matcher": "ml_ranked",
    "ml_pair_ranking": True,
    "ml_top_k": 10,
    "ml_min_pairs_per_image": 5,
    "ml_similarity_threshold": 0.0,
    "ml_ensure_connected": True,
}

PRESET_NEURALANGELO = {
    "name": "Neuralangelo High Fidelity",
    "image_max_size": 4096,
    "requires_cuda": True,
    "min_vram_gb": 12.0,
    "description": "High-fidelity neural surface reconstruction via multi-resolution 3D hash grids and numerical gradients."
}

QUALITY_PRESETS = {
    "Fast": PRESET_FAST,
    "Standard": PRESET_STANDARD,
    "High Density": PRESET_HIGH_DENSITY,
    "Neuralangelo High Fidelity": PRESET_NEURALANGELO
}


def get_active_config(preset_name="Standard", overrides=None) -> dict:
    """
    Get merged configuration for the chosen preset with any user overrides.
    """
    base = dict(QUALITY_PRESETS.get(preset_name, PRESET_STANDARD))
    if overrides and isinstance(overrides, dict):
        base.update(overrides)
    return base